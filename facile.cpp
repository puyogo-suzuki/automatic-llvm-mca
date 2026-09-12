#include "facile.h"
#include "mca_common.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <queue>
#include <set>
#include <vector>

namespace facile {

namespace {

struct DependencyEdge {
    size_t Target;
    double Latency;
    unsigned Distance; // 0 for intra-iteration, 1 for inter-iteration (loop-carried)
};

// Extract instruction latency with a minimum threshold of 1.0 cycle
double getInstLatency(const llvm::mca::Instruction &Inst) {
    double Lat = static_cast<double>(Inst.getLatency());
    return std::max(Lat, 1.0);
}

// ---------------------------------------------------------------------------
// Latency to put on a RAW dependency edge  (Writer defines Reg) -> (Reader
// reads Reg through use slot Use).
//
// Refines two coarse approximations that the plain getInstLatency(Writer)
// form made.  Both are cases where the scheduling model already carries the
// right number and facile was simply not reading it.
//
// (a) PER-DEFINITION latency.  mca::Instruction::getLatency() is
//     InstrDesc::MaxLatency, i.e. the MAXIMUM over all of an instruction's
//     writes, so every out-edge of a multi-def instruction was charged the
//     slowest one.  AArch64 pre/post-index memory ops are exactly that shape:
//     `ldr x2, [x1], #8` is Sched<[WriteAdr, WriteLD]> - 1c for the x1 base
//     update, 4c for the x2 load result - so the base-pointer recurrence that
//     every post-increment / stride loop carries (x1 -> x1 across the
//     backedge, distance 1) was being given the LOAD's 4c and produced a
//     precedence bound of 4 cycles/iteration for a loop whose address
//     recurrence really closes in 1.  WriteState::getLatency() is that
//     individual write's own latency.
//
// (b) READADVANCE (operand bypass / late forwarding).  A scheduling model
//     states, per (consumer sched class, consumer operand slot, producer
//     write-resource), how many cycles earlier than write-back that operand
//     is actually needed.  MCA's own RegisterFile::addRegisterRead applies it
//     via WriteState::addUser(IID, Use, ReadAdvance), giving an effective
//     dependence latency of WS.getLatency() - ReadAdvance.  facile never
//     consulted it, so all operands of a multi-input instruction were assumed
//     to need the producer's result at the same (worst) cycle - the textbook
//     way to over-predict a multiply-accumulate recurrence.
//
//     The case that actually matters here is the ACCUMULATOR of a
//     multiply-accumulate.  Arm Cortex-A78 Software Optimization Guide
//     (r1p2, PJDOC-466751330-9691) Table 3-24 lists
//        "FP multiply accumulate | FMADD, FMSUB, FNMADD, FNMSUB |
//         Execution Latency 4 (2) | Throughput 2 | Pipeline V"
//     with note 3: "FP multiply-accumulate pipelines support late-forwarding
//     of accumulate operands from similar uOPs, allowing a typical sequence
//     of multiply-accumulate uOPs to issue one every N cycles (accumulate
//     latency N shown in parentheses)."  Table 3-7 likewise gives
//     "Multiply accumulate, W-form / X-form | MADD, MSUB | 2(1)".
//     AArch64SchedNeoverseN2.td (the model installed for cortex-a78) encodes
//     precisely that:
//        def N2Wr_FMA : SchedWriteRes<[N2UnitV]> { let Latency = 4; }
//        def N2Rd_FMA : SchedReadAdvance<2, [WriteFMul, N2Wr_FMA]>;
//        def : InstRW<[N2Wr_FMA, ReadDefault, ReadDefault, N2Rd_FMA],
//                     (instregex "^FN?M(ADD|SUB)[HSD]rrr$")>;
//     (AArch64SchedNeoverseN1.td, used for cortex-a76, is identical.)  So an
//     FMADD -> FMADD accumulator recurrence - the shape of every FP
//     reduction, dot product and stencil accumulation in SPEC FP - really
//     closes in 2 cycles, and facile was calling it 4: a clean 2x
//     over-prediction of the precedence bound on the exact loops where the
//     precedence bound is the binding constraint.
//
//     Apple's two models declare ReadAdvance 0 everywhere, so (b) is a no-op
//     for icestorm/firestorm today; it is what makes it POSSIBLE to encode
//     the forwarding paths Dougall Johnson documents for them ("MADD's output
//     can be passed to its third operand (the addend) with 1c latency, but if
//     it's chained with other instructions it has 3c latency", "Loads may be
//     passed to the base address of other loads with 3c latency ... but
//     chaining with ALU operations gives a latency of 4c"), which are
//     per-operand facts that no WriteRes latency can express.
//
// The 1-cycle floor is kept from getInstLatency(): a dependent operation can
// never issue in the same cycle as its producer in this model, and MCA's own
// zero-latency writes (register-renamed MOVs) already relied on it.
// ---------------------------------------------------------------------------
double getEdgeLatency(const llvm::MCSubtargetInfo &STI,
                      const llvm::mca::Instruction &Writer,
                      const llvm::mca::ReadState &Use,
                      unsigned Reg) {
    const llvm::mca::WriteState *WS = nullptr;
    for (const llvm::mca::WriteState &W : Writer.getDefs()) {
        if (W.getRegisterID() == Reg) {
            WS = &W;
            break;
        }
    }
    if (!WS) // implicit/elided def: fall back to the instruction-wide latency
        return getInstLatency(Writer);

    double Lat = static_cast<double>(WS->getLatency());

    const llvm::MCSchedModel &SM = STI.getSchedModel();
    const llvm::mca::ReadDescriptor &RD = Use.getDescriptor();
    if (const llvm::MCSchedClassDesc *SC = SM.getSchedClassDesc(RD.SchedClassID)) {
        // Variant classes carry no ReadAdvance entries of their own; MCA's
        // InstrBuilder has already stored the RESOLVED class id in RD, so a
        // still-variant descriptor here means the model has nothing to say.
        if (!SC->isVariant())
            Lat -= static_cast<double>(
                STI.getReadAdvanceCycles(SC, RD.UseIndex, WS->getWriteResourceID()));
    }

    return std::max(Lat, 1.0);
}

// Resolve dynamic variant scheduling classes via subtarget info and MCInst if available
const llvm::MCSchedClassDesc *resolveSchedClass(const llvm::MCSubtargetInfo &STI,
                                                const llvm::MCInstrInfo &MCII,
                                                unsigned SchedClass,
                                                const llvm::MCInst *MCI) {
    const llvm::MCSchedModel &SM = STI.getSchedModel();
    const llvm::MCSchedClassDesc *SCDesc = SM.getSchedClassDesc(SchedClass);
    unsigned PrevSchedClass = SchedClass;
    unsigned CurrSchedClass = SchedClass;

    while (SCDesc && SCDesc->isVariant()) {
        if (MCI) {
            CurrSchedClass = STI.resolveVariantSchedClass(CurrSchedClass, MCI, &MCII, SM.getProcessorID());
        }
        if (CurrSchedClass == PrevSchedClass) break;
        PrevSchedClass = CurrSchedClass;
        SCDesc = SM.getSchedClassDesc(CurrSchedClass);
    }
    return SCDesc;
}

// Cortex-A78 Software Optimization Guide sec. 4.14 "Instruction fusion":
// specific adjacent AArch64 instruction pairs execute as a single fused
// operation on Cortex-A78 ("These instruction pairs must be adjacent to each
// other in program code"). The second instruction of a fused pair consumes
// no additional issue-width or execution-port resource. Confirmed this is
// A78-specific: the Cortex-A76 SOG documents no equivalent CMP/CSEL or
// CMP/B.cond fusion (only unrelated FP fused-multiply-accumulate), so this
// must not be applied there.
//   1. CMP/CMN (immediate) + B.cond      6. CMP (register) + CSET
//   2. CMP/CMN (register) + B.cond       7. TST (immediate) + B.cond
//   3. CMP (immediate) + CSEL            8. TST (register) + B.cond
//   4. CMP (register) + CSEL             9. BICS (register) + B.cond
//   5. CMP (immediate) + CSET            10. NOP + Any instruction
// CMP/CMN/TST are flag-setting SUBS/ADDS/ANDS with a discarded (WZR/XZR)
// destination; only those forms qualify, not general flag-setting ops with
// a real destination register.
bool isA78FusionCandidate(const llvm::MCSubtargetInfo &STI) {
    return STI.getCPU() == "cortex-a78" || STI.getCPU() == "cortex-a78ae" || STI.getCPU() == "cortex-a78c";
}

// mca::Instruction::getDefs() elides writes to the discarded zero register
// (nothing can ever depend on WZR/XZR's value, so the register-renaming
// dependency tracker doesn't bother recording it) -- so whether a
// SUBS/ADDS/ANDS is a true CMP/CMN/TST (destination discarded) vs. a general
// flag-setting op with a real destination can only be checked on the raw
// MCInst operand list, not on the mca::Instruction.
bool writesZeroReg(const llvm::MCInst *MCI) {
    if (!MCI || MCI->getNumOperands() == 0 || !MCI->getOperand(0).isReg())
        return false;
    llvm::MCRegister R = MCI->getOperand(0).getReg();
    return R == llvm::AArch64::WZR || R == llvm::AArch64::XZR;
}

bool isA78FusibleProducer(const llvm::MCInstrInfo &MCII, const llvm::mca::Instruction &Inst,
                          const llvm::MCInst *MCI) {
    llvm::StringRef Name = MCII.getName(Inst.getOpcode());
    bool IsCmpCmnTst = Name.starts_with_insensitive("SUBS") || Name.starts_with_insensitive("ADDS") ||
                        Name.starts_with_insensitive("ANDS");
    bool IsBics = Name.starts_with_insensitive("BICS");
    if (IsBics) return true; // BICS(register)+B.cond: always the register form
    if (!IsCmpCmnTst) return false;
    return writesZeroReg(MCI);
}

bool isA78FusibleConsumer(const llvm::MCInstrInfo &MCII, const llvm::mca::Instruction &Inst) {
    llvm::StringRef Name = MCII.getName(Inst.getOpcode());
    // CSET is an alias for CSINC Wd, WZR, WZR, invert(cond) -- detecting the
    // alias precisely would need operand inspection; only the plain CSEL
    // form (which this codebase can verify directly) is matched here.
    return Name.equals_insensitive("Bcc") || Name.starts_with_insensitive("CSEL");
}

// Returns a mask, one entry per instruction, true if that instruction is the
// *second* half of an adjacent fused pair (and should be excluded from
// issue-width/port-resource accounting).
std::vector<bool> computeA78FusedMask(const llvm::MCSubtargetInfo &STI,
                                      const llvm::MCInstrInfo &MCII,
                                      llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                                      llvm::ArrayRef<const llvm::MCInst *> MCInsts) {
    std::vector<bool> Fused(SimInstrs.size(), false);
    if (!isA78FusionCandidate(STI) || SimInstrs.size() < 2)
        return Fused;
    for (size_t i = 0; i + 1 < SimInstrs.size(); ++i) {
        const llvm::MCInst *MCI = (i < MCInsts.size()) ? MCInsts[i] : nullptr;
        if (isA78FusibleProducer(MCII, *SimInstrs[i], MCI) && isA78FusibleConsumer(MCII, *SimInstrs[i + 1]))
            Fused[i + 1] = true;
    }
    return Fused;
}

// 1. Calculate Dispatch / Issue Width Limit
double calculateIssueBound(const llvm::MCSchedModel &SM,
                            llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                            unsigned &TotalUops,
                            unsigned DispatchWidthOverride,
                            const std::vector<bool> &FusedMask) {
    unsigned IssueWidth = DispatchWidthOverride > 0 ? DispatchWidthOverride
                         : (SM.IssueWidth > 0 ? SM.IssueWidth : 1);
    TotalUops = 0;
    for (size_t i = 0; i < SimInstrs.size(); ++i) {
        if (i < FusedMask.size() && FusedMask[i])
            continue;
        unsigned NumUops = SimInstrs[i]->getNumMicroOps();
        TotalUops += (NumUops > 0 ? NumUops : 1);
    }
    return static_cast<double>(TotalUops) / static_cast<double>(IssueWidth);
}

// 2. Calculate Execution Ports Contention Bound
double calculatePortUsageBound(const llvm::MCSubtargetInfo &STI,
                               const llvm::MCInstrInfo &MCII,
                               llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                               llvm::ArrayRef<const llvm::MCInst *> MCInsts,
                               std::string &BottleneckPortName,
                               const std::vector<bool> &FusedMask) {
    const llvm::MCSchedModel &SM = STI.getSchedModel();
    unsigned NumProcResources = SM.NumProcResourceKinds;
    std::vector<double> ProcResUsage(NumProcResources, 0.0);

    for (size_t i = 0; i < SimInstrs.size(); ++i) {
        if (i < FusedMask.size() && FusedMask[i])
            continue;
        const auto &Inst = SimInstrs[i];
        const llvm::MCInst *MCI = (i < MCInsts.size()) ? MCInsts[i] : nullptr;
        const llvm::MCInstrDesc &MCID = MCII.get(Inst->getOpcode());
        
        const llvm::MCSchedClassDesc *SCDesc = resolveSchedClass(STI, MCII, MCID.getSchedClass(), MCI);
        if (!SCDesc) continue;

        for (const llvm::MCWriteProcResEntry *WPR = STI.getWriteProcResBegin(SCDesc);
             WPR != STI.getWriteProcResEnd(SCDesc); ++WPR) {
            unsigned ProcResIdx = WPR->ProcResourceIdx;
            unsigned Cycles = WPR->ReleaseAtCycle - WPR->AcquireAtCycle;
            if (Cycles == 0) continue; // Skip entries that consume 0 resource cycles
            if (ProcResIdx < NumProcResources) {
                ProcResUsage[ProcResIdx] += Cycles;
            }
        }
    }

    double MaxPortBound = 0.0;
    BottleneckPortName = "None";

    for (unsigned r = 1; r < NumProcResources; ++r) {
        const llvm::MCProcResourceDesc *PRD = SM.getProcResource(r);
        if (!PRD || PRD->NumUnits == 0) continue;
        double PortCycles = ProcResUsage[r] / static_cast<double>(PRD->NumUnits);
        if (PortCycles > MaxPortBound) {
            MaxPortBound = PortCycles;
            BottleneckPortName = PRD->Name ? PRD->Name : ("Port" + std::to_string(r));
        }
    }

    return MaxPortBound;
}

// 3. Build Read-After-Write (RAW) Register Dependency Graph
std::vector<std::vector<DependencyEdge>> buildDependencyGraph(
    const llvm::MCSubtargetInfo &STI,
    llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs) {

    size_t N = SimInstrs.size();
    std::vector<std::vector<DependencyEdge>> Adj(N);
    std::map<unsigned, size_t> LastWriter;
    // Instruction index + the index of the USE SLOT within that instruction's
    // getUses(), so that the loop-carried edge built in Pass 2 can be given
    // the same per-operand ReadAdvance treatment as the intra-iteration ones.
    std::map<unsigned, std::pair<size_t, size_t>> FirstReader;
    std::set<unsigned> DefinedRegs;

    // Pass 1: Intra-iteration dependencies & Live-In reads
    for (size_t i = 0; i < N; ++i) {
        const auto &Inst = SimInstrs[i];

        size_t UseSlot = 0;
        for (const auto &Op : Inst->getUses()) {
            unsigned Reg = Op.getRegisterID();
            size_t ThisSlot = UseSlot++;
            if (Reg == 0) continue;

            if (DefinedRegs.find(Reg) == DefinedRegs.end()) {
                if (FirstReader.find(Reg) == FirstReader.end()) {
                    FirstReader[Reg] = {i, ThisSlot};
                }
            }

            auto it = LastWriter.find(Reg);
            if (it != LastWriter.end()) {
                size_t WriterIdx = it->second;
                if (WriterIdx < i) {
                    double Lat = getEdgeLatency(STI, *SimInstrs[WriterIdx], Op, Reg);
                    Adj[WriterIdx].push_back({i, Lat, 0});
                }
            }
        }

        for (const auto &Op : Inst->getDefs()) {
            unsigned Reg = Op.getRegisterID();
            if (Reg != 0) {
                LastWriter[Reg] = i;
                DefinedRegs.insert(Reg);
            }
        }
    }

    // Pass 2: Loop-carried dependencies (LastWriter -> FirstReader across iterations)
    for (const auto &entry : FirstReader) {
        unsigned Reg = entry.first;
        size_t ReaderIdx = entry.second.first;
        size_t ReaderSlot = entry.second.second;
        auto it = LastWriter.find(Reg);
        if (it != LastWriter.end()) {
            size_t WriterIdx = it->second;
            const auto &Uses = SimInstrs[ReaderIdx]->getUses();
            double Lat = (ReaderSlot < Uses.size())
                             ? getEdgeLatency(STI, *SimInstrs[WriterIdx], Uses[ReaderSlot], Reg)
                             : getInstLatency(*SimInstrs[WriterIdx]);
            Adj[WriterIdx].push_back({ReaderIdx, Lat, 1});
        }
    }

    return Adj;
}

// ---------------------------------------------------------------------------
// Memory (store -> load) RAW dependences.
//
// WHY THIS EXISTS.  buildDependencyGraph() above tracks only REGISTER RAW
// dependences, and it deliberately extends the published Facile model by
// adding *inter-iteration* (loop-carried) register edges in its Pass 2, so
// that calculatePrecedenceBound() computes a Maximum-Cycle-Ratio recurrence
// bound on steady-state loop throughput rather than a single basic block's
// critical path.  Once loop-carried register recurrences are modelled,
// omitting loop-carried MEMORY recurrences is an inconsistency: a value that
// round-trips through memory (store in iteration k, load in iteration k+1) is
// just as much a recurrence, and bounds throughput just as hard.  Facile as
// published assumes basic blocks are compute-bound and that "loads and stores
// may or may not alias" is unknowable (sec. 3.3), which is defensible for
// single-block throughput but not for the loop-recurrence bound this tool
// actually computes.
//
// WHY IT MATTERS ASYMMETRICALLY, AND WHY IT SHOWED UP AS A FireStorm-ONLY
// ERROR.  EstimatedCycles = max(IssueBound, PortBound, PrecedenceBound).  A
// missing precedence bound is invisible on a narrow core, because IssueBound
// (uops / issue width) is larger anyway and still wins the max; it becomes
// visible exactly on the WIDEST core of a pair, whose IssueBound is smallest.
// 456.hmmer's P7Viterbi inner loop is the textbook case.  The block at
// 0xcc2c..0xcd18 (59 instructions, 17 loads, 10 stores) carries the serial
// HMMER D-state recurrence
//     str w2, [x5, x0, lsl #2]   ; dc[k]        (iteration k)
//     ...
//     ldr w3, [x5, x1]           ; dc[k-1]      (iteration k+1)
//     add w3, w3, w2  /  cmp  /  csel  /  cmp  /  csel  /  str
// i.e. a memory round trip plus a 5-deep flag/select chain, which no register
// edge can see because the value never stays in a register across the
// backedge.  Measured on real M1 (macbook/456.hmmer): IceStorm needs 15.9
// cycles for the block and its IssueBound of 59/4 = 14.75 already covers that,
// so IceStorm's prediction (15) is accidentally right.  FireStorm needs 11.7
// cycles but its IssueBound is only 59/8 = 7.375, so the model predicted 7 --
// and the missing cycles are precisely the ones that show up in FireStorm's
// MAP_STALL_DISPATCH counter (30% of its cycles on hmmer, versus 4.5% on
// IceStorm, the largest such ratio in the 24-benchmark suite): the mapper
// cannot dispatch because the scheduler is backed up behind a recurrence the
// model does not know about.  Dougall Johnson's measured port throughputs
// cannot explain that gap and were confirmed innocent: on FireStorm every
// resource bound for this block is at or below the issue bound (LDR TP 0.333
// on u8-10 -> 17/3 = 5.67; STR TP 0.5 on u7/8 -> 10/2 = 5; CMP/CSEL TP 0.333
// on u1-3 -> 18/3 = 6; ADD TP 0.167 on u1-6 -> 12/6 = 2).  The bottleneck is
// a dependence, not a port.
//
// MAY-ALIAS RULE.  Deliberately the same conservative-but-cheap rule the MLP
// analyser's SeenStoreAddrs already uses, with no new tunable:
//   * different base register            -> assume no alias (the same
//     AssumeNoAlias-style assumption the rest of the tool makes; without it
//     every store would fence every load and the bound would explode);
//   * same base register, BOTH accesses constant-offset -> alias iff the
//     offsets are equal (this is what keeps stack spill/reload traffic and
//     struct-field access from generating false edges);
//   * same base register, at least one register-indexed (LDRWroX / STRWroX,
//     i.e. an a[i]-style access) -> may alias.  hmmer's recurrence is exactly
//     this case: str [x5, x0, lsl #2] vs. ldr [x5, x1].
//   * a base register redefined between the two accesses breaks the match,
//     since the register no longer denotes the same address; for a
//     loop-carried edge the base must be loop-invariant (never redefined in
//     the region).  This mirrors SeenStoreAddrs::reset().
//
// EDGE LATENCY.  The producer's own latency, exactly as the register RAW path
// does (getInstLatency of the store, i.e. the .td WriteST latency = 1 cycle on
// both M1 models).  The load's own latency then applies on the load's outgoing
// edge, so a store->load round trip costs WriteST + WriteLD = 1 + 3 = 4
// cycles, i.e. store-to-load forwarding is modelled as no slower than an L1
// hit.  That is the optimistic end of the plausible range and introduces NO
// new constant; a larger, separately-measured store-forwarding latency would
// only raise the bound further, so this errs toward under- rather than
// over-prediction.
// ---------------------------------------------------------------------------

// Index register of an AArch64 register-offset access (LDRWroX / STRWroX and
// friends), or 0 for constant-offset forms.  getMemAccessInfo() records only
// the base register for these forms, but distinguishing a[i] from a[j] needs
// the index too -- see sameAddressExpr() below.  Mirrors mlp_logic.cpp's
// register-offset detection (operand 1 = base, operand 2 = index).
unsigned getIndexReg(const llvm::MCInstrInfo &MCII, const llvm::MCInst *MCI) {
    if (!MCI) return 0;
    llvm::StringRef Name = MCII.getName(MCI->getOpcode());
    if (Name.find_insensitive("ro") == llvm::StringRef::npos) return 0;
    if (MCI->getNumOperands() < 3 || !MCI->getOperand(2).isReg()) return 0;
    return MCI->getOperand(2).getReg();
}

// Do two accesses compute the SAME address expression (same base, same index
// register, same displacement)?  This is the classic dependence-distance
// (ZIV/SIV) test: if two accesses in a loop have identical subscript
// expressions and that subscript advances every iteration, then the
// dependence distance between them is 0 -- they touch the same address WITHIN
// an iteration and can never touch the same address ACROSS iterations.  The
// canonical case is an array read-modify-write,
//     ldr w0, [x1, x2, lsl #2]   ; t = a[i]
//     ...
//     str w0, [x1, x2, lsl #2]   ; a[i] = f(t)
// where the next iteration reads a[i+1], which the previous iteration's store
// to a[i] did not write.  Such a pair must NOT get a loop-carried edge.
// hmmer's real recurrence is precisely the opposite case -- the store indexes
// with x0 (k) and the load with x1 (4*(k-1)), so the expressions differ, a
// one-iteration lag cannot be ruled out, and the edge is kept.
bool sameAddressExpr(const MemAccessInfo &A, unsigned IdxA,
                     const MemAccessInfo &B, unsigned IdxB) {
    return A.base_reg == B.base_reg && A.offset == B.offset &&
           A.is_constant_offset() == B.is_constant_offset() && IdxA == IdxB;
}

bool mayAlias(const MemAccessInfo &A, const MemAccessInfo &B) {
    if (!A.valid() || !B.valid()) return false;
    if (A.is_pc_relative() || B.is_pc_relative()) return false;
    if (A.base_reg == 0 || B.base_reg == 0) return false;
    if (A.base_reg != B.base_reg) return false;
    if (A.is_constant_offset() && B.is_constant_offset())
        return A.offset == B.offset;
    return true;
}

// Does instruction `i` write a register overlapping `BaseReg`?  Uses the raw
// MCInst/MCInstrDesc rather than mca::Instruction::getDefs() so that
// sub-register writes (w4 redefining x4) and writeback base updates are both
// caught via MCRegisterInfo::regsOverlap.
bool definesReg(const llvm::MCInstrInfo &MCII, const llvm::MCRegisterInfo &MRI,
                const llvm::MCInst *MCI, unsigned BaseReg) {
    if (!MCI) return true; // unknown instruction: assume it clobbers
    const llvm::MCInstrDesc &MCID = MCII.get(MCI->getOpcode());
    for (unsigned k = 0, e = MCID.getNumDefs(); k < e && k < MCI->getNumOperands(); ++k) {
        const llvm::MCOperand &Op = MCI->getOperand(k);
        if (Op.isReg() && Op.getReg() != 0 && MRI.regsOverlap(Op.getReg(), BaseReg))
            return true;
    }
    for (llvm::MCPhysReg R : MCID.implicit_defs()) {
        if (MRI.regsOverlap(R, BaseReg)) return true;
    }
    return false;
}

void addMemoryDependencies(const llvm::MCInstrInfo &MCII,
                           const llvm::MCRegisterInfo &MRI,
                           llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                           llvm::ArrayRef<const llvm::MCInst *> MCInsts,
                           llvm::ArrayRef<MemAccessInfo> MemInfos,
                           std::vector<std::vector<DependencyEdge>> &Adj) {
    size_t N = SimInstrs.size();
    if (MemInfos.size() < N || MCInsts.size() < N) return;

    bool HasStore = false, HasLoad = false;
    for (size_t i = 0; i < N && !(HasStore && HasLoad); ++i) {
        const MemAccessInfo &M = MemInfos[i];
        if (!M.valid() || M.is_pc_relative() || M.base_reg == 0) continue;
        HasStore |= M.is_store();
        HasLoad |= M.is_load();
    }
    if (!HasStore || !HasLoad) return;

    std::vector<unsigned> IdxRegs(N, 0);
    for (size_t i = 0; i < N; ++i)
        IdxRegs[i] = getIndexReg(MCII, MCInsts[i]);

    // Prefix counts of register redefinitions, memoized per register, so the
    // "does this register still hold the same value" test is O(1) per pair.
    // Redef[r][i] = number of instructions at index < i writing a register
    // overlapping r.  Always reached through countRedefs() so a missing row is
    // built rather than default-constructed empty.
    std::map<unsigned, std::vector<unsigned>> Redef;
    auto countRedefs = [&](unsigned R) -> const std::vector<unsigned> & {
        auto It = Redef.find(R);
        if (It != Redef.end()) return It->second;
        std::vector<unsigned> &P = Redef[R];
        P.assign(N + 1, 0);
        for (size_t i = 0; i < N; ++i)
            P[i + 1] = P[i] + (definesReg(MCII, MRI, MCInsts[i], R) ? 1 : 0);
        return P;
    };

    // Register unchanged strictly between indices lo and hi (exclusive both ends).
    auto stableBetween = [&](unsigned R, size_t lo, size_t hi) {
        if (hi <= lo + 1) return true;
        const std::vector<unsigned> &P = countRedefs(R);
        return P[hi] == P[lo + 1];
    };
    auto loopInvariant = [&](unsigned R) { return countRedefs(R)[N] == 0; };

    for (size_t j = 0; j < N; ++j) {
        const MemAccessInfo &L = MemInfos[j];
        if (!L.valid() || !L.is_load() || L.is_pc_relative() || L.base_reg == 0)
            continue;

        // Pass 1: nearest preceding aliasing store in the same iteration.
        bool FoundIntra = false;
        for (size_t i = j; i-- > 0;) {
            const MemAccessInfo &S = MemInfos[i];
            if (!S.is_store() || !mayAlias(S, L)) continue;
            if (!stableBetween(L.base_reg, i, j)) break; // base changed: earlier matches are stale
            Adj[i].push_back({j, getInstLatency(*SimInstrs[i]), 0});
            FoundIntra = true;
            break;
        }
        if (FoundIntra) continue;

        // Pass 2: loop-carried.  A load with no preceding aliasing store in the
        // region reads a value that, in steady state, the region's LAST
        // aliasing store produced in the previous iteration -- the memory
        // analogue of the LastWriter -> FirstReader edge in
        // buildDependencyGraph()'s Pass 2.  Requires a loop-invariant base.
        if (!loopInvariant(L.base_reg)) continue;
        for (size_t i = N; i-- > 0;) {
            const MemAccessInfo &S = MemInfos[i];
            if (!S.is_store() || !mayAlias(S, L)) continue;
            // Dependence-distance test (see sameAddressExpr): identical
            // subscript expressions with a loop-varying index cannot alias
            // across iterations, so no loop-carried edge.
            if (sameAddressExpr(S, IdxRegs[i], L, IdxRegs[j]) &&
                IdxRegs[j] != 0 && !loopInvariant(IdxRegs[j]))
                break;
            Adj[i].push_back({j, getInstLatency(*SimInstrs[i]), 1});
            break;
        }
    }
}

// Check if a cycle ratio mu is achievable using SPFA negative cycle detection on transformed weights: c = mu * distance - latency
bool isAchievableRatio(double mu,
                       size_t N,
                       const std::vector<std::vector<DependencyEdge>> &Adj) {
    std::vector<double> dist(N, 0.0);
    std::vector<unsigned> count(N, 0);
    std::vector<bool> inQueue(N, true);
    std::queue<size_t> q;
    for (size_t i = 0; i < N; ++i) {
        q.push(i);
    }

    size_t maxOps = N * std::max<size_t>(10, N);
    size_t ops = 0;

    while (!q.empty()) {
        size_t u = q.front();
        q.pop();
        inQueue[u] = false;

        if (++ops > maxOps) {
            return true;
        }

        for (const auto &E : Adj[u]) {
            double weight = mu * E.Distance - E.Latency;
            if (dist[u] + weight < dist[E.Target] - 1e-9) {
                dist[E.Target] = dist[u] + weight;
                count[E.Target] = count[u] + 1;
                if (count[E.Target] >= N) {
                    return true;
                }
                if (!inQueue[E.Target]) {
                    q.push(E.Target);
                    inQueue[E.Target] = true;
                }
            }
        }
    }

    return false;
}

// 4. Calculate Maximum Cycle Ratio (MCR) for Precedence Constraints
double calculatePrecedenceBound(
    size_t N,
    const std::vector<std::vector<DependencyEdge>> &Adj) {
    
    if (N == 0) return 0.0;

    double MaxRatio = 0.0;

    // Fast check for self-loops (e.g., ADD X0, X0, #1)
    for (size_t u = 0; u < N; ++u) {
        for (const auto &E : Adj[u]) {
            if (E.Target == u && E.Distance > 0) {
                double Ratio = E.Latency / E.Distance;
                if (Ratio > MaxRatio) MaxRatio = Ratio;
            }
        }
    }

    // Determine upper bound for binary search.
    // Any cycle includes intra-iteration edges and at least one inter-iteration edge.
    // Therefore MCR <= sum of all edge latencies in the graph.
    double maxLatencySum = 0.0;
    for (size_t u = 0; u < N; ++u) {
        for (const auto &E : Adj[u]) {
            if (E.Latency > 0.0) {
                maxLatencySum += E.Latency;
            }
        }
    }

    if (maxLatencySum <= 0.0) {
        return MaxRatio;
    }

    double low = MaxRatio;
    double high = std::max({MaxRatio + 1.0, maxLatencySum, 1.0});

    // Binary search for exact MCR
    for (int iter = 0; iter < 30; ++iter) {
        double mid = low + (high - low) / 2.0;
        if (isAchievableRatio(mid, N, Adj)) {
            low = mid;
        } else {
            high = mid;
        }
    }

    return low;
}

// 5. Determine Dominant Bottleneck Category Name
std::string determineDominantBottleneck(double EstimatedCycles,
                                        double PrecedenceBound,
                                        double PortBound,
                                        const std::string &PortBottleneckName) {
    if (EstimatedCycles == PrecedenceBound && PrecedenceBound > 0.0) {
        return "Precedence Constraints (Dependency Chain)";
    }
    if (EstimatedCycles == PortBound && PortBound > 0.0) {
        return "Execution Ports (" + PortBottleneckName + ")";
    }
    return "Issue Width (Dispatch Limit)";
}

} // namespace

FacileResult computeFacilePrediction(const llvm::MCSubtargetInfo &STI,
                                     const llvm::MCInstrInfo &MCII,
                                     const llvm::MCRegisterInfo &MRI,
                                     llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                                     llvm::ArrayRef<const llvm::MCInst *> MCInsts,
                                     unsigned DispatchWidth,
                                     llvm::ArrayRef<MemAccessInfo> MemInfos) {
    FacileResult Res;
    if (SimInstrs.empty()) return Res;

    Res.TotalInstructions = SimInstrs.size();

    // For Coalesced ROB CPUs (Apple Firestorm / Icestorm), normalize NumMicroOps to 1 (MOP-based)
    if (isCoalescedROBCPU(STI.getCPU())) {
        for (const auto &Inst : SimInstrs) {
            llvm::mca::InstrDesc &MutableDesc = const_cast<llvm::mca::InstrDesc &>(Inst->getDesc());
            MutableDesc.NumMicroOps = 1;
        }
    }

    std::vector<bool> FusedMask = computeA78FusedMask(STI, MCII, SimInstrs, MCInsts);

    // 1. Issue Limit
    Res.IssueBound = calculateIssueBound(STI.getSchedModel(), SimInstrs, Res.TotalMicroOps, DispatchWidth, FusedMask);

    // 2. Execution Ports Limit
    Res.PortBound = calculatePortUsageBound(STI, MCII, SimInstrs, MCInsts, Res.PortBottleneckName, FusedMask);

    // 3. Precedence Constraints Limit
    auto Adj = buildDependencyGraph(STI, SimInstrs);
    if (!MemInfos.empty())
        addMemoryDependencies(MCII, MRI, SimInstrs, MCInsts, MemInfos, Adj);
    Res.PrecedenceBound = calculatePrecedenceBound(SimInstrs.size(), Adj);

    // 4. Overall Max Bottleneck Prediction
    Res.EstimatedCycles = std::max({Res.IssueBound, Res.PortBound, Res.PrecedenceBound});
    Res.EstimatedCPI = Res.EstimatedCycles / static_cast<double>(Res.TotalInstructions);
    Res.DominantBottleneck = determineDominantBottleneck(Res.EstimatedCycles, Res.PrecedenceBound, Res.PortBound, Res.PortBottleneckName);

    if (Res.EstimatedCycles == Res.PrecedenceBound && Res.PrecedenceBound > 0.0) {
        Res.FacileReason = "prec";
    } else if (Res.EstimatedCycles == Res.PortBound && Res.PortBound > Res.IssueBound) {
        Res.FacileReason = "exec";
    } else {
        Res.FacileReason = "inst";
    }

    return Res;
}

void printFacileResult(const FacileResult &Res, llvm::StringRef CPUName, llvm::raw_ostream &OS) {
    OS << "==================================================\n";
    OS << "Facile Static Analytical Throughput Prediction (AArch64)\n";
    OS << "==================================================\n";
    OS << "Target CPU:             " << CPUName << "\n";
    OS << "Total Instructions:     " << Res.TotalInstructions << "\n";
    OS << "Total MicroOps:         " << Res.TotalMicroOps << "\n";
    OS << "--------------------------------------------------\n";
    OS << "1. Issue Limit:         " << llvm::format("%.2f", Res.IssueBound) << " cycles/iter\n";
    OS << "2. Execution Ports:     " << llvm::format("%.2f", Res.PortBound) << " cycles/iter";
    if (!Res.PortBottleneckName.empty() && Res.PortBottleneckName != "None") {
        OS << "  [Bottleneck: " << Res.PortBottleneckName << "]";
    }
    OS << "\n";
    OS << "3. Precedence (RAW):    " << llvm::format("%.2f", Res.PrecedenceBound) << " cycles/iter\n";
    OS << "--------------------------------------------------\n";
    OS << "Dominant Bottleneck:    " << Res.DominantBottleneck << "\n";
    OS << "Estimated Throughput:   " << llvm::format("%.2f", Res.EstimatedCycles) << " cycles/iter\n";
    OS << "Estimated CPI:          " << llvm::format("%.3f", Res.EstimatedCPI) << "\n";
    OS << "==================================================\n";
}

} // namespace facile
