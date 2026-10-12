#include "facile/facile.h"
#include "cpu/cpu_traits.h"
#include "cpu/sched_model.h"
#include "facile/fusion.h"
#include "facile/n2_corrections.h"
#include "mlp/mlp_types.h"
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

// Is this a CPU whose scheduling model uses a zero-latency write to mean
// "renamed away, so there is no dependence at all"?  See point (c) in the
// getEdgeLatency() comment below for why this must be asked per-target rather
// than by just trusting any Latency==0 write.
bool isZeroLatencyMovTarget(const llvm::MCSubtargetInfo &STI) {
    return getCpuTraits(STI.getCPU()).ZeroLatencyMov;
}

// ---------------------------------------------------------------------------
// Latency of a RAW dependence edge (Writer defines Reg -> Reader reads Reg
// through use slot Use): the individual write's own latency (not the
// instruction-wide MaxLatency), minus the consumer's ReadAdvance, floored at one
// cycle.  See docs.md section 4.A.
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

    // Rename-time zero-latency MOVs: a Latency==0 write in the model means "no
    // dependence at all" - but only for CPUs whose model uses Latency==0 for
    // nothing else (CpuTraits::ZeroLatencyMov).  See docs.md section 4.B.
    if (Lat == 0.0 && isZeroLatencyMovTarget(STI))
        return 0.0;

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

// 1. Issue bound: max(uops / uop dispatch width, Mops / Mop dispatch width).
// The Mop bound is skipped when MopWidth == 0 (CPU not verified, or Apple's
// coalesced-ROB cores).  Why there are two bounds: docs.md section 4.E.
double calculateIssueBound(const llvm::MCSchedModel &SM,
                            llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                            unsigned &TotalUops,
                            unsigned DispatchWidthOverride,
                            unsigned MopWidth,
                            const std::vector<bool> &FusedMask,
                            const std::vector<bool> &NoAluMask = {}) {
    unsigned IssueWidth = DispatchWidthOverride > 0 ? DispatchWidthOverride
                         : (SM.IssueWidth > 0 ? SM.IssueWidth : 1);
    TotalUops = 0;
    unsigned NumMops = 0;
    for (size_t i = 0; i < SimInstrs.size(); ++i) {
        if (i < FusedMask.size() && FusedMask[i])
            continue;
        unsigned NumUops = SimInstrs[i]->getNumMicroOps();
        if (i < NoAluMask.size() && NoAluMask[i] && NumUops > 1) --NumUops;  // A78 SOG reg-offset correction
        TotalUops += (NumUops > 0 ? NumUops : 1);
        ++NumMops; // one Mop per non-fused instruction (see docs.md section 4.E)
    }
    double UopBound = static_cast<double>(TotalUops) / static_cast<double>(IssueWidth);
    double MopBound = MopWidth > 0
                         ? static_cast<double>(NumMops) / static_cast<double>(MopWidth)
                         : 0.0;
    return std::max(UopBound, MopBound);
}

// 2. Calculate Execution Ports Contention Bound
double calculatePortUsageBound(const llvm::MCSubtargetInfo &STI,
                               const llvm::MCInstrInfo &MCII,
                               llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                               llvm::ArrayRef<const llvm::MCInst *> MCInsts,
                               std::string &BottleneckPortName,
                               const std::vector<bool> &FusedMask,
                               const std::vector<bool> &NoAluMask = {}) {
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

        A78ResOverride Ovr = classifyA78ResOverride(STI, MCII, *Inst);
        if (Ovr == A78ResOverride::LDPSW) {  // A78 SOG: LDPSW "I, L", 3/2 per cycle
            for (unsigned r = 1; r < NumProcResources; ++r) {
                const llvm::MCProcResourceDesc *P = SM.getProcResource(r);
                if (!P || !P->Name) continue;
                llvm::StringRef PN(P->Name);
                if (PN == "N2UnitI") ProcResUsage[r] += 1;
                if (PN == "N2UnitL") ProcResUsage[r] += 2;
            }
            continue;
        }
        for (const llvm::MCWriteProcResEntry *WPR = STI.getWriteProcResBegin(SCDesc);
             WPR != STI.getWriteProcResEnd(SCDesc); ++WPR) {
            unsigned ProcResIdx = WPR->ProcResourceIdx;
            unsigned Cycles = WPR->ReleaseAtCycle - WPR->AcquireAtCycle;
            if (Cycles == 0) continue; // Skip entries that consume 0 resource cycles
            if (Ovr == A78ResOverride::OccupancyCap1) Cycles = std::min(Cycles, 1u);
            if (Ovr == A78ResOverride::DivCap5) Cycles = std::min(Cycles, 5u);
            if (i < NoAluMask.size() && NoAluMask[i]) {  // A78 SOG reg-offset correction
                const llvm::MCProcResourceDesc *P = SM.getProcResource(ProcResIdx);
                if (P && P->Name && llvm::StringRef(P->Name) == "N2UnitI") continue;
            }
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

// A store with a writeback base update (STR/STP post- or pre-index, ST1.. post-index)
// writes the base register it also reads; no other def of a store is read by the
// store itself (STXR's status register, the only other kind, is not).  Returns the
// operand index of that base register, i.e. the first ADDRESS operand: every
// register use with a smaller explicit operand index is store DATA.  Returns -1 if
// `Inst` is not such a store, or has no store data to separate from.
//
// The writeback is produced by the address-generation side (the SOGs split a store
// into an address and a data uOP), so it does not wait for the data.  See docs.md
// section 4.F.
int storeWritebackAddressOperand(const llvm::mca::Instruction &Inst) {
    if (!Inst.getMayStore()) return -1;
    for (const llvm::mca::WriteState &W : Inst.getDefs()) {
        if (W.getRegisterID() == 0) continue;
        for (const llvm::mca::ReadState &R : Inst.getUses()) {
            if (R.getRegisterID() != W.getRegisterID() || R.getDescriptor().OpIndex < 0) continue;
            const int Base = R.getDescriptor().OpIndex;
            for (const llvm::mca::ReadState &D : Inst.getUses())
                if (D.getRegisterID() != 0 && D.getDescriptor().OpIndex >= 0 && D.getDescriptor().OpIndex < Base) return Base;
        }
    }
    return -1;
}

// True if `W` is the base-register update of a store (see above).
bool isStoreWritebackDef(const llvm::mca::Instruction &Inst, const llvm::mca::WriteState &W) {
    if (!Inst.getMayStore() || W.getRegisterID() == 0) return false;
    for (const llvm::mca::ReadState &R : Inst.getUses())
        if (R.getRegisterID() == W.getRegisterID() && R.getDescriptor().OpIndex >= 0) return true;
    return false;
}

// 3. Build Read-After-Write (RAW) Register Dependency Graph
//
// One node per instruction, plus one extra node per store with a writeback (see
// storeWritebackAddressOperand()): that node reads only the address registers and
// is the producer of the writeback register, so the base update does not appear
// to wait for the store data.  The extra nodes are numbered after the
// instructions, so instruction indices are unchanged.
std::vector<std::vector<DependencyEdge>> buildDependencyGraph(
    const llvm::MCSubtargetInfo &STI,
    llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs) {

    size_t N = SimInstrs.size();
    std::vector<std::vector<DependencyEdge>> Adj(N);
    // For each register: the node that last wrote it and the INSTRUCTION whose
    // def latency applies (they differ for a writeback node).
    struct Writer { size_t Node; size_t Inst; };
    std::map<unsigned, Writer> LastWriter;
    // Instruction index + the index of the USE SLOT within that instruction's
    // getUses(), so that the loop-carried edge built in Pass 2 can be given
    // the same per-operand ReadAdvance treatment as the intra-iteration ones.
    std::map<unsigned, std::pair<size_t, size_t>> FirstReader;
    std::set<unsigned> DefinedRegs;
    // Address-register reads of writeback nodes that happen before the register is
    // redefined in the iteration: they read the previous iteration's value.
    struct EarlyRead { size_t Node; size_t Inst; size_t Slot; unsigned Reg; };
    std::vector<EarlyRead> EarlyReads;

    // Pass 1: Intra-iteration dependencies & Live-In reads
    for (size_t i = 0; i < N; ++i) {
        const auto &Inst = SimInstrs[i];
        const int AddrOp = storeWritebackAddressOperand(*Inst);
        size_t WbNode = 0;
        if (AddrOp >= 0) {
            WbNode = Adj.size();
            Adj.emplace_back();
        }

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

            const bool IsAddressUse = AddrOp >= 0 && Op.getDescriptor().OpIndex >= AddrOp;
            if (IsAddressUse && DefinedRegs.find(Reg) == DefinedRegs.end())
                EarlyReads.push_back({WbNode, i, ThisSlot, Reg});

            auto it = LastWriter.find(Reg);
            if (it != LastWriter.end() && it->second.Inst < i) {
                double Lat = getEdgeLatency(STI, *SimInstrs[it->second.Inst], Op, Reg);
                Adj[it->second.Node].push_back({i, Lat, 0});
                if (IsAddressUse) Adj[it->second.Node].push_back({WbNode, Lat, 0});
            }
        }

        for (const auto &Op : Inst->getDefs()) {
            unsigned Reg = Op.getRegisterID();
            if (Reg != 0) {
                LastWriter[Reg] = {(AddrOp >= 0 && isStoreWritebackDef(*Inst, Op)) ? WbNode : i, i};
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
            const auto &Uses = SimInstrs[ReaderIdx]->getUses();
            double Lat = (ReaderSlot < Uses.size())
                             ? getEdgeLatency(STI, *SimInstrs[it->second.Inst], Uses[ReaderSlot], Reg)
                             : getInstLatency(*SimInstrs[it->second.Inst]);
            Adj[it->second.Node].push_back({ReaderIdx, Lat, 1});
        }
    }
    // The writeback nodes' own loop-carried reads (the base register updates itself).
    for (const EarlyRead &E : EarlyReads) {
        auto it = LastWriter.find(E.Reg);
        if (it == LastWriter.end()) continue;
        const auto &Uses = SimInstrs[E.Inst]->getUses();
        double Lat = (E.Slot < Uses.size())
                         ? getEdgeLatency(STI, *SimInstrs[it->second.Inst], Uses[E.Slot], E.Reg)
                         : getInstLatency(*SimInstrs[it->second.Inst]);
        Adj[it->second.Node].push_back({E.Node, Lat, 1});
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
// A dynamic-programming inner loop with a memory-carried D-state recurrence is
// the textbook case; such a block carries the serial recurrence
//     str w2, [x5, x0, lsl #2]   ; dc[k]        (iteration k)
//     ...
//     ldr w3, [x5, x1]           ; dc[k-1]      (iteration k+1)
//     add w3, w3, w2  /  cmp  /  csel  /  cmp  /  csel  /  str
// i.e. a memory round trip plus a 5-deep flag/select chain, which no register
// edge can see because the value never stays in a register across the
// backedge.  Such a recurrence bounds the block from below regardless of the
// issue width, so it becomes visible exactly on the widest core, where the
// IssueBound is smallest: the bottleneck is a dependence, not a port.
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
//     i.e. an a[i]-style access) -> may alias.  The recurrence above is exactly
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
// new constant; a larger store-forwarding latency would
// only raise the bound further, so this errs toward under- rather than
// over-prediction.
// ---------------------------------------------------------------------------

// Index register of an AArch64 register-offset access (LDRWroX / STRWroX and
// friends), or 0 for constant-offset forms.  getMemAccessInfo() records only
// the base register for these forms, but distinguishing a[i] from a[j] needs
// the index too -- see sameAddressExpr() below.  Mirrors mlp/seen_regs.cpp's
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
// The recurrence above is precisely the opposite case -- the store indexes
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

// Size in bytes of the access made by an AArch64 register-offset load/store
// (LDR/STR ... roW/roX); 0 if the opcode is not one of them.
unsigned regOffsetAccessSize(llvm::StringRef Name) {
    if (!Name.ends_with("roW") && !Name.ends_with("roX")) return 0;
    llvm::StringRef B = Name.drop_back(3);
    static const std::pair<const char *, unsigned> Sizes[] = {
        {"STRBB", 1}, {"STRHH", 2}, {"STRW", 4}, {"STRX", 8},
        {"LDRBB", 1}, {"LDRHH", 2}, {"LDRW", 4}, {"LDRX", 8},
        {"LDRSBW", 1}, {"LDRSBX", 1}, {"LDRSHW", 2}, {"LDRSHX", 2}, {"LDRSW", 4},
        {"STRB", 1}, {"STRH", 2}, {"STRS", 4}, {"STRD", 8}, {"STRQ", 16},
        {"LDRB", 1}, {"LDRH", 2}, {"LDRS", 4}, {"LDRD", 8}, {"LDRQ", 16}};
    for (const auto &E : Sizes)
        if (B == E.first) return E.second;
    return 0;
}

// The signed total added to `IdxReg` by the instructions strictly between `Lo` and
// `Hi`, if every one of them that writes IdxReg is `add/sub IdxReg, IdxReg, #imm`
// (no shift).  Returns false if any other instruction writes it (or an
// instruction is unknown): the value is then not statically known.
bool indexDeltaBetween(const llvm::MCInstrInfo &MCII, const llvm::MCRegisterInfo &MRI,
                       llvm::ArrayRef<const llvm::MCInst *> MCInsts, unsigned IdxReg,
                       size_t Lo, size_t Hi, int64_t &Delta) {
    Delta = 0;
    for (size_t k = Lo + 1; k < Hi; ++k) {
        const llvm::MCInst *MCI = MCInsts[k];
        if (!definesReg(MCII, MRI, MCI, IdxReg)) continue;
        llvm::StringRef N = MCII.getName(MCI->getOpcode());
        const bool IsAdd = N == "ADDXri" || N == "ADDWri";
        const bool IsSub = N == "SUBXri" || N == "SUBWri";
        if (!(IsAdd || IsSub) || MCI->getNumOperands() < 4 || !MCI->getOperand(0).isReg() ||
            !MCI->getOperand(1).isReg() || !MCI->getOperand(2).isImm() || !MCI->getOperand(3).isImm() ||
            MCI->getOperand(0).getReg() != IdxReg || MCI->getOperand(1).getReg() != IdxReg ||
            MCI->getOperand(3).getImm() != 0)
            return false;
        Delta += IsAdd ? MCI->getOperand(2).getImm() : -MCI->getOperand(2).getImm();
    }
    return true;
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

    // Store `i` and load `j` use the same register-offset addressing form
    // (`[xB, xI{, ext #s}]`), and the only thing between them that changes the index
    // register is `add/sub xI, xI, #imm`, so the load address minus the store address
    // is a known number of bytes.  If the two accesses do not overlap, the load
    // cannot take its data from the store (the SOG's store-to-load forwarding needs
    // the load to start at the start or the middle of the older store), so there is
    // no memory dependence.  Anything that is not statically known - a different
    // form, another kind of index update, an unknown size - returns false and the
    // dependence is kept.  See docs.md section 4.G.
    auto provablyDisjoint = [&](size_t i, size_t j) {
        const llvm::MCInst *SI = MCInsts[i], *LI = MCInsts[j];
        if (!SI || !LI || IdxRegs[i] == 0 || IdxRegs[i] != IdxRegs[j]) return false;
        const unsigned SizeS = regOffsetAccessSize(MCII.getName(SI->getOpcode()));
        const unsigned SizeL = regOffsetAccessSize(MCII.getName(LI->getOpcode()));
        if (SizeS == 0 || SizeL == 0) return false;
        if (SI->getNumOperands() < 5 || LI->getNumOperands() < 5) return false;
        // Same extend type and same scaled/unscaled flag (operands 3 and 4).
        for (unsigned k : {3u, 4u}) {
            if (!SI->getOperand(k).isImm() || !LI->getOperand(k).isImm() ||
                SI->getOperand(k).getImm() != LI->getOperand(k).getImm())
                return false;
        }
        const bool Scaled = SI->getOperand(4).getImm() != 0;
        if (Scaled && SizeS != SizeL) return false;  // different scales of the index
        int64_t Delta;
        if (!indexDeltaBetween(MCII, MRI, MCInsts, IdxRegs[i], i, j, Delta) || Delta == 0) return false;
        const int64_t Bytes = Delta * (Scaled ? static_cast<int64_t>(SizeS) : 1);  // load address - store address
        return Bytes >= static_cast<int64_t>(SizeS) || Bytes <= -static_cast<int64_t>(SizeL);
    };

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
            if (provablyDisjoint(i, j)) continue;        // statically different bytes: no forwarding
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
                                     llvm::ArrayRef<MemAccessInfo> MemInfos,
                                     unsigned MopWidth) {
    FacileResult Res;
    if (SimInstrs.empty()) return Res;

    Res.TotalInstructions = SimInstrs.size();

    // For Coalesced ROB CPUs (Apple Firestorm / Icestorm), normalize NumMicroOps to 1 (MOP-based)
    if (isCoalescedROBSubtarget(STI)) {
        for (const auto &Inst : SimInstrs) {
            llvm::mca::InstrDesc &MutableDesc = const_cast<llvm::mca::InstrDesc &>(Inst->getDesc());
            MutableDesc.NumMicroOps = 1;
        }
    }

    std::vector<bool> FusedMask = computeFusedMask(STI, MCII, SimInstrs, MCInsts);
    std::vector<bool> NoAluMask(SimInstrs.size(), false);
    for (size_t i = 0; i < SimInstrs.size(); ++i)
        NoAluMask[i] = isA78RegOffsetNoAlu(STI, MCII, *SimInstrs[i], i < MCInsts.size() ? MCInsts[i] : nullptr);

    // 1. Issue Limit
    Res.IssueBound = calculateIssueBound(STI.getSchedModel(), SimInstrs, Res.TotalMicroOps, DispatchWidth, MopWidth, FusedMask, NoAluMask);

    // 2. Execution Ports Limit
    Res.PortBound = calculatePortUsageBound(STI, MCII, SimInstrs, MCInsts, Res.PortBottleneckName, FusedMask, NoAluMask);

    // 3. Precedence Constraints Limit
    auto Adj = buildDependencyGraph(STI, SimInstrs);
    if (!MemInfos.empty())
        addMemoryDependencies(MCII, MRI, SimInstrs, MCInsts, MemInfos, Adj);
    Res.PrecedenceBound = calculatePrecedenceBound(Adj.size(), Adj);

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
