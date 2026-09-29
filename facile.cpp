#include "facile.h"
#include "cpu_traits.h"
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

// Cortex-A78 macro-op fusion (SOG sec. 4.14).  The SOG's list is NOT the cross
// product of its producers and consumers: classifyA78Producer(),
// classifyA78Consumer() and isA78FusiblePair() transcribe it row by row.  The
// AES pair is deliberately not part of the mask.  See docs.md section 4.C.
// A78 SOG correction: integer register-offset loads/stores do not use an integer
// (I) pipe on the A78 (only halfword forms with a shift do), unlike
// NeoverseN2Model's WriteLDIdx/WriteSTIdx.  See docs.md section 4.D.

// A78 SOG correction: NeoverseN2Model occupies M0 for several cycles for
// SCVTF/UCVTF (gen->vec), FMOV (gen->vec) and DUP (from GPR), which the SOG lists
// as fully pipelined; it models LDPSW with no load pipe; and it occupies the
// divider for the worst case where NeoverseN1Model (A76) uses the best case.
// classifyA78ResOverride() corrects these.  See docs.md section 4.D.


enum class A78ResOverride { None, OccupancyCap1, LDPSW, DivCap5 };
static A78ResOverride classifyA78ResOverride(const llvm::MCSubtargetInfo &STI, const llvm::MCInstrInfo &MCII,
                                             const llvm::mca::Instruction &Inst) {
    if (!getCpuTraits(STI.getCPU()).N2ModelCorrections) return A78ResOverride::None;
    llvm::StringRef N = MCII.getName(Inst.getOpcode());
    if ((N == "SDIVWr" || N == "UDIVWr" || N == "SDIVXr" || N == "UDIVXr"))
        return A78ResOverride::DivCap5;
    if (N == "LDPSWi" || N == "LDPSWpost" || N == "LDPSWpre") return A78ResOverride::LDPSW;
    if (N == "FMOVWHr" || N == "FMOVXHr" || N == "FMOVWSr" || N == "FMOVXDr") return A78ResOverride::OccupancyCap1;
    if ((N.starts_with("SCVTF") || N.starts_with("UCVTF")) && N.ends_with("ri") && N.size() == 10)
        return A78ResOverride::OccupancyCap1;  // [SU]CVTF[SU][WX][HSD]ri
    if (N.starts_with("DUPv") && N.ends_with("gpr")) return A78ResOverride::OccupancyCap1;
    return A78ResOverride::None;
}

static bool isA78RegOffsetNoAlu(const llvm::MCSubtargetInfo &STI, const llvm::MCInstrInfo &MCII,
                                const llvm::mca::Instruction &Inst, const llvm::MCInst *MCI) {
    if (!getCpuTraits(STI.getCPU()).N2ModelCorrections) return false;
    llvm::StringRef N = MCII.getName(Inst.getOpcode());
    if (!(N.ends_with("roW") || N.ends_with("roX"))) return false;
    llvm::StringRef B = N.drop_back(3);
    static const char *always[] = {"LDRBB", "LDRW", "LDRX", "LDRSBW", "LDRSBX", "LDRSW", "PRFM",
                                   "STRBB", "STRW", "STRX"};
    for (const char *a : always) if (B == a) return true;
    static const char *half[] = {"LDRHH", "LDRSHW", "LDRSHX", "STRHH"};
    for (const char *h : half) {
        if (B == h) {
            // operand 4 = "do shift" flag of the addressing mode (Rt, Rn, Rm, sign-ext, shift)
            if (!MCI || MCI->getNumOperands() < 5 || !MCI->getOperand(4).isImm()) return false;
            return MCI->getOperand(4).getImm() == 0;  // unscaled halfword: L only
        }
    }
    return false;  // FP/SIMD reg-offset forms: left to the model (see SOG Table 3-18)
}

// Cortex-X1 macro-op fusion: the A78 table without the CMP + CSEL / CMP + CSET
// rows (FusionTable::X1).  See docs.md section 4.C.

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

// The four producers sec. 4.14 names, plus NOP (item 10, which fuses with
// *any* successor and so is handled as its own kind rather than through the
// pairing table).
enum class A78Producer { None, CMP, CMN, TST, BICS, NOP };
// The three consumers sec. 4.14 names.
enum class A78Consumer { None, BCOND, CSEL, CSET };

A78Producer classifyA78Producer(const llvm::MCInstrInfo &MCII, const llvm::mca::Instruction &Inst,
                                const llvm::MCInst *MCI) {
    llvm::StringRef Name = MCII.getName(Inst.getOpcode());
    // Item 10's "NOP".  AArch64 gives `nop` (encoding 0xd503201f) its OWN
    // opcode, with no operands: dumping MCII.getName() over a whole compiled
    // binary yields "NOP" (0 operands) and "HINT" (1 immediate) as two
    // distinct entries, so matching only "HINT" leaves item 10 dead code.
    // HINT #0 occupies the same architectural encoding space and is accepted
    // too, for input that decodes that way.
    if (Name.equals_insensitive("NOP"))
        return A78Producer::NOP;
    if (Name.equals_insensitive("HINT")) {
        if (MCI && MCI->getNumOperands() >= 1 && MCI->getOperand(0).isImm() &&
            MCI->getOperand(0).getImm() == 0)
            return A78Producer::NOP;
        return A78Producer::None;
    }
    // AArch64 has no BICS-immediate encoding (a logical immediate with an
    // inverted mask is assembled as ANDS), so every BICS is "BICS (register)"
    // as item 9 requires.  Unlike CMP/CMN/TST, BICS is named in the SOG by
    // its own mnemonic, so a real (non-ZR) destination still qualifies.
    if (Name.starts_with_insensitive("BICS"))
        return A78Producer::BICS;
    // CMP / CMN / TST are SUBS / ADDS / ANDS with the destination discarded
    // into the zero register; a flag-setting op with a real destination is a
    // different instruction and is not listed.
    if (!writesZeroReg(MCI))
        return A78Producer::None;
    if (Name.starts_with_insensitive("SUBS")) return A78Producer::CMP;
    if (Name.starts_with_insensitive("ADDS")) return A78Producer::CMN;
    if (Name.starts_with_insensitive("ANDS")) return A78Producer::TST;
    return A78Producer::None;
}

A78Consumer classifyA78Consumer(const llvm::MCInstrInfo &MCII, const llvm::mca::Instruction &Inst,
                                const llvm::MCInst *MCI) {
    llvm::StringRef Name = MCII.getName(Inst.getOpcode());
    if (Name.equals_insensitive("Bcc")) return A78Consumer::BCOND;
    if (Name.starts_with_insensitive("CSEL")) return A78Consumer::CSEL;
    // CSET Wd, cond is an alias for CSINC Wd, WZR, WZR, invert(cond): the
    // operand test below is what distinguishes it from a general CSINC, and
    // is exactly the condition the AArch64 AsmPrinter uses to print `cset`.
    // CSETM (CSINV Wd, WZR, WZR, ...) is deliberately NOT matched: sec. 4.14
    // items 5 and 6 say CSET, and nothing else.
    if (Name.starts_with_insensitive("CSINC")) {
        if (MCI && MCI->getNumOperands() >= 3 && MCI->getOperand(1).isReg() &&
            MCI->getOperand(2).isReg()) {
            llvm::MCRegister Rn = MCI->getOperand(1).getReg();
            llvm::MCRegister Rm = MCI->getOperand(2).getReg();
            bool RnZero = (Rn == llvm::AArch64::WZR || Rn == llvm::AArch64::XZR);
            bool RmZero = (Rm == llvm::AArch64::WZR || Rm == llvm::AArch64::XZR);
            if (RnZero && RmZero) return A78Consumer::CSET;
        }
        return A78Consumer::None;
    }
    return A78Consumer::None;
}

// The sec. 4.14 pairing table, transcribed row by row.  Producers down,
// consumers across; see docs.md section 4.C for why this is NOT the cross
// product of the two sets.
//
// `HasSelectFusion` is false for the Cortex-X1, whose SOG omits items 3-6
// (CMP + CSEL / CSET); see docs.md section 4.C.
bool isA78FusiblePair(A78Producer P, A78Consumer C, bool HasSelectFusion = true) {
    switch (P) {
    case A78Producer::CMP:  // items 1,2 (+B.cond), 3,4 (+CSEL), 5,6 (+CSET)
        return C == A78Consumer::BCOND ||
               (HasSelectFusion && (C == A78Consumer::CSEL || C == A78Consumer::CSET));
    case A78Producer::CMN:  // items 1,2 only -- CMN is absent from items 3-6
        return C == A78Consumer::BCOND;
    case A78Producer::TST:  // items 7,8 only
        return C == A78Consumer::BCOND;
    case A78Producer::BICS: // item 9 only
        return C == A78Consumer::BCOND;
    default:
        return false;
    }
}

// Returns a mask, one entry per instruction, true if that instruction is the
// half of an adjacent fused pair that disappears (and should be excluded from
// issue-width/port-resource accounting).
//
// For the compare-and-branch/select pairs (items 1-9) that is the SECOND
// instruction: the flag-setting compare is what survives as the fused
// operation's own work.  For item 10 ("NOP + Any instruction") it is the
// FIRST: a NOP is what gets absorbed into its successor, and masking the
// successor instead would wrongly delete a real instruction's port usage and
// keep the NOP's.
//
// Pairing is greedy and NON-OVERLAPPING (an instruction already consumed by
// one fusion cannot also start another), which matters only for runs of
// consecutive NOPs: `nop; nop; ldr` is one fused pair plus a separate `nop;
// ldr` pair, not a single three-way fusion.  For items 1-9 the producer and
// consumer sets are disjoint, so this is a no-op there.
//
// `HasSelectFusion` = false gives the Cortex-X1 table (A78 minus CSEL/CSET).
std::vector<bool> computeA78FusedMask(const llvm::MCInstrInfo &MCII,
                                      llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                                      llvm::ArrayRef<const llvm::MCInst *> MCInsts,
                                      bool HasSelectFusion) {
    std::vector<bool> Fused(SimInstrs.size(), false);
    if (SimInstrs.size() < 2)
        return Fused;
    auto McAt = [&](size_t i) -> const llvm::MCInst * {
        return (i < MCInsts.size()) ? MCInsts[i] : nullptr;
    };
    for (size_t i = 0; i + 1 < SimInstrs.size(); ++i) {
        A78Producer P = classifyA78Producer(MCII, *SimInstrs[i], McAt(i));
        if (P == A78Producer::None)
            continue;
        if (P == A78Producer::NOP) {
            // Item 10: fuses with whatever follows, whatever that is.
            Fused[i] = true;
            ++i; // the successor is consumed by this pair
            continue;
        }
        if (isA78FusiblePair(P, classifyA78Consumer(MCII, *SimInstrs[i + 1], McAt(i + 1)),
                             HasSelectFusion)) {
            Fused[i + 1] = true;
            ++i;
        }
    }
    return Fused;
}

// Cortex-A720 macro-op fusion (SOG sec. 4.11).  Its table is NOT the A78 table:
// three rows differ and further rows exist that are deliberately not modelled.
// The row-by-row comparison and the reasons: docs.md section 4.C.

// The four producers A720 sec. 4.11 names among the rows modelled here.  Note
// the absence of NOP: sec. 4.11 has no "NOP + Any instruction" row (see
// difference (1) in docs.md section 4.C), so unlike the A78 enum this one has
// no NOP member and computeA720FusedMask() has no NOP path.
enum class A720Producer { None, CMP, CMN, TST, BICS };
// The three consumers sec. 4.11 names among the rows modelled here.
enum class A720Consumer { None, BCOND, CSEL, CSET };

// Everything the pairing table needs to know about a matched producer.  The
// register-vs-immediate form and whether Rn is the zero register have to be
// carried separately (rather than folded into the enum) because sec. 4.11
// prints the "Rn != ZR" qualifier on the B.cond row ONLY - see (2) above.
struct A720ProducerInfo {
    A720Producer Kind = A720Producer::None;
    bool RegisterForm = false; // register (incl. shifted/extended) vs immediate
    bool RnIsZeroReg = false;  // first source operand is WZR/XZR
};

// AArch64 spells the immediate form of these flag-setting ops with a trailing
// "ri" (SUBSWri, ADDSXri, ANDSWri, ...); every other form is a register form
// (SUBSWrr / SUBSWrs / SUBSWrx / SUBSXrx64 / ...).  Sec. 4.11's note "For CMP,
// CMN, TST fusion is allowed for shifted and/or extended register forms"
// confirms rs/rx belong on the register side rather than being excluded.
bool isA720ImmediateForm(llvm::StringRef Name) {
    return Name.ends_with_insensitive("ri");
}

A720ProducerInfo classifyA720Producer(const llvm::MCInstrInfo &MCII,
                                      const llvm::mca::Instruction &Inst,
                                      const llvm::MCInst *MCI) {
    A720ProducerInfo Info;
    llvm::StringRef Name = MCII.getName(Inst.getOpcode());

    // "BICS ZR (register)": unlike A78, A720 names the destination, so the
    // result must be discarded into WZR/XZR.  AArch64 has no BICS-immediate
    // encoding (a logical immediate with an inverted mask assembles as ANDS),
    // so every BICS is already a register form as the row requires.
    if (Name.starts_with_insensitive("BICS")) {
        if (!writesZeroReg(MCI))
            return Info;
        Info.Kind = A720Producer::BICS;
        Info.RegisterForm = true;
        return Info;
    }

    // CMP / CMN / TST are SUBS / ADDS / ANDS with the destination discarded
    // into the zero register; a flag-setting op with a real destination is a
    // different instruction and is not listed in sec. 4.11.
    if (!writesZeroReg(MCI))
        return Info;
    if (Name.starts_with_insensitive("SUBS")) Info.Kind = A720Producer::CMP;
    else if (Name.starts_with_insensitive("ADDS")) Info.Kind = A720Producer::CMN;
    else if (Name.starts_with_insensitive("ANDS")) Info.Kind = A720Producer::TST;
    else return Info;

    Info.RegisterForm = !isA720ImmediateForm(Name);
    // Operand 0 is the (discarded) destination, operand 1 is Rn.
    if (MCI && MCI->getNumOperands() >= 2 && MCI->getOperand(1).isReg()) {
        llvm::MCRegister Rn = MCI->getOperand(1).getReg();
        Info.RnIsZeroReg = (Rn == llvm::AArch64::WZR || Rn == llvm::AArch64::XZR);
    }
    return Info;
}

// Consumers are spelled identically to A78's, so classifyA78Consumer() is
// reused rather than duplicated; only the PAIRING table differs between the
// two cores.  (CSET is CSINC Rd, ZR, ZR, invert(cond); CSETM is deliberately
// not matched, exactly as on A78 - sec. 4.11 says CSET and nothing else.)
A720Consumer classifyA720Consumer(const llvm::MCInstrInfo &MCII,
                                  const llvm::mca::Instruction &Inst,
                                  const llvm::MCInst *MCI) {
    switch (classifyA78Consumer(MCII, Inst, MCI)) {
    case A78Consumer::BCOND: return A720Consumer::BCOND;
    case A78Consumer::CSEL:  return A720Consumer::CSEL;
    case A78Consumer::CSET:  return A720Consumer::CSET;
    default:                 return A720Consumer::None;
    }
}

// The sec. 4.11 pairing table, transcribed row by row.  As on A78 this is NOT
// the cross product of the producer and consumer sets: CSEL and CSET are
// listed with CMP only ("CMP (immediate) + CSEL", not "CMP/CMN"), while TST
// and BICS appear only with B.cond.
bool isA720FusiblePair(const A720ProducerInfo &P, A720Consumer C) {
    switch (P.Kind) {
    case A720Producer::CMP:
        if (C == A720Consumer::BCOND)
            // "CMP/CMN (immediate) + B.cond" has no Rn qualifier;
            // "CMP/CMN (register Rn != ZR) + B.cond" does.
            return !(P.RegisterForm && P.RnIsZeroReg);
        return C == A720Consumer::CSEL || C == A720Consumer::CSET;
    case A720Producer::CMN:
        // CMN appears on the B.cond row only - the CSEL and CSET rows say
        // "CMP", not "CMP/CMN".
        if (C == A720Consumer::BCOND)
            return !(P.RegisterForm && P.RnIsZeroReg);
        return false;
    case A720Producer::TST:
        return C == A720Consumer::BCOND;
    case A720Producer::BICS:
        return C == A720Consumer::BCOND;
    default:
        return false;
    }
}

// Mask of instructions that disappear into an adjacent fused pair on A720.
// Always the SECOND instruction: every row modelled here pairs a flag-setting
// compare with a consumer of those flags, and the compare is what survives as
// the fused operation's own work.  (A78's computeA78FusedMask() also has to
// handle the opposite case because of its "NOP + Any instruction" row, where
// the FIRST instruction is the one absorbed; A720 sec. 4.11 has no such row,
// so there is no first-instruction path here.)
//
// Pairing is greedy and non-overlapping, as on A78; since the producer and
// consumer sets are disjoint for every row modelled here, that is a no-op in
// practice and is kept only for structural parity.
std::vector<bool> computeA720FusedMask(const llvm::MCInstrInfo &MCII,
                                       llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                                       llvm::ArrayRef<const llvm::MCInst *> MCInsts) {
    std::vector<bool> Fused(SimInstrs.size(), false);
    if (SimInstrs.size() < 2)
        return Fused;
    auto McAt = [&](size_t i) -> const llvm::MCInst * {
        return (i < MCInsts.size()) ? MCInsts[i] : nullptr;
    };
    for (size_t i = 0; i + 1 < SimInstrs.size(); ++i) {
        A720ProducerInfo P = classifyA720Producer(MCII, *SimInstrs[i], McAt(i));
        if (P.Kind == A720Producer::None)
            continue;
        if (isA720FusiblePair(P, classifyA720Consumer(MCII, *SimInstrs[i + 1], McAt(i + 1)))) {
            Fused[i + 1] = true;
            ++i;
        }
    }
    return Fused;
}

// Per-target dispatch.  Each core gets its OWN SOG's table; there is
// deliberately no shared "Arm fusion" path, because A78 sec. 4.14 and A720
// sec. 4.11 disagree on three rows (see docs.md section 4.C).  The X1 is the
// A78 table with the CSEL/CSET rows removed.  Which table a CPU uses is
// CpuTraits::Fusion (cpu_traits.cpp).
std::vector<bool> computeFusedMask(const llvm::MCSubtargetInfo &STI,
                                   const llvm::MCInstrInfo &MCII,
                                   llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                                   llvm::ArrayRef<const llvm::MCInst *> MCInsts) {
    switch (getCpuTraits(STI.getCPU()).Fusion) {
    case FusionTable::A78:  return computeA78FusedMask(MCII, SimInstrs, MCInsts, /*HasSelectFusion=*/true);
    case FusionTable::X1:   return computeA78FusedMask(MCII, SimInstrs, MCInsts, /*HasSelectFusion=*/false);
    case FusionTable::A720: return computeA720FusedMask(MCII, SimInstrs, MCInsts);
    case FusionTable::None: break;
    }
    return std::vector<bool>(SimInstrs.size(), false);
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
                                     llvm::ArrayRef<MemAccessInfo> MemInfos,
                                     unsigned MopWidth) {
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
