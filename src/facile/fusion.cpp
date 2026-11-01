#include "facile/fusion.h"
#include "cpu/cpu_traits.h"
#include "llvm-source/llvm/lib/Target/AArch64/MCTargetDesc/AArch64MCTargetDesc.h"
#include <string>

namespace facile {

namespace {

// Cortex-A78 macro-op fusion (SOG sec. 4.14).  The SOG's list is NOT the cross
// product of its producers and consumers: classifyA78Producer(),
// classifyA78Consumer() and isA78FusiblePair() transcribe it row by row.  The
// AES pair is deliberately not part of the mask.  See docs/docs.md section 4.C.

// Cortex-X1 macro-op fusion: the A78 table without the CMP + CSEL / CMP + CSET
// rows (FusionTable::X1).  See docs/docs.md section 4.C.

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
// consumers across; see docs/docs.md section 4.C for why this is NOT the cross
// product of the two sets.
//
// `HasSelectFusion` is false for the Cortex-X1, whose SOG omits items 3-6
// (CMP + CSEL / CSET); see docs/docs.md section 4.C.
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
// The row-by-row comparison and the reasons: docs/docs.md section 4.C.

// The four producers A720 sec. 4.11 names among the rows modelled here.  Note
// the absence of NOP: sec. 4.11 has no "NOP + Any instruction" row (see
// difference (1) in docs/docs.md section 4.C), so unlike the A78 enum this one has
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

// [a720_fusion_simd] docs/a720-sog.md C-5: the sec. 4.11 rows "SHL + SRI (both
// scalar or both vector)" and "FCMP + AXFLAG" (l.3165-3166).  SHL and SRI by
// immediate are SHL<form> / SRI<form> with the same form suffix family: the
// scalar forms SHLd / SRId, the vector forms SHLv*_shift / SRIv*_shift.  The pair
// need not be dependent (the SOG states no such condition).  FCMP is every
// FCMP / FCMPE form (register or #0.0, H/S/D).
bool isA720ShlSriPair(llvm::StringRef First, llvm::StringRef Second) {
    if (First == "SHLd")
        return Second == "SRId";
    if (First.starts_with("SHLv") && First.ends_with("_shift"))
        return Second.starts_with("SRIv") && Second.ends_with("_shift");
    return false;
}

bool isA720FcmpAxflagPair(llvm::StringRef First, llvm::StringRef Second) {
    return First.starts_with("FCMP") && Second == "AXFLAG";
}

// Mask of instructions that disappear into an adjacent fused pair on A720.
// Always the SECOND instruction: the compare-and-branch/select rows pair a
// flag-setting compare with a consumer of those flags, and the compare is what
// survives as the fused operation's own work; SHL + SRI and FCMP + AXFLAG are
// treated the same way (docs/a720-sog.md D-11).  (A78's computeA78FusedMask()
// also has to handle the opposite case because of its "NOP + Any instruction"
// row, where the FIRST instruction is the one absorbed; A720 sec. 4.11 has no
// such row, so there is no first-instruction path here.)
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
        llvm::StringRef First = MCII.getName(SimInstrs[i]->getOpcode());
        llvm::StringRef Second = MCII.getName(SimInstrs[i + 1]->getOpcode());
        if (isA720ShlSriPair(First, Second) || isA720FcmpAxflagPair(First, Second)) {
            Fused[i + 1] = true;
            ++i;
            continue;
        }
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

} // namespace

// Per-target dispatch.  Each core gets its OWN SOG's table; there is
// deliberately no shared "Arm fusion" path, because A78 sec. 4.14 and A720
// sec. 4.11 disagree on three rows (see docs/docs.md section 4.C).  The X1 is the
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

} // namespace facile
