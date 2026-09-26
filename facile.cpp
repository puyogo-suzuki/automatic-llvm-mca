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

// Is this a CPU whose scheduling model uses a zero-latency write to mean
// "renamed away, so there is no dependence at all"?  See point (c) in the
// getEdgeLatency() comment below for why this must be asked per-target rather
// than by just trusting any Latency==0 write.
bool isZeroLatencyMovTarget(const llvm::MCSubtargetInfo &STI) {
    return STI.getCPU() == "cortex-a78" || STI.getCPU() == "cortex-a78ae" || STI.getCPU() == "cortex-a78c" ||
           STI.getCPU() == "cortex-a720" || STI.getCPU() == "cortex-a720ae";
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
//     reduction, dot product and stencil accumulation in real FP workloads - really
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

    // (c) TRUE ZERO-LATENCY WRITES (rename-time move elimination / zero
    //     idioms).  Arm Cortex-A78 SOG sec. 4.15 "Zero Latency MOVs": "A
    //     subset of register-to-register move operations and move immediate
    //     operations are executed with zero latency.  These instructions do
    //     not utilize the scheduling and execution resources of the machine.
    //     These are as follows: MOV Xd,#0 / MOV Xd,XZR / MOV Wd,#0 /
    //     MOV Wd,WZR / MOV Rd,#0 (AArch32) / MOV Wd,Wn / MOV Xd,Xn /
    //     MOV Rd,Rn (AArch32)".  Such a MOV is resolved by pointing the
    //     destination's rename entry at the source's physical register, so a
    //     consumer of the MOV's result reads the SAME physical register the
    //     producer wrote and waits exactly zero additional cycles.
    //
    //     The scheduling model ALREADY carries this and facile was, again,
    //     simply not reading it.  AArch64SchedNeoverseN2.td (the model
    //     installed for cortex-a78, see AArch64Processors.td) has
    //        def N2Write_0c : SchedWriteRes<[]> { let Latency = 0; }
    //        def N2Write_0or1c_1I : SchedWriteVariant<[
    //              SchedVar<NeoverseZeroMove, [N2Write_0c]>,
    //              SchedVar<NoSchedPred,      [N2Write_1c_1I]>]>;
    //        def : InstRW<[N2Write_0or1c_1I], (instregex "^MOVZ[WX]i$")>;
    //        def : InstRW<[N2Write_0or1c_1I], (instregex "^ORR[WX]rs$")>;
    //     and AArch64SchedPredNeoverse.td's NeoverseZeroMove enumerates
    //     precisely sec. 4.15's list (MOVZ[WX]i with both immediates zero,
    //     i.e. MOV Wd/Xd,#0; ORR[WX]rs with Rn == ZR and shift 0, which is
    //     how MOV Wd,WZR / MOV Xd,XZR / MOV Wd,Wn / MOV Xd,Xn all encode).
    //     SchedWriteRes<[]> is also exactly the SOG's "do not utilize the
    //     scheduling and execution resources" - no ProcResource is listed -
    //     and calculatePortUsageBound() already gets that part right,
    //     because resolveSchedClass() resolves the variant.  Only the
    //     dependence latency was being lost, to the max(Lat, 1.0) floor
    //     below.
    //
    //     The floor is right for every other write (a dependent operation
    //     cannot issue in the same cycle as a producer that really has to
    //     compute something), but a zero-latency MOV is not computed at all:
    //     there is no producer to wait for.  Charging it a cycle inflates
    //     every dependence chain that passes through a register copy - and
    //     copies are the single most common mnemonic in this corpus.
    //     WHY THIS IS GATED ON THE TARGET AND NOT JUST ON "Latency == 0".
    //     A zero-latency write means "renamed away" only in the N2 model.
    //     Checked every other model this tool uses, and a blanket rule would
    //     have been wrong in two of them:
    //       * N2 (cortex-a78): N2Write_0c appears at exactly three places,
    //         all three of them the NeoverseZeroMove arm of a
    //         SchedWriteVariant.  Here Latency==0 <=> SOG sec. 4.15 move.
    //       * N1 (cortex-a76): N1Write_0c_0Z (Latency 0, NumMicroOps 0) is
    //         used as the SECOND write of LDPWi / LDNPWi / LDPSWi, i.e. the
    //         second destination register of a load pair.  It is a micro-op
    //         accounting marker, NOT a statement that the second loaded
    //         register is available immediately - that register comes out of
    //         the same 4-cycle L1 access as the first.  A 0-cycle edge there
    //         would be a flat error, so A76 must keep the floor.  (A76's SOG
    //         documents no zero-latency-MOV section either.)
    //       * Firestorm / Icestorm: WriteZeroFire/WriteZeroIce covers MOVZ,
    //         MOVN, *MOVK*, FMOVDr, FMOVSr and NOP.  MOVK is a read-modify-
    //         write of its own destination (it merges a 16-bit field into the
    //         existing value) and so genuinely carries a dependence; these
    //         models also predate this reasoning and their CSVs are not
    //         regenerated here.  Left alone.
    //       * A720 (cortex-a720/-a720ae): see the A720 paragraph below.  Its
    //         only Latency==0 write is A720Write_0c, which - like N2's - exists
    //         solely as the zero-move arm of a SchedWriteVariant.
    //       * A55 does not run through facile at all (analyze.bash drives it
    //         with --dependency dependency and no --facile), and A520/V1 define
    //         no Latency==0 write.
    //     So the gate is the honest scope of what has actually been verified,
    //     not a convenience.
    //
    //     CORTEX-A720.  This was a known, unfixed gap until the sec. 4.12 data
    //     was added to AArch64SchedA720.td; the paragraphs below record what was
    //     missing, what was added, and why the gate can now honestly include it.
    //     The gap was never "the core has no zero-latency moves" - that was a
    //     fact about this project's SCHEDULING MODEL, not about the core.  A720
    //     does have zero-latency moves, and its SOG documents MORE of them than
    //     A78's does.  Arm Cortex-A720 Core
    //     Software Optimization Guide (109720, Issue 7.0) sec. 4.12 "Zero
    //     Latency Instructions" - note the title is "Instructions", not A78
    //     sec. 4.15's "MOVs" - reads: "A subset of register-to-register move
    //     operations, move immediate operations, predicates operations are
    //     executed with zero latency.  These instructions do not utilize the
    //     scheduling and execution resources of the machine.  These are as
    //     follows: MOV Xd,#{12{1'b0},imm[3:0]} / MOV Xd,XZR /
    //     MOV Wd,#{12{1'b0},imm[3:0]} / MOV Wd,WZR / MOV Hd,WZR / MOV Hd,XZR /
    //     MOV Sd,WZR / MOV Dd,XZR / MOVI Dd,#0 / MOVI Vd.2D,#0 / MOV Wd,Wn /
    //     MOV Xd,Xn / FMOV Sd,Sn / FMOV Dd,Dn / MOV Vd,Vn (vector) /
    //     MOV Zd.D,Zn.D / PTRUE / PFALSE / SETFFR", with the caveat "The
    //     MOV Wd,Wn, MOV Xd,Xn and FMOV Sd,Sn, FMOV Dd,Dn, MOV Vd,Vn (vector),
    //     MOV Zd.D,Zn.D instructions may not be executed with zero latency
    //     under certain conditions."  That is a strict superset of A78
    //     sec. 4.15's list: A720 adds the GPR<->FP/ASIMD zero moves
    //     (MOV Hd/Sd/Dd from WZR/XZR), the vector zero idioms (MOVI Dd,#0 and
    //     MOVI Vd.2D,#0), FP/vector register-to-register copies (FMOV Sd,Sn /
    //     FMOV Dd,Dn / MOV Vd,Vn), and the SVE predicate forms
    //     (MOV Zd.D,Zn.D / PTRUE / PFALSE / SETFFR).  It also narrows the
    //     immediate form: A78 says "MOV Xd,#0", A720 says
    //     "MOV Xd,#{12{1'b0},imm[3:0]}", i.e. any 16-bit immediate whose top
    //     12 bits are zero (0..15), not just zero.
    //
    //     WHY THE GATE COULD NOT SIMPLY BE WIDENED.  The A78 fix worked because
    //     the data was already in the model and facile was merely not reading
    //     it: AArch64SchedNeoverseN2.td defines N2Write_0c and gates it on the
    //     NeoverseZeroMove predicate, so `Lat == 0.0` is a reliable signal and
    //     the only change needed was to stop applying the max(Lat,1.0) floor.
    //     A720 was the opposite case.  cortex-a720/cortex-a720ae resolve to
    //     CortexA720Model in AArch64SchedA720.td (AArch64Processors.td:1448-
    //     1450), which was a hand-written WriteRes-only model: ZERO InstRW
    //     entries, no SchedWriteRes<[]>, no `Latency = 0` anywhere, and no
    //     zero-move predicate.  So `Lat == 0.0` could never be true for A720,
    //     and adding "cortex-a720" to isZeroLatencyMovTarget() on its own would
    //     have been pure dead code - it would have READ as a fix while changing
    //     nothing.
    //
    //     WHAT WAS ADDED, AND WHY THE GATE IS NOW CORRECT.
    //     AArch64SchedA720.td now carries sec. 4.12 directly (see the long
    //     commentary there for the opcode-by-opcode derivation):
    //        def A720Write_0c : SchedWriteRes<[]> { let Latency = 0; }
    //        def A720Write_0or1c_1I / _0or2c_1V / _0or3c_1M0 : SchedWriteVariant
    //            selecting A720Write_0c under a new A720ZeroMove predicate and
    //            otherwise the SOG's own non-eliminated write (1c/I for
    //            MOVZ[WX]i and ORR[WX]rs, 3c/M0 for the GPR-sourced
    //            FMOV[WX][HSD]r, 2c/V for FMOV[SD]r, MOVID/MOVIv2d_ns and
    //            ORRv16i8/ORRv8i8);
    //        InstRW rows binding those to MOVZWi, MOVZXi, ORRWrs, ORRXrs,
    //            FMOVWHr, FMOVXHr, FMOVWSr, FMOVXDr, MOVID, MOVIv2d_ns,
    //            ORRv16i8, ORRv8i8, FMOVSr, FMOVDr.
    //     A720ZeroMove is deliberately NOT stock LLVM's NeoverseZeroMove: that
    //     predicate is Neoverse N2/V1's and is too narrow for A720 in two ways
    //     (it requires MOVZ imm == 0, where sec. 4.12 documents imm[3:0], i.e.
    //     0..15; and it has no FMOV Sd,Sn / FMOV Dd,Dn arm at all) and too wide
    //     in one (its SVE ORR_ZZZ arm is unreachable under A720's
    //     UnsupportedFeatures = SVEUnsupported.F).  AArch64SchedPredNeoverse.td
    //     is left untouched so the eight Neoverse models sharing it do not move.
    //
    //     The condition this gate depends on therefore holds for A720 the same
    //     way it holds for N2, and was re-checked rather than assumed:
    //     A720Write_0c is referenced ONLY as the A720ZeroMove arm of those three
    //     variants and nowhere else in the file, so within CortexA720Model
    //     "Latency == 0" is equivalent to "this is a sec. 4.12 zero-latency
    //     form".  None of the newly bound writes is an accounting marker of the
    //     N1Write_0c_0Z kind (that one is the second destination of a load pair,
    //     where the value genuinely is not available early), and none is a
    //     read-modify-write of the Firestorm/Icestorm MOVK kind: every opcode
    //     listed above fully overwrites its destination from a source that the
    //     rename stage can point at (or from a constant it can fabricate).
    //     MOVK is specifically NOT bound, and neither is FMOVHr (FMOV Hd,Hn),
    //     which sec. 4.12 does not list.
    //
    //     STILL NOT VALIDATED AGAINST HARDWARE.  No measured Cortex-A720 dataset
    //     is available (no benchmark run was collected on an A720 core), so this
    //     change cannot be checked against real CPI in
    //     either direction.  It rests entirely on SOG sec. 4.12 being transcribed
    //     correctly, which is what the unit tests in tests/mlp_test.cpp assert:
    //     that the documented forms carry 0 and the deliberately excluded
    //     neighbours (non-ZR ORR, shifted ORR, MOVZ with imm > 15) still carry
    //     their normal latency.  Accuracy is NOT claimed.
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

// Cortex-A78 Software Optimization Guide sec. 4.14 "Instruction fusion":
// specific adjacent AArch64 instruction pairs execute as a single fused
// operation on Cortex-A78 ("These instruction pairs must be adjacent to each
// other in program code"). The second instruction of a fused pair consumes
// no additional issue-width or execution-port resource. Confirmed this is
// A78-specific: the Cortex-A76 SOG documents no equivalent CMP/CSEL or
// CMP/B.cond fusion (only unrelated FP fused-multiply-accumulate), so this
// must not be applied there.
//
// THE SOG's LIST IS NOT A CROSS PRODUCT.  Quoting sec. 4.14 verbatim:
//   1. CMP/CMN (immediate) + B.cond      6. CMP (register) + CSET
//   2. CMP/CMN (register) + B.cond       7. TST (immediate) + B.cond
//   3. CMP (immediate) + CSEL            8. TST (register) + B.cond
//   4. CMP (register) + CSEL             9. BICS (register) + B.cond
//   5. CMP (immediate) + CSET            10. NOP + Any instruction
// Note which producer each consumer is paired with.  B.cond accepts all four
// producers (CMP, CMN, TST, BICS), but CSEL and CSET are listed with CMP
// ONLY - items 3-6 say "CMP", not "CMP/CMN" the way items 1-2 do, and TST
// and BICS appear only with B.cond.  The previous implementation treated the
// producer set and the consumer set as independent and so also fused
// CMN+CSEL, TST+CSEL and BICS+CSEL, none of which the SOG lists.  The table
// below encodes the pairing exactly as printed.
//
// CMP/CMN/TST are flag-setting SUBS/ADDS/ANDS with a discarded (WZR/XZR)
// destination; only those forms qualify, not general flag-setting ops with
// a real destination register.
//
// NOT MODELLED HERE, DELIBERATELY: sec. 4.14's SECOND list ("The following
// instruction pairs are fused in both Aarch32 and Aarch64 modes: 1. AESE +
// AESMC, 2. AESD + AESIMC (see Section 4.6 on AES Encryption/Decryption)").
// That cross-reference resolves to sec. 4.7 "AES encryption/decryption",
// which states "Cortex-A78 can issue TWO AESE/AESMC/AESD/AESIMC instruction
// every cycle (fully pipelined) with an execution latency of two cycles" and
// "Pairs of dependent AESE/AESMC and AESD/AESIMC instructions exhibit higher
// performance when they are adjacent in the program code and both
// instructions use the same destination register".  So the AES "fusion" is a
// LATENCY / forwarding effect on a dependent pair, not the issue-slot and
// port elimination that FusedMask expresses: both halves still occupy V-pipe
// throughput (two AES instructions per cycle, per the quote), so masking the
// second one out of calculatePortUsageBound() would claim four AES
// instructions per cycle and contradict the SOG.  Wiring it through this
// mask would therefore be a modelling error, not a fix; and the benchmark
// corpus this tool has been used on contains no AES instructions at all, so
// there is nothing to validate a separate latency-path change against.
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
// consumers across; see the big comment on isA78FusionCandidate() for why
// this is NOT the cross product of the two sets.
bool isA78FusiblePair(A78Producer P, A78Consumer C) {
    switch (P) {
    case A78Producer::CMP:  // items 1,2 (+B.cond), 3,4 (+CSEL), 5,6 (+CSET)
        return C == A78Consumer::BCOND || C == A78Consumer::CSEL || C == A78Consumer::CSET;
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
std::vector<bool> computeA78FusedMask(const llvm::MCSubtargetInfo &STI,
                                      const llvm::MCInstrInfo &MCII,
                                      llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                                      llvm::ArrayRef<const llvm::MCInst *> MCInsts) {
    std::vector<bool> Fused(SimInstrs.size(), false);
    if (!isA78FusionCandidate(STI) || SimInstrs.size() < 2)
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
        if (isA78FusiblePair(P, classifyA78Consumer(MCII, *SimInstrs[i + 1], McAt(i + 1)))) {
            Fused[i + 1] = true;
            ++i;
        }
    }
    return Fused;
}

// ---------------------------------------------------------------------------
// Cortex-A720 instruction fusion.
//
// Arm Cortex-A720 Core Software Optimization Guide (109720, Issue 7.0)
// sec. 4.11 "Instruction fusion": "Cortex-A720 core can accelerate certain
// instruction pairs in an operation called fusion.  Specific instruction pairs
// that can be fused are as follows: [...] These instruction pairs must be
// adjacent to each other in program code."
//
// WHY THIS IS A SEPARATE FUNCTION AND NOT `isA78FusionCandidate() ||
// cortex-a720`.  A720 is A78's architectural successor, so widening the A78
// gate is the obvious-looking move - and it is wrong.  The two SOGs print
// DIFFERENT tables.  Side by side, A78 sec. 4.14 vs. A720 sec. 4.11:
//
//   A78 sec. 4.14                      A720 sec. 4.11
//   --------------------------------   -------------------------------------
//   CMP/CMN (immediate) + B.cond       CMP/CMN (immediate) + B.cond
//   CMP/CMN (register)  + B.cond       CMP/CMN (register Rn != ZR) + B.cond
//   CMP (immediate) + CSEL             CMP (immediate) + CSEL
//   CMP (register)  + CSEL             CMP (register)  + CSEL
//   CMP (immediate) + CSET             CMP (immediate) + CSET
//   CMP (register)  + CSET             CMP (register)  + CSET
//   TST (immediate) + B.cond           TST (immediate) + B.cond
//   TST (register)  + B.cond           TST (register)  + B.cond
//   BICS (register) + B.cond           BICS ZR (register) + B.cond
//   NOP + Any instruction              -- ABSENT --
//   -- absent --                       BTI + Integer DP/BR/BLR/RET/B uncond/
//                                          CBZ/TBZ
//   -- absent --                       SHL + SRI (both scalar or both vector)
//   -- absent --                       FCMP + AXFLAG
//   AESE + AESMC / AESD + AESIMC       AESE + AESMC / AESD + AESIMC
//   -- absent --                       MOVPRFX + supported SVE instruction
//
// THREE load-bearing differences, each of which a widened A78 gate gets wrong,
// all three in the direction of OVER-fusing (i.e. under-predicting cycles):
//
//  (1) A720 DOES NOT DOCUMENT "NOP + Any instruction".  A78 sec. 4.14 item 10
//      has no counterpart anywhere in A720 sec. 4.11.  This is not a rounding
//      difference: counting adjacent static pairs over a corpus of 25 real
//      benchmark binaries (6,744,507 instructions)
//      finds 92,102 NOP+Any pairs - by far the most frequent A78 fusion.
//      Reusing A78's table for A720 would silently delete 92,102 instructions'
//      worth of issue slots and port cycles that A720's SOG never says are
//      free.
//
//  (2) A720 restricts the register form of CMP/CMN + B.cond to "Rn != ZR";
//      A78 prints the same row with no such qualifier.  22 static pairs in
//      the corpus are CMP/CMN (register) with Rn == ZR followed by B.cond -
//      fusible on A78, NOT fusible on A720.  Note the qualifier is printed on
//      the B.cond row ONLY; the "CMP (register) + CSEL" and "CMP (register) +
//      CSET" rows carry no Rn restriction, so - transcribing exactly as
//      printed, which is the same discipline the A78 table needed - the
//      restriction is applied to the B.cond pairing only.
//
//  (3) A720 says "BICS ZR (register)", A78 says "BICS (register)".  The A78
//      implementation above deliberately accepts a real (non-ZR) destination
//      because A78's SOG names BICS by its own mnemonic with no destination
//      qualifier.  A720 names the destination explicitly, so on A720 the BICS
//      must discard its result into WZR/XZR.  10 static pairs in the corpus
//      are non-ZR-destination BICS + B.cond: fusible on A78, not on A720.
//
// NOT MODELLED HERE, DELIBERATELY - and why, row by row:
//
//   * AESE+AESMC / AESD+AESIMC.  Identical reasoning to the A78 case above,
//     against A720's own text.  Sec. 4.11's cross-reference resolves to A720
//     sec. 4.6 "AES encryption/decryption": "Cortex-A720 core can issue two
//     AESE/AESMC/AESD/AESIMC instruction every cycle (fully pipelined) with an
//     execution latency of two cycles.  Plus note, pairs of dependent
//     AESE/AESMC and AESD/AESIMC instructions are higher performance when they
//     are adjacent in the program code and both instructions use the same
//     destination register since they are fused".  As on A78, that is a
//     latency/forwarding effect on a dependent pair, not the issue-slot and
//     port elimination FusedMask expresses - both halves still occupy V-pipe
//     throughput, so masking the second out would claim four AES instructions
//     per cycle and contradict sec. 4.6.  The corpus contains no AES
//     instructions at all, so there is nothing to validate a latency-path
//     change against either.
//
//   * SHL + SRI, and FCMP + AXFLAG.  Both are genuine issue-slot fusions and
//     would be correct to model, but neither can ever fire on this corpus:
//     scanning all 25 binaries finds ZERO SRI instructions and ZERO AXFLAG
//     instructions (AXFLAG is Armv8.5 FEAT_FlagM2, which these builds predate).
//     Implementing them would be unexecutable code that no test or dataset
//     available could distinguish from a no-op, so they are recorded here
//     rather than written.  If a future corpus contains SRI or AXFLAG, add
//     them as SHL->SRI (matched on both operands being scalar, or both vector)
//     and FCMP->AXFLAG rows in isA720FusiblePair().
//
//   * BTI + Integer DP/BR/BLR/RET/B uncond/CBZ/TBZ.  This one DOES occur - 57
//     BTI instructions corpus-wide, 52 of them followed by a listed consumer -
//     but every single occurrence is a function-entry landing pad in CRT or
//     libgcc glue (_start, frame_dummy, __eqtf2, __letf2), i.e. never inside a
//     loop body, which is the only thing facile is ever asked to analyse.
//     Modelling it would therefore change no prediction this tool can make,
//     while requiring a definition of "Integer DP" that the SOG does not give
//     precisely enough to encode without guessing.  Recorded, not written.
//
//   * MOVPRFX + supported SVE instruction.  CortexA720Model in
//     AArch64SchedA720.td declares `UnsupportedFeatures = SVEUnsupported.F`,
//     so SVE instructions carry no scheduling data in this model at all, and
//     the corpus contains zero MOVPRFX instructions.  Out of scope twice over.
// ---------------------------------------------------------------------------
bool isA720FusionCandidate(const llvm::MCSubtargetInfo &STI) {
    return STI.getCPU() == "cortex-a720" || STI.getCPU() == "cortex-a720ae";
}

// The four producers A720 sec. 4.11 names among the rows modelled here.  Note
// the absence of NOP: sec. 4.11 has no "NOP + Any instruction" row (see
// difference (1) above), so unlike the A78 enum this one has no NOP member and
// computeA720FusedMask() has no NOP path.
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
std::vector<bool> computeA720FusedMask(const llvm::MCSubtargetInfo &STI,
                                       const llvm::MCInstrInfo &MCII,
                                       llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                                       llvm::ArrayRef<const llvm::MCInst *> MCInsts) {
    std::vector<bool> Fused(SimInstrs.size(), false);
    if (!isA720FusionCandidate(STI) || SimInstrs.size() < 2)
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
// sec. 4.11 disagree on three rows (see the table above).
std::vector<bool> computeFusedMask(const llvm::MCSubtargetInfo &STI,
                                   const llvm::MCInstrInfo &MCII,
                                   llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                                   llvm::ArrayRef<const llvm::MCInst *> MCInsts) {
    if (isA720FusionCandidate(STI))
        return computeA720FusedMask(STI, MCII, SimInstrs, MCInsts);
    return computeA78FusedMask(STI, MCII, SimInstrs, MCInsts);
}

// 1. Calculate Dispatch / Issue Width Limit
//
// TWO INDEPENDENT front-end bounds, both documented for every Arm core this
// tool models via a locally-modified InstRW-bearing .td (Cortex-A76/A78/A720):
// their Software Optimization Guides each state a pair of numbers of the form
// "the dispatch stage can process up to N Mops per cycle and dispatch up to
// M uops per cycle" (e.g. A76 SOG sec 4.1 p.41: N=4, M=8). A Mop
// (macro-operation) is what one non-fused instruction decodes into 1:1 (the
// SOGs describe splitting only going the OTHER way, Mop -> up to two uops,
// at the dispatch stage - see frontend.cpp's MopDispatchWidth comment); M
// (the uop cap) is what DispatchWidthOverride already carries. Since AArch64
// averages ~1.1-1.3 uops/instruction, N/4 is normally the TIGHTER of the two
// bounds and was, before this change, never computed at all - only the uop
// bound M was. MopWidth==0 (a CPU this hasn't been verified for, or Apple's
// coalesced-ROB cores where DispatchWidthOverride is already a Mop-level
// width) skips this second bound entirely, leaving prior behavior unchanged.
double calculateIssueBound(const llvm::MCSchedModel &SM,
                            llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                            unsigned &TotalUops,
                            unsigned DispatchWidthOverride,
                            unsigned MopWidth,
                            const std::vector<bool> &FusedMask) {
    unsigned IssueWidth = DispatchWidthOverride > 0 ? DispatchWidthOverride
                         : (SM.IssueWidth > 0 ? SM.IssueWidth : 1);
    TotalUops = 0;
    unsigned NumMops = 0;
    for (size_t i = 0; i < SimInstrs.size(); ++i) {
        if (i < FusedMask.size() && FusedMask[i])
            continue;
        unsigned NumUops = SimInstrs[i]->getNumMicroOps();
        TotalUops += (NumUops > 0 ? NumUops : 1);
        ++NumMops; // one Mop per non-fused instruction (see comment above)
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
// A dynamic-programming inner loop with a memory-carried D-state recurrence is
// the textbook case.  The block at
// 0xcc2c..0xcd18 (59 instructions, 17 loads, 10 stores) carries the serial
// recurrence
//     str w2, [x5, x0, lsl #2]   ; dc[k]        (iteration k)
//     ...
//     ldr w3, [x5, x1]           ; dc[k-1]      (iteration k+1)
//     add w3, w3, w2  /  cmp  /  csel  /  cmp  /  csel  /  str
// i.e. a memory round trip plus a 5-deep flag/select chain, which no register
// edge can see because the value never stays in a register across the
// backedge.  Measured on real M1: IceStorm needs 15.9
// cycles for the block and its IssueBound of 59/4 = 14.75 already covers that,
// so IceStorm's prediction (15) is accidentally right.  FireStorm needs 11.7
// cycles but its IssueBound is only 59/8 = 7.375, so the model predicted 7 --
// and the missing cycles are precisely the ones that show up in FireStorm's
// MAP_STALL_DISPATCH counter (30% of its cycles on this benchmark, versus 4.5% on
// IceStorm, the largest such ratio in the benchmark suite tested): the mapper
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

    // 1. Issue Limit
    Res.IssueBound = calculateIssueBound(STI.getSchedModel(), SimInstrs, Res.TotalMicroOps, DispatchWidth, MopWidth, FusedMask);

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
