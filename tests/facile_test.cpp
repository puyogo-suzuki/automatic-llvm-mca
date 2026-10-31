#include "test_support.h"

// ============================================================================
// Facile Predictor Unit & Corner Case Tests
// ============================================================================

static facile::FacileResult runFacileAArch64(const AArch64TestContext &TC, const std::string &asm_code) {
    initLLVMAArch64();
    auto instrs = parseAsm(TC, asm_code);
    mca::InstrumentManager IM(*TC.STI, *TC.MCII);
    mca::InstrBuilder IB(*TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), IM, 0);
    std::vector<std::unique_ptr<mca::Instruction>> SimInstrs;
    std::vector<const MCInst *> MCInsts;
    for (const auto &I : instrs) {
        auto ExpectedInst = IB.createInstruction(I.Inst, {});
        if (ExpectedInst) {
            SimInstrs.push_back(std::move(*ExpectedInst));
            MCInsts.push_back(&I.Inst);
        }
    }
    return facile::computeFacilePrediction(*TC.STI, *TC.MCII, *TC.MRI, SimInstrs, MCInsts);
}


// Same as runFacileAArch64 but also supplies the per-instruction
// MemAccessInfo array, enabling facile's store->load memory RAW edges.
static facile::FacileResult runFacileAArch64Mem(const AArch64TestContext &TC, const std::string &asm_code) {
    initLLVMAArch64();
    auto instrs = parseAsm(TC, asm_code);
    auto Analyzer = MLPAnalyzer::create(*TC.STI);
    mca::InstrumentManager IM(*TC.STI, *TC.MCII);
    mca::InstrBuilder IB(*TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), IM, 0);
    std::vector<std::unique_ptr<mca::Instruction>> SimInstrs;
    std::vector<const MCInst *> MCInsts;
    std::vector<MemAccessInfo> MemInfos;
    for (const auto &I : instrs) {
        auto ExpectedInst = IB.createInstruction(I.Inst, {});
        if (ExpectedInst) {
            SimInstrs.push_back(std::move(*ExpectedInst));
            MCInsts.push_back(&I.Inst);
            const MCInstrDesc &MCID = TC.MCII->get(I.Inst.getOpcode());
            MemInfos.push_back(Analyzer->getMemAccessInfo(I.Inst, MCID, *TC.MRI, *TC.MCII));
        }
    }
    return facile::computeFacilePrediction(*TC.STI, *TC.MCII, *TC.MRI, SimInstrs, MCInsts, 0, MemInfos);
}


// A value that round-trips through memory across the backedge is a real
// loop-carried recurrence.  The store and the load index the same base with
// DIFFERENT registers (x1 vs x2), so a one-iteration lag cannot be ruled out
// and the edge must be created.  This is the shape of a memory-carried
// D-state recurrence, which the register-only graph cannot see at all.
TEST(FacileTest, MemoryLoopCarriedRecurrence) {
    initLLVMAArch64();
    AArch64TestContext TC("firestorm");
    std::string code =
        "ldr w3, [x5, x1]\n"
        "add w3, w3, #1\n"
        "str w3, [x5, x2, lsl #2]\n";
    auto WithMem = runFacileAArch64Mem(TC, code);
    auto NoMem = runFacileAArch64(TC, code);

    EXPECT_EQ(WithMem.TotalInstructions, 3u);
    // Without memory edges there is no cycle at all in the graph.
    EXPECT_DOUBLE_EQ(NoMem.PrecedenceBound, 0.0);
    // With them the recurrence is store(1) + load(3) + add(1) = 5 cycles.
    EXPECT_GT(WithMem.PrecedenceBound, 0.0);
    EXPECT_EQ(WithMem.FacileReason, "prec");
}


// Dependence-distance test: identical subscript expressions with a
// loop-varying index are an array read-modify-write (a[i] = f(a[i])).  Their
// dependence distance is 0, so there must be NO loop-carried edge -- the next
// iteration reads a[i+1], which this iteration's store did not write.
TEST(FacileTest, MemoryReadModifyWriteHasNoLoopCarriedEdge) {
    initLLVMAArch64();
    AArch64TestContext TC("firestorm");
    std::string code =
        "ldr w0, [x1, x2, lsl #2]\n"
        "add w0, w0, #1\n"
        "str w0, [x1, x2, lsl #2]\n"
        "add x2, x2, #1\n";
    auto WithMem = runFacileAArch64Mem(TC, code);
    auto NoMem = runFacileAArch64(TC, code);

    EXPECT_EQ(WithMem.TotalInstructions, 4u);
    // x2's own self-recurrence (add x2, x2, #1) is the only cycle, and the
    // memory edges must not add to it.
    EXPECT_DOUBLE_EQ(WithMem.PrecedenceBound, NoMem.PrecedenceBound);
}


// Same base register, both accesses constant-offset with DIFFERENT
// displacements: provably disjoint, so no edge (this is what keeps stack
// spill/reload and struct-field traffic from generating false recurrences).
TEST(FacileTest, MemoryDistinctConstantOffsetsDoNotAlias) {
    initLLVMAArch64();
    AArch64TestContext TC("firestorm");
    std::string code =
        "ldr w0, [x1, #8]\n"
        "add w0, w0, #1\n"
        "str w0, [x1, #16]\n";
    auto WithMem = runFacileAArch64Mem(TC, code);
    auto NoMem = runFacileAArch64(TC, code);

    EXPECT_DOUBLE_EQ(WithMem.PrecedenceBound, NoMem.PrecedenceBound);
    EXPECT_DOUBLE_EQ(WithMem.PrecedenceBound, 0.0);
}


// Different base registers are assumed not to alias, matching the
// AssumeNoAlias convention the rest of the tool uses.
TEST(FacileTest, MemoryDifferentBaseRegistersDoNotAlias) {
    initLLVMAArch64();
    AArch64TestContext TC("firestorm");
    std::string code =
        "ldr w0, [x1, x3]\n"
        "add w0, w0, #1\n"
        "str w0, [x2, x3, lsl #2]\n";
    auto WithMem = runFacileAArch64Mem(TC, code);
    EXPECT_DOUBLE_EQ(WithMem.PrecedenceBound, 0.0);
}


TEST(FacileTest, EmptyInstructions) {
    initLLVMAArch64();
    AArch64TestContext TC;
    std::vector<std::unique_ptr<mca::Instruction>> SimInstrs;
    std::vector<const MCInst *> MCInsts;
    auto Res = facile::computeFacilePrediction(*TC.STI, *TC.MCII, *TC.MRI, SimInstrs, MCInsts);

    EXPECT_EQ(Res.TotalInstructions, 0u);
    EXPECT_EQ(Res.TotalMicroOps, 0u);
    EXPECT_DOUBLE_EQ(Res.IssueBound, 0.0);
    EXPECT_DOUBLE_EQ(Res.PortBound, 0.0);
    EXPECT_DOUBLE_EQ(Res.PrecedenceBound, 0.0);
    EXPECT_DOUBLE_EQ(Res.EstimatedCycles, 0.0);
    EXPECT_DOUBLE_EQ(Res.EstimatedCPI, 0.0);
}


TEST(FacileTest, IndependentInstructionsIssueBound) {
    AArch64TestContext TC;
    std::string code = "add x0, x1, x2\nadd x3, x4, x5\nadd x6, x7, x8\nadd x9, x10, x11\n";
    auto Res = runFacileAArch64(TC, code);

    EXPECT_EQ(Res.TotalInstructions, 4u);
    EXPECT_DOUBLE_EQ(Res.PrecedenceBound, 0.0);
    EXPECT_GT(Res.IssueBound, 0.0);
    EXPECT_GT(Res.EstimatedCycles, 0.0);
}


TEST(FacileTest, SelfLoopDependency) {
    AArch64TestContext TC;
    std::string code = "add x0, x0, #1\n";
    auto Res = runFacileAArch64(TC, code);

    EXPECT_EQ(Res.TotalInstructions, 1u);
    EXPECT_GE(Res.PrecedenceBound, 1.0);
    EXPECT_EQ(Res.DominantBottleneck, "Precedence Constraints (Dependency Chain)");
}


TEST(FacileTest, TwoInstructionLoopDependencyChain) {
    AArch64TestContext TC;
    std::string code = "add x0, x1, #1\nadd x1, x0, #1\n";
    auto Res = runFacileAArch64(TC, code);

    EXPECT_EQ(Res.TotalInstructions, 2u);
    EXPECT_NEAR(Res.PrecedenceBound, 2.0, 0.01);
}


TEST(FacileTest, FloatingPointDependencyChain) {
    AArch64TestContext TC;
    std::string code = "fadd d0, d0, d1\n";
    auto Res = runFacileAArch64(TC, code);

    EXPECT_EQ(Res.TotalInstructions, 1u);
    EXPECT_GT(Res.PrecedenceBound, 1.0);
}


TEST(FacileTest, DominantBottleneckSelection) {
    AArch64TestContext TC;
    std::string code = "add x0, x0, #1\nadd x0, x0, #1\nadd x0, x0, #1\n";
    auto Res = runFacileAArch64(TC, code);

    EXPECT_EQ(Res.TotalInstructions, 3u);
    EXPECT_NEAR(Res.PrecedenceBound, 3.0, 0.01);
    EXPECT_EQ(Res.DominantBottleneck, "Precedence Constraints (Dependency Chain)");
}


TEST(FacileTest, LargeInstructionSequenceScale) {
    AArch64TestContext TC;
    std::string code;
    for (int i = 0; i < 60; ++i) {
        code += "add x" + std::to_string(i % 30) + ", x30, x30\n";
    }
    auto Res = runFacileAArch64(TC, code);

    EXPECT_EQ(Res.TotalInstructions, 60u);
    EXPECT_GT(Res.IssueBound, 0.0);
    EXPECT_DOUBLE_EQ(Res.PrecedenceBound, 0.0);
}


TEST(FacileTest, VariantSchedClassResolution) {
    AArch64TestContext TC;
    std::string code = "add x0, x1, x2, lsl #2\n";
    auto Res = runFacileAArch64(TC, code);

    EXPECT_EQ(Res.TotalInstructions, 1u);
    EXPECT_GT(Res.IssueBound, 0.0);
}


TEST(FacileTest, MultipleIndependentChainsMaxSelection) {
    AArch64TestContext TC;
    std::string code = "add x0, x0, #1\nfadd d0, d0, d1\n";
    auto Res = runFacileAArch64(TC, code);

    EXPECT_EQ(Res.TotalInstructions, 2u);
    EXPECT_GT(Res.PrecedenceBound, 1.0);
}


TEST(FacileTest, NoLoopCarriedDependency) {
    AArch64TestContext TC;
    std::string code = "mov x0, #1\nadd x1, x0, #2\n";
    auto Res = runFacileAArch64(TC, code);

    EXPECT_EQ(Res.TotalInstructions, 2u);
    EXPECT_DOUBLE_EQ(Res.PrecedenceBound, 0.0);
}


TEST(FacileTest, FacileReasonValues) {
    AArch64TestContext TC;
    std::string prec_code = "add x0, x0, #1\n";
    auto ResPrec = runFacileAArch64(TC, prec_code);
    EXPECT_EQ(ResPrec.FacileReason, "prec");

    std::string inst_code = "mov x0, #1\nadd x1, x2, #2\n";
    auto ResInst = runFacileAArch64(TC, inst_code);
    EXPECT_TRUE(ResInst.FacileReason == "inst" || ResInst.FacileReason == "exec");
}


TEST(FacileTest, FirestormCoalescedMOP) {
    AArch64TestContext TC("firestorm");
    std::string code = "ldp x0, x1, [x2]\nldp x3, x4, [x5]\n";
    auto Res = runFacileAArch64(TC, code);

    EXPECT_EQ(Res.TotalInstructions, 2u);
    // On Firestorm (Coalesced ROB / 8-MOP width), 2 MOPs = 2 micro-ops normalized
    EXPECT_EQ(Res.TotalMicroOps, 2u);
    EXPECT_DOUBLE_EQ(Res.IssueBound, 2.0 / 8.0);
}


TEST(FacileTest, IcestormCoalescedMOP) {
    AArch64TestContext TC("icestorm");
    std::string code = "ldp x0, x1, [x2]\nldp x3, x4, [x5]\n";
    auto Res = runFacileAArch64(TC, code);

    EXPECT_EQ(Res.TotalInstructions, 2u);
    // On Icestorm (Coalesced ROB / 4-MOP width), 2 MOPs = 2 micro-ops normalized
    EXPECT_EQ(Res.TotalMicroOps, 2u);
    EXPECT_DOUBLE_EQ(Res.IssueBound, 2.0 / 4.0);
}


// ============================================================================
// Cortex-A720 instruction fusion (A720 SOG 109720 Issue 7.0 sec. 4.11)
//
// These tests exist mainly to prove the A720 table is NOT A78's table.  Each
// one runs the same instruction pair on both cores and asserts they disagree
// where the two SOGs disagree.  TotalMicroOps is the observable: a fused pair
// contributes only its surviving half to calculateIssueBound().
// ============================================================================

// Baseline: the rows the two SOGs share really do fuse on A720.
TEST(FacileTest, A720FusionCmpImmediateBcond) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a720");
    auto Res = runFacileAArch64(TC, "cmp w0, #1\nb.eq .Lt\n.Lt:\n");
    EXPECT_EQ(Res.TotalInstructions, 2u);
    EXPECT_EQ(Res.TotalMicroOps, 1u); // b.eq absorbed into the cmp
}


TEST(FacileTest, A720FusionCmpImmediateCset) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a720");
    auto Res = runFacileAArch64(TC, "cmp w0, #1\ncset w1, eq\n");
    EXPECT_EQ(Res.TotalInstructions, 2u);
    EXPECT_EQ(Res.TotalMicroOps, 1u);
}


TEST(FacileTest, A720FusionCmpRegisterCsel) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a720");
    auto Res = runFacileAArch64(TC, "cmp w0, w1\ncsel w2, w3, w4, eq\n");
    EXPECT_EQ(Res.TotalInstructions, 2u);
    EXPECT_EQ(Res.TotalMicroOps, 1u);
}


// DIFFERENCE (1): A78 sec. 4.14 item 10 is "NOP + Any instruction"; A720
// sec. 4.11 has no such row.  NOP is common in real code, so getting it
// wrong would be a large modelling error of a naively-widened A78 gate.
TEST(FacileTest, A720DoesNotFuseNopWithAnything) {
    initLLVMAArch64();
    AArch64TestContext A78("cortex-a78");
    AArch64TestContext A720("cortex-a720");
    const std::string Code = "nop\nadd x0, x1, x2\n";
    EXPECT_EQ(runFacileAArch64(A78, Code).TotalMicroOps, 1u);  // A78: fused
    EXPECT_EQ(runFacileAArch64(A720, Code).TotalMicroOps, 2u); // A720: not
}


// DIFFERENCE (2): A720 prints "CMP/CMN (register Rn != ZR) + B.cond"; A78
// prints the same row unqualified.
TEST(FacileTest, A720CmpRegisterWithZeroRnIsNotFused) {
    initLLVMAArch64();
    AArch64TestContext A78("cortex-a78");
    AArch64TestContext A720("cortex-a720");
    const std::string Code = "cmp wzr, w1\nb.eq .Lt\n.Lt:\n";
    EXPECT_EQ(runFacileAArch64(A78, Code).TotalMicroOps, 1u);  // A78: fused
    EXPECT_EQ(runFacileAArch64(A720, Code).TotalMicroOps, 2u); // A720: Rn==ZR
    // The qualifier is printed on the REGISTER row only, which is consistent:
    // the immediate form encodes Rn in a GPR32sp/GPR64sp slot, where that bit
    // pattern means SP rather than ZR, so "CMP (immediate) with Rn == ZR" is
    // not an encodable instruction at all (`cmp wzr, #1` is rejected by the
    // assembler).  A normal immediate CMP is unaffected and still fuses:
    EXPECT_EQ(runFacileAArch64(A720, "cmp w0, #1\nb.eq .Lt\n.Lt:\n").TotalMicroOps, 1u);
}


// DIFFERENCE (3): A720 prints "BICS ZR (register)", A78 prints "BICS
// (register)" -- so a BICS with a real destination fuses on A78 only.
TEST(FacileTest, A720BicsRequiresZeroDestination) {
    initLLVMAArch64();
    AArch64TestContext A78("cortex-a78");
    AArch64TestContext A720("cortex-a720");
    const std::string RealDest = "bics w2, w0, w1\nb.eq .Lt\n.Lt:\n";
    EXPECT_EQ(runFacileAArch64(A78, RealDest).TotalMicroOps, 1u);  // A78: fused
    EXPECT_EQ(runFacileAArch64(A720, RealDest).TotalMicroOps, 2u); // A720: not
    // BICS ZR is the form A720 does list.
    EXPECT_EQ(runFacileAArch64(A720, "bics wzr, w0, w1\nb.eq .Lt\n.Lt:\n").TotalMicroOps, 1u);
}


// The A720 table is not a cross product either: CSEL/CSET are listed with CMP
// only, and TST/BICS only with B.cond.
TEST(FacileTest, A720TableIsNotACrossProduct) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a720");
    // CMN + CSEL is absent from sec. 4.11 (the CSEL row says "CMP").
    EXPECT_EQ(runFacileAArch64(TC, "cmn w0, #1\ncsel w2, w3, w4, eq\n").TotalMicroOps, 2u);
    // TST + CSEL is absent too (TST appears only on the B.cond row)...
    EXPECT_EQ(runFacileAArch64(TC, "tst w0, #1\ncsel w2, w3, w4, eq\n").TotalMicroOps, 2u);
    // ...while TST + B.cond is listed and does fuse.
    EXPECT_EQ(runFacileAArch64(TC, "tst w0, #1\nb.eq .Lt\n.Lt:\n").TotalMicroOps, 1u);
    // CMN + B.cond is listed and does fuse.
    EXPECT_EQ(runFacileAArch64(TC, "cmn w0, #1\nb.eq .Lt\n.Lt:\n").TotalMicroOps, 1u);
}


// A flag-setting op with a real destination is not a CMP/CMN/TST at all.
TEST(FacileTest, A720FlagSettingWithRealDestinationIsNotFused) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a720");
    EXPECT_EQ(runFacileAArch64(TC, "subs w5, w0, #1\nb.eq .Lt\n.Lt:\n").TotalMicroOps, 2u);
}


// Fusion must stay off cores whose SOGs document no such table.
TEST(FacileTest, A720FusionDoesNotLeakToOtherCores) {
    initLLVMAArch64();
    AArch64TestContext A76("cortex-a76");
    EXPECT_EQ(runFacileAArch64(A76, "cmp w0, #1\nb.eq .Lt\n.Lt:\n").TotalMicroOps, 2u);
}


// ============================================================================
// Cortex-A720 zero-latency instructions (A720 SOG 109720 Issue 7.0 sec. 4.12)
//
// AArch64SchedA720.td now models sec. 4.12 with an A720-local A720ZeroMove
// predicate selecting A720Write_0c (Latency 0, SchedWriteRes<[]>), and
// facile.cpp's getEdgeLatency() drops the max(Lat,1.0) floor for A720.  Two
// observables are used below:
//
//   PrecedenceBound - the cost of a loop-carried recurrence.  Putting the MOV
//     INSIDE the recurrence makes its dependence latency directly readable:
//     the cycle costs (producer latency + MOV latency), so a zero-latency MOV
//     shows up as the producer's latency alone.
//   PortBound - sec. 4.12's other half, "do not utilize the scheduling and
//     execution resources of the machine".  A720Write_0c lists no ProcResource,
//     so a block of qualifying moves has to show ZERO port pressure while the
//     same moves in a non-qualifying form occupy their pipeline.
//
// These numbers are derived from the SOG, not fitted, so nothing here is an
// accuracy claim.
// ============================================================================

// MOV Xd, Xn (encoded ORRXrs with Rn == XZR) inside a recurrence.  The cycle is
// add(1c) -> mov -> add, so A720 must show 1.0 and a core without the model
// must show 2.0.  cortex-a76 (NeoverseN1Model) is the control: N1 has no
// zero-latency-MOV section in its SOG and is explicitly excluded from
// isZeroLatencyMovTarget().
TEST(FacileTest, A720ZeroLatencyMovGpr) {
    initLLVMAArch64();
    AArch64TestContext A720("cortex-a720");
    AArch64TestContext A76("cortex-a76");
    const std::string Code = "add x0, x1, #1\nmov x1, x0\n";
    EXPECT_NEAR(runFacileAArch64(A720, Code).PrecedenceBound, 1.0, 0.01);
    EXPECT_NEAR(runFacileAArch64(A76,  Code).PrecedenceBound, 2.0, 0.01);
}


// Same shape, but the ORR is a real logical OR (Rn != ZR), which sec. 4.12 does
// not list.  A720ZeroMove's CheckIsReg1Zero must reject it: 1c + 1c = 2.0.
TEST(FacileTest, A720ZeroLatencyOrrRequiresZeroRn) {
    initLLVMAArch64();
    AArch64TestContext A720("cortex-a720");
    EXPECT_NEAR(runFacileAArch64(A720, "add x0, x1, #1\norr x1, x0, x2\n").PrecedenceBound,
                2.0, 0.01);
}


// Rn IS ZR but the source is shifted, so the result is not a copy of any
// register and cannot be renamed away.  CheckImmOperand<3, 0> must reject it.
TEST(FacileTest, A720ZeroLatencyOrrRequiresNoShift) {
    initLLVMAArch64();
    AArch64TestContext A720("cortex-a720");
    EXPECT_NEAR(runFacileAArch64(A720, "add x0, x1, #1\norr x1, xzr, x0, lsl #1\n").PrecedenceBound,
                2.0, 0.01);
}


// FMOV Dd, Dn / FMOV Sd, Sn - the forms stock LLVM's NeoverseZeroMove does NOT
// cover (N2 and V1 give FMOVDr/FMOVSr an unconditional 2-cycle V write), and
// the reason A720 needed its own predicate.  The recurrence is fadd(3c) -> fmov
// -> fadd, so zero latency shows as 3.0 and the unmodelled case would be 5.0.
TEST(FacileTest, A720ZeroLatencyFmovFpToFp) {
    initLLVMAArch64();
    AArch64TestContext A720("cortex-a720");
    EXPECT_NEAR(runFacileAArch64(A720, "fadd d0, d1, d2\nfmov d1, d0\n").PrecedenceBound,
                3.0, 0.01);
    EXPECT_NEAR(runFacileAArch64(A720, "fadd s0, s1, s2\nfmov s1, s0\n").PrecedenceBound,
                3.0, 0.01);
}


// FMOV Hd, Hn is deliberately absent from sec. 4.12 and so from A720ZeroMove;
// it keeps the SOG's normal "FP move, register" latency.  This pins the
// distinction, which is the easiest thing to get wrong by pattern-matching
// FMOV[HSD]r as a family (NeoverseN3 draws the same line).
TEST(FacileTest, A720FmovHalfPrecisionIsNotZeroLatency) {
    initLLVMAArch64();
    AArch64TestContext A720("cortex-a720");
    const double H = runFacileAArch64(A720, "fadd h0, h1, h2\nfmov h1, h0\n").PrecedenceBound;
    const double D = runFacileAArch64(A720, "fadd d0, d1, d2\nfmov d1, d0\n").PrecedenceBound;
    EXPECT_NEAR(D, 3.0, 0.01); // fadd's 3c alone: the D-form copy is eliminated
    EXPECT_GT(H, D);           // the H-form copy still carries a dependence
    // Deliberately not an absolute for H.  FMOVHr is bound to this model's
    // generic WriteFCopy, which is 1c, while the A720 SOG's Table 3-12
    // "FP move, register" is 2c - a pre-existing discrepancy in the hand-written
    // WriteRes block that is not this change's subject (WriteFCopy covers many
    // more instructions than FMOVHr).  What matters here is only that the H form
    // is NOT eliminated.
}


// MOV Vd, Vn (vector), encoded ORRv16i8/ORRv8i8 with Rn == Rm.
TEST(FacileTest, A720ZeroLatencyMovVector) {
    initLLVMAArch64();
    AArch64TestContext A720("cortex-a720");
    // The recurrence is add(2c) -> mov -> add, so an eliminated copy leaves the
    // vector add's 2 cycles alone.  Absolute values, not a GT: under a broken
    // model where the "zero" arm still costs a cycle the GT form would still
    // hold and the test would not discriminate.
    EXPECT_NEAR(runFacileAArch64(A720, "add v0.16b, v1.16b, v2.16b\nmov v1.16b, v0.16b\n").PrecedenceBound,
                2.0, 0.01);
    // A real vector ORR of two different registers is not a move:
    // CheckSameRegOperand<1, 2> must reject it, leaving the SOG's 2c
    // "ASIMD logical" (Table 3-14), for 2 + 2 = 4.
    EXPECT_NEAR(runFacileAArch64(A720, "add v0.16b, v1.16b, v2.16b\norr v1.16b, v0.16b, v2.16b\n").PrecedenceBound,
                4.0, 0.01);
}


// THE IMMEDIATE-RANGE DIFFERENCE.  sec. 4.12 says MOV Xd, #{12{1'b0},imm[3:0]},
// i.e. a MOVZ immediate of 0..15 - NOT just 0, which is all stock LLVM's
// NeoverseZeroMove accepts (CheckImmOperand<1, 0>).  Read through PortBound,
// because a move-immediate has no source register and so cannot sit in a
// recurrence: a qualifying MOVZ occupies no pipeline at all, a non-qualifying
// one occupies the I pipes.
TEST(FacileTest, A720MovzZeroLatencyImmediateRange) {
    initLLVMAArch64();
    AArch64TestContext A720("cortex-a720");
    auto PortOf = [&](const std::string &Code) {
        return runFacileAArch64(A720, Code).PortBound;
    };
    // imm 0 and imm 15 are both inside sec. 4.12's imm[3:0].
    EXPECT_DOUBLE_EQ(PortOf("mov x0, #0\nmov x1, #0\nmov x2, #0\nmov x3, #0\n"), 0.0);
    EXPECT_DOUBLE_EQ(PortOf("mov x0, #15\nmov x1, #15\nmov x2, #15\nmov x3, #15\n"), 0.0);
    // imm 16 is the first value outside it, and must cost a pipeline slot.
    EXPECT_GT(PortOf("mov x0, #16\nmov x1, #16\nmov x2, #16\nmov x3, #16\n"), 0.0);
    EXPECT_GT(PortOf("mov x0, #4096\nmov x1, #4096\nmov x2, #4096\nmov x3, #4096\n"), 0.0);
    // A shifted MOVZ is excluded even though its imm16 field is in range: the
    // value produced is large, not imm[3:0].
    EXPECT_GT(PortOf("movz x0, #1, lsl #16\nmovz x1, #1, lsl #16\n"
                     "movz x2, #1, lsl #16\nmovz x3, #1, lsl #16\n"), 0.0);
}


// MOV Hd/Sd/Dd from WZR/XZR and the MOVI zero idioms, likewise read through
// PortBound (all are sources-free or ZR-sourced, so none can form a cycle).
TEST(FacileTest, A720ZeroLatencyGprToFpAndMoviIdioms) {
    initLLVMAArch64();
    AArch64TestContext A720("cortex-a720");
    auto PortOf = [&](const std::string &Code) {
        return runFacileAArch64(A720, Code).PortBound;
    };
    // FMOVWSr / FMOVXDr with Rn == ZR: sec. 4.12's MOV Sd,WZR and MOV Dd,XZR.
    EXPECT_DOUBLE_EQ(PortOf("fmov s0, wzr\nfmov s1, wzr\nfmov d2, xzr\nfmov d3, xzr\n"), 0.0);
    // The same opcodes with a real GPR source are ordinary 3c/M0 transfers.
    EXPECT_GT(PortOf("fmov s0, w4\nfmov s1, w4\nfmov d2, x4\nfmov d3, x4\n"), 0.0);
    // MOVI Dd,#0 and MOVI Vd.2D,#0.
    EXPECT_DOUBLE_EQ(PortOf("movi d0, #0\nmovi d1, #0\nmovi v2.2d, #0\nmovi v3.2d, #0\n"), 0.0);
    // A non-zero MOVI is an ordinary ASIMD move-immediate.
    EXPECT_GT(PortOf("movi v0.4s, #1\nmovi v1.4s, #1\nmovi v2.4s, #1\nmovi v3.4s, #1\n"), 0.0);
}


// MOVK is a read-modify-write of its own destination: it merges a 16-bit field
// into the existing value, so it genuinely carries a dependence and is NOT in
// sec. 4.12.  This is the exact trap that made the Firestorm/Icestorm zero
// write unusable as a "renamed away" signal (see getEdgeLatency()'s comment),
// so it is pinned here for A720 too.
TEST(FacileTest, A720MovkIsNotZeroLatency) {
    initLLVMAArch64();
    AArch64TestContext A720("cortex-a720");
    EXPECT_NEAR(runFacileAArch64(A720, "add x0, x1, #1\nmov x1, x0\nmovk x1, #7, lsl #16\n").PrecedenceBound,
                2.0, 0.01);
}


// The sec. 4.12 modelling must not leak onto cores whose own SOG does not
// document it.  cortex-a76 keeps the 1-cycle floor on the very same MOV.
TEST(FacileTest, A720ZeroLatencyDoesNotLeakToOtherCores) {
    initLLVMAArch64();
    AArch64TestContext A76("cortex-a76");
    EXPECT_NEAR(runFacileAArch64(A76, "add x0, x1, #1\nmov x1, x0\n").PrecedenceBound, 2.0, 0.01);
}


// ============================================================================
// Cortex-X1 (X1 SOG PJDOC-466751330-12804 Issue 4.0)
//
// The X1 runs on the locally modified NeoverseV1Model.  What differs from the
// A78 (NeoverseN2Model) and is pinned here:
//   * machine parameters: ROB 224, uOP dispatch width 16, 8 Mops/cycle;
//   * "Instruction fusion" is the A78 table WITHOUT the CMP + CSEL/CSET rows;
//   * "Zero Latency MOVs" is the A78 list, and must not turn the second
//     destination of a load pair (V1Write_0c_0Z in stock V1) into a free value.
// ============================================================================

// facile with an explicit dispatch / Mop width, as frontend.cpp passes them.
static facile::FacileResult runFacileAArch64Widths(const AArch64TestContext &TC, const std::string &asm_code,
                                                   unsigned DispatchWidth, unsigned MopWidth) {
    initLLVMAArch64();
    auto instrs = parseAsm(TC, asm_code);
    mca::InstrumentManager IM(*TC.STI, *TC.MCII);
    mca::InstrBuilder IB(*TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), IM, 0);
    std::vector<std::unique_ptr<mca::Instruction>> SimInstrs;
    std::vector<const MCInst *> MCInsts;
    for (const auto &I : instrs) {
        auto ExpectedInst = IB.createInstruction(I.Inst, {});
        if (ExpectedInst) {
            SimInstrs.push_back(std::move(*ExpectedInst));
            MCInsts.push_back(&I.Inst);
        }
    }
    return facile::computeFacilePrediction(*TC.STI, *TC.MCII, *TC.MRI, SimInstrs, MCInsts, DispatchWidth, {}, MopWidth);
}


TEST(FacileTest, X1MachineParameters) {
    initLLVMAArch64();
    for (const char *cpu : {"cortex-x1", "cortex-x1c"}) {
        AArch64TestContext TC(cpu);
        const MCSchedModel &SM = TC.STI->getSchedModel();
        EXPECT_EQ(SM.MicroOpBufferSize, 224) << cpu;  // ROB
        EXPECT_EQ(SM.IssueWidth, 16u) << cpu;         // uOPs dispatched per cycle
    }
}


// 16 independent single-uop ALU instructions: 16 uops / 16 = 1.0 cycles but
// 16 Mops / 8 = 2.0, so the Mop cap is the binding front-end bound.
TEST(FacileTest, X1MopDispatchBound) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-x1");
    std::string Code;
    for (int i = 0; i < 16; ++i)
        Code += "add x" + std::to_string(i) + ", x" + std::to_string(i + 1) + ", #1\n";
    EXPECT_NEAR(runFacileAArch64Widths(TC, Code, 16, 8).IssueBound, 2.0, 0.01);
    EXPECT_NEAR(runFacileAArch64Widths(TC, Code, 16, 0).IssueBound, 1.0, 0.01);  // Mop cap off
}


// Rows the X1 shares with the A78 (its items 1,2,7,8,9,10).
TEST(FacileTest, X1FusionSharedRows) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-x1");
    EXPECT_EQ(runFacileAArch64(TC, "cmp w0, #1\nb.eq .Lt\n.Lt:\n").TotalMicroOps, 1u);
    EXPECT_EQ(runFacileAArch64(TC, "cmp w0, w1\nb.eq .Lt\n.Lt:\n").TotalMicroOps, 1u);
    EXPECT_EQ(runFacileAArch64(TC, "cmn w0, #1\nb.eq .Lt\n.Lt:\n").TotalMicroOps, 1u);
    EXPECT_EQ(runFacileAArch64(TC, "tst w0, #1\nb.eq .Lt\n.Lt:\n").TotalMicroOps, 1u);
    EXPECT_EQ(runFacileAArch64(TC, "tst w0, w1\nb.eq .Lt\n.Lt:\n").TotalMicroOps, 1u);
    EXPECT_EQ(runFacileAArch64(TC, "bics wzr, w0, w1\nb.eq .Lt\n.Lt:\n").TotalMicroOps, 1u);
    EXPECT_EQ(runFacileAArch64(TC, "nop\nadd x0, x1, x2\n").TotalMicroOps, 1u);  // NOP + Any
}


// The A78 rows the X1 SOG does NOT print: CMP + CSEL and CMP + CSET.
TEST(FacileTest, X1DoesNotFuseCmpWithCselOrCset) {
    initLLVMAArch64();
    AArch64TestContext A78("cortex-a78");
    AArch64TestContext X1("cortex-x1");
    AArch64TestContext X1C("cortex-x1c");
    const std::string Csel = "cmp w0, w1\ncsel w2, w3, w4, eq\n";
    const std::string CselI = "cmp w0, #1\ncsel w2, w3, w4, eq\n";
    const std::string Cset = "cmp w0, #1\ncset w1, eq\n";
    for (const std::string &Code : {Csel, CselI, Cset}) {
        EXPECT_EQ(runFacileAArch64(A78, Code).TotalMicroOps, 1u) << Code;  // A78: fused
        EXPECT_EQ(runFacileAArch64(X1, Code).TotalMicroOps, 2u) << Code;   // X1: not
        EXPECT_EQ(runFacileAArch64(X1C, Code).TotalMicroOps, 2u) << Code;
    }
}


TEST(FacileTest, X1FlagSettingWithRealDestinationIsNotFusedExceptBics) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-x1");
    EXPECT_EQ(runFacileAArch64(TC, "subs w5, w0, #1\nb.eq .Lt\n.Lt:\n").TotalMicroOps, 2u);
    // BICS is named by its own mnemonic in the SOG, so a real destination is fine.
    EXPECT_EQ(runFacileAArch64(TC, "bics w5, w0, w1\nb.eq .Lt\n.Lt:\n").TotalMicroOps, 1u);
}


// Sec. "Zero Latency MOVs": MOV Xd,Xn inside a recurrence costs nothing, so the
// add(1c) -> mov -> add cycle is 1.0 on the X1 and 2.0 on the A76 control.
TEST(FacileTest, X1ZeroLatencyMov) {
    initLLVMAArch64();
    AArch64TestContext X1("cortex-x1");
    AArch64TestContext A76("cortex-a76");
    const std::string Code = "add x0, x1, #1\nmov x1, x0\n";
    EXPECT_NEAR(runFacileAArch64(X1, Code).PrecedenceBound, 1.0, 0.01);
    EXPECT_NEAR(runFacileAArch64(A76, Code).PrecedenceBound, 2.0, 0.01);
    // Not in the SOG list: a MOV with a non-zero shifted ORR, MOVK, and MOVZ #imm != 0.
    EXPECT_NEAR(runFacileAArch64(X1, "add x0, x1, #1\norr x1, xzr, x0, lsl #1\n").PrecedenceBound, 2.0, 0.01);
    EXPECT_NEAR(runFacileAArch64(X1, "add x0, x1, #1\nmov x1, x0\nmovk x1, #7, lsl #16\n").PrecedenceBound, 2.0, 0.01);
}


// The load-pair trap: stock V1 gives the high destination of LDPW / LDPSW /
// LDP[SD] Latency 0 (V1Write_0c_0Z).  With the zero-latency gate on, that would
// make it free.  The recurrence x3 -> ldp -> (high dest) -> mov -> x3 must cost
// the load's latency instead.
TEST(FacileTest, X1LoadPairHighHalfIsNotZeroLatency) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-x1");
    // LDPSW: high half (x2) is 5c.  mov x3, x2 is a zero-latency MOV, so the
    // cycle costs exactly the load.
    EXPECT_NEAR(runFacileAArch64(TC, "ldpsw x1, x2, [x3]\nmov x3, x2\n").PrecedenceBound, 5.0, 0.01);
    // S/D-form pair: high half (d2) is 6c, then fmov x3, d2 (vec->gen, 2c) closes the cycle.
    EXPECT_NEAR(runFacileAArch64(TC, "ldp d1, d2, [x3]\nfmov x3, d2\n").PrecedenceBound, 8.0, 0.01);
}


// X1 SOG Table 3-34: AES ops run on V01 only, throughput 2 per cycle.  Four
// independent AESE therefore need 2.0 cycles (stock V would say 1.0).
TEST(FacileTest, X1AesRunsOnV01Only) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-x1");
    EXPECT_NEAR(runFacileAArch64(TC, "aese v0.16b, v8.16b\naese v1.16b, v8.16b\n"
                                     "aese v2.16b, v8.16b\naese v3.16b, v8.16b\n").PortBound, 2.0, 0.01);
    EXPECT_NEAR(runFacileAArch64(TC, "aesmc v0.16b, v4.16b\naesmc v1.16b, v5.16b\n"
                                     "aesmc v2.16b, v6.16b\naesmc v3.16b, v7.16b\n").PortBound, 2.0, 0.01);
}


// X1 SOG Table 3-12: LDPSW is "5, 1.5, I, L" - 1.5 per cycle, i.e. 2/3 cycle each
// on the three L pipes (stock V1: 1/3).
TEST(FacileTest, X1LdpswThroughput) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-x1");
    EXPECT_NEAR(runFacileAArch64(TC, "ldpsw x1, x2, [x0]\n").PortBound, 2.0 / 3.0, 0.01);
    EXPECT_NEAR(runFacileAArch64(TC, "ldp w1, w2, [x0]\n").PortBound, 1.0 / 3.0, 0.01);   // W-form: 3/cycle
    EXPECT_NEAR(runFacileAArch64(TC, "ldp x1, x2, [x0]\n").PortBound, 2.0 / 3.0, 0.01);   // X-form: 1.5/cycle
}


// Rows where the X1 SOG differs from the stock V1 model it started from.  Each
// asserts the pipe set / throughput the X1 SOG prints, via PortBound on
// independent instructions (PortBound = worst resource occupancy / units).
TEST(FacileTest, X1PipeAssignmentsFollowSog) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-x1");
    auto Port = [&](const char *Code) { return runFacileAArch64(TC, Code).PortBound; };
    // FCSEL: V02 (2 pipes) -> 4 of them need 2.0 cycles.
    EXPECT_NEAR(Port("fcsel d0, d8, d9, eq\nfcsel d1, d8, d9, eq\nfcsel d2, d8, d9, eq\nfcsel d3, d8, d9, eq\n"), 2.0, 0.01);
    // FP convert, vec -> gen: V02 but 1 per cycle -> 2 of them need 2.0 cycles.
    EXPECT_NEAR(Port("fcvtzs w0, d8\nfcvtzs w1, d9\n"), 2.0, 0.01);
    // TBL (1 table reg): one uOP on V01 -> 4 need 2.0 cycles.
    EXPECT_NEAR(Port("tbl v0.16b, {v8.16b}, v9.16b\ntbl v1.16b, {v8.16b}, v9.16b\n"
                     "tbl v2.16b, {v8.16b}, v9.16b\ntbl v3.16b, {v8.16b}, v9.16b\n"), 2.0, 0.01);
    // PMULL (64x64): V01, 2 per cycle -> 2 of them need 1.0 cycle.
    EXPECT_NEAR(Port("pmull v0.1q, v8.1d, v9.1d\npmull v1.1q, v8.1d, v9.1d\n"), 1.0, 0.01);
    // SHA1SU0 / SHA256SU1 (three-register forms): V0 only -> 2 need 2.0 cycles.
    EXPECT_NEAR(Port("sha1su0 v0.4s, v8.4s, v9.4s\nsha1su0 v1.4s, v8.4s, v9.4s\n"), 2.0, 0.01);
    EXPECT_NEAR(Port("sha256su1 v0.4s, v8.4s, v9.4s\nsha256su1 v1.4s, v8.4s, v9.4s\n"), 2.0, 0.01);
    // SQSHRUN (shift by immed, complex): V13 -> 4 need 2.0 cycles.
    EXPECT_NEAR(Port("sqshrun v0.8b, v8.8h, #3\nsqshrun v1.8b, v8.8h, #3\n"
                     "sqshrun v2.8b, v8.8h, #3\nsqshrun v3.8b, v8.8h, #3\n"), 2.0, 0.01);
}


// ASIMD dot product "2 (1)": 2c latency, 1c through the accumulator.
TEST(FacileTest, X1DotProductAccumulatorLatency) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-x1");
    EXPECT_NEAR(runFacileAArch64(TC, "sdot v0.4s, v1.16b, v2.16b\n").PrecedenceBound, 1.0, 0.01);
    // Through a NON-accumulator operand the full 2c applies; the vector MOV is
    // an ordinary 2c ORR on the X1 (its zero-latency list is GPR moves only).
    EXPECT_NEAR(runFacileAArch64(TC, "sdot v0.4s, v1.16b, v2.16b\nmov v1.16b, v0.16b\n").PrecedenceBound, 4.0, 0.01);
}

// ---------------------------------------------------------------------------
// Stores with a writeback base update (docs/docs.md section 4.F).  The base update is
// produced by the address side of the store, so it must not wait for the store
// data.  The recurrences below are written without a branch: facile adds the
// loop-carried edges itself.
// ---------------------------------------------------------------------------

// Data comes from a load off the same base register that the store updates.  The
// only recurrence is the 1-cycle base update; the ldr -> sub -> str data path is
// not part of a cycle.  Before the fix this was ldr(4) + sub(1) + str(1) = 6.
TEST(FacileTest, PostIndexStoreWritebackDoesNotDependOnStoreData) {
    initLLVMAArch64();
    for (const char *cpu : {"cortex-a78", "cortex-x1", "cortex-a720", "cortex-a76", "firestorm", "icestorm"}) {
        AArch64TestContext TC(cpu);
        auto Res = runFacileAArch64(TC, "ldr w1, [x0]\nsub w1, w1, #10\nstr w1, [x0], #4\ncmp x0, x21\n");
        EXPECT_LE(Res.PrecedenceBound, 1.0) << cpu;
        EXPECT_GE(Res.PrecedenceBound, 1.0) << cpu;  // the base update itself stays a 1-cycle recurrence
    }
}

TEST(FacileTest, PreIndexAndPairStoreWritebackDoNotDependOnStoreData) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a78");
    EXPECT_NEAR(runFacileAArch64(TC, "ldr w1, [x0]\nsub w1, w1, #10\nstr w1, [x0, #4]!\ncmp x0, x21\n").PrecedenceBound, 1.0, 0.01);
    EXPECT_NEAR(runFacileAArch64(TC, "ldp w1, w2, [x0]\nadd w1, w1, w2\nstp w1, w2, [x0], #8\ncmp x0, x21\n").PrecedenceBound, 1.0, 0.01);
}

// What was not wrong stays as it was: a load with writeback, a store whose base is
// updated by a separate add, and a data recurrence that does not involve the base.
TEST(FacileTest, StoreWritebackChangeLeavesOtherRecurrencesAlone) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a78");
    EXPECT_NEAR(runFacileAArch64(TC, "ldr w1, [x0], #4\nsub w1, w1, #10\nstr w1, [x2], #4\ncmp x0, x21\n").PrecedenceBound, 1.0, 0.01);
    EXPECT_LE(runFacileAArch64(TC, "ldr w1, [x0]\nsub w1, w1, #10\nstr w1, [x0]\nadd x0, x0, #4\ncmp x0, x21\n").PrecedenceBound, 1.0);
    EXPECT_NEAR(runFacileAArch64(TC, "ldr w1, [x0], #4\nadd w2, w2, w1\ncmp x0, x21\n").PrecedenceBound, 1.0, 0.01);
}

// The address side still counts: if the base register is itself loaded (a pointer
// chase), the writeback depends on that load and the recurrence is ldr (4) + base
// update (1).
TEST(FacileTest, StoreWritebackStillDependsOnItsBaseRegister) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a78");
    EXPECT_NEAR(runFacileAArch64(TC, "ldr x0, [x0]\nstr w1, [x0], #8\n").PrecedenceBound, 5.0, 0.01);
}

// ---------------------------------------------------------------------------
// Memory dependence when the index register changes between store and load
// (docs/docs.md section 4.G).  The loops are written so that the memory edge is part
// of a cycle: the loaded value feeds the data of the store.
// ---------------------------------------------------------------------------
namespace {
// PrecedenceBound with the memory edges minus without them: > 0 iff a memory
// dependence closes a recurrence.
double memoryEdgeEffect(const std::string &body) {
    AArch64TestContext TC("cortex-a78");
    return runFacileAArch64Mem(TC, body).PrecedenceBound - runFacileAArch64(TC, body).PrecedenceBound;
}
}  // namespace

// The index moves by 2 and the accesses are 2 bytes wide: the load reads the NEXT
// element, so there is no memory dependence (x264 quant_4x4).
TEST(FacileTest, MemoryNoEdgeWhenIndexAdvancesPastTheStore) {
    initLLVMAArch64();
    EXPECT_DOUBLE_EQ(memoryEdgeEffect("add w3, w3, w5\nstrh w3, [x0, x4]\nadd x4, x4, #2\nldrsh w5, [x0, x4]\n"), 0.0);
    // Scaled form: str w [x0, x4, lsl #2] then x4 += 1 is 4 bytes further.
    EXPECT_DOUBLE_EQ(memoryEdgeEffect("add w3, w3, w5\nstr w3, [x0, x4, lsl #2]\nadd x4, x4, #1\nldr w5, [x0, x4, lsl #2]\n"), 0.0);
    // Moving back is just as disjoint.
    EXPECT_DOUBLE_EQ(memoryEdgeEffect("add w3, w3, w5\nstrh w3, [x0, x4]\nsub x4, x4, #2\nldrsh w5, [x0, x4]\n"), 0.0);
}

// Everything that is not statically disjoint keeps the edge.
TEST(FacileTest, MemoryEdgeKeptUnlessProvablyDisjoint) {
    initLLVMAArch64();
    // Same address: the index is not updated between them.
    EXPECT_GT(memoryEdgeEffect("add w3, w3, w5\nstrh w3, [x0, x4]\nldrsh w5, [x0, x4]\n"), 0.0);
    // The index moves by 1 byte but the accesses are 2 bytes: they overlap.
    EXPECT_GT(memoryEdgeEffect("add w3, w3, w5\nstrh w3, [x0, x4]\nadd x4, x4, #1\nldrsh w5, [x0, x4]\n"), 0.0);
    // The index is rewritten by something that is not add/sub of an immediate.
    EXPECT_GT(memoryEdgeEffect("add w3, w3, w5\nstrh w3, [x0, x4]\nmov x4, x6\nldrsh w5, [x0, x4]\n"), 0.0);
    // Different scaling of the index in the two accesses: not comparable.
    EXPECT_GT(memoryEdgeEffect("add w3, w3, w5\nstrh w3, [x0, x4, lsl #1]\nadd x4, x4, #1\nldr w5, [x0, x4, lsl #2]\n"), 0.0);
}

// ---------------------------------------------------------------------------
// FP multiply-accumulate (SOGs: FMADD / FMLA "4 (2)"): result latency 4, and the
// accumulate operand is needed 2 cycles early, so a chain of accumulations costs
// 2 cycles each.  The A720 model used to treat FMADD as an FMUL (3, no early read).
// ---------------------------------------------------------------------------
TEST(FacileTest, FmaAccumulateChainIsTwoCyclesPerLink) {
    initLLVMAArch64();
    // (cortex-a76 / cortex-a78 are not listed: this fixture runs them on libLLVM's stock
    // tables, not on the models the tool installs - see AArch64TestContext.)
    for (const char *cpu : {"cortex-x1", "cortex-a720"}) {
        AArch64TestContext TC(cpu);
        EXPECT_NEAR(runFacileAArch64(TC, "fmadd d0, d1, d2, d0\n").PrecedenceBound, 2.0, 0.01) << cpu;
        EXPECT_NEAR(runFacileAArch64(TC, "fmadd d0, d1, d2, d0\nfmadd d0, d3, d4, d0\n").PrecedenceBound, 4.0, 0.01) << cpu;
    }
    AArch64TestContext A720("cortex-a720");
    EXPECT_NEAR(runFacileAArch64(A720, "fmla v0.4s, v1.4s, v2.4s\n").PrecedenceBound, 2.0, 0.01);
}

// A720 SOG: the result of an FP multiply is forwarded late to the accumulate operand
// (FMUL 3 - 2 = 1), but a non-accumulate operand sees the full latency (4).
TEST(FacileTest, A720FmulToAccumulateForwarding) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a720");
    // fmadd (4, through a multiplicand) -> fmul (3, to the accumulator: 1) = 4 + 1.
    EXPECT_NEAR(runFacileAArch64(TC, "fmadd d0, d1, d2, d3\nfmul d3, d0, d4\n").PrecedenceBound, 5.0, 0.01);
}

// ---------------------------------------------------------------------------
// FP divide / square root latency, one value per form (docs/docs.md section 4.I).
// A self-dependent chain makes the precedence bound equal to the latency.
// ---------------------------------------------------------------------------
namespace {
double fpChainLatency(const char *cpu, const std::string &insn) {
    AArch64TestContext TC(cpu);
    return runFacileAArch64(TC, insn).PrecedenceBound;
}
}  // namespace

TEST(FacileTest, A720FpDivSqrtLatencyPerFormFromSog) {
    initLLVMAArch64();
    // Scalar: H 5, S 7, D 12 for both.
    EXPECT_NEAR(fpChainLatency("cortex-a720", "fdiv h0, h0, h1\n"), 5.0, 0.01);
    EXPECT_NEAR(fpChainLatency("cortex-a720", "fdiv s0, s0, s1\n"), 7.0, 0.01);
    EXPECT_NEAR(fpChainLatency("cortex-a720", "fdiv d0, d0, d1\n"), 12.0, 0.01);
    EXPECT_NEAR(fpChainLatency("cortex-a720", "fsqrt s0, s0\n"), 7.0, 0.01);
    EXPECT_NEAR(fpChainLatency("cortex-a720", "fsqrt d0, d0\n"), 12.0, 0.01);
    // ASIMD: D-form F32 8, Q-form F32 10, Q-form F64 13.
    EXPECT_NEAR(fpChainLatency("cortex-a720", "fdiv v0.2s, v0.2s, v1.2s\n"), 8.0, 0.01);
    EXPECT_NEAR(fpChainLatency("cortex-a720", "fdiv v0.4s, v0.4s, v1.4s\n"), 10.0, 0.01);
    EXPECT_NEAR(fpChainLatency("cortex-a720", "fsqrt v0.2d, v0.2d\n"), 13.0, 0.01);
}

// X1: the best latency of the SOG range (7 to 10 / 7 to 15 / 7 to 16).
TEST(FacileTest, X1FpDivSqrtUsesBestLatencyOfTheSogRange) {
    initLLVMAArch64();
    EXPECT_NEAR(fpChainLatency("cortex-x1", "fdiv s0, s0, s1\n"), 7.0, 0.01);
    EXPECT_NEAR(fpChainLatency("cortex-x1", "fdiv d0, d0, d1\n"), 7.0, 0.01);
    EXPECT_NEAR(fpChainLatency("cortex-x1", "fsqrt d0, d0\n"), 7.0, 0.01);
    EXPECT_NEAR(fpChainLatency("cortex-x1", "fdiv v0.2d, v0.2d, v1.2d\n"), 7.0, 0.01);
    // Q-form F16: FDIV 10 to 13, FSQRT 11 to 13.
    EXPECT_NEAR(fpChainLatency("cortex-x1", "fdiv v0.8h, v0.8h, v1.8h\n"), 10.0, 0.01);
    EXPECT_NEAR(fpChainLatency("cortex-x1", "fsqrt v0.8h, v0.8h\n"), 11.0, 0.01);
}

// Apple: per form, from Dougall Johnson's tables.
TEST(FacileTest, AppleFpDivSqrtLatencyPerForm) {
    initLLVMAArch64();
    EXPECT_NEAR(fpChainLatency("firestorm", "fdiv s0, s0, s1\n"), 8.0, 0.01);
    EXPECT_NEAR(fpChainLatency("firestorm", "fdiv d0, d0, d1\n"), 10.0, 0.01);
    EXPECT_NEAR(fpChainLatency("firestorm", "fsqrt d0, d0\n"), 13.0, 0.01);
    EXPECT_NEAR(fpChainLatency("firestorm", "fdiv v0.4s, v0.4s, v1.4s\n"), 8.0, 0.01);
    EXPECT_NEAR(fpChainLatency("icestorm", "fdiv v0.4s, v0.4s, v1.4s\n"), 9.0, 0.01);
    EXPECT_NEAR(fpChainLatency("icestorm", "fdiv v0.2d, v0.2d, v1.2d\n"), 11.0, 0.01);
    EXPECT_NEAR(fpChainLatency("icestorm", "fsqrt v0.2d, v0.2d\n"), 15.0, 0.01);
}

// Occupancy of the divider (docs/docs.md section 4.I): the BEST throughput of the SOG range,
// rounded up to whole cycles.  Two independent instructions -> PortBound = twice the
// per-instruction time on the unit(s).
TEST(FacileTest, FpDivSqrtOccupancyIsBestThroughputRoundedUp) {
    initLLVMAArch64();
    auto port = [](const char *cpu, const std::string &code) {
        AArch64TestContext TC(cpu);
        return runFacileAArch64(TC, code).PortBound;
    };
    // X1 (V0/V2 pair): scalar D best 4/7 -> 1.75 cycles -> 2 per instruction; S 8/7 -> 1;
    // ASIMD Q-form F32 2/7 -> 3.5.
    EXPECT_NEAR(port("cortex-x1", "fdiv d0, d8, d9\nfdiv d1, d8, d9\n"), 4.0, 0.01);
    EXPECT_NEAR(port("cortex-x1", "fdiv s0, s8, s9\nfdiv s1, s8, s9\n"), 2.0, 0.01);
    EXPECT_NEAR(port("cortex-x1", "fdiv v0.4s, v8.4s, v9.4s\nfdiv v1.4s, v8.4s, v9.4s\n"), 7.0, 0.01);
    // A720: scalar fully pipelined (1); ASIMD as the table prints it (Q-form F16 1/8, Q-form F64 1/2).
    EXPECT_NEAR(port("cortex-a720", "fdiv d0, d8, d9\nfdiv d1, d8, d9\n"), 2.0, 0.01);
    EXPECT_NEAR(port("cortex-a720", "fdiv v0.8h, v8.8h, v9.8h\nfdiv v1.8h, v8.8h, v9.8h\n"), 16.0, 0.01);
    EXPECT_NEAR(port("cortex-a720", "fdiv v0.2d, v8.2d, v9.2d\nfdiv v1.2d, v8.2d, v9.2d\n"), 4.0, 0.01);
}
