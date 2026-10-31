// Cortex-A76: values pinned to the Arm Cortex-A76 Software Optimization Guide
// (PJDOC-466751330-7215, Version 10.0, "SOG" below; text copy in tmp/a76sog.txt, "l.N" = line N).
// The items (C-n, D-n, ...) are those of docs/a76-sog.md.
//
// Two kinds of checks:
//  * a76Desc(): the per-instruction data InstrBuilder takes from
//    ModifiedTarget/AArch64/AArch64SchedNeoverseN1.td (latency per def, uops, resource cycles);
//  * a76Facile(): the Facile prediction (src/facile/facile.cpp) of a loop body as mca_tool
//    --dependency ooo --facile computes it for cortex-a76 (uop width 8, Mop width 4 from
//    cpu_traits.cpp, store -> load memory edges on), with the expected value worked out by hand
//    from the SOG in each comment.
//
// The context is built here rather than with AArch64TestContext("cortex-a76") alone: that
// constructor leaves cortex-a76 on libLLVM's stock NeoverseN1Model (tests/test_support.h,
// docs/a76-sog.md X-1), while mca_tool installs the local model.

#include "test_support.h"

#include <map>
#include <string>
#include <vector>

namespace {

// AArch64TestContext with the scheduling model mca_tool installs for cortex-a76
// (what frontend.cpp's initTarget() does: remap, install, wrap).
struct A76Context : AArch64TestContext {
    A76Context() : AArch64TestContext("cortex-a76") {
        llvm::remapSchedClassIndices(*MCII, "cortex-a76");
        llvm::installLocalSchedModel(*STI, "cortex-a76");
        STI = llvm::wrapCustomSubtargetInfo(std::move(STI), "cortex-a76");
    }
};

struct A76InstInfo {
    unsigned MaxLatency = 0;
    unsigned NumMicroOps = 0;
    std::vector<unsigned> WriteLatencies;  // per def, in operand order (implicit defs last)
    // Total resource cycles per resource name, as the scheduling tables list them (a use of a
    // pipeline also appears under every group that contains it, e.g. M under I and D).
    std::map<std::string, unsigned> ResourceCycles;
    bool Valid = false;
};

A76InstInfo a76Desc(const std::string &AsmLine) {
    initLLVMAArch64();
    A76Context TC;
    auto Instrs = parseAsm(TC, AsmLine);
    A76InstInfo Info;
    if (Instrs.size() != 1)
        return Info;
    mca::InstrumentManager IM(*TC.STI, *TC.MCII);
    mca::InstrBuilder IB(*TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), IM, 0);
    auto ExpectedInst = IB.createInstruction(Instrs[0].Inst, {});
    if (!ExpectedInst) {
        consumeError(ExpectedInst.takeError());
        return Info;
    }
    const mca::Instruction &IS = **ExpectedInst;
    Info.MaxLatency = IS.getLatency();
    Info.NumMicroOps = IS.getNumMicroOps();
    for (const mca::WriteState &WS : IS.getDefs())
        Info.WriteLatencies.push_back(WS.getLatency());
    const MCSchedModel &SM = TC.STI->getSchedModel();
    const MCSchedClassDesc *SCD = SM.getSchedClassDesc(IS.getDesc().SchedClassID);
    if (SCD) {
        for (const MCWriteProcResEntry *E = TC.STI->getWriteProcResBegin(SCD),
                                       *End = TC.STI->getWriteProcResEnd(SCD);
             E != End; ++E)
            Info.ResourceCycles[SM.getProcResource(E->ProcResourceIdx)->Name] +=
                E->ReleaseAtCycle - E->AcquireAtCycle;
    }
    Info.Valid = true;
    return Info;
}

// Facile prediction for the loop body `Body` (one instruction per line, no loop branch), with the
// widths and the memory edges of `mca_tool --mcpu cortex-a76 --dependency ooo --facile`.
facile::FacileResult a76Facile(const std::string &Body) {
    initLLVMAArch64();
    A76Context TC;
    auto Instrs = parseAsm(TC, Body);
    auto Analyzer = MLPAnalyzer::create(*TC.STI);
    mca::InstrumentManager IM(*TC.STI, *TC.MCII);
    mca::InstrBuilder IB(*TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), IM, 0);
    std::vector<std::unique_ptr<mca::Instruction>> SimInstrs;
    std::vector<const MCInst *> MCInsts;
    std::vector<MemAccessInfo> MemInfos;
    for (const auto &I : Instrs) {
        auto ExpectedInst = IB.createInstruction(I.Inst, {});
        if (!ExpectedInst) {
            consumeError(ExpectedInst.takeError());
            continue;
        }
        SimInstrs.push_back(std::move(*ExpectedInst));
        MCInsts.push_back(&I.Inst);
        MemInfos.push_back(Analyzer->getMemAccessInfo(I.Inst, TC.MCII->get(I.Inst.getOpcode()),
                                                      *TC.MRI, *TC.MCII));
    }
    const CpuTraits &T = getCpuTraits("cortex-a76");
    return facile::computeFacilePrediction(*TC.STI, *TC.MCII, *TC.MRI, SimInstrs, MCInsts,
                                           T.UopDispatchWidth, MemInfos, T.MopDispatchWidth);
}

double a76Cycles(const std::string &Body) { return a76Facile(Body).EstimatedCycles; }

constexpr double kTol = 0.01;

} // namespace

// ---------------------------------------------------------------------------
// C-1, C-2 Dispatch: 4 Mops and 8 uops per cycle (SOG 4.1, l.2044); eight pipelines, three of
// them integer (S0, S1, M) (sec. 2.1).  U-1: the ROB size is not in the SOG; 128 is kept.
// ---------------------------------------------------------------------------
TEST(A76SogTest, DispatchAndPipelines) {
    const CpuTraits &T = getCpuTraits("cortex-a76");
    EXPECT_EQ(T.UopDispatchWidth, 8u);
    EXPECT_EQ(T.MopDispatchWidth, 4u);
    EXPECT_FALSE(T.ZeroLatencyMov);           // the SOG has no zero-latency MOV section (C-4)
    EXPECT_EQ(T.Fusion, FusionTable::None);   // nor an instruction-fusion section (C-4)
    initLLVMAArch64();
    A76Context TC;
    EXPECT_EQ(TC.STI->getSchedModel().MicroOpBufferSize, 128u);  // U-1
    // Eight independent ADDs on three integer pipelines: 8/3 (the Mop bound would be 2).
    auto R = a76Facile("add x1, x1, #1\nadd x2, x2, #1\nadd x3, x3, #1\nadd x4, x4, #1\n"
                       "add x5, x5, #1\nadd x6, x6, #1\nadd x7, x7, #1\nadd x8, x8, #1");
    EXPECT_NEAR(R.IssueBound, 2.0, kTol);
    EXPECT_NEAR(R.EstimatedCycles, 8.0 / 3.0, kTol);
    // Four NOPs: no pipeline, one Mop each: 1 cycle (C-3).
    EXPECT_TRUE(a76Desc("nop").ResourceCycles.empty());
    EXPECT_NEAR(a76Cycles("nop\nnop\nnop\nnop"), 1.0, kTol);
}

// ---------------------------------------------------------------------------
// C-7 Branches (SOG 3.3, l.313-323), incl. the fixed BLR x30 latency 2.
// ---------------------------------------------------------------------------
TEST(A76SogTest, Branches) {
    A76InstInfo B = a76Desc("b #8");
    EXPECT_EQ(B.ResourceCycles["N1UnitB"], 1u);
    A76InstInfo Bl = a76Desc("bl #8");
    EXPECT_EQ(Bl.ResourceCycles["N1UnitB"], 1u);
    EXPECT_EQ(Bl.ResourceCycles["N1UnitI"], 1u);
    // "Branch and link, register (reg == lr) BLR 2" (l.321-322), in the scheduling class.  MCA's
    // InstrBuilder gives every call latency 0 whatever the model says (X-2), so the class latency is
    // read with MCSchedModel::computeInstrLatency.
    initLLVMAArch64();
    A76Context TC;
    auto Blr = parseAsm(TC, "blr x1\nblr x30");
    ASSERT_EQ(Blr.size(), 2u);
    auto classLatency = [&](const MCInst &MI) {
        const MCSchedModel &SM = TC.STI->getSchedModel();
        unsigned SC = TC.MCII->get(MI.getOpcode()).getSchedClass();
        while (SM.getSchedClassDesc(SC)->isVariant())
            SC = TC.STI->resolveVariantSchedClass(SC, &MI, TC.MCII.get(), SM.getProcessorID());
        return MCSchedModel::computeInstrLatency(*TC.STI, *SM.getSchedClassDesc(SC));
    };
    EXPECT_EQ(classLatency(Blr[0].Inst), 1);
    EXPECT_EQ(classLatency(Blr[1].Inst), 2);
    A76InstInfo BlrLr = a76Desc("blr x30");
    EXPECT_EQ(BlrLr.MaxLatency, 0u);  // X-2
    EXPECT_EQ(BlrLr.ResourceCycles["N1UnitB"], 1u);
    EXPECT_EQ(BlrLr.ResourceCycles["N1UnitI"], 1u);
}

// ---------------------------------------------------------------------------
// C-8 Arithmetic and logical (SOG 3.4, l.348-362), incl. the fixed ANDS/BICS without a shift.
// ---------------------------------------------------------------------------
TEST(A76SogTest, ArithmeticAndLogical) {
    EXPECT_EQ(a76Desc("adds x1, x2, x3").MaxLatency, 1u);
    EXPECT_EQ(a76Desc("add x1, x2, x3, lsl #4").MaxLatency, 1u);
    A76InstInfo Lsl5 = a76Desc("add x1, x2, x3, lsl #5");
    EXPECT_EQ(Lsl5.MaxLatency, 2u);
    EXPECT_EQ(Lsl5.ResourceCycles["N1UnitM"], 1u);
    EXPECT_EQ(a76Desc("add x1, x2, x3, lsr #1").MaxLatency, 2u);
    EXPECT_EQ(a76Desc("add x1, x2, w3, uxtw #2").MaxLatency, 2u);
    // "Logical, basic AND{S}, BIC{S}" 1 3 I (l.359); "Logical, shift, flagset" 2 1 M (l.362).
    for (const char *Asm : {"tst x1, x2", "ands w1, w2, w3", "bics x1, x2, x3"}) {
        A76InstInfo I = a76Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 1u) << Asm;
        EXPECT_EQ(I.ResourceCycles.count("N1UnitM"), 0u) << Asm;
    }
    A76InstInfo AndsSh = a76Desc("ands x1, x2, x3, lsl #2");
    EXPECT_EQ(AndsSh.MaxLatency, 2u);
    EXPECT_EQ(AndsSh.ResourceCycles["N1UnitM"], 1u);
    // TST + CSEL through NZCV and x1: 1 + 1 (it was 2 + 1).
    EXPECT_NEAR(a76Cycles("tst x1, x2\ncsel x1, x1, x3, eq"), 2.0, kTol);
    EXPECT_NEAR(a76Cycles("tst x1, x2, lsl #2\ncsel x1, x1, x3, eq"), 3.0, kTol);
}

// ---------------------------------------------------------------------------
// C-9 Divide and multiply (SOG 3.6, l.425-501); fixed divide latency (best case 5, D-1) and D-3
// (the divider is a resource of its own; M is taken for one cycle).
// ---------------------------------------------------------------------------
TEST(A76SogTest, DivideAndMultiply) {
    for (const char *Asm : {"sdiv w1, w2, w3", "udiv w1, w2, w3", "sdiv x1, x2, x3", "udiv x1, x2, x3"}) {
        A76InstInfo D = a76Desc(Asm);
        EXPECT_EQ(D.MaxLatency, 5u) << Asm;
        EXPECT_EQ(D.ResourceCycles["N1UnitIDiv"], 5u) << Asm;
        EXPECT_EQ(D.ResourceCycles["N1UnitM"], 1u) << Asm;
    }
    EXPECT_NEAR(a76Cycles("sdiv x1, x1, x2"), 5.0, kTol);
    EXPECT_NEAR(a76Cycles("udiv x1, x2, x3\nudiv x4, x5, x6"), 10.0, kTol);
    // A divide and three X-form MULs (M held 3 cycles each, note 4): M-bound 1 + 9 = 10.
    EXPECT_NEAR(a76Cycles("udiv x1, x2, x3\nmul x4, x5, x6\nmul x7, x5, x6\nmul x8, x5, x6"), 10.0, kTol);
    // Multiply accumulate W 2(1), X 4(3) and M held 3 cycles, long 2(1), high 5 and M held 4.
    EXPECT_EQ(a76Desc("madd w1, w2, w3, w4").MaxLatency, 2u);
    A76InstInfo MaddX = a76Desc("madd x1, x2, x3, x4");
    EXPECT_EQ(MaddX.MaxLatency, 4u);
    EXPECT_EQ(MaddX.ResourceCycles["N1UnitM"], 3u);
    EXPECT_EQ(a76Desc("smaddl x1, w2, w3, x4").MaxLatency, 2u);
    A76InstInfo Mh = a76Desc("smulh x1, x2, x3");
    EXPECT_EQ(Mh.MaxLatency, 5u);
    EXPECT_EQ(Mh.ResourceCycles["N1UnitM"], 4u);
    EXPECT_NEAR(a76Cycles("madd w1, w2, w3, w1"), 1.0, kTol);
    EXPECT_NEAR(a76Cycles("madd x1, x2, x3, x1"), 3.0, kTol);
    EXPECT_NEAR(a76Cycles("smulh x1, x2, x1"), 5.0, kTol);  // the "(3)" is a stall, not a forward
}

// ---------------------------------------------------------------------------
// C-10 Miscellaneous data processing (SOG 3.8, l.548-569).
// ---------------------------------------------------------------------------
TEST(A76SogTest, MiscDataProcessing) {
    EXPECT_EQ(a76Desc("ror x1, x2, #3").MaxLatency, 1u);
    A76InstInfo Extr = a76Desc("extr x1, x2, x3, #3");
    EXPECT_EQ(Extr.MaxLatency, 3u);
    EXPECT_EQ(Extr.NumMicroOps, 2u);
    EXPECT_EQ(a76Desc("ubfx x1, x2, #3, #4").MaxLatency, 1u);
    EXPECT_EQ(a76Desc("bfi x1, x2, #3, #4").MaxLatency, 2u);
    EXPECT_EQ(a76Desc("lsl x1, x2, x3").MaxLatency, 1u);
    EXPECT_EQ(a76Desc("rbit x1, x2").MaxLatency, 1u);
    EXPECT_EQ(a76Desc("crc32x w1, w2, x3").MaxLatency, 2u);
    EXPECT_NEAR(a76Cycles("crc32x w1, w1, x3"), 1.0, kTol);  // CRC note 1 (l.2023-2024)
}

// ---------------------------------------------------------------------------
// C-11 Loads (SOG 3.9, l.601-660); fixed LDRH/LDRSH scale by 2, the second register of
// LDP W / LDPSW, LDPSW throughput 1.
// ---------------------------------------------------------------------------
TEST(A76SogTest, Loads) {
    EXPECT_EQ(a76Desc("ldr x1, [x2, #8]").MaxLatency, 4u);
    A76InstInfo Post = a76Desc("ldr x1, [x2], #8");
    EXPECT_EQ(Post.ResourceCycles["N1UnitL"], 1u);
    EXPECT_EQ(Post.ResourceCycles["N1UnitI"], 1u);
    A76InstInfo Ldrh2 = a76Desc("ldrh w1, [x2, x3, lsl #1]");
    EXPECT_EQ(Ldrh2.MaxLatency, 5u);
    EXPECT_EQ(Ldrh2.ResourceCycles["N1UnitI"], 1u);
    A76InstInfo Ldrh = a76Desc("ldrh w1, [x2, x3]");
    EXPECT_EQ(Ldrh.MaxLatency, 4u);
    EXPECT_EQ(Ldrh.ResourceCycles.count("N1UnitI"), 0u);
    EXPECT_EQ(a76Desc("ldrsh x1, [x2, w3, sxtw #1]").MaxLatency, 5u);
    EXPECT_EQ(a76Desc("ldr x1, [x2, x3, lsl #3]").MaxLatency, 4u);
    // Load pair W: both registers 4 (the second used to be 0, seen as 1 by Facile).
    EXPECT_EQ(a76Desc("ldp w1, w2, [x0]").WriteLatencies, (std::vector<unsigned>{4, 4}));
    EXPECT_NEAR(a76Cycles("ldp w1, w2, [x0]\nadd x0, x0, x2"), 5.0, kTol);
    A76InstInfo Ldpsw = a76Desc("ldpsw x1, x2, [x0]");
    EXPECT_EQ(Ldpsw.WriteLatencies, (std::vector<unsigned>{5, 5}));
    EXPECT_EQ(Ldpsw.ResourceCycles["N1UnitL"], 2u);  // throughput 1
    EXPECT_NEAR(a76Cycles("ldpsw x1, x2, [x0]\nldpsw x3, x4, [x0, #8]"), 2.0, kTol);
    EXPECT_EQ(a76Desc("ldp x1, x2, [x0]").ResourceCycles["N1UnitL"], 2u);
}

// ---------------------------------------------------------------------------
// C-12 Stores (SOG 3.10, l.737-787); fixed: the store-data pipelines D are S1 and M (sec. 3.2, C-2),
// STRH scaled by 2, STNP W; D-6 / I-3: no I uop for the base update of integer stores.
// ---------------------------------------------------------------------------
TEST(A76SogTest, Stores) {
    A76InstInfo Str = a76Desc("str x1, [x2, #8]");
    EXPECT_EQ(Str.ResourceCycles["N1UnitL"], 1u);
    EXPECT_EQ(Str.ResourceCycles["N1UnitD"], 1u);
    EXPECT_EQ(Str.ResourceCycles["N1UnitI"], 1u);   // D is part of the integer pipes
    // Two ADDs and two STRs: four integer-pipe uops on S0, S1, M (it was 2/3 + L).
    EXPECT_NEAR(a76Cycles("add x1, x1, #1\nadd x2, x2, #1\nstr x4, [x0]\nstr x5, [x0, #8]"), 4.0 / 3.0, kTol);
    // Integer STR post-index "1 2 L, D": no I uop for the base update (I-3), two per cycle.
    A76InstInfo StrPost = a76Desc("str x1, [x2], #8");
    EXPECT_EQ(StrPost.ResourceCycles["N1UnitI"], 1u);  // the data uop only
    EXPECT_EQ(StrPost.WriteLatencies, (std::vector<unsigned>{1}));
    EXPECT_NEAR(a76Cycles("str x1, [x0], #8\nstr x2, [x3], #8"), 1.0, kTol);
    // STRH register offset scaled by 2: "2 3/2 I, L, D".
    A76InstInfo Strh2 = a76Desc("strh w1, [x2, x3, lsl #1]");
    EXPECT_EQ(Strh2.MaxLatency, 2u);
    EXPECT_EQ(Strh2.ResourceCycles["N1UnitI"], 2u);
    EXPECT_NEAR(a76Cycles("strh w1, [x0, x2, lsl #1]\nstrh w1, [x0, x3, lsl #1]\nstrh w1, [x0, x4, lsl #1]"),
                2.0, kTol);
    EXPECT_EQ(a76Desc("strh w1, [x2, x3]").ResourceCycles["N1UnitI"], 1u);
    // STP/STNP W "1 2 L, D"; STP X "1 1 L, D".
    EXPECT_EQ(a76Desc("stnp w1, w2, [x0]").ResourceCycles["N1UnitL"], 1u);
    EXPECT_EQ(a76Desc("stp w1, w2, [x0]").ResourceCycles["N1UnitL"], 1u);
    EXPECT_EQ(a76Desc("stp x1, x2, [x0]").ResourceCycles["N1UnitL"], 2u);
}

// ---------------------------------------------------------------------------
// C-13, C-14 FP data processing and miscellaneous (SOG 3.11, 3.12); D-1 best case of the divide
// ranges, D-3 the FP divide unit; fixed FMUL -> FMA forwarding, FCVT, FCVTXN, fixed-point FCVTZS; D-4 scalar SIMD-register
// conversions.
// ---------------------------------------------------------------------------
TEST(A76SogTest, FpScalar) {
    EXPECT_EQ(a76Desc("fadd d0, d1, d2").MaxLatency, 2u);
    EXPECT_EQ(a76Desc("fmul d0, d1, d2").MaxLatency, 3u);
    EXPECT_EQ(a76Desc("fcmp d0, d1").ResourceCycles["N1UnitV0"], 1u);
    A76InstInfo Fdiv = a76Desc("fdiv d0, d1, d2");
    EXPECT_EQ(Fdiv.MaxLatency, 7u);
    EXPECT_EQ(Fdiv.ResourceCycles["N1UnitFDiv"], 4u);   // 2/7 -> 4 cycles
    EXPECT_EQ(Fdiv.ResourceCycles["N1UnitV0"], 1u);
    EXPECT_EQ(a76Desc("fdiv s0, s1, s2").ResourceCycles["N1UnitFDiv"], 2u);  // 4/7
    EXPECT_EQ(a76Desc("fsqrt d0, d1").ResourceCycles["N1UnitFDiv"], 4u);
    // A divide does not block the other V0 uops (D-3): FDIV D (4 on the divider) with FCVTZS (V0 + V1)
    // and FRINTA (V0): 4 (it was 4 + 1 + 1 = 6 cycles of V0).
    EXPECT_NEAR(a76Cycles("fdiv d0, d2, d1\nfcvtzs w1, d3\nfrinta d4, d5"), 4.0, kTol);
    // FMA accumulator 4(2); FMUL -> FMA accumulator 1 (notes 2, 3).
    EXPECT_NEAR(a76Cycles("fmadd d0, d1, d2, d0"), 2.0, kTol);
    // fmul (-> accumulator 3 - 2 = 1) + fmadd (4) + fmov (2) around the loop: 7 (the accumulator edge
    // used to keep 3: 9).  The vector forms likewise (FMUL 3 -> FMLA 1, FMLA 4, MOV 2).
    EXPECT_NEAR(a76Cycles("fmul d0, d1, d2\nfmadd d3, d4, d5, d0\nfmov d1, d3"), 7.0, kTol);
    EXPECT_NEAR(a76Cycles("fmul v0.4s, v1.4s, v2.4s\nfmla v0.4s, v4.4s, v5.4s\nmov v1.16b, v0.16b"), 7.0, kTol);
    // FP convert vec to vec "3 1 V0" (l.936).
    for (const char *Asm : {"fcvt s0, d1", "fcvt d0, h1", "fcvtxn s0, d1"}) {
        A76InstInfo I = a76Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 3u) << Asm;
        EXPECT_EQ(I.ResourceCycles["N1UnitV0"], 1u) << Asm;
    }
    // FP convert vec to gen "4 1 V0, V1", also the fixed-point form.
    for (const char *Asm : {"fcvtzs w0, d1", "fcvtzs w0, d1, #3", "fcvtzu x0, s1, #10"}) {
        A76InstInfo I = a76Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 4u) << Asm;
        EXPECT_EQ(I.ResourceCycles["N1UnitV1"], 1u) << Asm;
    }
    EXPECT_EQ(a76Desc("scvtf d0, x1").MaxLatency, 6u);
    // D-4: SCVTF/FCVTZS with operand and result in FP registers: 3 1 V0.
    for (const char *Asm : {"scvtf d0, d1", "ucvtf s0, s1", "fcvtzs d0, d1", "fcvtzs s0, s1, #4"}) {
        A76InstInfo I = a76Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 3u) << Asm;
        EXPECT_EQ(I.ResourceCycles["N1UnitV0"], 1u) << Asm;
    }
    EXPECT_EQ(a76Desc("fmov d0, x1").MaxLatency, 3u);
    EXPECT_EQ(a76Desc("fmov x0, d1").ResourceCycles["N1UnitV1"], 1u);
}

// ---------------------------------------------------------------------------
// C-15, C-16 FP loads and stores (SOG 3.13, 3.14); fixed: second register of LDP S/D/Q, LDP Q
// without I, LDNP Q, H/Q register offset scale, STR H/Q register offset, STP S pre/post.
// ---------------------------------------------------------------------------
TEST(A76SogTest, FpLoadsAndStores) {
    EXPECT_EQ(a76Desc("ldr d0, [x1, #8]").MaxLatency, 5u);
    EXPECT_EQ(a76Desc("ldp d0, d1, [x0]").WriteLatencies, (std::vector<unsigned>{5, 5}));
    A76InstInfo LdpQ = a76Desc("ldp q0, q1, [x0]");
    EXPECT_EQ(LdpQ.WriteLatencies, (std::vector<unsigned>{7, 7}));
    EXPECT_EQ(LdpQ.ResourceCycles.count("N1UnitI"), 0u);
    EXPECT_EQ(LdpQ.ResourceCycles["N1UnitL"], 2u);
    EXPECT_EQ(a76Desc("ldnp q0, q1, [x0]").WriteLatencies, (std::vector<unsigned>{7, 7}));
    EXPECT_EQ(a76Desc("ldp q0, q1, [x0], #32").WriteLatencies, (std::vector<unsigned>{1, 7, 7}));
    EXPECT_EQ(a76Desc("ldr q0, [x1, x2]").MaxLatency, 5u);
    EXPECT_EQ(a76Desc("ldr q0, [x1, x2, lsl #4]").MaxLatency, 6u);
    EXPECT_EQ(a76Desc("ldr h0, [x1, x2, lsl #1]").MaxLatency, 6u);
    EXPECT_EQ(a76Desc("str h0, [x1, x2]").ResourceCycles.count("N1UnitI"), 0u);
    EXPECT_EQ(a76Desc("str h0, [x1, x2, lsl #1]").ResourceCycles["N1UnitI"], 1u);
    EXPECT_EQ(a76Desc("str q0, [x1, x2]").ResourceCycles.count("N1UnitI"), 0u);
    EXPECT_EQ(a76Desc("str q0, [x1, x2, lsl #4]").ResourceCycles["N1UnitI"], 1u);
    EXPECT_EQ(a76Desc("str q0, [x1]").ResourceCycles["N1UnitL"], 2u);       // 2 1 L, V
    EXPECT_EQ(a76Desc("stp q0, q1, [x1]").ResourceCycles["N1UnitL"], 4u);   // 3 1/2 L, V
    EXPECT_EQ(a76Desc("stp q0, q1, [x1, #32]!").ResourceCycles["N1UnitL"], 4u);   // pre-index 1/2
    EXPECT_EQ(a76Desc("stp q0, q1, [x1], #32").ResourceCycles["N1UnitL"], 2u);    // post-index 1 (I-5)
    EXPECT_EQ(a76Desc("stp s0, s1, [x1]").ResourceCycles["N1UnitL"], 1u);   // 2 2
    EXPECT_EQ(a76Desc("stp s0, s1, [x1], #8").ResourceCycles["N1UnitL"], 2u);  // 2 1
}

// ---------------------------------------------------------------------------
// C-17 to C-19 ASIMD (SOG 3.15-3.17); fixed by-element multiplies, SQSHRUN; D-5 scalar forms.
// ---------------------------------------------------------------------------
TEST(A76SogTest, Asimd) {
    EXPECT_EQ(a76Desc("add v0.4s, v1.4s, v2.4s").MaxLatency, 2u);
    A76InstInfo MulQ = a76Desc("mul v0.4s, v1.4s, v2.s[1]");
    EXPECT_EQ(MulQ.MaxLatency, 5u);
    EXPECT_EQ(MulQ.ResourceCycles["N1UnitV0"], 2u);
    EXPECT_EQ(a76Desc("mul v0.4h, v1.4h, v2.h[1]").MaxLatency, 4u);
    EXPECT_EQ(a76Desc("sqdmulh v0.4s, v1.4s, v2.s[1]").MaxLatency, 5u);
    // MLA by element, D-form: 4(1) on V0.
    EXPECT_NEAR(a76Cycles("mla v0.4h, v1.4h, v2.h[1]"), 1.0, kTol);
    EXPECT_EQ(a76Desc("mla v0.4h, v1.4h, v2.h[1]").ResourceCycles["N1UnitV0"], 1u);
    EXPECT_NEAR(a76Cycles("mla v0.4s, v1.4s, v2.4s"), 2.0, kTol);
    A76InstInfo Sqshrun = a76Desc("sqshrun v0.8b, v1.8h, #3");
    EXPECT_EQ(Sqshrun.MaxLatency, 4u);
    EXPECT_EQ(Sqshrun.ResourceCycles["N1UnitV1"], 1u);
    EXPECT_EQ(a76Desc("shl d0, d1, #3").ResourceCycles["N1UnitV1"], 1u);
    EXPECT_EQ(a76Desc("srshr d0, d1, #3").MaxLatency, 4u);
    EXPECT_EQ(a76Desc("sqdmlal s0, h1, h2").MaxLatency, 4u);
    EXPECT_NEAR(a76Cycles("saba v0.16b, v1.16b, v2.16b"), 1.0, kTol);   // 4(1)
    EXPECT_EQ(a76Desc("addv b0, v1.16b").MaxLatency, 6u);
    EXPECT_EQ(a76Desc("addv s0, v1.4s").MaxLatency, 3u);
    EXPECT_EQ(a76Desc("dup v0.4s, w1").ResourceCycles["N1UnitM"], 1u);
    EXPECT_EQ(a76Desc("ins v0.s[1], w1").MaxLatency, 5u);
    EXPECT_EQ(a76Desc("tbl v0.16b, {v1.16b, v2.16b, v3.16b}, v4.16b").ResourceCycles["N1UnitV"], 4u);
    EXPECT_EQ(a76Desc("fcvtzs v0.4s, v1.4s, #3").MaxLatency, 4u);
    EXPECT_NEAR(a76Cycles("fmla v0.4s, v1.4s, v2.4s"), 2.0, kTol);
}

// ---------------------------------------------------------------------------
// C-20, C-21 ASIMD loads and stores (SOG 3.18, 3.19); fixed throughput of LD3, LD4 multiple and
// most stores (D-7: expressed as occupancy of the L pipes).
// ---------------------------------------------------------------------------
TEST(A76SogTest, AsimdLoadsAndStores) {
    EXPECT_EQ(a76Desc("ld1 {v0.16b}, [x0]").MaxLatency, 5u);
    EXPECT_EQ(a76Desc("ld1 {v0.b}[1], [x0]").MaxLatency, 7u);
    // L cycles = 2 / throughput.
    EXPECT_EQ(a76Desc("ld3 {v0.16b, v1.16b, v2.16b}, [x0]").ResourceCycles["N1UnitL"], 4u);   // 1/2
    EXPECT_EQ(a76Desc("ld4 {v0.8b, v1.8b, v2.8b, v3.8b}, [x0]").ResourceCycles["N1UnitL"], 7u);  // 2/7
    EXPECT_EQ(a76Desc("ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0]").ResourceCycles["N1UnitL"], 10u);  // 1/5
    EXPECT_EQ(a76Desc("st1 {v0.8b}, [x0]").ResourceCycles["N1UnitL"], 1u);    // 2
    EXPECT_EQ(a76Desc("st1 {v0.16b}, [x0]").ResourceCycles["N1UnitL"], 2u);   // 1
    EXPECT_EQ(a76Desc("st1 {v0.16b, v1.16b}, [x0]").ResourceCycles["N1UnitL"], 4u);   // 1/2
    EXPECT_EQ(a76Desc("st1 {v0.b}[1], [x0]").ResourceCycles["N1UnitL"], 2u);  // 1
    EXPECT_EQ(a76Desc("st4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0]").ResourceCycles["N1UnitL"], 12u);  // 1/6
    EXPECT_NEAR(a76Cycles("st1 {v0.16b}, [x0]\nst1 {v1.16b}, [x1]"), 2.0, kTol);
    // The uop counts are unchanged.
    EXPECT_EQ(a76Desc("st4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0]").NumMicroOps, 12u);
}

// ---------------------------------------------------------------------------
// C-22, C-23 Crypto and CRC (SOG 3.20, 3.21).
// ---------------------------------------------------------------------------
TEST(A76SogTest, CryptoAndCrc) {
    A76InstInfo Aese = a76Desc("aese v0.16b, v1.16b");
    EXPECT_EQ(Aese.MaxLatency, 2u);
    EXPECT_EQ(Aese.ResourceCycles["N1UnitV0"], 1u);
    EXPECT_EQ(a76Desc("pmull v0.1q, v1.1d, v2.1d").MaxLatency, 2u);
    EXPECT_EQ(a76Desc("sha1c q0, s1, v2.4s").MaxLatency, 4u);
    EXPECT_EQ(a76Desc("sha256su1 v0.4s, v1.4s, v2.4s").MaxLatency, 2u);
    A76InstInfo Crc = a76Desc("crc32cw w1, w2, w3");
    EXPECT_EQ(Crc.MaxLatency, 2u);
    EXPECT_EQ(Crc.ResourceCycles["N1UnitM"], 1u);
}

// ---------------------------------------------------------------------------
// B-1 Store data is read one cycle late (the A55/A520 decision, not SOG-based): a value that goes
// through memory from a load to the next iteration's load costs 3 + 1 (store) instead of 4 + 1.
// ---------------------------------------------------------------------------
TEST(A76SogTest, BorrowedStoreDataNeededLater) {
    EXPECT_NEAR(a76Cycles("ldr x1, [x0]\nstr x1, [x0]"), 4.0, kTol);
    EXPECT_NEAR(a76Cycles("ldr x1, [x0]\nstp x1, x2, [x0]"), 4.0, kTol);
    EXPECT_NEAR(a76Cycles("ldr d1, [x0]\nfadd d1, d1, d2\nstr d1, [x0]"), 8.0, kTol);  // 5 + 2 - 1 + 2
    // A 1-cycle producer is not changed (Facile's 1-cycle floor): 4 + 1 + 1.
    EXPECT_NEAR(a76Cycles("ldr x1, [x0]\nadd x1, x1, #1\nstr x1, [x0]"), 6.0, kTol);
}
