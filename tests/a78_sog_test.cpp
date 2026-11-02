// Cortex-A78: values pinned to the Arm Cortex-A78 Core Software Optimization Guide
// (PJDOC-466751330-9691, Issue 4.0, "SOG" below; text copy in tmp/a78sog.txt, "l.N" = line N).
// The items (C-n, D-n, ...) are those of docs/a78-sog.md.
//
// Two kinds of checks:
//  * a78Desc(): the per-instruction data InstrBuilder takes from
//    ModifiedTarget/AArch64/AArch64SchedNeoverseN2.td (latency per def, uops, resource cycles);
//  * a78Facile(): the Facile prediction (src/facile/facile.cpp) of a loop body as mca_tool
//    --dependency ooo --facile computes it for cortex-a78 (uop width 12, Mop width 6, the A78
//    fusion table and zero-latency MOVs from cpu_traits.cpp, store -> load memory edges on), with
//    the expected value worked out by hand from the SOG in each comment.
//
// The context is built here rather than with AArch64TestContext("cortex-a78") alone: that
// constructor leaves cortex-a78 on libLLVM's stock NeoverseN2Model (tests/test_support.h,
// docs/a78-sog.md X-1), while mca_tool installs the local model.

#include "test_support.h"

#include <map>
#include <string>
#include <vector>

namespace {

// AArch64TestContext with the scheduling model mca_tool installs for cortex-a78
// (what frontend.cpp's initTarget() does: remap, install, wrap).
struct A78Context : AArch64TestContext {
    A78Context() : AArch64TestContext("cortex-a78") {
        llvm::remapSchedClassIndices(*MCII, "cortex-a78");
        llvm::installLocalSchedModel(*STI, "cortex-a78");
        STI = llvm::wrapCustomSubtargetInfo(std::move(STI), "cortex-a78");
    }
};

struct A78InstInfo {
    unsigned MaxLatency = 0;
    unsigned NumMicroOps = 0;
    std::vector<unsigned> WriteLatencies;  // per def, in operand order (implicit defs last)
    // Total resource cycles per resource name, as the scheduling tables list them (a use of a
    // pipeline also appears under every group that contains it, e.g. M0 under M and I).
    std::map<std::string, unsigned> ResourceCycles;
    bool Valid = false;
};

A78InstInfo a78Desc(const std::string &AsmLine) {
    initLLVMAArch64();
    A78Context TC;
    auto Instrs = parseAsm(TC, AsmLine);
    A78InstInfo Info;
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
// widths, fusion and memory edges of `mca_tool --mcpu cortex-a78 --dependency ooo --facile`.
facile::FacileResult a78Facile(const std::string &Body) {
    initLLVMAArch64();
    A78Context TC;
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
    const CpuTraits &T = getCpuTraits("cortex-a78");
    return facile::computeFacilePrediction(*TC.STI, *TC.MCII, *TC.MRI, SimInstrs, MCInsts,
                                           T.UopDispatchWidth, MemInfos, T.MopDispatchWidth);
}

double a78Cycles(const std::string &Body) { return a78Facile(Body).EstimatedCycles; }

constexpr double kTol = 0.01;

} // namespace

// ---------------------------------------------------------------------------
// C-1, C-2, C-3 Dispatch: 6 Mops and 12 uops per cycle (sec. 4.1, l.2455); thirteen issue pipelines
// (sec. 2.1, l.389-390), four of them integer (S0, S1, M0, M1).  U-1: no ROB size in the SOG.
// ---------------------------------------------------------------------------
TEST(A78SogTest, DispatchAndPipelines) {
    const CpuTraits &T = getCpuTraits("cortex-a78");
    EXPECT_EQ(T.UopDispatchWidth, 12u);
    EXPECT_EQ(T.MopDispatchWidth, 6u);
    EXPECT_TRUE(T.ZeroLatencyMov);           // sec. 4.15 (C-4)
    EXPECT_EQ(T.Fusion, FusionTable::A78);   // sec. 4.14 (C-5)
    initLLVMAArch64();
    A78Context TC;
    EXPECT_EQ(TC.STI->getSchedModel().MicroOpBufferSize, 160u);  // U-1
    // Eight independent ADDs on four integer pipelines: 2 (the Mop bound is 8/6).
    auto R = a78Facile("add x1, x1, #1\nadd x2, x2, #1\nadd x3, x3, #1\nadd x4, x4, #1\n"
                       "add x5, x5, #1\nadd x6, x6, #1\nadd x7, x7, #1\nadd x8, x8, #1");
    EXPECT_NEAR(R.IssueBound, 8.0 / 6.0, kTol);
    EXPECT_NEAR(R.EstimatedCycles, 2.0, kTol);
    // Flag-setting ALU: "ALU, basic, flagset ... 1 3 I" (l.561-563): three per cycle (D-9).
    EXPECT_NEAR(a78Cycles("adds x1, x1, #1\nadds x2, x2, #1\nadds x3, x3, #1\nadds x4, x4, #1\n"
                          "adds x5, x5, #1\nadds x6, x6, #1"), 2.0, kTol);
    // NOP: no pipeline (C-3).
    EXPECT_TRUE(a78Desc("nop").ResourceCycles.empty());
}

// ---------------------------------------------------------------------------
// C-4 Zero-latency MOVs (sec. 4.15, l.2829-2858): MOV #0, MOV ZR, MOV Wd,Wn / Xd,Xn only.
// ---------------------------------------------------------------------------
TEST(A78SogTest, ZeroLatencyMovs) {
    for (const char *Asm : {"mov x1, #0", "mov w1, #0", "mov x1, xzr", "mov w1, wzr", "mov x1, x2",
                            "mov w1, w2"}) {
        A78InstInfo I = a78Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 0u) << Asm;
        EXPECT_TRUE(I.ResourceCycles.empty()) << Asm;
    }
    // Not in the A78 list: FMOV from WZR/XZR (3 1 M0, l.1310), MOVI #0 (2 2 V, l.1947).
    A78InstInfo Fz = a78Desc("fmov s0, wzr");
    EXPECT_EQ(Fz.MaxLatency, 3u);
    EXPECT_EQ(Fz.ResourceCycles["N2UnitM0"], 1u);
    for (const char *Asm : {"movi d0, #0", "movi v0.2d, #0"}) {
        A78InstInfo I = a78Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 2u) << Asm;
        EXPECT_EQ(I.ResourceCycles["N2UnitV"], 1u) << Asm;
    }
    EXPECT_EQ(a78Desc("mov x1, #1").MaxLatency, 1u);
    EXPECT_EQ(a78Desc("mov v0.16b, v1.16b").MaxLatency, 2u);
    // A chain through a register move costs nothing: add (1) + mov (0).
    EXPECT_NEAR(a78Cycles("add x1, x2, #1\nmov x2, x1"), 1.0, kTol);
}

// ---------------------------------------------------------------------------
// C-5 Instruction fusion (sec. 4.14, l.2805-2825): the second instruction of a pair (the NOP of
// "NOP + Any") takes no Mop, uop or pipeline (D-11).
// ---------------------------------------------------------------------------
TEST(A78SogTest, Fusion) {
    EXPECT_EQ(a78Facile("cmp x1, #3\nb.eq #8").TotalMicroOps, 1u);
    EXPECT_EQ(a78Facile("cmp x1, x2\ncsel x3, x4, x5, eq").TotalMicroOps, 1u);
    EXPECT_EQ(a78Facile("cmp x1, #3\ncset x3, eq").TotalMicroOps, 1u);
    EXPECT_EQ(a78Facile("tst x1, #3\nb.eq #8").TotalMicroOps, 1u);
    EXPECT_EQ(a78Facile("bics x1, x2, x3\nb.eq #8").TotalMicroOps, 1u);
    EXPECT_EQ(a78Facile("nop\nadd x1, x2, x3").TotalMicroOps, 1u);
    // Not listed: CMN/TST + CSEL, a flag-setting op with a real destination + B.cond.
    EXPECT_EQ(a78Facile("cmn x1, #3\ncsel x3, x4, x5, eq").TotalMicroOps, 2u);
    EXPECT_EQ(a78Facile("tst x1, #3\ncsel x3, x4, x5, eq").TotalMicroOps, 2u);
    EXPECT_EQ(a78Facile("subs x1, x1, #3\nb.eq #8").TotalMicroOps, 2u);
    // AESE + AESMC is not masked (D-11, U-6): both take a V pipe.
    EXPECT_EQ(a78Facile("aese v0.16b, v1.16b\naesmc v0.16b, v0.16b").TotalMicroOps, 2u);
}

// ---------------------------------------------------------------------------
// C-8 Branches (sec. 3.3, l.528-537): two branch pipelines; BL/BLR B, S.
// ---------------------------------------------------------------------------
TEST(A78SogTest, Branches) {
    EXPECT_EQ(a78Desc("b #8").ResourceCycles["N2UnitB"], 1u);
    A78InstInfo Bl = a78Desc("bl #8");
    EXPECT_EQ(Bl.ResourceCycles["N2UnitB"], 1u);
    EXPECT_EQ(Bl.ResourceCycles["N2UnitS"], 1u);
    EXPECT_EQ(Bl.MaxLatency, 0u);  // X-2: MCA gives every call latency 0
    EXPECT_NEAR(a78Cycles("cbz x1, #8\ncbz x2, #8\ncbz x3, #8\ncbz x4, #8"), 2.0, kTol);
}

// ---------------------------------------------------------------------------
// C-9 Arithmetic and logical (sec. 3.4, l.553-586).
// ---------------------------------------------------------------------------
TEST(A78SogTest, ArithmeticAndLogical) {
    EXPECT_EQ(a78Desc("add x1, x2, x3, lsl #4").MaxLatency, 1u);
    A78InstInfo Lsl5 = a78Desc("add x1, x2, x3, lsl #5");
    EXPECT_EQ(Lsl5.MaxLatency, 2u);
    EXPECT_EQ(Lsl5.ResourceCycles["N2UnitM"], 1u);
    EXPECT_EQ(a78Desc("add x1, x2, w3, uxtw #2").MaxLatency, 2u);
    A78InstInfo Tst = a78Desc("tst x1, x2");
    EXPECT_EQ(Tst.MaxLatency, 1u);
    EXPECT_EQ(Tst.ResourceCycles["N2UnitFlg"], 1u);
    EXPECT_EQ(Tst.ResourceCycles.count("N2UnitM"), 0u);
    A78InstInfo AndsSh = a78Desc("ands x1, x2, x3, lsl #2");
    EXPECT_EQ(AndsSh.MaxLatency, 2u);
    EXPECT_EQ(AndsSh.ResourceCycles["N2UnitM"], 1u);
    EXPECT_EQ(a78Desc("csel x1, x2, x3, eq").MaxLatency, 1u);
    EXPECT_EQ(a78Desc("ccmp x1, x2, #0, eq").ResourceCycles["N2UnitFlg"], 1u);
}

// ---------------------------------------------------------------------------
// C-10 Divide and multiply (sec. 3.6, l.695-791): divide best case (D-1) on a divider of its own
// (D-3); SMADDL family on M.
// ---------------------------------------------------------------------------
TEST(A78SogTest, DivideAndMultiply) {
    for (const char *Asm : {"sdiv w1, w2, w3", "udiv w1, w2, w3", "sdiv x1, x2, x3", "udiv x1, x2, x3"}) {
        A78InstInfo D = a78Desc(Asm);
        EXPECT_EQ(D.MaxLatency, 5u) << Asm;
        EXPECT_EQ(D.ResourceCycles["A78UnitIDiv"], 5u) << Asm;
        EXPECT_EQ(D.ResourceCycles["N2UnitM0"], 1u) << Asm;
    }
    EXPECT_NEAR(a78Cycles("sdiv w1, w1, w2"), 5.0, kTol);
    EXPECT_NEAR(a78Cycles("udiv x1, x2, x3\nudiv x4, x5, x6"), 10.0, kTol);
    // A divide and two MADDs (M0, 1 per cycle): divider-bound 5, not M0-bound.
    EXPECT_NEAR(a78Cycles("udiv x1, x2, x3\nmadd x4, x5, x6, x7\nmadd x8, x5, x6, x7"), 5.0, kTol);
    // MUL 2 2 M; MADD 2(1) 1 M0; SMADDL 2(1) 2 M; SMULH 3 2 M; SMULL 2 2 M.
    EXPECT_EQ(a78Desc("mul x1, x2, x3").ResourceCycles.count("N2UnitM0"), 0u);
    EXPECT_EQ(a78Desc("madd x1, x2, x3, x4").ResourceCycles["N2UnitM0"], 1u);
    A78InstInfo Maddl = a78Desc("smaddl x1, w2, w3, x4");
    EXPECT_EQ(Maddl.MaxLatency, 2u);
    EXPECT_EQ(Maddl.ResourceCycles["N2UnitM"], 1u);
    EXPECT_EQ(Maddl.ResourceCycles.count("N2UnitM0"), 0u);
    EXPECT_NEAR(a78Cycles("smaddl x1, w2, w3, x4\nsmaddl x5, w2, w3, x6\nsmaddl x7, w2, w3, x8\n"
                          "smaddl x9, w2, w3, x10"), 2.0, kTol);
    EXPECT_EQ(a78Desc("smulh x1, x2, x3").MaxLatency, 3u);
    // Accumulator forwarding (note 2): 1 per link, from MADD and from SMADDL (U-5).
    EXPECT_NEAR(a78Cycles("madd x1, x2, x3, x1"), 1.0, kTol);
    EXPECT_NEAR(a78Cycles("smaddl x1, w2, w3, x1"), 1.0, kTol);
    EXPECT_NEAR(a78Cycles("madd x1, x2, x3, x4\nsmaddl x4, w2, w3, x1"), 2.0, kTol);
    // No forwarding from multiply high (note 2, l.791): 3 + 2.
    EXPECT_NEAR(a78Cycles("smulh x1, x2, x3\nmadd x3, x4, x5, x1"), 5.0, kTol);
}

// ---------------------------------------------------------------------------
// C-11 Miscellaneous data processing (sec. 3.8, l.881-899), incl. EXTR imm == 0 (note 1).
// ---------------------------------------------------------------------------
TEST(A78SogTest, MiscDataProcessing) {
    EXPECT_EQ(a78Desc("ror x1, x2, #3").MaxLatency, 1u);
    EXPECT_EQ(a78Desc("extr x1, x2, x3, #0").MaxLatency, 1u);
    A78InstInfo Extr = a78Desc("extr x1, x2, x3, #3");
    EXPECT_EQ(Extr.MaxLatency, 3u);
    EXPECT_EQ(Extr.NumMicroOps, 2u);
    EXPECT_EQ(a78Desc("ubfx x1, x2, #3, #4").MaxLatency, 1u);
    A78InstInfo Bfi = a78Desc("bfi x1, x2, #3, #4");
    EXPECT_EQ(Bfi.MaxLatency, 2u);
    EXPECT_EQ(Bfi.ResourceCycles["N2UnitM"], 1u);
    EXPECT_EQ(a78Desc("adrp x1, #4096").MaxLatency, 1u);
    EXPECT_EQ(a78Desc("rbit x1, x2").MaxLatency, 1u);
}

// ---------------------------------------------------------------------------
// C-12 Loads (sec. 3.9, l.956-1020): register offset without I, LDRH/LDRSH scaled 5 (I, L), LDPSW
// 5 3/2 I, L, load-pair second register (D-10).
// ---------------------------------------------------------------------------
TEST(A78SogTest, Loads) {
    A78InstInfo Post = a78Desc("ldr x1, [x2], #8");
    EXPECT_EQ(Post.WriteLatencies, (std::vector<unsigned>{1, 4}));
    EXPECT_EQ(Post.ResourceCycles["N2UnitI"], 1u);
    for (const char *Asm : {"ldr w1, [x2, x3]", "ldr x1, [x2, x3, lsl #3]", "ldrb w1, [x2, w3, uxtw]",
                            "ldrsw x1, [x2, w3, sxtw #2]", "ldrh w1, [x2, x3]"}) {
        A78InstInfo I = a78Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 4u) << Asm;
        EXPECT_EQ(I.NumMicroOps, 1u) << Asm;
        EXPECT_EQ(I.ResourceCycles.count("N2UnitI"), 0u) << Asm;
    }
    for (const char *Asm : {"ldrh w1, [x2, x3, lsl #1]", "ldrsh x1, [x2, w3, sxtw #1]"}) {
        A78InstInfo I = a78Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 5u) << Asm;
        EXPECT_EQ(I.ResourceCycles["N2UnitI"], 1u) << Asm;
    }
    EXPECT_EQ(a78Desc("ldp w1, w2, [x0]").NumMicroOps, 1u);
    A78InstInfo LdpX = a78Desc("ldp x1, x2, [x0]");
    EXPECT_EQ(LdpX.WriteLatencies, (std::vector<unsigned>{4, 4}));
    EXPECT_EQ(LdpX.ResourceCycles["N2UnitL"], 2u);
    A78InstInfo Ldpsw = a78Desc("ldpsw x1, x2, [x0]");
    EXPECT_EQ(Ldpsw.WriteLatencies, (std::vector<unsigned>{5, 5}));
    EXPECT_EQ(Ldpsw.ResourceCycles["N2UnitI"], 1u);
    EXPECT_EQ(Ldpsw.ResourceCycles["N2UnitL"], 2u);
    EXPECT_NEAR(a78Cycles("ldpsw x1, x2, [x0]\nadd x0, x0, x2"), 6.0, kTol);
    // Three LDPSW: 6 L-cycles on three L pipes = 2 (3/2 per cycle).
    EXPECT_NEAR(a78Cycles("ldpsw x1, x2, [x0]\nldpsw x3, x4, [x0, #8]\nldpsw x5, x6, [x0, #16]"), 2.0,
                kTol);
}

// ---------------------------------------------------------------------------
// C-13 Stores (sec. 3.10, l.1101-1141): store data on its own pipes D; register offset without I;
// STRH scaled 2 (I, L01, D).
// ---------------------------------------------------------------------------
TEST(A78SogTest, Stores) {
    A78InstInfo Str = a78Desc("str x1, [x2]");
    EXPECT_EQ(Str.ResourceCycles["N2UnitL01"], 1u);
    EXPECT_EQ(Str.ResourceCycles["N2UnitD"], 1u);
    EXPECT_EQ(Str.ResourceCycles.count("N2UnitI"), 0u);
    A78InstInfo Ro = a78Desc("str w1, [x2, x3, lsl #2]");
    EXPECT_EQ(Ro.NumMicroOps, 2u);
    EXPECT_EQ(Ro.ResourceCycles.count("N2UnitI"), 0u);
    A78InstInfo Strh = a78Desc("strh w1, [x2, x3, lsl #1]");
    EXPECT_EQ(Strh.MaxLatency, 2u);
    EXPECT_EQ(Strh.ResourceCycles["N2UnitI"], 1u);
    EXPECT_EQ(a78Desc("strh w1, [x2, x3]").ResourceCycles.count("N2UnitI"), 0u);
    A78InstInfo Post = a78Desc("str x1, [x2], #8");
    EXPECT_EQ(Post.ResourceCycles["N2UnitI"], 1u);  // "L01, D, I" (l.1111)
    // Two stores per cycle (D, L01), integer pipes free: four ADDs + two STR = 1.
    EXPECT_NEAR(a78Cycles("str x1, [x9]\nstr x2, [x10]\nadd x3, x3, #1\nadd x4, x4, #1\n"
                          "add x5, x5, #1\nadd x6, x6, #1"), 1.0, kTol);
}

// ---------------------------------------------------------------------------
// C-14, C-15 FP data processing and FP miscellaneous (sec. 3.11, 3.12, l.1205-1314).
// ---------------------------------------------------------------------------
TEST(A78SogTest, FpScalar) {
    EXPECT_EQ(a78Desc("fadd d0, d1, d2").MaxLatency, 2u);
    EXPECT_EQ(a78Desc("fcmp d0, d1").ResourceCycles["N2UnitV0"], 1u);
    EXPECT_EQ(a78Desc("fmul d0, d1, d2").MaxLatency, 3u);
    // FMADD 4(2) and the FMUL -> FMADD forwarding (notes 2, 3): 1 + 4 + 2 (FMOV).
    EXPECT_NEAR(a78Cycles("fmadd d0, d1, d2, d0"), 2.0, kTol);
    EXPECT_NEAR(a78Cycles("fmul d0, d1, d2\nfmadd d3, d4, d5, d0\nfmov d1, d3"), 7.0, kTol);
    A78InstInfo Frint = a78Desc("frinta d0, d1");
    EXPECT_EQ(Frint.MaxLatency, 2u);
    EXPECT_EQ(Frint.ResourceCycles["N2UnitV0"], 1u);
    // FDIV / FSQRT: best latency, divide unit held ceil(1 / best throughput) (D-1, D-3).
    A78InstInfo Fdiv = a78Desc("fdiv d0, d1, d2");
    EXPECT_EQ(Fdiv.MaxLatency, 7u);
    EXPECT_EQ(Fdiv.ResourceCycles["A78UnitFDiv"], 4u);
    EXPECT_EQ(Fdiv.ResourceCycles["N2UnitV0"], 1u);
    EXPECT_EQ(a78Desc("fsqrt s0, s1").ResourceCycles["A78UnitFDiv"], 2u);
    EXPECT_NEAR(a78Cycles("fdiv d0, d2, d1\nfcvtzs w1, d3\nfrinta d4, d5"), 4.0, kTol);
    // Converts: vec to vec 2 1 V0; vec to gen 3 1 V0; gen to vec 3 1 M0 (one cycle).
    A78InstInfo Fcvt = a78Desc("fcvt s0, d1");
    EXPECT_EQ(Fcvt.MaxLatency, 2u);
    EXPECT_EQ(Fcvt.ResourceCycles["N2UnitV0"], 1u);
    EXPECT_EQ(a78Desc("fcvtxn s0, d1").ResourceCycles["N2UnitV0"], 1u);
    for (const char *Asm : {"fcvtzs w1, d0", "fcvtau x1, s0", "fcvtzs w1, d0, #3"}) {
        A78InstInfo I = a78Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 3u) << Asm;
        EXPECT_EQ(I.ResourceCycles["N2UnitV0"], 1u) << Asm;
    }
    for (const char *Asm : {"scvtf d0, x1", "ucvtf s0, w1", "fmov d0, x1", "fmov s0, w1"}) {
        A78InstInfo I = a78Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 3u) << Asm;
        EXPECT_EQ(I.ResourceCycles["N2UnitM0"], 1u) << Asm;
    }
    EXPECT_NEAR(a78Cycles("scvtf d0, x1\nscvtf d1, x2\nscvtf d2, x3"), 3.0, kTol);
    EXPECT_EQ(a78Desc("fmov v0.d[1], x1").MaxLatency, 5u);
    // vec to gen FMOV 2 1 V1.
    A78InstInfo FmovX = a78Desc("fmov x1, d0");
    EXPECT_EQ(FmovX.MaxLatency, 2u);
    EXPECT_EQ(FmovX.ResourceCycles["N2UnitV1"], 1u);
    // D-4: scalar FP -> FP converts 3 1 V0.
    for (const char *Asm : {"scvtf d0, d1", "fcvtzs s0, s1", "ucvtf d0, d1, #3"}) {
        A78InstInfo I = a78Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 3u) << Asm;
        EXPECT_EQ(I.ResourceCycles["N2UnitV0"], 1u) << Asm;
    }
}

// ---------------------------------------------------------------------------
// C-16, C-17 FP loads and stores (sec. 3.13, 3.14, l.1366-1523).
// ---------------------------------------------------------------------------
TEST(A78SogTest, FpLoadsAndStores) {
    EXPECT_EQ(a78Desc("ldr q0, [x0, #16]").MaxLatency, 5u);
    A78InstInfo Post = a78Desc("ldr q0, [x0], #16");
    EXPECT_EQ(Post.WriteLatencies, (std::vector<unsigned>{1, 5}));
    EXPECT_EQ(Post.ResourceCycles["N2UnitI"], 1u);
    // Register offset: basic (X, no scale) 5 L (the printed I is taken as a misprint, I-4); scale S/D 5 L;
    // scale H/Q 6 I, L; extend 5 L.
    struct Row { const char *Asm; unsigned Lat; unsigned I; };
    for (const Row &R : {Row{"ldr d0, [x0, x1]", 5, 0}, Row{"ldr b0, [x0, x1]", 5, 0},
                         Row{"ldr d0, [x0, x1, lsl #3]", 5, 0}, Row{"ldr s0, [x0, w1, uxtw]", 5, 0},
                         Row{"ldr q0, [x0, x1]", 5, 0}, Row{"ldr q0, [x0, x1, lsl #4]", 6, 1},
                         Row{"ldr q0, [x0, w1, sxtw]", 5, 0}, Row{"ldr h0, [x0, w1, sxtw #1]", 6, 1}}) {
        A78InstInfo I = a78Desc(R.Asm);
        EXPECT_EQ(I.MaxLatency, R.Lat) << R.Asm;
        EXPECT_EQ(I.ResourceCycles["N2UnitI"], R.I) << R.Asm;
    }
    // Pairs: both registers 5 (D-10); pre/post-index one I.
    EXPECT_EQ(a78Desc("ldp d0, d1, [x0]").WriteLatencies, (std::vector<unsigned>{5, 5}));
    A78InstInfo LdpQ = a78Desc("ldp q0, q1, [x0], #32");
    EXPECT_EQ(LdpQ.WriteLatencies, (std::vector<unsigned>{1, 5, 5}));
    EXPECT_EQ(LdpQ.ResourceCycles["N2UnitI"], 1u);
    EXPECT_EQ(LdpQ.ResourceCycles["N2UnitL"], 2u);
    EXPECT_NEAR(a78Cycles("ldp d0, d1, [x0]\nfadd d1, d1, d1\nfmov x0, d1"), 9.0, kTol);
    // Stores.
    A78InstInfo StrPost = a78Desc("str q0, [x0], #16");
    EXPECT_EQ(StrPost.WriteLatencies, (std::vector<unsigned>{1}));
    EXPECT_EQ(StrPost.ResourceCycles["N2UnitI"], 1u);
    EXPECT_EQ(a78Desc("str h0, [x0, x1, lsl #1]").ResourceCycles["N2UnitI"], 1u);
    EXPECT_EQ(a78Desc("str h0, [x0, x1]").ResourceCycles.count("N2UnitI"), 0u);
    EXPECT_EQ(a78Desc("str q0, [x0, w1, sxtw #4]").ResourceCycles["N2UnitI"], 1u);
    EXPECT_EQ(a78Desc("str q0, [x0, x1, lsl #4]").ResourceCycles.count("N2UnitI"), 0u);
    // STP Q immediate offset 2 per cycle (I-6); post/pre-index 1 per cycle, base update latency 1.
    EXPECT_EQ(a78Desc("stp q0, q1, [x0]").ResourceCycles["N2UnitV"], 1u);
    EXPECT_NEAR(a78Cycles("stp q0, q1, [x0]\nstp q2, q3, [x1]"), 1.0, kTol);
    for (const char *Asm : {"stp q0, q1, [x0], #32", "stp q0, q1, [x0, #32]!"}) {
        A78InstInfo I = a78Desc(Asm);
        EXPECT_EQ(I.WriteLatencies, (std::vector<unsigned>{1})) << Asm;
        EXPECT_EQ(I.ResourceCycles["N2UnitI"], 1u) << Asm;
        EXPECT_EQ(I.ResourceCycles["N2UnitV"], 2u) << Asm;
    }
    EXPECT_EQ(a78Desc("stp d0, d1, [x0], #16").ResourceCycles["N2UnitI"], 1u);
}

// ---------------------------------------------------------------------------
// C-18, C-19, C-20 ASIMD integer, FP and miscellaneous (sec. 3.15-3.17, l.1554-1994).
// ---------------------------------------------------------------------------
TEST(A78SogTest, Asimd) {
    EXPECT_EQ(a78Desc("add v0.4s, v1.4s, v2.4s").MaxLatency, 2u);
    // SABA 3(1), SDOT 2(1), MLA 4(1).
    EXPECT_EQ(a78Desc("saba v0.8b, v1.8b, v2.8b").MaxLatency, 3u);
    EXPECT_NEAR(a78Cycles("saba v0.8b, v1.8b, v2.8b\nsaba v0.8b, v3.8b, v4.8b"), 2.0, kTol);
    EXPECT_EQ(a78Desc("sdot v0.4s, v1.16b, v2.16b").MaxLatency, 2u);
    EXPECT_NEAR(a78Cycles("sdot v0.4s, v1.16b, v2.16b\nsdot v0.4s, v3.16b, v4.16b"), 2.0, kTol);
    EXPECT_NEAR(a78Cycles("mla v0.4s, v1.4s, v2.4s\nmla v0.4s, v3.4s, v4.4s"), 2.0, kTol);
    // SQRDMLAH 4, no accumulator forwarding.
    EXPECT_NEAR(a78Cycles("sqrdmlah v0.4s, v1.4s, v2.4s"), 4.0, kTol);
    // ADDV 16B 4 1/2 V1; SMULL 3 1 V0.
    EXPECT_EQ(a78Desc("addv b0, v1.16b").ResourceCycles["N2UnitV1"], 2u);
    A78InstInfo Smull = a78Desc("smull v0.2d, v1.2s, v2.2s");
    EXPECT_EQ(Smull.MaxLatency, 3u);
    EXPECT_EQ(Smull.ResourceCycles["N2UnitV0"], 1u);
    // D-5: scalar shift forms take their mnemonic's row (2 1 V1).
    EXPECT_EQ(a78Desc("shl d0, d1, #3").ResourceCycles["N2UnitV1"], 1u);
    EXPECT_EQ(a78Desc("sqshrn s0, d1, #3").MaxLatency, 4u);
    // ASIMD FP: FMAXV 4 1 V; FMLA 4(2); fixed-point convert Q F32 4 1/2 V0.
    EXPECT_EQ(a78Desc("fmaxv s0, v1.4s").ResourceCycles["N2UnitV"], 2u);
    EXPECT_NEAR(a78Cycles("fmla v0.4s, v1.4s, v2.4s"), 2.0, kTol);
    A78InstInfo Cvt = a78Desc("fcvtzs v0.4s, v1.4s, #3");
    EXPECT_EQ(Cvt.MaxLatency, 4u);
    EXPECT_EQ(Cvt.ResourceCycles["N2UnitV0"], 2u);
    EXPECT_EQ(a78Desc("fdiv v0.2d, v1.2d, v2.2d").ResourceCycles["A78UnitFDiv"], 7u);
    // Misc: DUP from GPR 3 1 M0; UMOV 2 1 V1; TBL 4 regs 4 2/3 V; TBX 4 regs 6 2/5 V.
    A78InstInfo Dup = a78Desc("dup v0.4s, w1");
    EXPECT_EQ(Dup.MaxLatency, 3u);
    EXPECT_EQ(Dup.ResourceCycles["N2UnitM0"], 1u);
    EXPECT_EQ(a78Desc("umov w1, v0.s[1]").ResourceCycles["N2UnitV1"], 1u);
    EXPECT_EQ(a78Desc("tbl v0.16b, {v1.16b, v2.16b, v3.16b, v4.16b}, v5.16b").ResourceCycles["N2UnitV"],
              3u);
    EXPECT_EQ(a78Desc("tbx v0.16b, {v1.16b, v2.16b, v3.16b, v4.16b}, v5.16b").ResourceCycles["N2UnitV"],
              5u);
    EXPECT_EQ(a78Desc("ins v0.s[1], w1").MaxLatency, 5u);
}

// ---------------------------------------------------------------------------
// C-21, C-22 ASIMD loads and stores (sec. 3.18, 3.19, l.2060-2314); D-6.
// ---------------------------------------------------------------------------
TEST(A78SogTest, AsimdLoadsAndStores) {
    EXPECT_EQ(a78Desc("ld1 {v0.16b}, [x0]").MaxLatency, 5u);
    EXPECT_EQ(a78Desc("ld1 {v0.16b, v1.16b, v2.16b}, [x0]").MaxLatency, 5u);
    A78InstInfo Ld1D4 = a78Desc("ld1 {v0.8b, v1.8b, v2.8b, v3.8b}, [x0]");
    EXPECT_EQ(Ld1D4.MaxLatency, 5u);
    EXPECT_EQ(Ld1D4.ResourceCycles["N2UnitL"], 2u);  // 3/2 per cycle
    EXPECT_EQ(a78Desc("ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0]").MaxLatency, 6u);
    EXPECT_EQ(a78Desc("ld1 {v0.s}[1], [x0]").MaxLatency, 7u);
    EXPECT_EQ(a78Desc("ld1r {v0.4s}, [x0]").MaxLatency, 7u);
    EXPECT_EQ(a78Desc("ld2 {v0.4s, v1.4s}, [x0]").MaxLatency, 7u);
    EXPECT_EQ(a78Desc("ld3 {v0.4s, v1.4s, v2.4s}, [x0]").MaxLatency, 8u);
    A78InstInfo Ld4Q = a78Desc("ld4 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0]");
    EXPECT_EQ(Ld4Q.MaxLatency, 10u);
    EXPECT_EQ(Ld4Q.ResourceCycles["N2UnitL"], 12u);  // 1/4 per cycle
    A78InstInfo Post = a78Desc("ld1 {v0.16b}, [x0], #16");
    EXPECT_EQ(Post.WriteLatencies, (std::vector<unsigned>{1, 5}));
    // Stores: L01 held 2/T cycles where the row's throughput is below the µOPs'.
    struct Row { const char *Asm; unsigned Lat; unsigned L01; };
    for (const Row &R : {Row{"st1 {v0.16b}, [x0]", 2, 1}, Row{"st1 {v0.s}[1], [x0]", 4, 2},
                         Row{"st2 {v0.8b, v1.8b}, [x0]", 4, 2}, Row{"st2 {v0.4s, v1.4s}, [x0]", 5, 4},
                         Row{"st3 {v0.8b, v1.8b, v2.8b}, [x0]", 5, 4},
                         Row{"st3 {v0.4s, v1.4s, v2.4s}, [x0]", 6, 6},
                         Row{"st3 {v0.s, v1.s, v2.s}[1], [x0]", 5, 4},
                         Row{"st4 {v0.8b, v1.8b, v2.8b, v3.8b}, [x0]", 6, 6},
                         Row{"st4 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0]", 7, 12},
                         Row{"st4 {v0.2d, v1.2d, v2.2d, v3.2d}, [x0]", 5, 8},
                         Row{"st4 {v0.d, v1.d, v2.d, v3.d}[1], [x0]", 4, 4}}) {
        A78InstInfo I = a78Desc(R.Asm);
        EXPECT_EQ(I.MaxLatency, R.Lat) << R.Asm;
        EXPECT_EQ(I.ResourceCycles["N2UnitL01"], R.L01) << R.Asm;
    }
    EXPECT_NEAR(a78Cycles("st2 {v0.4s, v1.4s}, [x0]\nst2 {v2.4s, v3.4s}, [x1]"), 4.0, kTol);
}

// ---------------------------------------------------------------------------
// C-23, C-24 Cryptography and CRC (sec. 3.20, 3.21, l.2367-2436).
// ---------------------------------------------------------------------------
TEST(A78SogTest, CryptoAndCrc) {
    A78InstInfo Aes = a78Desc("aese v0.16b, v1.16b");
    EXPECT_EQ(Aes.MaxLatency, 2u);
    EXPECT_EQ(Aes.ResourceCycles["N2UnitV"], 1u);
    EXPECT_EQ(a78Desc("pmull v0.1q, v1.1d, v2.1d").ResourceCycles["N2UnitV0"], 1u);
    EXPECT_EQ(a78Desc("sha256h q0, q1, v2.4s").MaxLatency, 4u);
    EXPECT_EQ(a78Desc("crc32x w1, w2, x3").ResourceCycles["N2UnitM0"], 1u);
    EXPECT_NEAR(a78Cycles("crc32x w1, w1, x3"), 1.0, kTol);
}

// ---------------------------------------------------------------------------
// B-1 Store data read one cycle late (NOT SOG-based; the A55/A520/A76 decision).
// ---------------------------------------------------------------------------
TEST(A78SogTest, BorrowedStoreDataNeededLater) {
    // ldr -> str data: 4 - 1 (advance); str -> ldr memory edge: the store's latency 1 (D-7): 3 + 1.
    // FP: 5 - 1 + 2.
    EXPECT_NEAR(a78Cycles("ldr x1, [x0]\nstr x1, [x0]"), 4.0, kTol);
    EXPECT_NEAR(a78Cycles("ldr q1, [x0]\nstr q1, [x0]"), 6.0, kTol);
}

// D-12 [a78_pac_hint]: the HINT-space pointer-authentication instructions are NOPs on the A78
// (Armv8.2-A, no FEAT_PAuth); cortex-a78c has PAuth and keeps the model's rows (U-12).
TEST(A78SogTest, PointerAuthenticationHints) {
    for (const char *Asm : {"paciasp", "autiasp", "paciaz", "autiaz", "pacia1716", "autib1716",
                            "hint #25", "hint #29", "xpaclri"}) {
        A78InstInfo I = a78Desc(Asm);
        ASSERT_TRUE(I.Valid) << Asm;
        EXPECT_TRUE(I.ResourceCycles.empty()) << Asm;
        EXPECT_EQ(I.MaxLatency, 1u) << Asm;
    }
    // They still read and write LR with latency 1 (LLVM's implicit operands): a chain of 1 each.
    EXPECT_NEAR(a78Cycles("paciasp\nautiasp"), 2.0, kTol);
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a78c");
    llvm::remapSchedClassIndices(*TC.MCII, "cortex-a78c");
    llvm::installLocalSchedModel(*TC.STI, "cortex-a78c");
    TC.STI = llvm::wrapCustomSubtargetInfo(std::move(TC.STI), "cortex-a78c");
    auto Instrs = parseAsm(TC, "paciasp");
    ASSERT_EQ(Instrs.size(), 1u);
    mca::InstrumentManager IM(*TC.STI, *TC.MCII);
    mca::InstrBuilder IB(*TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), IM, 0);
    auto E = IB.createInstruction(Instrs[0].Inst, {});
    ASSERT_TRUE(static_cast<bool>(E));
    EXPECT_EQ((*E)->getLatency(), 5u);
}
