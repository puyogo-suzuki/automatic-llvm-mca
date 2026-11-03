// Cortex-X1: values pinned to the Arm Cortex-X1 Core Software Optimization Guide
// (PJDOC-466751330-12804, Issue 4.0, "SOG" below; "l.N" = line N of a text extraction of the guide).
// The items (C-n, D-n, ...) are those of docs/x1-sog.md.
//
// Two kinds of checks:
//  * x1Desc(): the per-instruction data InstrBuilder takes from
//    ModifiedTarget/AArch64/AArch64SchedNeoverseV1.td (latency per def, uops, resource cycles);
//  * x1Facile(): the Facile prediction (src/facile/facile.cpp) of a loop body as mca_tool
//    --dependency ooo --facile computes it for cortex-x1 (uop width 16, Mop width 8, the X1
//    fusion table and zero-latency MOVs from cpu_traits.cpp, store -> load memory edges on), with
//    the expected value worked out by hand from the SOG in each comment.
//
// AArch64TestContext("cortex-x1") already installs the local model (tests/test_support.h).

#include "sog_test_support.h"

namespace {

constexpr const char *kCpu = "cortex-x1";

using X1InstInfo = SogInstInfo;

X1InstInfo x1Desc(const std::string &Asm) { return sogDesc(kCpu, Asm); }
facile::FacileResult x1Facile(const std::string &Body) { return sogFacile(kCpu, Body); }
double x1Cycles(const std::string &Body) { return x1Facile(Body).EstimatedCycles; }

constexpr double kTol = kSogFacileTol;

} // namespace

// ---------------------------------------------------------------------------
// Not a check: prints the model's data for the instructions in $X1_DUMP (separated by ';'), for
// the reproduction of docs/x1-sog.md (section 10).  Passes when the variable is unset.
// ---------------------------------------------------------------------------
TEST(X1SogTest, Dump) {
    sogDump(kCpu, "X1_DUMP");
}

// ---------------------------------------------------------------------------
// C-1, C-2, C-3 Dispatch: 8 Mops and 16 uOPs per cycle (sec. 4.1, l.2474); fifteen issue
// pipelines (sec. 2.1, l.388-389).  U-1: the ROB size is the owner's value, not a SOG value.
// ---------------------------------------------------------------------------
TEST(X1SogTest, DispatchAndPipelines) {
    const CpuTraits &T = getCpuTraits("cortex-x1");
    EXPECT_EQ(T.Model, CpuModel::X1);
    EXPECT_EQ(T.UopDispatchWidth, 16u);
    EXPECT_EQ(T.MopDispatchWidth, 8u);
    EXPECT_TRUE(T.ZeroLatencyMov);          // sec. 4.15 (C-4)
    EXPECT_EQ(T.Fusion, FusionTable::X1);   // sec. 4.14 (C-5)
    initLLVMAArch64();
    AArch64TestContext TC("cortex-x1");
    EXPECT_EQ(TC.STI->getSchedModel().MicroOpBufferSize, 224u);  // U-1
    // Eight independent ADDs on the four I pipelines: 2 (the Mop bound is 8/8).
    auto R = x1Facile("add x1, x1, #1\nadd x2, x2, #1\nadd x3, x3, #1\nadd x4, x4, #1\n"
                      "add x5, x5, #1\nadd x6, x6, #1\nadd x7, x7, #1\nadd x8, x8, #1");
    EXPECT_NEAR(R.IssueBound, 1.0, kTol);
    EXPECT_NEAR(R.EstimatedCycles, 2.0, kTol);
    // "ALU, basic, flagset ... 1 3 I" (l.565-567): three per cycle (D-9); CCMP "1 4 I" (l.574).
    EXPECT_NEAR(x1Cycles("adds x1, x1, #1\nadds x2, x2, #1\nadds x3, x3, #1\nadds x4, x4, #1\n"
                         "adds x5, x5, #1\nadds x6, x6, #1"), 2.0, kTol);
    EXPECT_EQ(x1Desc("ccmp x1, x2, #0, eq").ResourceCycles.count("V1UnitFlg"), 0u);
    // Integer store data on D: two STR and four ADD in one cycle.
    EXPECT_NEAR(x1Cycles("str x1, [x0]\nstr x2, [x0, #8]\nadd x3, x3, #1\nadd x4, x4, #1\n"
                         "add x5, x5, #1\nadd x6, x6, #1"), 1.0, kTol);
    // NOP: no pipeline (C-3).
    EXPECT_TRUE(x1Desc("nop").ResourceCycles.empty());
}

// ---------------------------------------------------------------------------
// C-4 Zero-latency MOVs (sec. 4.15, l.2849-2879): MOV #0, MOV ZR, MOV Wd,Wn / Xd,Xn only.
// ---------------------------------------------------------------------------
TEST(X1SogTest, ZeroLatencyMovs) {
    for (const char *Asm : {"mov x1, #0", "mov w1, #0", "mov x1, xzr", "mov w1, wzr", "mov x1, x2",
                            "mov w1, w2"}) {
        X1InstInfo I = x1Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 0u) << Asm;
        EXPECT_TRUE(I.ResourceCycles.empty()) << Asm;
    }
    // Not in the list: FMOV from WZR (3 1 M0, l.1319), MOVI #0 (2 4 V, l.1961), MOV #1, MOV V.
    X1InstInfo Fz = x1Desc("fmov s0, wzr");
    EXPECT_EQ(Fz.MaxLatency, 3u);
    EXPECT_EQ(Fz.ResourceCycles["V1UnitM0"], 1u);
    EXPECT_EQ(x1Desc("movi d0, #0").MaxLatency, 2u);
    EXPECT_EQ(x1Desc("mov x1, #1").MaxLatency, 1u);
    EXPECT_EQ(x1Desc("mov v0.16b, v1.16b").MaxLatency, 2u);
    EXPECT_NEAR(x1Cycles("add x1, x2, #1\nmov x2, x1"), 1.0, kTol);
}

// ---------------------------------------------------------------------------
// C-5 Instruction fusion (sec. 4.14, l.2831-2845): the A78 table without CMP + CSEL / CSET.
// ---------------------------------------------------------------------------
TEST(X1SogTest, Fusion) {
    EXPECT_EQ(x1Facile("cmp x1, #3\nb.eq #8").TotalMicroOps, 1u);
    EXPECT_EQ(x1Facile("cmn x1, x2\nb.eq #8").TotalMicroOps, 1u);
    EXPECT_EQ(x1Facile("tst x1, #3\nb.eq #8").TotalMicroOps, 1u);
    EXPECT_EQ(x1Facile("tst x1, x2\nb.ne #8").TotalMicroOps, 1u);
    EXPECT_EQ(x1Facile("bics x1, x2, x3\nb.eq #8").TotalMicroOps, 1u);
    EXPECT_EQ(x1Facile("nop\nadd x1, x2, x3").TotalMicroOps, 1u);
    // Not listed for the X1: CMP + CSEL, CMP + CSET; AES pairs are not masked (D-11, U-6).
    EXPECT_EQ(x1Facile("cmp x1, x2\ncsel x3, x4, x5, eq").TotalMicroOps, 2u);
    EXPECT_EQ(x1Facile("cmp x1, #3\ncset x3, eq").TotalMicroOps, 2u);
    EXPECT_EQ(x1Facile("aese v0.16b, v1.16b\naesmc v0.16b, v0.16b").TotalMicroOps, 2u);
}

// ---------------------------------------------------------------------------
// C-8 Branches (sec. 3.3, l.532-541).
// ---------------------------------------------------------------------------
TEST(X1SogTest, Branches) {
    EXPECT_EQ(x1Desc("b #8").ResourceCycles["V1UnitB"], 1u);
    X1InstInfo Bl = x1Desc("bl #8");
    EXPECT_EQ(Bl.ResourceCycles["V1UnitB"], 1u);
    EXPECT_EQ(Bl.ResourceCycles["V1UnitS"], 1u);
    EXPECT_EQ(Bl.MaxLatency, 0u);  // X-1: MCA gives every call latency 0
    EXPECT_NEAR(x1Cycles("cbz x1, #8\ncbz x2, #8\ncbz x3, #8\ncbz x4, #8"), 2.0, kTol);
}

// ---------------------------------------------------------------------------
// C-9 Arithmetic and logical (sec. 3.4, l.557-590).
// ---------------------------------------------------------------------------
TEST(X1SogTest, ArithmeticAndLogical) {
    EXPECT_EQ(x1Desc("add x1, x2, x3, lsl #4").MaxLatency, 1u);
    X1InstInfo Lsl5 = x1Desc("add x1, x2, x3, lsl #5");
    EXPECT_EQ(Lsl5.MaxLatency, 2u);
    EXPECT_EQ(Lsl5.ResourceCycles["V1UnitM"], 1u);
    EXPECT_EQ(x1Desc("adds x1, x2, x3, lsl #2").MaxLatency, 1u);
    EXPECT_EQ(x1Desc("adds x1, x2, x3, asr #2").MaxLatency, 2u);
    // [x1_ext_flagset] every extended form is "ALU, extend and shift 2 2 M" (l.568), also UXTB.
    for (const char *Asm : {"add x1, x2, w3, uxtb", "cmp w1, w2, uxtb", "adds x1, x2, w3, uxtb",
                            "cmp x1, w2, uxtw #2", "cmp x1, x2, uxtx"}) {
        X1InstInfo I = x1Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 2u) << Asm;
        EXPECT_EQ(I.ResourceCycles["V1UnitM"], 1u) << Asm;
    }
    // [x1_logical_flagset] unshifted ANDS/BICS/TST: "ALU, basic, flagset 1 3 I" (l.565-567);
    // shifted: "Logical, shift, flagset 2 2 M" (l.590).
    for (const char *Asm : {"tst x1, x2", "ands x1, x2, x3", "bics w1, w2, w3"}) {
        X1InstInfo I = x1Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 1u) << Asm;
        EXPECT_EQ(I.ResourceCycles.count("V1UnitM"), 0u) << Asm;
        EXPECT_EQ(I.ResourceCycles["V1UnitFlg"], 1u) << Asm;
    }
    EXPECT_EQ(x1Desc("ands x1, x2, x3, lsl #2").MaxLatency, 2u);
    EXPECT_NEAR(x1Cycles("ands x1, x1, x2"), 1.0, kTol);
    // "Logical, shift, no flagset 1 4 I" (l.588-589), any shift.
    EXPECT_EQ(x1Desc("eor x1, x2, x3, ror #7").MaxLatency, 1u);
}

// ---------------------------------------------------------------------------
// C-10 Divide and multiply (sec. 3.6, l.699-795); D-1, D-3; I-5; U-5.
// ---------------------------------------------------------------------------
TEST(X1SogTest, DivideAndMultiply) {
    for (const char *Asm : {"sdiv w1, w2, w3", "udiv x1, x2, x3"}) {
        X1InstInfo I = x1Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 5u) << Asm;
        EXPECT_EQ(I.ResourceCycles["V1UnitM0"], 1u) << Asm;
        EXPECT_EQ(I.ResourceCycles["X1UnitIDiv"], 5u) << Asm;
    }
    EXPECT_NEAR(x1Cycles("sdiv w1, w1, w2"), 5.0, kTol);
    // The divider, not M0, is held: two MADD beside a divide still fit.
    EXPECT_NEAR(x1Cycles("udiv x1, x2, x3\nmadd x4, x5, x6, x4\nmadd x7, x5, x6, x7"), 5.0, kTol);
    // MUL (Ra = ZR) on M; MADD on M0 with the accumulator forwarded (2(1), note 2).
    EXPECT_EQ(x1Desc("mul x1, x2, x3").ResourceCycles.count("V1UnitM0"), 0u);
    EXPECT_EQ(x1Desc("madd x1, x2, x3, x4").ResourceCycles["V1UnitM0"], 1u);
    EXPECT_NEAR(x1Cycles("madd x1, x2, x3, x1"), 1.0, kTol);
    // [x1_maddl] SMADDL family on M, 2 per cycle (I-5); forwarding from and to MADD (U-5).
    X1InstInfo L = x1Desc("smaddl x1, w2, w3, x4");
    EXPECT_EQ(L.MaxLatency, 2u);
    EXPECT_EQ(L.ResourceCycles["V1UnitM"], 1u);
    EXPECT_EQ(L.ResourceCycles.count("V1UnitM0"), 0u);
    EXPECT_NEAR(x1Cycles("smaddl x1, w2, w3, x4\nsmaddl x5, w2, w3, x6\nsmaddl x7, w2, w3, x8\n"
                         "smaddl x9, w2, w3, x10"), 2.0, kTol);
    EXPECT_NEAR(x1Cycles("smaddl x1, w2, w3, x1"), 1.0, kTol);
    EXPECT_NEAR(x1Cycles("madd x1, x2, x3, x1\numaddl x1, w4, w5, x1"), 2.0, kTol);
    // "Multiply high SMULH, UMULH 3 2 M" (l.710).
    EXPECT_EQ(x1Desc("smulh x1, x2, x3").MaxLatency, 3u);
    // [x1_mul_acc_fwd] U-5: a multiply without an accumulator (MUL, SMULL) forwards to the
    // accumulator of a MADD / SMADDL: 1 + 2.  SMULH does not (note 2, l.795): 3 + 2.
    EXPECT_NEAR(x1Cycles("mul x1, x1, x2\nmadd x1, x3, x4, x1"), 3.0, kTol);
    EXPECT_NEAR(x1Cycles("smull x1, w1, w2\nsmaddl x1, w3, w4, x1"), 3.0, kTol);
    EXPECT_NEAR(x1Cycles("smulh x1, x1, x2\nmadd x1, x3, x4, x1"), 5.0, kTol);
}

// ---------------------------------------------------------------------------
// C-11 Miscellaneous data processing (sec. 3.8, l.885-903).
// ---------------------------------------------------------------------------
TEST(X1SogTest, MiscDataProcessing) {
    // [x1_extr] one register when Rn == Rm or imm == 0 (note 1, l.903).
    EXPECT_EQ(x1Desc("extr x1, x2, x2, #3").MaxLatency, 1u);
    EXPECT_EQ(x1Desc("extr x1, x2, x3, #0").MaxLatency, 1u);
    X1InstInfo Two = x1Desc("extr x1, x2, x3, #3");
    EXPECT_EQ(Two.MaxLatency, 3u);
    EXPECT_EQ(Two.ResourceCycles["V1UnitM"], 1u);
    EXPECT_EQ(x1Desc("bfm x1, x2, #3, #5").MaxLatency, 2u);
    EXPECT_EQ(x1Desc("adrp x1, #4096").MaxLatency, 1u);
    EXPECT_EQ(x1Desc("rbit x1, x2").MaxLatency, 1u);
}

// ---------------------------------------------------------------------------
// C-12 Loads (sec. 3.9, l.960-1024); D-10.
// ---------------------------------------------------------------------------
TEST(X1SogTest, Loads) {
    EXPECT_EQ(x1Desc("ldr x1, [x2]").MaxLatency, 4u);
    EXPECT_EQ(x1Desc("ldr x1, [x2, x3, lsl #3]").NumMicroOps, 1u);
    X1InstInfo Post = x1Desc("ldr x1, [x2], #8");
    EXPECT_EQ(Post.WriteLatencies, (std::vector<unsigned>{1, 4}));
    // [x1_reg_offset_h] LDRH/LDRSH scaled by 2: 5 3 I, L (l.989-990, l.996-997).
    for (const char *Asm : {"ldrh w1, [x2, x3, lsl #1]", "ldrsh x1, [x2, w3, sxtw #1]"}) {
        X1InstInfo I = x1Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 5u) << Asm;
        EXPECT_EQ(I.ResourceCycles["V1UnitI"], 1u) << Asm;
    }
    for (const char *Asm : {"ldrh w1, [x2, x3]", "ldrh w1, [x2, w3, uxtw]"}) {
        X1InstInfo I = x1Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 4u) << Asm;
        EXPECT_EQ(I.ResourceCycles.count("V1UnitI"), 0u) << Asm;
    }
    // LDP: W one L, X two L (1.5 per cycle); LDPSW I + L held 2 cycles, both registers 5.
    EXPECT_EQ(x1Desc("ldp w1, w2, [x3]").ResourceCycles["V1UnitL"], 1u);
    EXPECT_EQ(x1Desc("ldp x1, x2, [x3]").ResourceCycles["V1UnitL"], 2u);
    X1InstInfo Psw = x1Desc("ldpsw x1, x2, [x3]");
    EXPECT_EQ(Psw.WriteLatencies, (std::vector<unsigned>{5, 5}));
    EXPECT_EQ(Psw.ResourceCycles["V1UnitL"], 2u);
    EXPECT_EQ(Psw.ResourceCycles["V1UnitI"], 1u);
    EXPECT_EQ(x1Desc("ldp w1, w2, [x3]").WriteLatencies, (std::vector<unsigned>{4, 4}));
    EXPECT_NEAR(x1Cycles("ldrh w1, [x0, x1, lsl #1]"), 5.0, kTol);
}

// ---------------------------------------------------------------------------
// C-13 Stores (sec. 3.10, l.1105-1145); I-3.
// ---------------------------------------------------------------------------
TEST(X1SogTest, Stores) {
    X1InstInfo S = x1Desc("str x1, [x2]");
    EXPECT_EQ(S.ResourceCycles["V1UnitL01"], 1u);
    EXPECT_EQ(S.ResourceCycles["V1UnitD"], 1u);
    EXPECT_EQ(x1Desc("str x1, [x2, x3, lsl #3]").ResourceCycles.count("V1UnitI"), 0u);
    // [x1_store_rows] STRH scaled by 2: 2 2 I, L01, D.
    for (const char *Asm : {"strh w1, [x2, x3, lsl #1]", "strh w1, [x2, w3, sxtw #1]"}) {
        X1InstInfo I = x1Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 2u) << Asm;
        EXPECT_EQ(I.ResourceCycles["V1UnitI"], 1u) << Asm;
    }
    EXPECT_EQ(x1Desc("strh w1, [x2, w3, sxtw]").ResourceCycles.count("V1UnitI"), 0u);
    EXPECT_EQ(x1Desc("stp x1, x2, [x3], #16").WriteLatencies, (std::vector<unsigned>{1}));
    // Four STR: two L01 pipes, 2 per cycle.
    EXPECT_NEAR(x1Cycles("str x1, [x0]\nstr x2, [x0, #8]\nstr x3, [x0, #16]\nstr x4, [x0, #24]"),
                2.0, kTol);
}

// ---------------------------------------------------------------------------
// C-14, C-15 FP data processing and miscellaneous (sec. 3.11, 3.12, l.1209-1323); D-1, D-3, D-4.
// ---------------------------------------------------------------------------
TEST(X1SogTest, FpScalar) {
    EXPECT_EQ(x1Desc("fadd d0, d1, d2").MaxLatency, 2u);
    EXPECT_EQ(x1Desc("fcmp d0, d1").ResourceCycles["V1UnitV0"], 1u);
    // FDIV / FSQRT: best latency, V02 one cycle, the divider ceil(2 / T) cycles.
    struct { const char *Asm; unsigned Lat, Occ; } Div[] = {
        {"fdiv h0, h1, h2", 7, 2}, {"fdiv s0, s1, s2", 7, 2}, {"fdiv d0, d1, d2", 7, 4},
        {"fsqrt s0, s1", 7, 2}, {"fsqrt d0, d1", 7, 4}, {"fdiv v0.2s, v1.2s, v2.2s", 7, 4},
        {"fdiv v0.4h, v1.4h, v2.4h", 7, 7}, {"fdiv v0.8h, v1.8h, v2.8h", 10, 10},
        {"fsqrt v0.8h, v1.8h", 11, 11}, {"fdiv v0.4s, v1.4s, v2.4s", 7, 7},
        {"fdiv v0.2d, v1.2d, v2.2d", 7, 7}};
    for (const auto &D : Div) {
        X1InstInfo I = x1Desc(D.Asm);
        EXPECT_EQ(I.MaxLatency, D.Lat) << D.Asm;
        EXPECT_EQ(I.ResourceCycles["V1UnitV02"], 1u) << D.Asm;
        EXPECT_EQ(I.ResourceCycles["X1UnitFDiv"], D.Occ) << D.Asm;
    }
    // Two FDIV D: 4 cycles on each of the two dividers (ceil(2 / (4/7)), D-2), 4 per iteration;
    // FRINT and FCVT (V02) beside them do not wait for the dividers (D-3).
    EXPECT_NEAR(x1Cycles("fdiv d0, d2, d1\nfdiv d3, d2, d1\nfrinta d4, d5\nfcvtzs w1, d6"), 4.0, kTol);
    // V02: FDIV 1 + FRINTA 1 + FCVTZS 2 + FRINTX 1 = 5 cycles on two pipes.
    EXPECT_NEAR(x1Cycles("fdiv d0, d2, d1\nfrinta d4, d5\nfcvtzs w1, d6\nfrintx d7, d8"), 2.5, kTol);
    // FMADD 4 (2) and the FMUL -> accumulator forwarding (notes 2, 3, l.1297-1302).
    EXPECT_NEAR(x1Cycles("fmadd d0, d1, d2, d0"), 2.0, kTol);
    EXPECT_NEAR(x1Cycles("fmul d0, d1, d2\nfmadd d3, d4, d5, d0\nfmov d1, d3"), 7.0, kTol);
    EXPECT_EQ(x1Desc("frinta d0, d1").ResourceCycles["V1UnitV02"], 1u);
    EXPECT_EQ(x1Desc("fcsel d0, d1, d2, eq").ResourceCycles["V1UnitV02"], 1u);
    // FP convert to GPR, also the fixed-point form: 3 1 V02 (V02 held 2 cycles, D-2).
    for (const char *Asm : {"fcvtzs x1, d0", "fcvtzs x1, d0, #3", "fcvtau w1, s0"}) {
        X1InstInfo I = x1Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 3u) << Asm;
        EXPECT_EQ(I.ResourceCycles["V1UnitV02"], 2u) << Asm;
    }
    // D-4: scalar conversions FP -> FP: 3 2 V02.
    for (const char *Asm : {"fcvtzs d0, d1", "fcvtzs s0, s1", "fcvtzs h0, h1", "scvtf s0, s1",
                            "scvtf h0, h1", "ucvtf d0, d1, #3"}) {
        X1InstInfo I = x1Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 3u) << Asm;
        EXPECT_EQ(I.ResourceCycles["V1UnitV02"], 1u) << Asm;
    }
    EXPECT_EQ(x1Desc("scvtf d0, x1").ResourceCycles["V1UnitM0"], 1u);
    EXPECT_EQ(x1Desc("fmov d0, x1").ResourceCycles["V1UnitM0"], 1u);
    EXPECT_EQ(x1Desc("fmov x1, d0").ResourceCycles["V1UnitV1"], 1u);
    EXPECT_EQ(x1Desc("fmov v0.d[1], x1").MaxLatency, 5u);
}

// ---------------------------------------------------------------------------
// C-16, C-17 FP loads and stores (sec. 3.13, 3.14, l.1373-1528); I-4, D-10.
// ---------------------------------------------------------------------------
TEST(X1SogTest, FpLoadsAndStores) {
    EXPECT_EQ(x1Desc("ldr q0, [x1]").MaxLatency, 6u);
    // [x1_reg_offset_h] H/Q with the scale: 7 I, L; without: 6 L (I-4 for the "basic" I).
    for (const char *Asm : {"ldr q0, [x1, x2, lsl #4]", "ldr h0, [x1, w2, uxtw #1]"}) {
        X1InstInfo I = x1Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 7u) << Asm;
        EXPECT_EQ(I.ResourceCycles["V1UnitI"], 1u) << Asm;
    }
    for (const char *Asm : {"ldr q0, [x1, x2]", "ldr q0, [x1, w2, uxtw]", "ldr h0, [x1, x2]",
                            "ldr d0, [x1, x2]", "ldr s0, [x1, w2, uxtw #2]"}) {
        X1InstInfo I = x1Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 6u) << Asm;
        EXPECT_EQ(I.NumMicroOps, 1u) << Asm;
    }
    // [x1_ldp_hi] LDP Q: both registers 6, L held 2 cycles (3/2 per cycle).
    X1InstInfo Q = x1Desc("ldp q0, q1, [x1]");
    EXPECT_EQ(Q.WriteLatencies, (std::vector<unsigned>{6, 6}));
    EXPECT_EQ(Q.ResourceCycles["V1UnitL"], 2u);
    EXPECT_EQ(x1Desc("ldp q0, q1, [x1], #32").WriteLatencies, (std::vector<unsigned>{1, 6, 6}));
    EXPECT_EQ(x1Desc("ldp d0, d1, [x1]").WriteLatencies, (std::vector<unsigned>{6, 6}));
    // FP stores: L01 and V01; [x1_store_rows] I only for the scaled H and Q register offsets.
    X1InstInfo St = x1Desc("str q0, [x1]");
    EXPECT_EQ(St.ResourceCycles["V1UnitL01"], 1u);
    EXPECT_EQ(St.ResourceCycles["V1UnitV01"], 1u);
    for (const char *Asm : {"str q0, [x1, x2, lsl #4]", "str h0, [x1, w2, uxtw #1]",
                            "str q0, [x1, w2, uxtw #4]"})
        EXPECT_EQ(x1Desc(Asm).ResourceCycles["V1UnitI"], 1u) << Asm;
    for (const char *Asm : {"str q0, [x1, x2]", "str q0, [x1, w2, uxtw]", "str h0, [x1, x2]",
                            "str s0, [x1, x2, lsl #2]"})
        EXPECT_EQ(x1Desc(Asm).ResourceCycles.count("V1UnitI"), 0u) << Asm;
    // STP Q immediate offset 2 per cycle (as printed); pre/post-index 1 per cycle.
    EXPECT_EQ(x1Desc("stp q0, q1, [x1]").ResourceCycles["V1UnitL01"], 1u);
    EXPECT_EQ(x1Desc("stp q0, q1, [x1], #32").ResourceCycles["V1UnitL01"], 2u);
}

// ---------------------------------------------------------------------------
// C-18, C-19, C-20 ASIMD integer, FP and miscellaneous (sec. 3.15-3.17, l.1559-2008); D-2, D-5.
// ---------------------------------------------------------------------------
TEST(X1SogTest, Asimd) {
    EXPECT_EQ(x1Desc("add v0.4s, v1.4s, v2.4s").ResourceCycles["V1UnitV"], 1u);
    EXPECT_NEAR(x1Cycles("saba v0.16b, v1.16b, v2.16b"), 1.0, kTol);   // 4(1)
    EXPECT_NEAR(x1Cycles("sdot v0.4s, v1.16b, v2.16b"), 1.0, kTol);    // 2(1)
    EXPECT_NEAR(x1Cycles("mla v0.4s, v1.4s, v2.4s"), 1.0, kTol);       // 4(1)
    EXPECT_NEAR(x1Cycles("fmla v0.4s, v1.4s, v2.4s"), 2.0, kTol);      // 4(2)
    EXPECT_EQ(x1Desc("sqrdmlah v0.4s, v1.4s, v2.4s").ReadAdvances.size(), 0u);
    EXPECT_EQ(x1Desc("addv b0, v1.16b").ResourceCycles["V1UnitV13"], 2u);  // 4 1 V13
    EXPECT_EQ(x1Desc("smull v0.4s, v1.4h, v2.4h").ResourceCycles["V1UnitV02"], 1u);
    // [x1_asimd_scalar] D-5: scalar forms take the row of their mnemonic.
    EXPECT_EQ(x1Desc("shl d0, d1, #3").ResourceCycles["V1UnitV13"], 1u);
    EXPECT_EQ(x1Desc("srshr d0, d1, #3").MaxLatency, 4u);
    EXPECT_EQ(x1Desc("sqshrn b0, h1, #3").MaxLatency, 4u);
    EXPECT_EQ(x1Desc("ssra d0, d1, #3").MaxLatency, 4u);
    EXPECT_NEAR(x1Cycles("ssra d0, d1, #3"), 1.0, kTol);
    EXPECT_EQ(x1Desc("sqdmull s0, h1, h2").MaxLatency, 3u);
    EXPECT_EQ(x1Desc("fmulx d0, d1, d2").MaxLatency, 3u);
    // Scalar FMULX "3 4 V" (l.1842), FRECPS / FRSQRTS "4 4 V" (l.1975): one V cycle.
    for (const char *Asm : {"fmulx s0, s1, s2", "fmulx d0, d1, v2.d[1]", "frecps d0, d1, d2",
                            "frsqrts s0, s1, s2", "frecps h0, h1, h2"}) {
        X1InstInfo I = x1Desc(Asm);
        EXPECT_EQ(I.MaxLatency, Asm[1] == 'm' ? 3u : 4u) << Asm;
        EXPECT_EQ(I.ResourceCycles["V1UnitV"], 1u) << Asm;
    }
    // [x1_mul_acc_fwd] U-5: ASIMD MUL and SMULL/UMULL forward to the accumulator of MLA/MLAL
    // (note 1, l.1758-1760), and MLA and MLAL to each other; SQDMULH does not.
    EXPECT_NEAR(x1Cycles("mul v0.4s, v0.4s, v1.4s\nmla v0.4s, v2.4s, v3.4s"), 5.0, kTol);    // 4 + 1
    EXPECT_NEAR(x1Cycles("smull v0.4s, v0.4h, v1.4h\nsmlal v0.4s, v2.4h, v3.4h"), 5.0, kTol); // 1 + 4
    EXPECT_NEAR(x1Cycles("mla v0.4s, v1.4s, v2.4s\nsmlal v0.4s, v3.4h, v4.4h"), 2.0, kTol);   // 1 + 1
    EXPECT_NEAR(x1Cycles("sqdmulh v0.4s, v0.4s, v1.4s\nmla v0.4s, v2.4s, v3.4s"), 8.0, kTol); // 4 + 4
    // [x1_v_throughput] Q-form F16 converts and rounds 6 1/2 V02: V02 4 cycles.
    for (const char *Asm : {"fcvtzs v0.8h, v1.8h", "scvtf v0.8h, v1.8h", "frintz v0.8h, v1.8h",
                            "frecpe v0.8h, v1.8h"}) {
        X1InstInfo I = x1Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 6u) << Asm;
        EXPECT_EQ(I.ResourceCycles["V1UnitV02"], 4u) << Asm;
    }
    // 4 1 V02: V02 2 cycles.
    for (const char *Asm : {"fcvtzs v0.4s, v1.4s", "frintp v0.4s, v1.4s", "ursqrte v0.4s, v1.4s",
                            "frecpe v0.4s, v1.4s", "frecpe v0.4h, v1.4h"})
        EXPECT_EQ(x1Desc(Asm).ResourceCycles["V1UnitV02"], 2u) << Asm;
    // FRECPE "D-form F32 and scalar forms 3 2 V02" (l.1966-1968): also the H form.
    EXPECT_EQ(x1Desc("frecpe h0, h1").MaxLatency, 3u);
    // UMOV/SMOV "2 1 V" (l.2002-2003): V held 4 cycles.
    EXPECT_EQ(x1Desc("umov w1, v0.s[1]").ResourceCycles["V1UnitV"], 4u);
    EXPECT_NEAR(x1Cycles("umov w1, v0.s[1]\numov w2, v1.s[1]"), 2.0, kTol);
    // TBL / TBX on V01.
    EXPECT_EQ(x1Desc("tbl v0.16b, {v1.16b, v2.16b, v3.16b, v4.16b}, v5.16b")
                  .ResourceCycles["V1UnitV01"], 3u);
    EXPECT_EQ(x1Desc("dup v0.4s, w1").ResourceCycles["V1UnitM0"], 1u);
    EXPECT_EQ(x1Desc("ins v0.s[1], w1").MaxLatency, 5u);
}

// ---------------------------------------------------------------------------
// C-21, C-22 ASIMD loads and stores (sec. 3.18, 3.19, l.2075-2329); D-6, I-6.
// ---------------------------------------------------------------------------
TEST(X1SogTest, AsimdLoadsAndStores) {
    EXPECT_EQ(x1Desc("ld1 {v0.16b}, [x1]").MaxLatency, 6u);
    EXPECT_EQ(x1Desc("ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1]").MaxLatency, 7u);
    EXPECT_EQ(x1Desc("ld1r {v0.4s}, [x1]").MaxLatency, 8u);
    // [x1_ls_throughput] LD4 Q 9 1/2: L held 6 cycles.
    X1InstInfo L4 = x1Desc("ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1]");
    EXPECT_EQ(L4.MaxLatency, 9u);
    EXPECT_EQ(L4.ResourceCycles["V1UnitL"], 6u);
    EXPECT_NEAR(x1Cycles("ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0]"), 2.0, kTol);
    // Stores on L01 and V01 (I-6); ST4 one lane B/H/S on L01, V01 2 cycles.
    X1InstInfo S4 = x1Desc("st4 {v0.s, v1.s, v2.s, v3.s}[1], [x1]");
    EXPECT_EQ(S4.MaxLatency, 6u);
    EXPECT_EQ(S4.ResourceCycles["V1UnitL01"], 2u);
    EXPECT_EQ(S4.ResourceCycles["V1UnitV01"], 2u);
    EXPECT_EQ(x1Desc("st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x1]").ResourceCycles["V1UnitL01"], 4u);
    EXPECT_EQ(x1Desc("st1 {v0.16b}, [x1], #16").WriteLatencies, (std::vector<unsigned>{1}));
}

// ---------------------------------------------------------------------------
// C-23, C-24 Cryptography and CRC (sec. 3.20, 3.21, l.2382-2455); U-6.
// ---------------------------------------------------------------------------
TEST(X1SogTest, CryptoAndCrc) {
    X1InstInfo A = x1Desc("aesmc v0.16b, v1.16b");
    EXPECT_EQ(A.MaxLatency, 2u);
    EXPECT_EQ(A.ResourceCycles["V1UnitV01"], 1u);
    EXPECT_TRUE(A.ReadAdvances.empty());  // [x1_aes] U-6: no forwarding
    EXPECT_NEAR(x1Cycles("aese v0.16b, v1.16b\naesmc v0.16b, v0.16b"), 4.0, kTol);
    EXPECT_EQ(x1Desc("pmull v0.1q, v1.1d, v2.1d").ResourceCycles["V1UnitV01"], 1u);
    EXPECT_EQ(x1Desc("sha256h q0, q1, v2.4s").MaxLatency, 4u);
    EXPECT_EQ(x1Desc("sha1h s0, s1").ResourceCycles["V1UnitV0"], 1u);
    EXPECT_NEAR(x1Cycles("crc32x w1, w1, x3"), 1.0, kTol);  // 2, note 1: one cycle less
}

// ---------------------------------------------------------------------------
// D-13 Hint-space pointer authentication: NOPs without FEAT_PAuth (cortex-x1), the stock rows
// with it (cortex-x1c, U-12).
// ---------------------------------------------------------------------------
TEST(X1SogTest, PointerAuthenticationHints) {
    for (const char *Asm : {"paciasp", "autiasp", "hint #25", "xpaclri"}) {
        X1InstInfo I = x1Desc(Asm);
        EXPECT_TRUE(I.ResourceCycles.empty()) << Asm;
        EXPECT_EQ(I.MaxLatency, 1u) << Asm;
    }
    initLLVMAArch64();
    AArch64TestContext TC("cortex-x1c");
    auto Instrs = parseAsm(TC, "paciasp");
    ASSERT_EQ(Instrs.size(), 1u);
    mca::InstrumentManager IM(*TC.STI, *TC.MCII);
    mca::InstrBuilder IB(*TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), IM, 0);
    auto E = IB.createInstruction(Instrs[0].Inst, {});
    ASSERT_TRUE(static_cast<bool>(E));
    EXPECT_EQ((*E)->getLatency(), 5u);
}

// ---------------------------------------------------------------------------
// B-1 Store data is read one cycle late (not SOG-based).
// ---------------------------------------------------------------------------
TEST(X1SogTest, BorrowedStoreDataNeededLater) {
    EXPECT_NEAR(x1Cycles("ldr x1, [x0]\nstr x1, [x0]"), 4.0, kTol);   // 4 - 1 + 1
    EXPECT_NEAR(x1Cycles("ldr q1, [x0]\nstr q1, [x0]"), 7.0, kTol);   // 6 - 1 + 2
    EXPECT_FALSE(x1Desc("str x1, [x2]").ReadAdvances.empty());
    EXPECT_FALSE(x1Desc("st1 {v0.s}[1], [x1]").ReadAdvances.empty());
}
