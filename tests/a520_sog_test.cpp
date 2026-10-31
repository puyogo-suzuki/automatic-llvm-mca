// Cortex-A520: values pinned to the Arm Cortex-A520 Core Software Optimization Guide
// (PJDOC-1505342170-671342, Issue 1.2, "SOG" below; text copy in tmp/a520sog.txt).
// The items (C-n, D-n, ...) are those of docs/a520-sog.md.
//
// Two kinds of checks:
//  * a520Desc(): the per-instruction data InstrBuilder takes from
//    ModifiedTarget/AArch64/AArch64SchedA520.td (latency, resource cycles, issue entries);
//  * a520CyclesPerIter(): steady-state cycles per iteration of a small loop body (no loop
//    branch) simulated by the Cortex-A520 issue stage (src/a520/a520_issue_stage.cpp), with
//    the expected value worked out by hand from the SOG in each comment.
//
// Configuration: the SOG's "A" column (2x128-bit vector datapath), docs/a520-sog.md D-1.

#include "test_support.h"

#include <map>
#include <string>
#include <vector>

namespace {

struct A520InstInfo {
    unsigned MaxLatency = 0;
    unsigned NumMicroOps = 0;
    std::vector<unsigned> WriteLatencies;  // per def, in operand order
    // Total resource cycles per resource name (e.g. "A520UnitMAC" -> 2).
    std::map<std::string, unsigned> ResourceCycles;
    bool BeginGroup = false;
    bool EndGroup = false;
    bool Valid = false;
};

A520InstInfo a520Desc(const std::string &AsmLine) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a520");
    auto Instrs = parseAsm(TC, AsmLine);
    A520InstInfo Info;
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
    Info.BeginGroup = IS.getBeginGroup();
    Info.EndGroup = IS.getEndGroup();
    for (const mca::WriteState &WS : IS.getDefs())
        Info.WriteLatencies.push_back(WS.getLatency());
    const MCSchedModel &SM = TC.STI->getSchedModel();
    const MCSchedClassDesc *SCD = SM.getSchedClassDesc(IS.getDesc().SchedClassID);
    if (SCD) {
        for (const MCWriteProcResEntry *E = TC.STI->getWriteProcResBegin(SCD),
                                       *End = TC.STI->getWriteProcResEnd(SCD);
             E != End; ++E)
            Info.ResourceCycles[SM.getProcResource(E->ProcResourceIdx)->Name] += E->ReleaseAtCycle;
    }
    Info.Valid = true;
    return Info;
}

// Steady-state cycles per iteration of `Body` (one instruction per line),
// simulated as a loop (loop-carried dependencies kept), --dependency dependency.
// With `IgnoreLoopCarried`, as a basic block under --ignore-loop-carried=default.
double a520CyclesPerIter(const std::string &Body, bool IgnoreLoopCarried = false) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a520");
    auto Instrs = parseAsm(TC, Body);
    AArch64MLPAnalyzer Analyzer;
    mca::PipelineOptions PO(0, 0, 0, 0, 0, 0, true);
    McaMetrics M = analyzeMcaRegion(Instrs, *TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), PO,
                                    200, 16, DependencyKind::Dependency,
                                    MLPWindowAssignmentKind::Forward, Analyzer,
                                    IgnoreLoopCarried, -1, /*mlpWindowLoop=*/false);
    if (!M.Valid || M.RetiredInstructions == 0 || Instrs.empty())
        return -1.0;
    return static_cast<double>(M.Cycles) /
           (static_cast<double>(M.RetiredInstructions) / static_cast<double>(Instrs.size()));
}

constexpr double kTol = 0.1;

} // namespace

// ---------------------------------------------------------------------------
// C-1 / C-2 Issue: three issue-queue entries per cycle, at most two ALU, two load, one
// store, two VPU instructions (SOG 4.1, p.72); an instruction with three or more
// GPR sources (MADD) takes two entries.
// ---------------------------------------------------------------------------
TEST(A520SogTest, IssueWidthAndEntries) {
    // Six independent ADDs, two ALUs: 3 cycles.
    EXPECT_NEAR(a520CyclesPerIter("add x1, x1, #1\nadd x2, x2, #1\nadd x3, x3, #1\n"
                                  "add x4, x4, #1\nadd x5, x5, #1\nadd x6, x6, #1"),
                3.0, kTol);
    // ADD, LDR, STR: one of each per cycle fits in three entries.
    EXPECT_NEAR(a520CyclesPerIter("add x1, x1, #1\nldr x2, [x0]\nstr x3, [x0, #8]"), 1.0, kTol);
    // MADD W: two entries.  In order, three entries per cycle: c0 madd + add; c1 add + madd;
    // c2 add + add (the next madd's two entries do not fit) -> two iterations in 3 cycles.
    // With one entry per MADD it would be 1.0.
    A520InstInfo Madd = a520Desc("madd w1, w2, w3, w4");
    EXPECT_EQ(Madd.NumMicroOps, 2u);
    EXPECT_NEAR(a520CyclesPerIter("madd w1, w2, w3, w4\nadd x5, x5, #1\nadd x6, x6, #1"), 1.5, kTol);
    // STR (register offset) and STP of GPRs: three GPR sources, two entries; LDP pre/post:
    // three GPR destinations, two entries.
    EXPECT_EQ(a520Desc("str x1, [x2, x3]").NumMicroOps, 2u);
    EXPECT_EQ(a520Desc("stp x1, x2, [x0]").NumMicroOps, 2u);
    EXPECT_EQ(a520Desc("ldp x1, x2, [x0], #16").NumMicroOps, 2u);
    EXPECT_EQ(a520Desc("ldp x1, x2, [x0]").NumMicroOps, 1u);
    EXPECT_EQ(a520Desc("str d1, [x2, x3]").NumMicroOps, 1u);  // FP data is not a GPR source
}

// ---------------------------------------------------------------------------
// D-14 NOP uses no execution resource (the SOG has no row for it); D-15 the zero register does not
// count as a GPR source, so MUL / MNEG / SMULL / UMULL (MADD ... with the zero accumulator) take one
// issue entry (docs/a520-sog.md).
// ---------------------------------------------------------------------------
TEST(A520SogTest, NopHasNoResourceAndMulTakesOneEntry) {
    A520InstInfo Nop = a520Desc("nop");
    EXPECT_EQ(Nop.NumMicroOps, 1u);
    EXPECT_TRUE(Nop.ResourceCycles.empty());
    // Four NOPs: three issue entries per cycle, no ALU limit (before: two ALUs, 2.0).
    EXPECT_NEAR(a520CyclesPerIter("nop\nnop\nnop\nnop\nnop\nnop"), 2.0, kTol);
    EXPECT_NEAR(a520CyclesPerIter("nop\nnop\nnop"), 1.0, kTol);
    // The other hints keep an ALU: YIELD is not a NOP.
    EXPECT_FALSE(a520Desc("yield").ResourceCycles.empty());
    // MUL W: one entry; MADD W with a real accumulator: two (above).
    EXPECT_EQ(a520Desc("mul w1, w2, w3").NumMicroOps, 1u);
    EXPECT_EQ(a520Desc("mul x1, x2, x3").NumMicroOps, 1u);
    EXPECT_EQ(a520Desc("mneg w1, w2, w3").NumMicroOps, 1u);
    EXPECT_EQ(a520Desc("smull x1, w2, w3").NumMicroOps, 1u);
    EXPECT_EQ(a520Desc("umull x1, w2, w3").NumMicroOps, 1u);
    EXPECT_EQ(a520Desc("smnegl x1, w2, w3").NumMicroOps, 1u);
    EXPECT_EQ(a520Desc("madd w1, w2, w3, w4").NumMicroOps, 2u);
    EXPECT_EQ(a520Desc("smaddl x1, w2, w3, x4").NumMicroOps, 2u);
    // Latency, MAC occupancy and the accumulator forwarding are those of the MADD rows.
    EXPECT_EQ(a520Desc("mul w1, w2, w3").MaxLatency, 3u);
    A520InstInfo MulX = a520Desc("mul x1, x2, x3");
    EXPECT_EQ(MulX.MaxLatency, 4u);
    EXPECT_EQ(MulX.ResourceCycles["A520UnitMAC"], 2u);
    // MUL + two ADDs: one entry each, three per cycle: 1.0 (MADD with accumulator: 1.5, above).
    EXPECT_NEAR(a520CyclesPerIter("mul w1, w2, w3\nadd x5, x5, #1\nadd x6, x6, #1"), 1.0, kTol);
    // A MUL result read as the accumulator of a MADD still gets the accumulator advance (W: 3 - 1),
    // one cycle earlier than the same result read as a multiplicand.
    EXPECT_LT(a520CyclesPerIter("mul w1, w2, w3\nmadd w4, w5, w6, w1"),
              a520CyclesPerIter("mul w1, w2, w3\nmadd w4, w5, w1, w7") - 0.5);
}

// ---------------------------------------------------------------------------
// D-9 (the former open point of docs/a55-sog.md X-1) No general write-back ordering: an integer instruction does not wait
// for an older FP instruction to write back.  FADD 4 (SOG 3.12), ADD 1: four independent
// instructions, three per cycle -> 4 / 3.  (The stock InOrderIssueStage gave 6.)
// ---------------------------------------------------------------------------
TEST(A520SogTest, NoWriteBackOrderStall) {
    EXPECT_NEAR(a520CyclesPerIter("fadd d0, d1, d1\nadd x1, x1, #1\nfadd d2, d1, d1\nadd x2, x2, #1"),
                4.0 / 3.0, kTol);
    // LDR (2) and ADD (1).
    EXPECT_NEAR(a520CyclesPerIter("ldr x3, [x0]\nadd x1, x1, #1\nldr x4, [x0, #8]\nadd x2, x2, #1"),
                4.0 / 3.0, kTol);
    // In-order issue is kept: fadd d2 waits for fadd d0 (4) and nothing passes it.
    // c0 fadd d0; c4 fadd d2 + add + add; c5 add + the next fadd d0 -> 5 cycles.
    EXPECT_NEAR(a520CyclesPerIter("fadd d0, d0, d1\nfadd d2, d0, d1\nadd x1, x1, #1\nadd x2, x2, #1\nadd x3, x3, #1"),
                5.0, kTol);
}

// ---------------------------------------------------------------------------
// Section 2.2 (the former open point of docs/a55-sog.md X-2) --ignore-loop-carried: the first instruction of a basic block
// no longer waits for the previous iteration (#12).  Two FADDs (two VPU per cycle): 1.0.
// ---------------------------------------------------------------------------
TEST(A520SogTest, IgnoreLoopCarriedAppliesToFirstInstruction) {
    EXPECT_NEAR(a520CyclesPerIter("fadd d0, d0, d1\nfadd d2, d3, d1", /*IgnoreLoopCarried=*/true), 1.0, kTol);
    EXPECT_NEAR(a520CyclesPerIter("fadd d2, d3, d1\nfadd d0, d0, d1", /*IgnoreLoopCarried=*/true), 1.0, kTol);
    // As a loop the dependency stays: 4.
    EXPECT_NEAR(a520CyclesPerIter("fadd d0, d0, d1\nfadd d2, d3, d1"), 4.0, kTol);
}

// ---------------------------------------------------------------------------
// C-11 Arithmetic and logical (SOG 3.3, Table 3-2).
// ---------------------------------------------------------------------------
TEST(A520SogTest, ArithmeticAndLogical) {
    EXPECT_EQ(a520Desc("add x1, x2, x3").MaxLatency, 1u);
    EXPECT_EQ(a520Desc("adcs x1, x2, x3").ResourceCycles["A520UnitALU"], 2u);  // throughput 1
    EXPECT_EQ(a520Desc("ccmp x1, x2, #0, eq").ResourceCycles["A520UnitALU"], 2u);
    // Note 1 "Latency=2 when the dependency is on Rm" (D-6): a chain through Rm of a
    // shifted ADD costs 2, through Rn 1.
    EXPECT_NEAR(a520CyclesPerIter("add x1, x2, x1, lsl #2\nadd x1, x2, x1, lsl #2"), 4.0, kTol);
    EXPECT_NEAR(a520CyclesPerIter("add x1, x1, x2, lsl #2\nadd x1, x1, x2, lsl #2"), 2.0, kTol);
    // Logical, shift: 1, no Rm note.
    EXPECT_NEAR(a520CyclesPerIter("orr x1, x2, x1, lsl #2\norr x1, x2, x1, lsl #2"), 2.0, kTol);
}

// ---------------------------------------------------------------------------
// C-14 Miscellaneous data processing (SOG 3.6, Table 3-5), incl. the fixed variable shifts.
// ---------------------------------------------------------------------------
TEST(A520SogTest, MiscDataProcessing) {
    // Variable shift ASRV, LSLV, LSRV, RORV: 1, 2, ALU (was 2).
    for (const char *Asm : {"lsl x1, x2, x3", "lsr w1, w2, w3", "asr x1, x2, x3", "ror w1, w2, w3"}) {
        A520InstInfo V = a520Desc(Asm);
        EXPECT_EQ(V.MaxLatency, 1u) << Asm;
        EXPECT_EQ(V.ResourceCycles["A520UnitALU"], 1u) << Asm;
    }
    EXPECT_NEAR(a520CyclesPerIter("lsl x1, x1, x2\nlsl x1, x1, x2"), 2.0, kTol);
    // Bitfield move, basic 2; the LSL/LSR/ASR/UXTB/SXTB aliases 1 (note 2); SXTH/SXTW/UXTH 2.
    EXPECT_EQ(a520Desc("ubfx x1, x2, #3, #5").MaxLatency, 2u);
    EXPECT_EQ(a520Desc("lsl x1, x2, #3").MaxLatency, 1u);
    EXPECT_EQ(a520Desc("lsr w1, w2, #3").MaxLatency, 1u);
    EXPECT_EQ(a520Desc("asr x1, x2, #3").MaxLatency, 1u);
    EXPECT_EQ(a520Desc("uxtb w1, w2").MaxLatency, 1u);
    EXPECT_EQ(a520Desc("sxtb x1, w2").MaxLatency, 1u);
    EXPECT_EQ(a520Desc("sxth x1, w2").MaxLatency, 2u);
    EXPECT_EQ(a520Desc("sxtw x1, w2").MaxLatency, 2u);
    EXPECT_EQ(a520Desc("uxth w1, w2").MaxLatency, 2u);
    // EXTR 2, its ROR alias 1 (note 1).
    EXPECT_EQ(a520Desc("extr x1, x2, x3, #7").MaxLatency, 2u);
    EXPECT_EQ(a520Desc("ror x1, x2, #7").MaxLatency, 1u);
    EXPECT_EQ(a520Desc("rbit x1, x2").MaxLatency, 2u);
    EXPECT_EQ(a520Desc("rev x1, x2").MaxLatency, 1u);
    EXPECT_EQ(a520Desc("clz x1, x2").MaxLatency, 1u);
}

// ---------------------------------------------------------------------------
// C-28 CRC (SOG 3.23, Table 3-22): CRC32, CRC32C 2, 1, MAC (was 1 on an ALU).
// ---------------------------------------------------------------------------
TEST(A520SogTest, CrcOnMac) {
    for (const char *Asm : {"crc32b w1, w1, w2", "crc32w w1, w1, w2", "crc32cx w1, w1, x2"}) {
        A520InstInfo C = a520Desc(Asm);
        EXPECT_EQ(C.MaxLatency, 2u) << Asm;
        EXPECT_EQ(C.ResourceCycles["A520UnitMAC"], 1u) << Asm;
        EXPECT_EQ(C.ResourceCycles.count("A520UnitALU"), 0u) << Asm;
    }
    EXPECT_NEAR(a520CyclesPerIter("crc32x w1, w1, x2\ncrc32x w1, w1, x2"), 4.0, kTol);
    EXPECT_NEAR(a520CyclesPerIter("crc32x w1, w1, x2\ncrc32x w3, w3, x2"), 2.0, kTol);  // one MAC
}

// ---------------------------------------------------------------------------
// C-12 / C-5 Divide and multiply (SOG 3.4, Table 3-3).
// ---------------------------------------------------------------------------
TEST(A520SogTest, DivideAndMultiply) {
    // Best case of [N + bits/4] (D-13): UDIV 3, SDIV 4, W and X alike; the divider is held for
    // the whole latency.
    for (const char *Udiv : {"udiv w1, w2, w3", "udiv x1, x2, x3"}) {
        A520InstInfo D = a520Desc(Udiv);
        EXPECT_EQ(D.MaxLatency, 3u) << Udiv;
        EXPECT_EQ(D.ResourceCycles["A520UnitDIV"], 3u) << Udiv;
    }
    for (const char *Sdiv : {"sdiv w1, w2, w3", "sdiv x1, x2, x3"}) {
        A520InstInfo D = a520Desc(Sdiv);
        EXPECT_EQ(D.MaxLatency, 4u) << Sdiv;
        EXPECT_EQ(D.ResourceCycles["A520UnitDIV"], 4u) << Sdiv;
    }
    EXPECT_EQ(a520Desc("mul w1, w2, w3").MaxLatency, 3u);
    A520InstInfo MaddX = a520Desc("madd x1, x2, x3, x4");
    EXPECT_EQ(MaddX.MaxLatency, 4u);
    EXPECT_EQ(MaddX.ResourceCycles["A520UnitMAC"], 2u);
    EXPECT_EQ(a520Desc("smaddl x1, w2, w3, x4").MaxLatency, 2u);
    A520InstInfo Smulh = a520Desc("smulh x1, x2, x3");
    EXPECT_EQ(Smulh.MaxLatency, 6u);
    EXPECT_EQ(Smulh.ResourceCycles["A520UnitMAC"], 4u);
    // Note 1: a multiply-accumulate chain through the accumulator issues every 2 cycles.
    EXPECT_NEAR(a520CyclesPerIter("madd w1, w2, w3, w1\nmadd w1, w2, w3, w1"), 4.0, kTol);
    EXPECT_NEAR(a520CyclesPerIter("madd x1, x2, x3, x1\nmadd x1, x2, x3, x1"), 4.0, kTol);
    // Through a multiplicand: the full latency (3).
    EXPECT_NEAR(a520CyclesPerIter("madd w1, w1, w3, w4\nmadd w1, w1, w3, w4"), 6.0, kTol);
}

// ---------------------------------------------------------------------------
// C-13 / C-7 Pointer authentication (SOG 3.5, Table 3-4; 4.6).
// ---------------------------------------------------------------------------
TEST(A520SogTest, PointerAuthentication) {
    for (const char *Asm : {"paciasp", "autiasp", "pacia x1, x2", "autib x1, x2", "pacga x1, x2, x3"}) {
        A520InstInfo P = a520Desc(Asm);
        EXPECT_EQ(P.MaxLatency, 5u) << Asm;
        EXPECT_EQ(P.ResourceCycles["A520UnitPAC"], 1u) << Asm;
    }
    A520InstInfo X = a520Desc("xpaclri");
    EXPECT_EQ(X.MaxLatency, 5u);
    EXPECT_EQ(X.ResourceCycles["A520UnitPAC"], 5u);  // throughput 1/5
    A520InstInfo Blraa = a520Desc("blraa x1, x2");
    ASSERT_TRUE(Blraa.Valid);
    EXPECT_EQ(Blraa.ResourceCycles["A520UnitPAC"], 1u);
    EXPECT_EQ(Blraa.ResourceCycles["A520UnitB"], 1u);
    // 4.6: AUT* and the directly following RET issue together (LR latency does not apply):
    // a chain through x30 paciasp (5) -> autiasp (5) -> ret (0) = 10.
    EXPECT_NEAR(a520CyclesPerIter("paciasp\nautiasp\nret"), 10.0, kTol);
    // Without the RET the LR chain is the same 10; a RET after an ADD to x30 waits 1.
    EXPECT_NEAR(a520CyclesPerIter("add x30, x30, #4\nret"), 1.0, kTol);
}

// ---------------------------------------------------------------------------
// C-15 / C-16 / C-19 / C-20 Loads and stores (SOG 3.7, 3.8, 3.14, 3.15).
// ---------------------------------------------------------------------------
TEST(A520SogTest, LoadsAndStores) {
    EXPECT_EQ(a520Desc("ldr x1, [x0]").MaxLatency, 2u);
    EXPECT_EQ(a520Desc("ldr d1, [x0]").MaxLatency, 3u);
    EXPECT_EQ(a520Desc("ldr q1, [x0, x2]").MaxLatency, 3u);
    // Load-use 2: a pointer chase costs 2 per load.
    EXPECT_NEAR(a520CyclesPerIter("ldr x1, [x1]\nldr x1, [x1]"), 4.0, kTol);
    // Two loads per cycle (Load/Store + Load pipes), LDP signed offset throughput 2.
    EXPECT_NEAR(a520CyclesPerIter("ldp x1, x2, [x0]\nldp x3, x4, [x0, #16]\nldp x5, x6, [x0, #32]\nldp x7, x8, [x0, #48]"),
                2.0, kTol);
    // LDP post-index throughput 1.
    EXPECT_NEAR(a520CyclesPerIter("ldp x1, x2, [x0], #16\nldp x3, x4, [x9], #16"), 2.0, kTol);
    // FP load pair throughput 1.
    EXPECT_NEAR(a520CyclesPerIter("ldp d1, d2, [x0]\nldp d3, d4, [x0, #16]"), 2.0, kTol);
    // One store per cycle.
    EXPECT_NEAR(a520CyclesPerIter("str x1, [x0]\nstr x2, [x0, #8]\nstr q3, [x0, #16]"), 3.0, kTol);
    // STP Q: throughput 1/2.
    A520InstInfo StpQ = a520Desc("stp q0, q1, [x0]");
    EXPECT_EQ(StpQ.ResourceCycles["A520UnitLdSt"], 2u);
    // Base register update in parallel, latency 1.
    A520InstInfo Post = a520Desc("ldr x1, [x0], #8");
    ASSERT_EQ(Post.WriteLatencies.size(), 2u);
    EXPECT_EQ(Post.WriteLatencies[0], 1u);
    EXPECT_EQ(Post.WriteLatencies[1], 2u);
}

// ---------------------------------------------------------------------------
// C-6 4.5 ADRP -> LDR (unsigned immediate) forwarding: no dependency stall.
// ---------------------------------------------------------------------------
TEST(A520SogTest, AdrpPointerForwarding) {
    // adrp + ldr [x1, #8] issue in the same cycle: four instructions, three per cycle
    // (4 / 3); the register-offset form waits for the ADRP (1): c0 adrp; c1 ldr + adrp;
    // c2 ldr + the next adrp -> 2.
    EXPECT_NEAR(a520CyclesPerIter("adrp x1, 0\nldr x2, [x1, #8]\nadrp x3, 0\nldr x4, [x3, #8]"), 4.0 / 3.0, kTol);
    EXPECT_NEAR(a520CyclesPerIter("adrp x1, 0\nldr x2, [x5, x1]\nadrp x3, 0\nldr x4, [x5, x3]"), 2.0, kTol);
}

// ---------------------------------------------------------------------------
// C-17 / C-18 FP scalar (SOG 3.12, 3.13).
// ---------------------------------------------------------------------------
TEST(A520SogTest, FpScalar) {
    EXPECT_EQ(a520Desc("fadd d0, d1, d2").MaxLatency, 4u);
    EXPECT_EQ(a520Desc("fmul d0, d1, d2").MaxLatency, 4u);
    EXPECT_EQ(a520Desc("fmadd d0, d1, d2, d3").MaxLatency, 4u);
    A520InstInfo Fcmp = a520Desc("fcmp d0, d1");
    EXPECT_EQ(Fcmp.MaxLatency, 1u);
    EXPECT_EQ(Fcmp.ResourceCycles["A520UnitVALU0"], 1u);  // throughput 1
    A520InstInfo Fccmp = a520Desc("fccmp d0, d1, #0, eq");
    EXPECT_EQ(Fccmp.MaxLatency, 5u);
    EXPECT_EQ(Fccmp.ResourceCycles["A520UnitVALU0"], 5u);  // throughput 1/5
    EXPECT_EQ(a520Desc("fcsel d0, d1, d2, eq").MaxLatency, 3u);
    // FDIV / FSQRT: two units of VMC (throughput 2/N).
    A520InstInfo FdivD = a520Desc("fdiv d0, d1, d2");
    EXPECT_EQ(FdivD.MaxLatency, 22u);
    EXPECT_EQ(FdivD.ResourceCycles["A520UnitVMC"], 19u);
    A520InstInfo FsqrtS = a520Desc("fsqrt s0, s1");
    EXPECT_EQ(FsqrtS.MaxLatency, 14u);
    EXPECT_EQ(FsqrtS.ResourceCycles["A520UnitVMC"], 9u);
    EXPECT_EQ(a520Desc("fsqrt d0, d1").MaxLatency, 25u);
    EXPECT_EQ(a520Desc("fcvtzs w0, d1").ResourceCycles["A520UnitVALU0"], 1u);
    EXPECT_EQ(a520Desc("scvtf d0, x1").MaxLatency, 4u);
    EXPECT_EQ(a520Desc("fmov d0, x1").MaxLatency, 3u);
    EXPECT_EQ(a520Desc("fmov x0, d1").ResourceCycles["A520UnitVALU0"], 1u);
    EXPECT_EQ(a520Desc("fmov d0, #1.0").MaxLatency, 3u);
}

// ---------------------------------------------------------------------------
// C-21 / C-23 ASIMD rows changed by the audit (SOG 3.16, 3.18, 3.22).
// ---------------------------------------------------------------------------
TEST(A520SogTest, AsimdFixedRows) {
    // "ASIMD arith, reduce ADDV 3 1" for 8B/16B/8H; 4H/4S and SADDLV/UADDLV 4.
    EXPECT_EQ(a520Desc("addv b0, v1.16b").MaxLatency, 3u);
    EXPECT_EQ(a520Desc("addv b0, v1.8b").MaxLatency, 3u);
    EXPECT_EQ(a520Desc("addv h0, v1.8h").MaxLatency, 3u);
    EXPECT_EQ(a520Desc("addv h0, v1.4h").MaxLatency, 4u);
    EXPECT_EQ(a520Desc("addv s0, v1.4s").MaxLatency, 4u);
    EXPECT_EQ(a520Desc("saddlv h0, v1.16b").MaxLatency, 4u);
    EXPECT_EQ(a520Desc("addv b0, v1.16b").ResourceCycles["A520UnitVALU0"], 1u);  // throughput 1
    // SMMLA / UMMLA / USMMLA 4.  (XAR 4 / EOR3 3 and AESE 3 are not checked here: the test
    // context's cortex-a520 has no SHA3 / AES feature, so the assembler rejects them; they are
    // checked with mca-insts-info, docs/a520-sog.md C-27 and X-2.)
    EXPECT_EQ(a520Desc("smmla v0.4s, v1.16b, v2.16b").MaxLatency, 4u);
    EXPECT_EQ(a520Desc("usmmla v0.4s, v1.16b, v2.16b").MaxLatency, 4u);
    // BFMMLA 14, throughput 1, VMAC + VALU.
    A520InstInfo Bfmmla = a520Desc("bfmmla v0.4s, v1.8h, v2.8h");
    EXPECT_EQ(Bfmmla.MaxLatency, 14u);
    EXPECT_EQ(Bfmmla.ResourceCycles["A520UnitVMAC"], 2u);
    EXPECT_EQ(Bfmmla.ResourceCycles["A520UnitVALU"], 2u);
}

// ---------------------------------------------------------------------------
// C-21..C-26 ASIMD rows unchanged (spot checks of the A column).
// ---------------------------------------------------------------------------
TEST(A520SogTest, AsimdSpotChecks) {
    EXPECT_EQ(a520Desc("add v0.4s, v1.4s, v2.4s").MaxLatency, 3u);
    EXPECT_EQ(a520Desc("sqadd v0.4s, v1.4s, v2.4s").MaxLatency, 4u);
    EXPECT_EQ(a520Desc("cmtst v0.4s, v1.4s, v2.4s").MaxLatency, 4u);
    EXPECT_EQ(a520Desc("mul v0.4s, v1.4s, v2.4s").MaxLatency, 4u);
    A520InstInfo Saba = a520Desc("saba v0.16b, v1.16b, v2.16b");
    EXPECT_EQ(Saba.MaxLatency, 6u);
    EXPECT_EQ(Saba.ResourceCycles["A520UnitVALU"], 7u);  // 2/7
    A520InstInfo Tbl4 = a520Desc("tbl v0.16b, {v1.16b, v2.16b, v3.16b, v4.16b}, v5.16b");
    EXPECT_EQ(Tbl4.MaxLatency, 16u);
    EXPECT_EQ(Tbl4.ResourceCycles["A520UnitVALU"], 18u);  // 1/9
    EXPECT_EQ(a520Desc("umov w0, v1.s[1]").ResourceCycles["A520UnitVALU0"], 1u);
    EXPECT_EQ(a520Desc("fdiv v0.2d, v1.2d, v2.2d").ResourceCycles["A520UnitVMC"], 38u);  // 1/19
    A520InstInfo Ld4 = a520Desc("ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0]");
    EXPECT_EQ(Ld4.MaxLatency, 5u);
    EXPECT_EQ(Ld4.ResourceCycles["A520UnitLdSt"], 3u);  // 1/3
    EXPECT_EQ(a520Desc("st4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0]").ResourceCycles["A520UnitLdSt"], 8u);
}

// ---------------------------------------------------------------------------
// C-8 4.7 SIMD MAC forwarding: MLA chain through the accumulator, latency 1.
// ---------------------------------------------------------------------------
TEST(A520SogTest, SimdMacForwarding) {
    EXPECT_NEAR(a520CyclesPerIter("mla v0.4s, v1.4s, v2.4s\nmla v0.4s, v1.4s, v2.4s\n"
                                  "mla v0.4s, v1.4s, v2.4s\nmla v0.4s, v1.4s, v2.4s"),
                4.0, kTol);
    EXPECT_NEAR(a520CyclesPerIter("sdot v0.4s, v1.16b, v2.16b\nsdot v0.4s, v1.16b, v2.16b"), 2.0, kTol);
    // FMLA: no forwarding, 4 per step.
    EXPECT_NEAR(a520CyclesPerIter("fmla v0.4s, v1.4s, v2.4s\nfmla v0.4s, v1.4s, v2.4s"), 8.0, kTol);
}

// ===========================================================================
// B: behaviours borrowed from the Cortex-A55 (docs/a520-sog.md chapter "Borrowed from the
// Cortex-A55").  The A520 SOG does not state them; the values are the A55 model's, so these
// tests pin a project decision, not the guide.
// ===========================================================================

// ---------------------------------------------------------------------------
// B-1 [a520_from_a55_atomics] LSE atomics: the A55's rows (A55 SOG 4.10, docs/a55-sog.md C-19,
// U-5) on the A520's load pipe and Load/Store pipe; issue entries by the A520 SOG 4.1 rule.
// Before: WriteAtomic Unsupported, dropped by InstrBuilder.
// ---------------------------------------------------------------------------
TEST(A520SogTest, BorrowedAtomicsFromA55) {
    struct Row { const char *Asm; unsigned Lat, Occ, Entries; bool Single; };
    for (const Row &R : {Row{"ldadd x1, x2, [x0]", 2, 2, 1, false},     // LD<OP>: 2, 1/2
                         Row{"ldclrb w1, w2, [x0]", 2, 2, 1, false},
                         Row{"ldsmaxa w1, w2, [x0]", 2, 2, 1, false},
                         Row{"ldaddal x1, x2, [x0]", 3, 3, 1, false},   // LD<OP>{A}L: 3, 1/3
                         Row{"ldumaxl h1, h2, [x0]", 0, 0, 0, false},   // (not an instruction: skipped)
                         Row{"stadd x1, [x0]", 1, 1, 1, false},         // ST<OP>: 1, 1
                         Row{"steorl w1, [x0]", 2, 2, 1, false},        // ST<OP>L: 2, 1/2
                         Row{"swp x1, x2, [x0]", 2, 2, 1, false},       // SWP{A}: 2, 1/2
                         Row{"swpal w1, w2, [x0]", 3, 3, 1, false},     // SWPL/SWPAL: 3, 1/3
                         Row{"cas x1, x2, [x0]", 4, 4, 2, false},       // CAS{A}: 4, 1/4; Rs, Rt, Rn -> 2 entries
                         Row{"casalh w1, w2, [x0]", 3, 3, 2, false},    // CASL/CASAL: 3, 1/3
                         Row{"casp x2, x3, x4, x5, [x0]", 5, 5, 2, true},   // CASP{A}: 5, 1/5, "00"
                         Row{"caspal w2, w3, w4, w5, [x0]", 4, 4, 2, true}}) {
        if (R.Lat == 0)
            continue;
        A520InstInfo I = a520Desc(R.Asm);
        ASSERT_TRUE(I.Valid) << R.Asm;                       // InstrBuilder accepts it
        EXPECT_EQ(I.MaxLatency, R.Lat) << R.Asm;
        EXPECT_EQ(I.ResourceCycles["A520UnitLd"], R.Occ) << R.Asm;
        EXPECT_EQ(I.ResourceCycles["A520UnitLdSt"], R.Occ) << R.Asm;
        EXPECT_EQ(I.NumMicroOps, R.Entries) << R.Asm;
        EXPECT_TRUE(I.BeginGroup) << R.Asm;                  // A55 "01" / "00"
        EXPECT_EQ(I.EndGroup, R.Single) << R.Asm;
    }
    // Four LDADDs, each holding both pipes 2 cycles: 8; four STADDs: 4.
    EXPECT_NEAR(a520CyclesPerIter("ldadd x1, x2, [x0]\nldadd x3, x4, [x0]\nldadd x5, x6, [x0]\nldadd x7, x8, [x0]"),
                8.0, kTol);
    EXPECT_NEAR(a520CyclesPerIter("stadd x1, [x0]\nstadd x3, [x0]\nstadd x5, [x0]\nstadd x7, [x0]"), 4.0, kTol);
    // An independent load waits for a free load pipe: c0 ldadd (Ld and LdSt c0-c1); c2 ldr;
    // c3 the next ldadd (needs both pipes) -> 3.
    EXPECT_NEAR(a520CyclesPerIter("ldadd x1, x2, [x0]\nldr x5, [x6]"), 3.0, kTol);
    // BeginGroup: the LDADD is the first instruction of its cycle; its occupancy (2) bounds the loop.
    EXPECT_NEAR(a520CyclesPerIter("add x9, x9, #1\nldadd x1, x2, [x0]"), 2.0, kTol);
    // The result is seen at the latency: ldadd (2) -> add -> next ldadd's Rs: 3.
    EXPECT_NEAR(a520CyclesPerIter("ldadd x1, x2, [x0]\nadd x1, x2, #1"), 3.0, kTol);
}

// ---------------------------------------------------------------------------
// B-2 [a520_from_a55_store_data] Store data read one cycle late (ReadST advance 1 from every
// producer, on the data of every store form), as on the A55 (docs/a55-sog.md C-18, U-9).
// Before: ReadST advance 0, and no ReadST at all on the forms with an InstRW row.
// ---------------------------------------------------------------------------
TEST(A520SogTest, BorrowedStoreDataNeededLater) {
    // 64-bit MUL (4, MAC 2 cycles): c0 mul; c3 str (4 - 1) + next mul -> 3 (before: 4), the same
    // for the register-offset store (two entries + the MUL's one fit in three).
    EXPECT_NEAR(a520CyclesPerIter("mul x1, x2, x3\nstr x1, [x0]"), 3.0, kTol);
    EXPECT_NEAR(a520CyclesPerIter("mul x1, x2, x3\nstr x1, [x0, x4]"), 3.0, kTol);
    // Load (2) -> store data: c0 ldr; c1 store + next ldr -> 1 (before: 2), every store family.
    for (const char *St : {"str x1, [x5]", "str x1, [x5, x6]", "stur x1, [x5, #-8]", "str x1, [x5], #8",
                           "str w1, [x5, #4]!", "sttr x1, [x5]", "strb w1, [x5]", "stp x1, x2, [x5]",
                           "stp x2, x1, [x5]", "stnp x1, x2, [x5]", "stp x1, x2, [x5, #16]!",
                           "stlr x1, [x5]", "stlrb w1, [x5]"}) {
        EXPECT_NEAR(a520CyclesPerIter(std::string("ldr x1, [x0]\n") + St), 1.0, kTol) << St;
    }
    // Two load -> store pairs: one store per cycle -> 2 (before: 4).
    EXPECT_NEAR(a520CyclesPerIter("ldr x1, [x0]\nstr x1, [x5]\nldr x2, [x0, #8]\nstr x2, [x5, #8]"), 2.0, kTol);
    // FP: c0 fadd (4); c3 store + next fadd -> 3 (before: 4), incl. STP Q.
    for (const char *St : {"str d0, [x5]", "str q0, [x5, x6]", "stur d0, [x5, #-8]", "str d0, [x5], #8",
                           "stp d0, d1, [x5]", "stp d1, d0, [x5, #16]!", "stp q1, q0, [x5]",
                           "stnp q0, q1, [x5]"}) {
        EXPECT_NEAR(a520CyclesPerIter(std::string("fadd d0, d2, d2\n") + St), 3.0, kTol) << St;
    }
    // ASIMD: ADD (3): c0 add; c2 st1 + next add -> 2 (before: 3), incl. one-lane and post-index forms.
    for (const char *St : {"st1 {v0.8b}, [x5]", "st1 {v0.8b}, [x5], #8", "st1 {v0.b}[3], [x5]",
                           "st1 {v0.b}[3], [x5], #1", "st2 {v0.8b, v1.8b}, [x5]", "st2 {v31.8b, v0.8b}, [x5]",
                           "st4 {v0.s, v1.s, v2.s, v3.s}[1], [x5]"}) {
        EXPECT_NEAR(a520CyclesPerIter(std::string("add v0.8b, v2.8b, v2.8b\n") + St), 2.0, kTol) << St;
    }
    // ASIMD load (3) -> ASIMD store: c0 ld1; c2 st1 + next ld1 -> 2 (before: 3).
    EXPECT_NEAR(a520CyclesPerIter("ld1 {v0.8b}, [x0]\nst1 {v0.8b}, [x5]"), 2.0, kTol);
    // The base register is an ordinary operand: an ADD -> base chain stays 1 per step and the
    // store cannot issue in the ADD's cycle: c0 add; c1 str + add; ... -> 1.
    EXPECT_NEAR(a520CyclesPerIter("add x5, x5, #8\nstr x1, [x5]"), 1.0, kTol);
    // Not on the atomics (as on the A55): ldadd's Rs waits for the load (2): c0 ldr; c2 ldadd,
    // which holds both pipes c2-c3; c4 the next ldr -> 4.
    EXPECT_NEAR(a520CyclesPerIter("ldr x1, [x0]\nldadd x1, x2, [x5]"), 4.0, kTol);
}

// ---------------------------------------------------------------------------
// B-3 [a520_from_a55_base_update] Base-register update latency 1/T of the row's A520
// throughput (rounded up; T >= 1 -> 1), the A55's "(1/T)" scheme (docs/a55-sog.md U-11).
// Before: 1 for every writeback form.
// ---------------------------------------------------------------------------
TEST(A520SogTest, BorrowedBaseUpdateLatency) {
    struct Row { const char *Asm; unsigned Base; };
    for (const Row &R : {Row{"stp q0, q1, [x0], #32", 2u},                           // STP Q, T = 1/2
                         Row{"stp q0, q1, [x0, #32]!", 2u},
                         Row{"st1 {v0.16b, v1.16b}, [x0], #32", 2u},                 // 1/2
                         Row{"st1 {v0.16b, v1.16b, v2.16b}, [x0], #48", 3u},         // 1/3
                         Row{"st1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64", 4u}, // 1/4
                         Row{"st3 {v0.16b, v1.16b, v2.16b}, [x0], #48", 6u},         // 1/6
                         Row{"st4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64", 8u}, // 1/8
                         Row{"st4 {v0.h, v1.h, v2.h, v3.h}[1], [x0], #8", 2u},       // 1/2
                         Row{"ld1 {v0.16b, v1.16b, v2.16b}, [x0], #48", 2u},         // 1/2
                         Row{"ld3 {v0.16b, v1.16b, v2.16b}, [x0], #48", 3u},         // 1/3
                         Row{"ld4 {v0.2d, v1.2d, v2.2d, v3.2d}, [x0], #64", 4u},     // 1/4
                         Row{"ld4 {v0.b, v1.b, v2.b, v3.b}[1], [x0], #4", 4u},       // 1/4
                         Row{"ld4r {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #16", 2u},    // 1/2
                         // Unchanged (T >= 1): 1.
                         Row{"st1 {v0.16b}, [x0], #16", 1u},
                         Row{"st2 {v0.8b, v1.8b}, [x0], #16", 1u},
                         Row{"ld1 {v0.16b, v1.16b}, [x0], #32", 1u},
                         Row{"ld1 {v0.16b}, [x0], #16", 1u},
                         Row{"ldp q0, q1, [x0], #32", 1u},
                         Row{"ldr q0, [x0], #16", 1u},
                         Row{"str x1, [x0], #8", 1u},
                         Row{"stp x1, x2, [x0], #16", 1u},
                         Row{"ldp x1, x2, [x0], #16", 1u}}) {
        A520InstInfo I = a520Desc(R.Asm);
        ASSERT_TRUE(I.Valid) << R.Asm;
        ASSERT_FALSE(I.WriteLatencies.empty()) << R.Asm;
        EXPECT_EQ(I.WriteLatencies[0], R.Base) << R.Asm;     // the base (wback) is the first def
    }
    // ADD chain from the updated base: c0 stp q (LdSt c0-c1); c2 add, c3 add, c4 add + next stp
    // -> 4 (before: base at c1, the adds c1-c3, the next stp at c3 -> 3).
    EXPECT_NEAR(a520CyclesPerIter("stp q0, q1, [x0], #32\nadd x1, x0, #1\nadd x1, x1, #1\nadd x1, x1, #1"),
                4.0, kTol);
    // A self-chain through the base is bounded by the occupancy anyway: 2 per STP Q (unchanged).
    EXPECT_NEAR(a520CyclesPerIter("stp q0, q1, [x0], #32\nstp q0, q1, [x0], #32"), 4.0, kTol);
}

// ---------------------------------------------------------------------------
// B-4 [a520_from_a55_flags] Same-cycle flag bypass from a flag setter to a conditional consumer,
// with the A55's exclusions (docs/a55-sog.md C-4, D-8): consumers ADC/ADCS/SBC/SBCS (NGC) and
// producers FCMP/FCCMP wait for the flags.  Before: the consumer saw the producer's latency (1).
// ---------------------------------------------------------------------------
TEST(A520SogTest, BorrowedFlagBypass) {
    // cmp + csel + csel: three ALU instructions, two ALUs -> 1.5 (before: c0 cmp; c1 csel + csel
    // -> 2).
    EXPECT_NEAR(a520CyclesPerIter("cmp x1, x2\ncsel x3, x4, x5, eq\ncsel x6, x4, x5, ne"), 1.5, kTol);
    // A flag chain through CCMP (ALU 2 cycles, T = 1): c0 cmp + ccmp; c1 csel; c2 cmp + ccmp -> 2
    // (before: c0 cmp; c1 ccmp; c2 csel + cmp; c3 ccmp ... -> 3).
    EXPECT_NEAR(a520CyclesPerIter("cmp x1, x2\nccmp x3, x4, #0, eq\ncsel x5, x6, x7, ne"), 2.0, kTol);
    // Conditional branch in the same cycle as the compare: c0 subs + b.ne + add -> 1 (before: 2).
    EXPECT_NEAR(a520CyclesPerIter("subs x1, x2, #1\nb.ne #8\nadd x3, x3, #1"), 1.0, kTol);
    // Load -> compare -> csel: c0 ldr; c2 cmp + csel + next ldr -> 2 (before: 3).
    EXPECT_NEAR(a520CyclesPerIter("ldr x1, [x0]\ncmp x1, #0\ncsel x3, x4, x5, eq"), 2.0, kTol);
    // Arithmetic with carry waits for the flags: c0 cmp; c1 adc + adc -> 2.
    EXPECT_NEAR(a520CyclesPerIter("cmp x1, x2\nadc x3, x3, x4\nadc x6, x6, x4"), 2.0, kTol);
    EXPECT_NEAR(a520CyclesPerIter("cmp x1, x2\nsbc w3, w3, w4\nsbc w6, w6, w4"), 2.0, kTol);
    EXPECT_NEAR(a520CyclesPerIter("cmp x1, x2\nngc x3, x4\nngc x6, x4"), 2.0, kTol);
    // A flag chain cmp -> csel -> cmp: 1 per iteration (before: 2).
    EXPECT_NEAR(a520CyclesPerIter("cmp x0, x1\ncsel x0, x4, x0, eq"), 1.0, kTol);
    // FP compares are excluded: the chain fcmp (1) -> csel (1) -> fmov to FP (3) -> fcmp costs
    // 1 + 1 + 3 = 5 (with the bypass it would be 4).
    EXPECT_NEAR(a520CyclesPerIter("fcmp s0, s1\ncsel w3, w4, w5, eq\nfmov s0, w3"), 5.0, kTol);
    // FCCMP (5, VALU0 5 cycles): the csels wait for its flags, 5 cycles in any case.
    EXPECT_NEAR(a520CyclesPerIter("fccmp s0, s1, #0, eq\ncsel x3, x4, x5, eq"), 5.0, kTol);
}
