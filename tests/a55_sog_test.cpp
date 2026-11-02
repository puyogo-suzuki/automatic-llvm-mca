// Cortex-A55: values pinned to the Arm Cortex-A55 Software Optimization Guide
// (ARM-EPM-128372 v3.0, "SOG" below; text copy in tmp/a55sog.txt).
//
// Two kinds of checks:
//  * a55Desc():  the per-instruction data InstrBuilder takes from
//    ModifiedTarget/AArch64/AArch64SchedA55.td (latency, resource cycles,
//    BeginGroup / EndGroup = the SOG "Dual-issue" column);
//  * a55CyclesPerIter(): steady-state cycles per iteration of a small loop body
//    simulated by the A55 issue stage (src/a55/a55_issue_stage.cpp), with the
//    expected value worked out by hand from the SOG tables in each comment.
//
// Project policy (docs/a55-sog.md, D-1): SOG ranges take the best (lowest-latency) value and
// resource occupancy is 1 / best throughput, rounded up.

#include "test_support.h"
#include "a55/a55_issue_stage.h"
#include "llvm-source/llvm/lib/Target/AArch64/MCTargetDesc/AArch64MCTargetDesc.h"

#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace {

struct A55InstInfo {
    unsigned MaxLatency = 0;
    std::vector<unsigned> WriteLatencies;  // per def, in operand order
    bool BeginGroup = false;
    bool EndGroup = false;
    // Total resource cycles per resource name (e.g. "CortexA55UnitLd" -> 2).
    std::map<std::string, unsigned> ResourceCycles;
    bool Valid = false;
};

A55InstInfo a55Desc(const std::string &AsmLine) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a55");
    auto Instrs = parseAsm(TC, AsmLine);
    A55InstInfo Info;
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
    for (const mca::WriteState &WS : IS.getDefs())
        Info.WriteLatencies.push_back(WS.getLatency());
    Info.BeginGroup = IS.getBeginGroup();
    Info.EndGroup = IS.getEndGroup();
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
// With `IgnoreLoopCarried`, as a basic block under --ignore-loop-carried=default
// (dependencies on the previous iteration ignored).
double a55CyclesPerIter(const std::string &Body, size_t *NumInstrs = nullptr,
                        uint64_t *Retired = nullptr, bool IgnoreLoopCarried = false) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a55");
    auto Instrs = parseAsm(TC, Body);
    if (NumInstrs) *NumInstrs = Instrs.size();
    AArch64MLPAnalyzer Analyzer;
    mca::PipelineOptions PO(0, 0, 0, 0, 0, 0, true);
    McaMetrics M = analyzeMcaRegion(Instrs, *TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), PO,
                                    200, 16, DependencyKind::Dependency,
                                    MLPWindowAssignmentKind::Forward, Analyzer,
                                    IgnoreLoopCarried, -1, /*mlpWindowLoop=*/false);
    if (Retired) *Retired = M.RetiredInstructions;
    if (!M.Valid || M.RetiredInstructions == 0 || Instrs.empty())
        return -1.0;
    return static_cast<double>(M.Cycles) /
           (static_cast<double>(M.RetiredInstructions) / static_cast<double>(Instrs.size()));
}

constexpr double kTol = 0.1;

} // namespace

// ---------------------------------------------------------------------------
// #1 No write-back ordering stall between independent instructions.
// SOG 4.3 (ALU basic 1, throughput 2, dual-issue 11), 4.11 (FADD 4, 2, 11),
// 4.8 (LDR 3 (2), 1, 11): a FADD/ADD or LDR/ADD pair issues every cycle.
// ---------------------------------------------------------------------------
TEST(A55SogTest, NoWriteBackOrderStallAfterLongLatencyOp) {
    EXPECT_NEAR(a55CyclesPerIter("fadd d0, d1, d1\nadd x1, x1, #1\nfadd d2, d1, d1\nadd x2, x2, #1"), 2.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ldr x3, [x0]\nadd x1, x1, #1\nldr x4, [x0, #8]\nadd x2, x2, #1"), 2.0, kTol);
    // 64-bit MUL: 4, throughput 1/2 (SOG 4.5 "4-5 (2-3)", "1/3 - 1/2", best case).
    EXPECT_NEAR(a55CyclesPerIter("mul x3, x9, x9\nadd x1, x1, #1\nmul x4, x9, x9\nadd x2, x2, #1"), 4.0, kTol);
}

// SOG 3.4: "Integer divide instructions are serializing and do not allow younger
// instructions to retire underneath."  UDIV W: 3 (best), DIV busy 3, dual-issue
// 01 (SOG 4.5).  Hand count: c0 udiv; the adds (latency 1) may not complete
// before the divide (c3), so c2 add+add, c3 add; the next udiv cannot be
// instruction-1 (01) -> c4.  4 cycles per iteration.
TEST(A55SogTest, IntegerDivideIsSerializing) {
    EXPECT_NEAR(a55CyclesPerIter("udiv w1, w2, w3\nadd x5, x5, #1\nadd x6, x6, #1\nadd x7, x7, #1"), 4.0, kTol);
}

// ---------------------------------------------------------------------------
// #2 Strictly in-order issue (SOG 5.1 "does not support any instruction
// reordering at issue", 3.5.1 "only issues instructions in-order").
// fadd d2 waits for fadd d0 (4 cycles) and nothing younger may pass it:
// c0 fadd d0; c4 fadd d2 + add; c5 add add; c6 add add; c7 add + next fadd d0
// -> 7 cycles.  (The former cross-pipe "decoupled" issue let the adds run
// underneath the stalled FP instruction.)
// ---------------------------------------------------------------------------
TEST(A55SogTest, StrictlyInOrderIssue) {
    EXPECT_NEAR(a55CyclesPerIter("fadd d0, d0, d1\nfadd d2, d0, d1\nadd x1, x1, #1\nadd x2, x2, #1\n"
                                 "add x3, x3, #1\nadd x4, x4, #1\nadd x5, x5, #1\nadd x6, x6, #1"),
                7.0, kTol);
    // Load-use 2 (SOG 3.3): c0 ldr x1; c2 add x2 + ldr x3; c4 add x4 + add x5;
    // c5 add x6 + next ldr x1 -> 5 cycles.
    EXPECT_NEAR(a55CyclesPerIter("ldr x1, [x0]\nadd x2, x1, #1\nldr x3, [x0, #8]\nadd x4, x3, #1\n"
                                 "add x5, x5, #1\nadd x6, x6, #1"),
                5.0, kTol);
}

// ---------------------------------------------------------------------------
// #3 Armv8.1 atomics are simulated, not dropped (SOG 4.10).
// ---------------------------------------------------------------------------
TEST(A55SogTest, AtomicsAreModelled) {
    A55InstInfo LdAdd = a55Desc("ldadd x1, x2, [x0]");
    ASSERT_TRUE(LdAdd.Valid);                         // InstrBuilder accepts it
    EXPECT_EQ(LdAdd.MaxLatency, 2u);                  // LD<OP>: 2, 1/2, 01
    EXPECT_TRUE(LdAdd.BeginGroup);
    EXPECT_FALSE(LdAdd.EndGroup);
    A55InstInfo LdAddAl = a55Desc("ldaddal x1, x2, [x0]");
    ASSERT_TRUE(LdAddAl.Valid);
    EXPECT_EQ(LdAddAl.MaxLatency, 3u);                // LD<OP>AL: 3, 1/3
    A55InstInfo Swp = a55Desc("swp x1, x2, [x0]");
    ASSERT_TRUE(Swp.Valid);
    EXPECT_EQ(Swp.MaxLatency, 2u);                    // SWP: 2, 1/2
    A55InstInfo Cas = a55Desc("cas x1, x2, [x0]");
    ASSERT_TRUE(Cas.Valid);
    EXPECT_EQ(Cas.MaxLatency, 4u);                    // CAS: 4, 1/4
    A55InstInfo Casp = a55Desc("casp x2, x3, x4, x5, [x0]");
    ASSERT_TRUE(Casp.Valid);
    EXPECT_EQ(Casp.MaxLatency, 5u);                   // CASP: 5, 1/5, 00
    EXPECT_TRUE(Casp.BeginGroup);
    EXPECT_TRUE(Casp.EndGroup);

    size_t N = 0;
    uint64_t Retired = 0;
    // Four LDADDs at throughput 1/2: 8 cycles, and every instruction retires.
    EXPECT_NEAR(a55CyclesPerIter("ldadd x1, x2, [x0]\nldadd x3, x4, [x0]\nldadd x5, x6, [x0]\nldadd x7, x8, [x0]",
                                 &N, &Retired),
                8.0, kTol);
    ASSERT_EQ(N, 4u);
    EXPECT_EQ(Retired % N, 0u);
    // ST<OP> (LD<OP> to XZR): 1, throughput 1.
    EXPECT_NEAR(a55CyclesPerIter("stadd x1, [x0]\nstadd x3, [x0]\nstadd x5, [x0]\nstadd x7, [x0]"), 4.0, kTol);
    // LD<OP>AL: throughput 1/3.
    EXPECT_NEAR(a55CyclesPerIter("ldaddal x1, x2, [x0]\nldaddal x3, x4, [x0]"), 6.0, kTol);
}

// ---------------------------------------------------------------------------
// #4 The dual-issue column (SOG 4.1: 00 / 01 / 10 / 11; 3.2 Tables 1-2).
// ---------------------------------------------------------------------------
TEST(A55SogTest, DualIssueColumn) {
    // LDP W-form "00" (SOG 4.8): c0 ldp; c1 add add; c2 add; next ldp cannot be
    // instruction-1 -> 3 cycles.  The same with an LDR ("11") pairs: 2 cycles.
    A55InstInfo LdpW = a55Desc("ldp w1, w2, [x0]");
    EXPECT_TRUE(LdpW.BeginGroup);
    EXPECT_TRUE(LdpW.EndGroup);
    EXPECT_NEAR(a55CyclesPerIter("ldp w1, w2, [x0]\nadd x5, x5, #1\nadd x6, x6, #1\nadd x7, x7, #1"), 3.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ldr w1, [x0]\nadd x5, x5, #1\nadd x6, x6, #1\nadd x7, x7, #1"), 2.0, kTol);
    // LDAR (01, 4.8) and FCSEL (01, 4.11) are both slot-0-only: they cannot pair.
    EXPECT_NEAR(a55CyclesPerIter("ldar x1, [x0]\nfcsel s0, s1, s2, eq"), 2.0, kTol);
    // BR "10" (4.2): may be instruction-1 but nothing issues after it in the
    // same cycle: c0 add add; c1 br -> 2 cycles.
    A55InstInfo Br = a55Desc("br x9");
    EXPECT_FALSE(Br.BeginGroup);
    EXPECT_TRUE(Br.EndGroup);
    EXPECT_NEAR(a55CyclesPerIter("add x1, x1, #1\nadd x2, x2, #1\nbr x9"), 2.0, kTol);
    // Control instructions cannot be dual issued (3.2 Table 1).
    A55InstInfo Mrs = a55Desc("mrs x1, tpidr_el0");
    EXPECT_TRUE(Mrs.BeginGroup);
    EXPECT_TRUE(Mrs.EndGroup);
    // ASIMD Q-form of a "2*, 11*" row: slot 0 only (4.15 note 1).
    EXPECT_TRUE(a55Desc("add v0.16b, v1.16b, v1.16b").BeginGroup);
    EXPECT_FALSE(a55Desc("add v0.8b, v1.8b, v1.8b").BeginGroup);
    // Divide 01 (4.5).
    EXPECT_TRUE(a55Desc("udiv w1, w2, w3").BeginGroup);
    // A writeback form counts as one instruction of the pair, not two micro-ops:
    // LDR post-index (11) + SUBS/B.NE-free body: one load per cycle.
    EXPECT_NEAR(a55CyclesPerIter("ldr x3, [x0], #8\nadd x5, x5, #1"), 1.0, kTol);
}

// ---------------------------------------------------------------------------
// #5 FSQRT per precision (SOG Table 4, 4.11) and the reciprocal estimate/step
// instructions (SOG 4.17), which the old "^.*SQRT.*" regexes mixed up.
// ---------------------------------------------------------------------------
TEST(A55SogTest, SquareRootAndReciprocalSqrt) {
    A55InstInfo S = a55Desc("fsqrt s0, s1");
    EXPECT_EQ(S.MaxLatency, 12u);
    EXPECT_EQ(S.ResourceCycles["CortexA55UnitFPDIV"], 9u);
    EXPECT_TRUE(S.BeginGroup);
    A55InstInfo D = a55Desc("fsqrt d0, d1");
    EXPECT_EQ(D.MaxLatency, 22u);
    EXPECT_EQ(D.ResourceCycles["CortexA55UnitFPDIV"], 19u);
    A55InstInfo H = a55Desc("fsqrt h0, h1");
    EXPECT_EQ(H.MaxLatency, 8u);
    EXPECT_EQ(H.ResourceCycles["CortexA55UnitFPDIV"], 5u);
    EXPECT_NEAR(a55CyclesPerIter("fsqrt s0, s1\nfsqrt s2, s3"), 18.0, kTol);
    // FRSQRTE / FRSQRTS / URSQRTE: 4, 2*, 11* -- no FP divide/sqrt unit.
    for (const char *Asm : {"frsqrte v0.2s, v1.2s", "frsqrts v0.4s, v1.4s, v2.4s", "ursqrte v0.4s, v1.4s",
                            "frsqrte s0, s1", "frsqrts d0, d1, d2"}) {
        A55InstInfo R = a55Desc(Asm);
        EXPECT_EQ(R.MaxLatency, 4u) << Asm;
        EXPECT_EQ(R.ResourceCycles.count("CortexA55UnitFPDIV"), 0u) << Asm;
    }
    EXPECT_NEAR(a55CyclesPerIter("frsqrte v0.2s, v0.2s\nfrsqrte v0.2s, v0.2s\nfrsqrte v0.2s, v0.2s\nfrsqrte v0.2s, v0.2s"),
                16.0, kTol);
}

// ---------------------------------------------------------------------------
// #6 Accumulator forwarding.  ASIMD MLA / SDOT "4 (1)" (SOG 4.15 notes 2-3):
// a chain through the accumulator issues every cycle.  FMLA / FMADD "4" with
// no accumulate latency (SOG Table 4, 4.11, 4.16): a chain costs 4 per step.
// ---------------------------------------------------------------------------
TEST(A55SogTest, AccumulatorForwarding) {
    EXPECT_NEAR(a55CyclesPerIter("mla v0.4s, v1.4s, v2.4s\nmla v0.4s, v1.4s, v2.4s\nmla v0.4s, v1.4s, v2.4s\nmla v0.4s, v1.4s, v2.4s"),
                4.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("sdot v0.4s, v1.16b, v2.16b\nsdot v0.4s, v1.16b, v2.16b\nsdot v0.4s, v1.16b, v2.16b\nsdot v0.4s, v1.16b, v2.16b"),
                4.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("smlal v0.4s, v1.4h, v2.4h\nsmlal v0.4s, v1.4h, v2.4h"), 2.0, kTol);
    // Not from a non-MAC producer: add (2) -> mla (4) chain = 2 + 4.
    EXPECT_NEAR(a55CyclesPerIter("add v0.2s, v0.2s, v3.2s\nmla v0.2s, v1.2s, v2.2s"), 6.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("fmla v0.4s, v1.4s, v2.4s\nfmla v0.4s, v1.4s, v2.4s\nfmla v0.4s, v1.4s, v2.4s\nfmla v0.4s, v1.4s, v2.4s"),
                16.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("fmadd d0, d1, d2, d0\nfmadd d0, d1, d2, d0\nfmadd d0, d1, d2, d0\nfmadd d0, d1, d2, d0"),
                16.0, kTol);
}

// ---------------------------------------------------------------------------
// #7 RBIT 2 (SOG 4.7), CRC32B/H/X/CB/CH/CX 2, CRC32W/CW 1 (SOG 4.21), and none
// of them is a 1-cycle ALU->ALU forwarding producer.
// ---------------------------------------------------------------------------
TEST(A55SogTest, RbitAndCrcLatency) {
    EXPECT_EQ(a55Desc("rbit x1, x1").MaxLatency, 2u);
    EXPECT_EQ(a55Desc("crc32b w1, w1, w2").MaxLatency, 2u);
    EXPECT_EQ(a55Desc("crc32cx w1, w1, x2").MaxLatency, 2u);
    EXPECT_EQ(a55Desc("crc32w w1, w1, w2").MaxLatency, 1u);
    EXPECT_NEAR(a55CyclesPerIter("rbit x1, x1\nrbit x1, x1\nrbit x1, x1\nrbit x1, x1"), 8.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("crc32b w1, w1, w2\ncrc32b w1, w1, w2\ncrc32b w1, w1, w2\ncrc32b w1, w1, w2"), 8.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("crc32w w1, w1, w2\ncrc32w w1, w1, w2\ncrc32w w1, w1, w2\ncrc32w w1, w1, w2"), 4.0, kTol);
}

// ---------------------------------------------------------------------------
// #9 Loads and stores (SOG 4.8, 4.9, 4.13, 4.14, 4.18, 4.19).
// ---------------------------------------------------------------------------
TEST(A55SogTest, ScalarAndPairLoadsStores) {
    // 9a LDR/LDUR Q: 4, 1/2, 01; pre/post "4, 2".
    for (const char *Asm : {"ldr q0, [x0]", "ldur q0, [x0, #1]", "ldr q0, [x0, x1]"}) {
        A55InstInfo Q = a55Desc(Asm);
        EXPECT_EQ(Q.MaxLatency, 4u) << Asm;
        EXPECT_EQ(Q.ResourceCycles["CortexA55UnitLd"], 2u) << Asm;
        EXPECT_TRUE(Q.BeginGroup) << Asm;
    }
    A55InstInfo QPost = a55Desc("ldr q0, [x0], #16");
    ASSERT_EQ(QPost.WriteLatencies.size(), 2u);
    EXPECT_EQ(QPost.WriteLatencies[0], 2u);   // base update
    EXPECT_EQ(QPost.WriteLatencies[1], 4u);   // data
    EXPECT_NEAR(a55CyclesPerIter("ldr q0, [x0]\nldr q1, [x0, #16]\nldr q2, [x0, #32]\nldr q3, [x0, #48]"), 8.0, kTol);
    // 9b STP/STNP Q: 2, 1/2, 01.
    EXPECT_NEAR(a55CyclesPerIter("stp q0, q1, [x0]\nstp q2, q3, [x0, #32]"), 4.0, kTol);
    EXPECT_EQ(a55Desc("stnp q0, q1, [x0]").ResourceCycles["CortexA55UnitSt"], 2u);
    // 9c LDP Q: 6, 1/4.
    A55InstInfo LdpQ = a55Desc("ldp q0, q1, [x0]");
    EXPECT_EQ(LdpQ.MaxLatency, 6u);
    EXPECT_EQ(LdpQ.ResourceCycles["CortexA55UnitLd"], 4u);
    // 9d LDNP has the LDP rows: W 3 (00), S 3 (01), Q 6.
    A55InstInfo LdnpW = a55Desc("ldnp w1, w2, [x0]");
    EXPECT_EQ(LdnpW.MaxLatency, 3u);
    EXPECT_TRUE(LdnpW.BeginGroup && LdnpW.EndGroup);
    EXPECT_EQ(a55Desc("ldnp s0, s1, [x0]").MaxLatency, 3u);
    EXPECT_EQ(a55Desc("ldnp q0, q1, [x0]").MaxLatency, 6u);
    // 9g Pointer chasing "3 (2)" on every LDR form, not only register offset.
    EXPECT_NEAR(a55CyclesPerIter("ldr x1, [x1]\nldr x1, [x1]\nldr x1, [x1]\nldr x1, [x1]"), 8.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ldr x1, [x1, #8]\nldr x1, [x1, #8]"), 4.0, kTol);
    // 9h Base update latency ", 1" (LDR) and ", 2" (LDP X, 00 and 1/2).
    A55InstInfo LdpXPost = a55Desc("ldp x1, x2, [x0], #16");
    ASSERT_EQ(LdpXPost.WriteLatencies.size(), 3u);
    EXPECT_EQ(LdpXPost.WriteLatencies[0], 2u);
    EXPECT_TRUE(LdpXPost.BeginGroup && LdpXPost.EndGroup);
    EXPECT_EQ(a55Desc("ldr x1, [x0], #8").WriteLatencies[0], 1u);
    EXPECT_NEAR(a55CyclesPerIter("ldr x3, [x0], #8\nldr x4, [x0], #8\nldr x5, [x0], #8\nldr x6, [x0], #8"), 4.0, kTol);
    // 9i LDAXP: 3, 1, 00.
    A55InstInfo Ldaxp = a55Desc("ldaxp x1, x2, [x0]");
    EXPECT_EQ(Ldaxp.MaxLatency, 3u);
    EXPECT_TRUE(Ldaxp.BeginGroup && Ldaxp.EndGroup);
    // 9j STLR 2, 1/2, 01; STLXR 4, 1/2, 01.
    A55InstInfo Stlr = a55Desc("stlr x1, [x0]");
    EXPECT_EQ(Stlr.MaxLatency, 2u);
    EXPECT_EQ(Stlr.ResourceCycles["CortexA55UnitSt"], 2u);
    EXPECT_TRUE(Stlr.BeginGroup);
    A55InstInfo Stlxr = a55Desc("stlxr w5, x1, [x0]");
    EXPECT_EQ(Stlxr.MaxLatency, 4u);
    EXPECT_EQ(Stlxr.ResourceCycles["CortexA55UnitSt"], 2u);
    // 9k PRFM 1.
    EXPECT_EQ(a55Desc("prfm pldl1keep, [x0]").MaxLatency, 1u);
}

TEST(A55SogTest, AsimdLoadsStores) {
    struct Row { const char *Asm; unsigned Lat; unsigned Occ; bool BG; const char *Unit; };
    const Row Rows[] = {
        // SOG 4.18 (p.37-38)
        {"ld1 {v0.8b}, [x0]", 3, 1, false, "CortexA55UnitLd"},
        {"ld1 {v0.16b}, [x0]", 4, 2, true, "CortexA55UnitLd"},
        {"ld1 {v0.16b, v1.16b}, [x0]", 6, 4, true, "CortexA55UnitLd"},
        {"ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0]", 10, 8, true, "CortexA55UnitLd"},
        {"ld1 {v0.s}[1], [x0]", 3, 1, true, "CortexA55UnitLd"},
        {"ld1r {v0.4s}, [x0]", 3, 1, true, "CortexA55UnitLd"},
        {"ld2 {v0.d, v1.d}[1], [x0]", 4, 2, true, "CortexA55UnitLd"},
        {"ld2 {v0.s, v1.s}[1], [x0]", 3, 1, true, "CortexA55UnitLd"},
        {"ld3 {v0.8b, v1.8b, v2.8b}, [x0]", 6, 4, true, "CortexA55UnitLd"},
        {"ld3 {v0.16b, v1.16b, v2.16b}, [x0]", 9, 7, true, "CortexA55UnitLd"},
        {"ld3 {v0.4s, v1.4s, v2.4s}, [x0]", 8, 6, true, "CortexA55UnitLd"},
        {"ld4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0]", 11, 9, true, "CortexA55UnitLd"},
        {"ld4 {v0.2s, v1.2s, v2.2s, v3.2s}, [x0]", 6, 4, true, "CortexA55UnitLd"},
        {"ld4r {v0.2d, v1.2d, v2.2d, v3.2d}, [x0]", 6, 4, true, "CortexA55UnitLd"},
        // SOG 4.19 (p.39-40)
        {"st1 {v0.16b}, [x0]", 1, 1, false, "CortexA55UnitSt"},
        {"st1 {v0.8b, v1.8b}, [x0]", 1, 1, true, "CortexA55UnitSt"},
        {"st1 {v0.16b, v1.16b}, [x0]", 2, 2, true, "CortexA55UnitSt"},
        {"st1 {v0.16b, v1.16b, v2.16b}, [x0]", 3, 3, true, "CortexA55UnitSt"},
        {"st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [x0]", 2, 2, true, "CortexA55UnitSt"},
        {"st2 {v0.8b, v1.8b}, [x0]", 1, 1, true, "CortexA55UnitSt"},
        {"st2 {v0.16b, v1.16b}, [x0]", 2, 2, true, "CortexA55UnitSt"},
        {"st2 {v0.d, v1.d}[1], [x0]", 2, 2, true, "CortexA55UnitSt"},
        {"st3 {v0.16b, v1.16b, v2.16b}, [x0]", 3, 3, true, "CortexA55UnitSt"},
        {"st4 {v0.8b, v1.8b, v2.8b, v3.8b}, [x0]", 3, 3, true, "CortexA55UnitSt"},
        {"st4 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0]", 5, 5, true, "CortexA55UnitSt"},
        {"st4 {v0.2d, v1.2d, v2.2d, v3.2d}, [x0]", 4, 4, true, "CortexA55UnitSt"},
    };
    for (const Row &R : Rows) {
        A55InstInfo I = a55Desc(R.Asm);
        ASSERT_TRUE(I.Valid) << R.Asm;
        EXPECT_EQ(I.MaxLatency, R.Lat) << R.Asm;
        EXPECT_EQ(I.ResourceCycles[R.Unit], R.Occ) << R.Asm;
        EXPECT_EQ(I.BeginGroup, R.BG) << R.Asm;
    }
    // Writeback "(1/T)": LD1 four Q registers, T = 1/8.
    A55InstInfo Post = a55Desc("ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0], #64");
    ASSERT_EQ(Post.WriteLatencies.size(), 2u);
    EXPECT_EQ(Post.WriteLatencies[0], 8u);
}

// ---------------------------------------------------------------------------
// #10 ASIMD rows that were on the WriteVd/WriteVq default of 4
// (SOG 4.15-4.17, 4.20).  {asm, latency, BeginGroup (Q-form / 01)}.
// ---------------------------------------------------------------------------
TEST(A55SogTest, AsimdMiscRows) {
    struct Row { const char *Asm; unsigned Lat; bool BG; };
    const Row Rows[] = {
        {"umov w1, v0.s[1]", 2, false},            // 4.17 SMOV/UMOV 2, 2, 11
        {"smov x1, v0.b[1]", 2, false},
        {"ext v0.8b, v1.8b, v2.8b, #1", 2, false}, // EXT 2, 2*, 11*
        {"ext v0.16b, v1.16b, v2.16b, #1", 2, true},
        {"rev64 v0.4s, v1.4s", 2, true},           // REV 2, 2*, 11*
        {"trn1 v0.2d, v1.2d, v2.2d", 2, true},     // TRN .2D 2, 1, 01
        {"zip1 v0.8b, v1.8b, v2.8b", 2, false},    // ZIP 2, 2*, 11*
        {"cnt v0.8b, v1.8b", 2, false},            // CNT 2
        {"clz v0.4s, v1.4s", 2, true},             // CLZ 2
        {"cls v0.4s, v1.4s", 3, true},             // CLS 3
        {"rbit v0.16b, v1.16b", 2, true},          // RBIT 2
        {"fcmeq v0.4s, v1.4s, v2.4s", 2, true},    // 4.16 FP compare 2, 2*, 11*
        {"fcmgt v0.2s, v1.2s, #0.0", 2, false},
        {"aese v0.16b, v1.16b", 2, true},          // 4.20 AESE 2, 1, 01
        {"aesmc v0.16b, v1.16b", 2, false},        //      AESMC 2, 1, 11
        {"sha1h s0, s1", 2, true},                 //      SHA1H 2, 1, 01
        {"sha256su0 v0.4s, v1.4s", 3, true},       //      SHA256SU0 3, 1, 01
        {"pmull v0.1q, v1.1d, v2.1d", 2, true},    //      PMULL 64-bit 2, 1, 01
        {"tbl v0.16b, {v1.16b}, v2.16b", 2, true}, // 4.17 TBL 2+N-1, 1/N, 01
        {"tbl v0.8b, {v1.16b, v2.16b}, v3.8b", 3, true},
        {"tbx v0.16b, {v1.16b, v2.16b, v3.16b, v4.16b}, v5.16b", 6, true}, // TBX 2+N, 1/(N+1)
        {"dup v0.4s, w1", 3, false},               // DUP gen reg 3, 2, 11
        {"dup v0.4s, v1.s[1]", 2, true},           // DUP element 2, 2*, 11*
        {"dup v0.2s, v1.s[1]", 2, false},
        {"sqshrn v0.8b, v1.8h, #2", 3, false},     // 4.15 SQSHRN 3, 2*, 11*
        {"sqshrn2 v0.16b, v1.8h, #2", 3, true},
        {"sli v0.4s, v1.4s, #2", 2, true},         // SLI 2, 2*, 11*
        {"sri v0.8b, v1.8b, #2", 2, false},
        {"fmov v0.4s, #1.0", 1, false},            // 4.17 FMOV FP immed 1, 2, 11
        {"fcvtl v0.4s, v1.4h", 4, true},           // 4.16 FCVTL 4, 1, 01
        {"fcvtn v0.4h, v1.4s", 4, true},           //      FCVTN 4, 1, 01
        {"fcvtzs v0.4s, v1.4s", 4, true},          //      FP convert other 4, 2*, 11*
        {"fmul v0.2s, v1.2s, v2.s[1]", 4, true},   //      FMUL by element 4, 1, 01
        {"saba v0.8b, v1.8b, v2.8b", 4, true},     // 4.15 SABA 4, 1/2, 01
        {"raddhn v0.8b, v1.8h, v2.8h", 4, true},   //      RADDHN 4, 1/2, 01
        {"srsra v0.4s, v1.4s, #2", 4, true},       //      SRSRA 4, 1/2, 01
        {"sadalp v0.4s, v1.8h", 4, true},          //      SADALP 4, 1/2, 01
        {"fcsel s0, s1, s2, eq", 2, true},         // 4.11 FCSEL 2, 1, 01
    };
    for (const Row &R : Rows) {
        A55InstInfo I = a55Desc(R.Asm);
        ASSERT_TRUE(I.Valid) << R.Asm;
        EXPECT_EQ(I.MaxLatency, R.Lat) << R.Asm;
        EXPECT_EQ(I.BeginGroup, R.BG) << R.Asm;
    }
    // Throughputs: TBX of 4 registers 1/5 -> FP ALU unit held 2/T = 10 cycles;
    // SABA 1/2 -> 4; FMOV FP immed Q-form throughput 2 -> 1.
    EXPECT_EQ(a55Desc("tbx v0.16b, {v1.16b, v2.16b, v3.16b, v4.16b}, v5.16b").ResourceCycles["CortexA55UnitFPALU"], 10u);
    EXPECT_EQ(a55Desc("saba v0.8b, v1.8b, v2.8b").ResourceCycles["CortexA55UnitFPALU"], 4u);
    EXPECT_EQ(a55Desc("fmov v0.4s, #1.0").ResourceCycles["CortexA55UnitFPALU"], 1u);
    EXPECT_NEAR(a55CyclesPerIter("saba v0.8b, v1.8b, v2.8b\nsaba v3.8b, v1.8b, v2.8b"), 4.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ext v0.16b, v0.16b, v1.16b, #4\next v0.16b, v0.16b, v1.16b, #4"), 4.0, kTol);
}

// ---------------------------------------------------------------------------
// #11 Integer divide, best case (SOG 4.5): latency 3, throughput 1/3.
// ---------------------------------------------------------------------------
TEST(A55SogTest, IntegerDivideBestCase) {
    A55InstInfo W = a55Desc("udiv w1, w2, w3");
    EXPECT_EQ(W.MaxLatency, 3u);
    EXPECT_EQ(W.ResourceCycles["CortexA55UnitDiv"], 3u);
    A55InstInfo X = a55Desc("sdiv x1, x2, x3");
    EXPECT_EQ(X.MaxLatency, 3u);
    EXPECT_EQ(X.ResourceCycles["CortexA55UnitDiv"], 3u);
    EXPECT_NEAR(a55CyclesPerIter("udiv w1, w2, w3\nudiv w4, w5, w6"), 6.0, kTol);
}

// ---------------------------------------------------------------------------
// [a55_alu_fwd] ALU -> ALU forwarding (SOG 3.1.1, 3.2, 4.3).
// "There is also a dedicated forwarding path from the ex1 stage of the ALU0 to
// the ex1 stage of the ALU1": a 1-cycle ALU instruction-0 and an ALU
// instruction-1 that reads its result in the ALU (not in the shifter) issue in
// the same cycle.  Otherwise an ALU result is seen at its exec latency (ALU
// basic 1, ALU extend and/or shift 2, SOG 4.3).  The rule lives in the issue
// stage (A55DecoupledIssueStage::isSameCycleALUForward); the .td no longer has
// an ALU -> ALU ReadAdvance.
// ---------------------------------------------------------------------------
TEST(A55SogTest, SameCycleAluForwarding) {
    // Four dependent adds: c0 add+add (ALU0 -> ALU1 path), c1 add+add (the
    // first one reads the ALU1 result of c0 one cycle later = latency 1) -> 2.
    EXPECT_NEAR(a55CyclesPerIter("add x1, x1, #1\nadd x1, x1, #1\nadd x1, x1, #1\nadd x1, x1, #1"), 2.0, kTol);
    // Without the dedicated path the shifted operand waits a cycle: x1 feeds
    // the shifter of the "ALU extend and/or shift" add (2).  c0 add; c1 add-lsl;
    // c3 add (waits for the 2-cycle result) -> 3.
    EXPECT_NEAR(a55CyclesPerIter("add x1, x1, #1\nadd x1, x3, x1, lsl #2"), 3.0, kTol);
    // Same pair, x1 now the unshifted operand (read by the ALU): c0 add +
    // add-lsl; c2 next add -> 2.
    EXPECT_NEAR(a55CyclesPerIter("add x1, x1, #1\nadd x1, x1, x3, lsl #2"), 2.0, kTol);
    // A 2-cycle producer is not on the path: c0 add-lsl; c2 add; c3 add-lsl -> 3.
    EXPECT_NEAR(a55CyclesPerIter("add x1, x2, x1, lsl #2\nadd x1, x1, #1"), 3.0, kTol);
    // Variable shift (1, SOG 4.4): both operands feed the shifter, so it is a
    // producer (1-cycle ALU result) but not a consumer of the path.
    // c0 add; c1 lslv + add (lslv -> add path); c2 lslv + add ... -> 1.
    EXPECT_NEAR(a55CyclesPerIter("add x1, x1, #1\nlslv x1, x1, x3"), 1.0, kTol);
    // Load -> ALU is the load-use latency 2 (SOG 3.3), never the ALU path:
    // c0 ldr; c2 add; next ldr dual-issues with it -> 2.
    EXPECT_NEAR(a55CyclesPerIter("ldr x1, [x0]\nadd x2, x1, #1"), 2.0, kTol);
    // ALU -> multiply operand (MAC operands at iss, SOG 3.1.1): latency 1, no
    // same-cycle pair.  c0 add; c1 mul (W, 3); c4 add -> 4.
    EXPECT_NEAR(a55CyclesPerIter("add w1, w1, #1\nmul w1, w1, w1"), 4.0, kTol);
}

namespace {
// Latency of each instruction of `Asm` (one per line) for cortex-a55, with its
// opcode; one context for the whole listing.
std::vector<std::pair<unsigned, unsigned>> a55OpcodeLatencies(const std::string &Asm) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a55");
    auto Instrs = parseAsm(TC, Asm);
    mca::InstrumentManager IM(*TC.STI, *TC.MCII);
    mca::InstrBuilder IB(*TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), IM, 0);
    std::vector<std::pair<unsigned, unsigned>> Out;
    for (const auto &I : Instrs) {
        auto ExpectedInst = IB.createInstruction(I.Inst, {});
        if (!ExpectedInst) {
            consumeError(ExpectedInst.takeError());
            Out.push_back({I.Inst.getOpcode(), 0});
            continue;
        }
        Out.push_back({I.Inst.getOpcode(), (*ExpectedInst)->getLatency()});
    }
    return Out;
}
} // namespace

// ---------------------------------------------------------------------------
// [a55_shift_imm] LSL/LSR/ASR by immediate: 1 cycle.  Project decision: the
// AArch32 row "Move, shift by immed ASR/LSL/LSR/ROR 1, 2, 11" (SOG 4.4, l.729)
// is applied to the AArch64 aliases of UBFM/SBFM (same shifter, SOG fig. 1);
// the literal AArch64 row is "Bitfield move, basic SBFM, UBFM 2" (SOG 4.7,
// l.909), which still applies to UBFX/SBFX/UBFIZ/SBFIZ; "Sign/zero extend" 1
// (l.907) to SXTB/UXTB/SXTH/UXTH/SXTW.
// ---------------------------------------------------------------------------
TEST(A55SogTest, ShiftImmediateAliasesTakeOneCycle) {
    EXPECT_EQ(a55Desc("lsl w1, w2, #3").MaxLatency, 1u);
    EXPECT_EQ(a55Desc("lsl x1, x2, #3").MaxLatency, 1u);
    EXPECT_EQ(a55Desc("lsl w1, w2, #31").MaxLatency, 1u);
    EXPECT_EQ(a55Desc("lsl x1, x2, #63").MaxLatency, 1u);
    EXPECT_EQ(a55Desc("lsl x1, x2, #1").MaxLatency, 1u);
    EXPECT_EQ(a55Desc("lsr w1, w2, #5").MaxLatency, 1u);
    EXPECT_EQ(a55Desc("lsr x1, x2, #63").MaxLatency, 1u);
    EXPECT_EQ(a55Desc("asr w1, w2, #5").MaxLatency, 1u);
    EXPECT_EQ(a55Desc("asr x1, x2, #1").MaxLatency, 1u);
    EXPECT_EQ(a55Desc("lsl x1, x2, #0").MaxLatency, 1u);       // UBFM #0, #63 (= LSR #0)
    // Not shifts: "Bitfield move, basic" 2.
    EXPECT_EQ(a55Desc("ubfx x1, x2, #3, #5").MaxLatency, 2u);   // UBFM #3, #7
    EXPECT_EQ(a55Desc("ubfx w1, w2, #1, #30").MaxLatency, 2u);  // UBFM #1, #30
    EXPECT_EQ(a55Desc("ubfiz x1, x2, #3, #5").MaxLatency, 2u);  // UBFM #61, #4
    EXPECT_EQ(a55Desc("ubfiz x1, x2, #59, #4").MaxLatency, 2u); // UBFM #5, #3 (immr = imms + 2)
    EXPECT_EQ(a55Desc("sbfx x1, x2, #3, #5").MaxLatency, 2u);
    EXPECT_EQ(a55Desc("sbfiz x1, x2, #60, #4").MaxLatency, 2u); // SBFM #4, #3: the LSL shape is UBFM only
    EXPECT_EQ(a55Desc("ubfx x1, x2, #1, #31").MaxLatency, 2u);  // UBFM X #1, #31: imms 31 is LSR only for W
    // Sign/zero extend 1 (unchanged).
    EXPECT_EQ(a55Desc("uxtb w1, w2").MaxLatency, 1u);
    EXPECT_EQ(a55Desc("sxth x1, w2").MaxLatency, 1u);
    EXPECT_EQ(a55Desc("sxtw x1, w2").MaxLatency, 1u);
    // BFM (insert) 2 (unchanged).
    EXPECT_EQ(a55Desc("bfi x1, x2, #3, #5").MaxLatency, 2u);

    // Every (immr, imms) of UBFM/SBFM W and X: 1 exactly for the shift and
    // extend aliases, 2 otherwise.
    for (int Size : {32, 64}) {
        const char R = Size == 32 ? 'w' : 'x';
        for (const char *Op : {"ubfm", "sbfm"}) {
            std::string Asm;
            std::vector<std::pair<int, int>> Pairs;
            for (int Immr = 0; Immr < Size; ++Immr)
                for (int Imms = 0; Imms < Size; ++Imms) {
                    Asm += std::string(Op) + " " + R + "1, " + R + "2, #" + std::to_string(Immr) + ", #" +
                           std::to_string(Imms) + "\n";
                    Pairs.push_back({Immr, Imms});
                }
            auto Lat = a55OpcodeLatencies(Asm);
            ASSERT_EQ(Lat.size(), Pairs.size());
            const bool IsU = std::string(Op) == "ubfm";
            for (size_t I = 0; I < Pairs.size(); ++I) {
                auto [Immr, Imms] = Pairs[I];
                bool ShiftRight = Imms == Size - 1;                       // LSR / ASR #immr
                bool Lsl = IsU && Imms != Size - 1 && Immr == Imms + 1;   // LSL #(Size - immr)
                bool Extend = Immr == 0 && (Imms == 7 || Imms == 15 || Imms == 31);
                unsigned Expected = (ShiftRight || Lsl || Extend) ? 1u : 2u;
                EXPECT_EQ(Lat[I].second, Expected) << Op << " " << R << " immr=" << Immr << " imms=" << Imms;
            }
        }
    }

    // A dependent LSL chain runs at latency 1 (the source feeds the shifter, so
    // no same-cycle pair): 4 cycles for 4.  UBFX stays 2: 8.
    EXPECT_NEAR(a55CyclesPerIter("lsl x1, x1, #3\nlsl x1, x1, #3\nlsl x1, x1, #3\nlsl x1, x1, #3"), 4.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ubfx x1, x1, #3, #5\nubfx x1, x1, #3, #5\nubfx x1, x1, #3, #5\nubfx x1, x1, #3, #5"), 8.0, kTol);
    // As a 1-cycle ALU result an LSL is a producer of the ALU0 -> ALU1 path:
    // c0 lsl + add; c1 lsl (reads the add, latency 1) + add -> 1.
    EXPECT_NEAR(a55CyclesPerIter("lsl x1, x1, #3\nadd x1, x1, #1"), 1.0, kTol);
}

// ---------------------------------------------------------------------------
// [a55_ror_imm] ROR by immediate = EXTR with Rn == Rm: 1 cycle, the same
// project decision as [a55_shift_imm] (AArch32 row "Move, shift by immed
// ASR/LSL/LSR/ROR 1, 2, 11", SOG 4.4 l.729).  The literal AArch64 row,
// "Bitfield extract EXTR 2, 2, 11" (SOG 4.7 l.905), still applies to EXTR with
// Rn != Rm (and would give 2 for ROR too).
// ---------------------------------------------------------------------------
TEST(A55SogTest, RorImmediateAliasTakesOneCycle) {
    // The latency comes from the MC instruction actually built for each line
    // (EXTRWrri / EXTRXrri, operands Rd, Rn, Rm, lsb).
    std::string Asm;
    std::vector<std::pair<std::string, unsigned>> Expected;
    for (int N : {0, 1, 3, 17, 31}) {
        Expected.push_back({"ror w1, w2, #" + std::to_string(N), 1u});
        Expected.push_back({"extr w1, w2, w2, #" + std::to_string(N), 1u});   // the same instruction
        Expected.push_back({"extr w1, w2, w3, #" + std::to_string(N), 2u});
    }
    for (int N : {0, 1, 3, 33, 63}) {
        Expected.push_back({"ror x1, x2, #" + std::to_string(N), 1u});
        Expected.push_back({"ror x1, x1, #" + std::to_string(N), 1u});
        Expected.push_back({"extr x1, x2, x3, #" + std::to_string(N), 2u});
        Expected.push_back({"extr x1, x1, x3, #" + std::to_string(N), 2u});    // Rd == Rn, Rm differs
        Expected.push_back({"extr x1, x3, x1, #" + std::to_string(N), 2u});
    }
    for (const auto &E : Expected)
        Asm += E.first + "\n";
    auto Lat = a55OpcodeLatencies(Asm);
    ASSERT_EQ(Lat.size(), Expected.size());
    for (size_t I = 0; I < Expected.size(); ++I) {
        const bool IsW = Expected[I].first.find(" w1") != std::string::npos;
        EXPECT_EQ(Lat[I].first, IsW ? unsigned(AArch64::EXTRWrri) : unsigned(AArch64::EXTRXrri)) << Expected[I].first;
        EXPECT_EQ(Lat[I].second, Expected[I].second) << Expected[I].first;
    }
    // Dependent chains: ROR 1 per step (like the LSL chain; the source feeds the
    // shifter, so no same-cycle ALU0 -> ALU1 pair): 4.  EXTR Rn != Rm 2: 8.
    EXPECT_NEAR(a55CyclesPerIter("ror x1, x1, #3\nror x1, x1, #3\nror x1, x1, #3\nror x1, x1, #3"), 4.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ror w1, w1, #3\nror w1, w1, #3\nror w1, w1, #3\nror w1, w1, #3"), 4.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("extr x1, x1, x2, #3\nextr x1, x1, x2, #3\nextr x1, x1, x2, #3\nextr x1, x1, x2, #3"),
                8.0, kTol);
    // As a 1-cycle ALU result a ROR is a producer of the ALU0 -> ALU1 path:
    // c0 ror + add; c1 ror (reads the add, latency 1) + add -> 1.
    EXPECT_NEAR(a55CyclesPerIter("ror x1, x1, #3\nadd x1, x1, #1"), 1.0, kTol);
}

// ---------------------------------------------------------------------------
// [a55_varshift] Variable shift ASRV/LSLV/LSRV/RORV: latency 1, throughput 1,
// dual-issue 11 (SOG 4.4 l.758), against throughput 2 for ALU basic (4.3).
// Modelled as one ALU + the one-unit CortexA55UnitVarShift for one cycle.
// ---------------------------------------------------------------------------
TEST(A55SogTest, VariableShiftThroughputOne) {
    for (const char *Asm : {"lsl x1, x2, x3", "lsr w1, w2, w3", "asr x1, x2, x3", "ror w1, w2, w3",
                            "lslv x1, x2, x3", "rorv x1, x2, x3"}) {
        A55InstInfo I = a55Desc(Asm);
        ASSERT_TRUE(I.Valid) << Asm;
        EXPECT_EQ(I.MaxLatency, 1u) << Asm;
        EXPECT_EQ(I.ResourceCycles["CortexA55UnitALU"], 1u) << Asm;
        EXPECT_EQ(I.ResourceCycles["CortexA55UnitVarShift"], 1u) << Asm;
        EXPECT_FALSE(I.BeginGroup) << Asm;
        EXPECT_FALSE(I.EndGroup) << Asm;
    }
    // Not on other ALU instructions (ALU basic, shift by immediate).
    EXPECT_EQ(a55Desc("add x1, x2, x3").ResourceCycles.count("CortexA55UnitVarShift"), 0u);
    EXPECT_EQ(a55Desc("lsl x1, x2, #3").ResourceCycles.count("CortexA55UnitVarShift"), 0u);
    // Four independent variable shifts: one per cycle -> 4 (was 2).
    EXPECT_NEAR(a55CyclesPerIter("lsl x1, x2, x9\nlsr x3, x4, x9\nasr x5, x6, x9\nror x7, x8, x9"), 4.0, kTol);
    // A variable shift dual-issues with an independent ALU instruction:
    // c0 lsl + add; c1 lsl + add -> 2.
    EXPECT_NEAR(a55CyclesPerIter("lsl x1, x2, x9\nadd x3, x4, #1\nlsl x5, x6, x9\nadd x7, x8, #1"), 2.0, kTol);
    // Dependent chain: latency 1 per step -> 4.
    EXPECT_NEAR(a55CyclesPerIter("lsl x1, x1, x9\nlsl x1, x1, x9\nlsl x1, x1, x9\nlsl x1, x1, x9"), 4.0, kTol);
    // Still a 1-cycle ALU-pipe producer of the ALU0 -> ALU1 path (SOG 3.1.1):
    // c0 lslv + add -> 1 (see SameCycleAluForwarding for the add -> lslv order).
    EXPECT_NEAR(a55CyclesPerIter("lsl x1, x1, x9\nadd x1, x1, #1"), 1.0, kTol);
}

// ---------------------------------------------------------------------------
// [a55_flag_bypass] SOG 3.2 Table 2 "Conditional": a conditional
// (flag-dependent) instruction-1 can be dual issued with a flag setting
// instruction-0, except arithmetic with carry (ADC/ADCS/SBC/SBCS, and their
// aliases NGC/NGCS).  SOG 5.11: the flag transfer from the FP flags takes a
// cycle, so FCMP{E} and FCCMP{E} are not same-cycle flag producers either.
// ---------------------------------------------------------------------------
TEST(A55SogTest, FlagBypassExceptCarryAndFpCompare) {
    // cmp + csel pair (unchanged): c0 cmp + csel; c1 csel + cmp; c2 csel + csel
    // -> 3 instructions per 1.5 cycles.
    EXPECT_NEAR(a55CyclesPerIter("cmp x1, x2\ncsel x3, x4, x5, eq\ncsel x6, x4, x5, ne"), 1.5, kTol);
    // Arithmetic with carry waits for the flags: c0 cmp; c1 adc + adc -> 2.
    EXPECT_NEAR(a55CyclesPerIter("cmp x1, x2\nadc x3, x3, x4\nadc x6, x6, x4"), 2.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("cmp x1, x2\nsbc w3, w3, w4\nsbc w6, w6, w4"), 2.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("cmp x1, x2\nngc x3, x4\nngc x6, x4"), 2.0, kTol);
    // ADCS -> ADC through the carry: c0 cmp; c1 adcs; c2 adc + next cmp;
    // c3 adcs -> 2.
    EXPECT_NEAR(a55CyclesPerIter("cmp x1, x2\nadcs x3, x3, x4\nadc x6, x6, x4"), 2.0, kTol);
    // adds -> adc, then an LDP W (00) that cannot be instruction-1:
    // c0 adds; c1 adc; c2 ldp -> 3 (2 with a same-cycle adds + adc pair).
    EXPECT_NEAR(a55CyclesPerIter("adds x1, x1, x2\nadc x3, x3, x4\nldp w5, w6, [x0]"), 3.0, kTol);
    // FP compares: c0 fccmp/fcmp; c1 csel + csel -> 2.
    EXPECT_NEAR(a55CyclesPerIter("fccmp s0, s1, #0, eq\ncsel x3, x4, x5, eq\ncsel x6, x4, x5, ne"), 2.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("fccmpe d0, d1, #0, eq\ncsel x3, x4, x5, eq\ncsel x6, x4, x5, ne"), 2.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("fcmp s0, s1\ncsel x3, x4, x5, eq\ncsel x6, x4, x5, ne"), 2.0, kTol);
}

// ---------------------------------------------------------------------------
// #12 --ignore-loop-carried=default: in a basic block the dependencies on the
// previous iteration are ignored, also for the FIRST instruction of the block
// (the tracker used to drop the previous iteration's producers only at that
// instruction's Dispatched event, after the issue stage's hazard check).
// fadd 4 (SOG 4.11); ALU basic 1, throughput 2.
// ---------------------------------------------------------------------------
TEST(A55SogTest, IgnoreLoopCarriedAppliesToFirstInstruction) {
    const char *First = "fadd d0, d0, d1\nadd x2, x2, #1\nadd x3, x3, #1";
    const char *Second = "add x2, x2, #1\nfadd d0, d0, d1\nadd x3, x3, #1";
    // Loop: the fadd chain, 4 per iteration, wherever the fadd is.
    EXPECT_NEAR(a55CyclesPerIter(First), 4.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter(Second), 4.0, kTol);
    // Basic block: no chain, 3 instructions per 1.5 cycles (c0 fadd + add;
    // c1 add + fadd; c2 add + add) in both positions.
    EXPECT_NEAR(a55CyclesPerIter(First, nullptr, nullptr, /*IgnoreLoopCarried=*/true), 1.5, kTol);
    EXPECT_NEAR(a55CyclesPerIter(Second, nullptr, nullptr, /*IgnoreLoopCarried=*/true), 1.5, kTol);
}

// ---------------------------------------------------------------------------
// [a55_sog_n5] FP compare FCCMP{E}, FCMP{E}: 1, 1, 11 (SOG 4.11 p.26, l.1355),
// against throughput 2 for the other FP ALU rows.  One FP ALU unit + the
// one-unit CortexA55UnitFPCmp for one cycle (docs/a55-sog.md C-21).
// ---------------------------------------------------------------------------
TEST(A55SogTest, FpCompareThroughputOne) {
    for (const char *Asm : {"fcmp s0, s1", "fcmpe d0, d1", "fcmp h0, #0.0", "fccmp s0, s1, #0, eq",
                            "fccmpe d0, d1, #4, ne"}) {
        A55InstInfo I = a55Desc(Asm);
        ASSERT_TRUE(I.Valid) << Asm;
        EXPECT_EQ(I.MaxLatency, 1u) << Asm;
        EXPECT_EQ(I.ResourceCycles["CortexA55UnitFPALU"], 1u) << Asm;
        EXPECT_EQ(I.ResourceCycles["CortexA55UnitFPCmp"], 1u) << Asm;
        EXPECT_FALSE(I.BeginGroup) << Asm;  // 11
        EXPECT_FALSE(I.EndGroup) << Asm;
    }
    // Not on the other FP ALU rows.
    EXPECT_EQ(a55Desc("fadd d0, d1, d2").ResourceCycles.count("CortexA55UnitFPCmp"), 0u);
    // Four independent compares: one per cycle -> 4 (was 2).
    EXPECT_NEAR(a55CyclesPerIter("fcmp s0, s1\nfcmp s2, s3\nfcmp d4, d5\nfcmpe d6, d7"), 4.0, kTol);
    // An FP compare still pairs with another FP ALU instruction (11):
    // c0 fcmp + fadd; c1 fcmp + fadd -> 2.
    EXPECT_NEAR(a55CyclesPerIter("fcmp s0, s1\nfadd d2, d3, d3\nfcmp s4, s5\nfadd d6, d3, d3"), 2.0, kTol);
}

// ---------------------------------------------------------------------------
// [a55_sog_n6] ASIMD logical MOV 2, 2*, 11* (SOG 4.15 A64 p.31, l.1709), against
// ORR 1, 2*, 11* (l.1707-1708).  MOV Vd, Vn is ORR Vd, Vn, Vn (docs/a55-sog.md I-8).
// ---------------------------------------------------------------------------
TEST(A55SogTest, AsimdMovRegisterLatencyTwo) {
    struct Row { const char *Asm; unsigned Lat; bool BG; };
    const Row Rows[] = {
        {"mov v0.8b, v1.8b", 2, false},              // MOV 2, 2*, 11*: D-form 11
        {"mov v0.16b, v1.16b", 2, true},             //                 Q-form slot 0 only
        {"orr v0.8b, v1.8b, v2.8b", 1, false},       // ORR 1, 2*, 11*
        {"orr v0.16b, v1.16b, v2.16b", 1, true},
        {"orr v0.4s, #1", 1, true},                  // ORR (immediate): unchanged
    };
    for (const Row &R : Rows) {
        A55InstInfo I = a55Desc(R.Asm);
        ASSERT_TRUE(I.Valid) << R.Asm;
        EXPECT_EQ(I.MaxLatency, R.Lat) << R.Asm;
        EXPECT_EQ(I.BeginGroup, R.BG) << R.Asm;
    }
    // A chain of four MOVs: 2 per step -> 8 (was 4).
    EXPECT_NEAR(a55CyclesPerIter("mov v0.16b, v0.16b\nmov v0.16b, v0.16b\nmov v0.16b, v0.16b\nmov v0.16b, v0.16b"),
                8.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("mov v0.8b, v0.8b\nmov v0.8b, v0.8b\nmov v0.8b, v0.8b\nmov v0.8b, v0.8b"), 8.0, kTol);
}

// ---------------------------------------------------------------------------
// [a55_sog_n7] ASIMD extract narrow, saturating SQXTN(2), SQXTUN(2), UQXTN(2):
// 4, 2, 11 with no Q-form note (SOG 4.17 p.35, l.1991-1992): one FP ALU unit,
// both slots, for every form.
// ---------------------------------------------------------------------------
TEST(A55SogTest, SaturatingExtractNarrowDualIssues) {
    for (const char *Asm : {"sqxtn v0.8b, v1.8h", "sqxtn2 v0.16b, v1.8h", "sqxtun v0.4h, v1.4s",
                            "sqxtun2 v0.4s, v1.2d", "uqxtn v0.2s, v1.2d", "uqxtn2 v0.8h, v1.4s",
                            "sqxtn b0, h1", "uqxtn s0, d1"}) {
        A55InstInfo I = a55Desc(Asm);
        ASSERT_TRUE(I.Valid) << Asm;
        EXPECT_EQ(I.MaxLatency, 4u) << Asm;
        EXPECT_EQ(I.ResourceCycles["CortexA55UnitFPALU"], 1u) << Asm;
        EXPECT_FALSE(I.BeginGroup) << Asm;
        EXPECT_FALSE(I.EndGroup) << Asm;
    }
    // Four independent narrows, two per cycle -> 2 (was 4: slot 0 only, both units).
    EXPECT_NEAR(a55CyclesPerIter("sqxtn v0.8b, v1.8h\nuqxtn v2.8b, v3.8h\nsqxtun v4.8b, v5.8h\nsqxtn v6.8b, v7.8h"),
                2.0, kTol);
}

// ---------------------------------------------------------------------------
// [a55_sog_n8] NOP: SOG 3.2 Table 2 (p.11, l.458) "Miscellaneous: IT,
// Architectural NOP" can be instruction-1; Table 1 does not restrict it as
// instruction-0; only branches may not pair with branches (Table 2,
// l.436-438).  The SOG names no pipe for a NOP: no resource.
// ---------------------------------------------------------------------------
TEST(A55SogTest, NopDualIssues) {
    A55InstInfo Nop = a55Desc("nop");
    ASSERT_TRUE(Nop.Valid);
    EXPECT_TRUE(Nop.ResourceCycles.empty());
    EXPECT_FALSE(Nop.BeginGroup);
    EXPECT_FALSE(Nop.EndGroup);
    // Two NOPs per cycle -> 2 (was 4: one per cycle on the branch unit).
    EXPECT_NEAR(a55CyclesPerIter("nop\nnop\nnop\nnop"), 2.0, kTol);
    // NOP + branch in one cycle: c0 nop + br (br 10 ends the cycle) -> 1 (was 2).
    EXPECT_NEAR(a55CyclesPerIter("nop\nbr x9"), 1.0, kTol);
    // NOP + ALU: c0 add + nop -> 1.
    EXPECT_NEAR(a55CyclesPerIter("add x1, x1, #1\nnop"), 1.0, kTol);
    // The other hints (HINT: YIELD, WFE, ...) are unchanged (docs/a55-sog.md U-13).
    EXPECT_EQ(a55Desc("yield").ResourceCycles["CortexA55UnitB"], 1u);
}

// ---------------------------------------------------------------------------
// [a55_sog_n2] Pointer chasing from load pairs (docs/a55-sog.md C-7).  SOG 3.3 Table 3 (p.11, l.494-495): for LDRD/LDP the fast path exists
// for "Only the first register of the pair if 32-bit load or the second
// register of the pair if 64-bit load"; SOG 4.8 (p.21, l.1078) gives LDPSW
// "3" with no number in parentheses.  One-instruction chains (two copies per
// body), cycles per iteration = 2 x the latency seen by the next base:
//   LDP X first register 4 (no fast path), second 4 - 1 = 3;
//   LDP W first register 3 - 1 = 2, second 3; LDPSW 3 for both.
// (Before: every register of every pair took the fast path: 6, 6, 4, 4, 4.)
// ---------------------------------------------------------------------------
TEST(A55SogTest, LoadPairPointerChasingPerRegister) {
    EXPECT_NEAR(a55CyclesPerIter("ldp x1, x2, [x1]\nldp x1, x2, [x1]"), 8.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ldp x1, x2, [x2]\nldp x1, x2, [x2]"), 6.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ldnp x1, x2, [x1]\nldnp x1, x2, [x1]"), 8.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ldp w1, w2, [x1]\nldp w1, w2, [x1]"), 4.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ldp w1, w2, [x2]\nldp w1, w2, [x2]"), 6.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ldnp w1, w2, [x2]\nldnp w1, w2, [x2]"), 6.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ldpsw x1, x2, [x1]\nldpsw x1, x2, [x1]"), 6.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ldpsw x1, x2, [x2]\nldpsw x1, x2, [x2]"), 6.0, kTol);
    // LDAXP "3(2)" (SOG 4.8 note 1) is not in Table 3: both registers keep the
    // fast path, as before (docs U-15): 2 per step.
    EXPECT_NEAR(a55CyclesPerIter("ldaxp x1, x2, [x1]\nldaxp x1, x2, [x1]"), 4.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ldaxp x1, x2, [x2]\nldaxp x1, x2, [x2]"), 4.0, kTol);
    // Single-register loads keep it (Table 3 "LDR & LDRT/LDTR (all variants)").
    EXPECT_NEAR(a55CyclesPerIter("ldr x1, [x1]\nldr x1, [x1]"), 4.0, kTol);
    // The ALU load-use latency 2 (SOG 3.3 l.484-485, C-6) is unchanged for LDPSW
    // and both LDP registers: c0 ldpsw; c2 add (3 - 1); c3 ldpsw (ALU -> base 1).
    EXPECT_NEAR(a55CyclesPerIter("ldpsw x1, x2, [x3]\nadd x3, x2, x0"), 3.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ldp x1, x2, [x3]\nadd x3, x1, x0"), 4.0, kTol);  // 4 - 1 + 1
    // LDPSW keeps its row: 3, 1, 00.
    A55InstInfo Ldpsw = a55Desc("ldpsw x1, x2, [x0]");
    ASSERT_EQ(Ldpsw.WriteLatencies.size(), 2u);
    EXPECT_EQ(Ldpsw.WriteLatencies[0], 3u);
    EXPECT_EQ(Ldpsw.WriteLatencies[1], 3u);
    EXPECT_EQ(Ldpsw.ResourceCycles["CortexA55UnitLd"], 1u);
    EXPECT_TRUE(Ldpsw.BeginGroup && Ldpsw.EndGroup);
    A55InstInfo LdpswPost = a55Desc("ldpsw x1, x2, [x0], #8");
    ASSERT_EQ(LdpswPost.WriteLatencies.size(), 3u);
    EXPECT_EQ(LdpswPost.WriteLatencies[0], 1u);
    EXPECT_EQ(LdpswPost.WriteLatencies[1], 3u);
}

// ---------------------------------------------------------------------------
// [a55_sog_n3] Pointer chasing into the base of stores and of the ASIMD
// structure loads/stores (docs/a55-sog.md C-7).  SOG 3.3 (p.11,
// l.486-489): "The first stage (ex1) of both the load and store pipeline
// contains an AGU ... load data ... can be forwarded from the beginning of the
// wr pipeline stage to either the load or store AGU base operand."
// "ldr x1, [x0]; <store> [x1]": c0 ldr; c2 store (3 - 1) + next ldr -> 2
// (before: the store base waited 3 -> 3).  The store data registers are
// independent here (store data: StoreDataNeededLater).
// ---------------------------------------------------------------------------
TEST(A55SogTest, StoreAndAsimdBasePointerChasing) {
    for (const char *St : {"str x2, [x1]", "str x2, [x1, #8]", "stur x2, [x1, #-8]", "sttr x2, [x1]",
                           "strb w2, [x1]", "strh w2, [x1, #2]", "str x2, [x1], #8", "str x2, [x1, #8]!",
                           "str d0, [x1]", "str q0, [x1, #16]", "stur s0, [x1, #-4]", "str b0, [x1], #1",
                           "stp x2, x3, [x1]", "stnp x2, x3, [x1]", "stp w2, w3, [x1, #8]!",
                           "stp d0, d1, [x1], #16", "stp q0, q1, [x1]", "stp q0, q1, [x1], #32",
                           "stlr x2, [x1]", "stlrb w2, [x1]", "stlxr w5, x2, [x1]", "stlxp w5, x2, x3, [x1]",
                           "str x2, [x1, x4]",  // register offset: had it already (stock Sched list)
                           "st1 {v0.8b}, [x1]", "st1 {v0.16b}, [x1], #16", "st1 {v0.16b, v1.16b}, [x1]",
                           "st1 {v0.b}[3], [x1]", "st1 {v0.d}[1], [x1], #8", "st2 {v0.8b, v1.8b}, [x1]",
                           "st2 {v0.8b, v1.8b}, [x1], x4", "st4 {v0.s, v1.s, v2.s, v3.s}[1], [x1]"}) {
        EXPECT_NEAR(a55CyclesPerIter(std::string("ldr x1, [x0]\n") + St), 2.0, kTol) << St;
    }
    // ASIMD loads: c0 ldr x1; c2 ld1 (3 - 1); c3 next ldr (one load per cycle)
    // -> 3 (before: c3 ld1, c4 ldr -> 4).
    for (const char *Ld : {"ld1 {v0.8b}, [x1]", "ld1 {v0.s}[1], [x1]", "ld1r {v0.4s}, [x1]",
                           "ld2 {v0.s, v1.s}[1], [x1]"}) {
        EXPECT_NEAR(a55CyclesPerIter(std::string("ldr x1, [x1]\n") + Ld), 3.0, kTol) << Ld;
    }
    // LD4R B "4, 1/2": the load pipe is held 2 cycles: c2-c3 ld4r, c4 ldr -> 4
    // (before: c3 ld4r, c5 ldr -> 5).
    EXPECT_NEAR(a55CyclesPerIter("ldr x1, [x1]\nld4r {v0.8b, v1.8b, v2.8b, v3.8b}, [x1]"), 4.0, kTol);
    // Only from the loads of Table 3: an LDP X first register (4, no fast path)
    // feeds a store base at 4: c0 ldp (01); c4 str; the next ldp cannot be
    // instruction-1 (01): c5 -> 5.  The second register: c3 str, c4 ldp -> 4.
    EXPECT_NEAR(a55CyclesPerIter("ldp x1, x2, [x0]\nstr x3, [x1]"), 5.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ldp x2, x1, [x0]\nstr x3, [x1]"), 4.0, kTol);  // second register: 3
}

// ---------------------------------------------------------------------------
// [a55_sog_u9] Store data is needed one cycle later than an ordinary operand
// (ReadST, advance 1 from every producer) for every store, not only the
// register-offset ones (docs/a55-sog.md U-9 decision).  SOG 4.9
// (p.22, l.1108-1109) data "required at wr"; 3.1.1 (l.351-352) ALU0 ex2 ->
// store ex2; 4.14 (p.28, l.1521-1522) FP store data "required at f2"; 4.19
// (l.2193) ASIMD stores "do not need to wait for data operands"; 5.7
// (l.2453-2456) the two kinds of store behave alike.
// ---------------------------------------------------------------------------
TEST(A55SogTest, StoreDataNeededLater) {
    // 64-bit MUL (4, MAC 2 cycles): c0 mul; c3 str (4 - 1) + next mul -> 3
    // (before: c4 -> 4), the same as the register-offset store.
    EXPECT_NEAR(a55CyclesPerIter("mul x1, x2, x3\nstr x1, [x0]"), 3.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("mul x1, x2, x3\nstr x1, [x0, x4]"), 3.0, kTol);
    // Load -> store data: c0 ldr; c2 str (3 - 1) + ldr; c4 str + next ldr -> 4
    // (before: 6), the same as the register-offset version.
    EXPECT_NEAR(a55CyclesPerIter("ldr x1, [x0]\nstr x1, [x5]\nldr x2, [x0, #8]\nstr x2, [x5, #8]"), 4.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ldr x1, [x0]\nstr x1, [x5, x6]\nldr x2, [x0, #8]\nstr x2, [x5, x7]"), 4.0, kTol);
    // c0 ldr; c2 store + next ldr -> 2 (before: 3) for each store family.
    for (const char *St : {"stur x1, [x5, #-8]", "str x1, [x5], #8", "str w1, [x5, #4]!", "sttr x1, [x5]",
                           "strb w1, [x5]", "stp x1, x2, [x5]", "stp x2, x1, [x5]", "stnp x1, x2, [x5]",
                           "stp x1, x2, [x5, #16]!", "stlr x1, [x5]", "stlxr w6, x1, [x5]",
                           "stlxp w6, x2, x1, [x5]"}) {
        EXPECT_NEAR(a55CyclesPerIter(std::string("ldr x1, [x0]\n") + St), 2.0, kTol) << St;
    }
    // FP: c0 fadd (4); c3 str + next fadd -> 3 (before: 4).
    for (const char *St : {"str d0, [x5]", "stur d0, [x5, #-8]", "str d0, [x5], #8", "stp d0, d1, [x5]",
                           "stp d1, d0, [x5, #16]!"}) {
        EXPECT_NEAR(a55CyclesPerIter(std::string("fadd d0, d2, d2\n") + St), 3.0, kTol) << St;
    }
    // ASIMD: ADD D-form (2): c0 add; c1 st1 + next add -> 1 (before: 2).
    for (const char *St : {"st1 {v0.8b}, [x5]", "st1 {v0.8b}, [x5], #8", "st1 {v0.b}[3], [x5]",
                           "st1 {v0.b}[3], [x5], #1", "st2 {v0.8b, v1.8b}, [x5]", "st2 {v31.8b, v0.8b}, [x5]"}) {
        EXPECT_NEAR(a55CyclesPerIter(std::string("add v0.8b, v2.8b, v2.8b\n") + St), 1.0, kTol) << St;
    }
    // ASIMD load -> ASIMD store: c0 ld1 (3); c2 st1 + next ld1 -> 2 (before: 3).
    EXPECT_NEAR(a55CyclesPerIter("ld1 {v0.8b}, [x0]\nst1 {v0.8b}, [x5]"), 2.0, kTol);
    // ALU -> store data in the same cycle (ALU0 ex2 -> store ex2, SOG 3.1.1):
    // c0 add + str; c1 add + str ... -> the adds' chain, 1 per pair.
    EXPECT_NEAR(a55CyclesPerIter("add x1, x1, #1\nstr x1, [x0]"), 1.0, kTol);
    // The base is not store data: an ALU result there is seen at 1.
    EXPECT_NEAR(a55CyclesPerIter("add x1, x1, #8\nstr x2, [x1]"), 1.0, kTol);
}

// ---------------------------------------------------------------------------
// ALU -> AGU: an address operand sees an ALU result at the ALU's exec latency (SOG
// 3.1.1 gives no number for the AGU; the 2-cycle reading was tried and dropped,
// docs/a55-sog.md U-3).
// ---------------------------------------------------------------------------
TEST(A55SogTest, AluToAguIsOneCycle) {
    // ALU -> base 1: c0 add; c1 ldr + add; c2 ldr + add ... -> 1.
    EXPECT_NEAR(a55CyclesPerIter("add x1, x1, #8\nldr x2, [x1]"), 1.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("add x1, x1, #8\nstr x2, [x1]"), 1.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("add x4, x4, #8\nldr x2, [x1, x4]"), 1.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("add x1, x1, x3, lsl #3\nldr x2, [x1]"), 2.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("ldpsw x1, x2, [x3]\nadd x3, x2, x0"), 3.0, kTol);
}

// ---------------------------------------------------------------------------
// C-5: ADRP -> LDR pointer forwarding (SOG 5.10, p.44): "adrp x0, <const>" followed by
// a load from "[x0, #lo12 <const>]" has no dependency stall.  Applied to the base of
// unsigned-immediate-offset loads only (docs/a55-sog.md C-5).
// ---------------------------------------------------------------------------
TEST(A55SogTest, AdrpPointerForwardingBaseOfUnsignedImmediateLoads) {
    // With the loop branch as third instruction (as in the analyzer's loops) the pair adrp +
    // load shows the stall: waived, adrp + ldr issue in one cycle and the branch in the next
    // (1.5 per instruction triple); not waived, the load waits one cycle (2.0).
    // The SOG's form, and the other LDR*ui widths:
    for (const char *Ld : {"ldr x2, [x1, #8]", "ldrsw x2, [x1, #8]", "ldrb w2, [x1, #8]",
                           "ldrh w2, [x1, #8]", "ldrsb x2, [x1, #8]", "ldr d0, [x1, #8]", "ldr s0, [x1, #4]"}) {
        EXPECT_NEAR(a55CyclesPerIter(std::string("adrp x1, 0\n") + Ld + "\nb #-8"), 1.5, kTol) << Ld;
    }
    // Not shown in the SOG: LDUR, pre/post-index, register offset (as base and as index
    // register).  The ADRP result is waited for (latency 1).
    for (const char *Ld : {"ldur x2, [x1, #1]", "ldr x2, [x1, #8]!", "ldr x2, [x1], #8",
                           "ldr x2, [x3, x1]", "ldr x2, [x1, x3, lsl #3]"}) {
        EXPECT_NEAR(a55CyclesPerIter(std::string("adrp x1, 0\n") + Ld + "\nb #-8"), 2.0, kTol) << Ld;
    }
}

// U-13 [a55_pac_hint]: the HINT-space pointer-authentication instructions are NOPs on the A55
// (Armv8.2-A, no FEAT_PAuth): no resource, no group bits, they pair like a NOP.
TEST(A55SogTest, PointerAuthenticationHints) {
    for (const char *Asm : {"paciasp", "autiasp", "paciaz", "autiaz", "pacia1716", "autib1716",
                            "hint #25", "hint #29", "xpaclri"}) {
        A55InstInfo I = a55Desc(Asm);
        ASSERT_TRUE(I.Valid) << Asm;
        EXPECT_TRUE(I.ResourceCycles.empty()) << Asm;
        EXPECT_FALSE(I.BeginGroup) << Asm;
        EXPECT_FALSE(I.EndGroup) << Asm;
    }
    // They still read and write LR with latency 1 (LLVM's implicit operands): a chain of 1 each.
    EXPECT_NEAR(a55CyclesPerIter("paciasp\nautiasp\npaciasp\nautiasp"), 4.0, kTol);
    EXPECT_NEAR(a55CyclesPerIter("paciasp\nadd x1, x1, #1"), 1.0, kTol);
    // The other hints are unchanged: BTI stays WriteHint on the branch unit.
    EXPECT_FALSE(a55Desc("bti c").ResourceCycles.empty());
}
