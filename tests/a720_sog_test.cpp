// Cortex-A720: values pinned to the Arm Cortex-A720 Core Software Optimization Guide
// (document 109720, Issue 7.0, "SOG" below; text copy in tmp/a720sog.txt, "l.N" = line N).
// The items (C-n, D-n, ...) are those of docs/a720-sog.md.
//
// Two kinds of checks:
//  * a720Desc(): the per-instruction data InstrBuilder takes from
//    ModifiedTarget/AArch64/AArch64SchedA720.td (latency per def, uops, resource cycles);
//  * a720Facile(): the Facile prediction (src/facile/facile.cpp) of a loop body as mca_tool
//    --dependency ooo --facile computes it for cortex-a720 (uop width 10, Mop width 5, the A720
//    fusion table and zero-latency instructions from cpu_traits.cpp, store -> load memory edges
//    on), with the expected value worked out by hand from the SOG in each comment.
//
// AArch64TestContext("cortex-a720") already installs the local model (tests/test_support.h).

#include "test_support.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct A720InstInfo {
    std::string Opcode;
    unsigned MaxLatency = 0;
    unsigned NumMicroOps = 0;
    std::vector<unsigned> WriteLatencies;  // per def, in operand order (implicit defs last)
    // Total resource cycles per resource name, as the scheduling tables list them (a use of a
    // pipeline also appears under every group that contains it, e.g. M0 under M and I).
    std::map<std::string, unsigned> ResourceCycles;
    // ReadAdvance entries of the sched class: "use index:cycles".
    std::vector<std::string> ReadAdvances;
    bool Valid = false;
};

A720InstInfo a720Desc(const std::string &AsmLine) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a720");
    auto Instrs = parseAsm(TC, AsmLine);
    A720InstInfo Info;
    if (Instrs.size() != 1)
        return Info;
    Info.Opcode = TC.MCII->getName(Instrs[0].Inst.getOpcode()).str();
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
        for (const MCReadAdvanceEntry &RA : TC.STI->getReadAdvanceEntries(*SCD))
            Info.ReadAdvances.push_back(std::to_string(RA.UseIdx) + ":" + std::to_string(RA.Cycles) +
                                        "(w" + std::to_string(RA.WriteResourceID) + ")");
    }
    Info.Valid = true;
    return Info;
}

// Facile prediction for the loop body `Body` (one instruction per line, no loop branch), with the
// widths, fusion and memory edges of `mca_tool --mcpu cortex-a720 --dependency ooo --facile`.
facile::FacileResult a720Facile(const std::string &Body) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a720");
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
    const CpuTraits &T = getCpuTraits("cortex-a720");
    return facile::computeFacilePrediction(*TC.STI, *TC.MCII, *TC.MRI, SimInstrs, MCInsts,
                                           T.UopDispatchWidth, MemInfos, T.MopDispatchWidth);
}

double a720Cycles(const std::string &Body) { return a720Facile(Body).EstimatedCycles; }

constexpr double kTol = 0.01;

} // namespace

// ---------------------------------------------------------------------------
// Not a check: prints the model's data for the instructions in $A720_DUMP (separated by ';'), for
// the reproduction of docs/a720-sog.md (section 10).  Passes when the variable is unset.
// ---------------------------------------------------------------------------
TEST(A720SogTest, Dump) {
    const char *Env = std::getenv("A720_DUMP");
    if (!Env)
        return;
    std::stringstream SS(Env);
    std::string Line;
    while (std::getline(SS, Line, ';')) {
        if (Line.empty())
            continue;
        A720InstInfo I = a720Desc(Line);
        std::cout << Line << " | " << I.Opcode << " | lat " << I.MaxLatency << " [";
        for (unsigned L : I.WriteLatencies)
            std::cout << " " << L;
        std::cout << " ] uops " << I.NumMicroOps << " |";
        for (const auto &[N, C] : I.ResourceCycles)
            std::cout << " " << N << ":" << C;
        std::cout << " | RA";
        for (const auto &R : I.ReadAdvances)
            std::cout << " " << R;
        std::cout << (I.Valid ? "" : " INVALID") << "\n";
    }
}

// ---------------------------------------------------------------------------
// C-1, C-2, C-3 Dispatch: 5 Mops and 10 uOPs per cycle (sec. 4.1, l.2828); thirteen issue
// pipelines (sec. 2.1, l.474-477), four of them integer (S0, S1, M0, M1).  U-1: no ROB size in the
// SOG.
// ---------------------------------------------------------------------------
TEST(A720SogTest, DispatchAndPipelines) {
    const CpuTraits &T = getCpuTraits("cortex-a720");
    EXPECT_EQ(T.UopDispatchWidth, 10u);
    EXPECT_EQ(T.MopDispatchWidth, 5u);
    EXPECT_TRUE(T.ZeroLatencyMov);            // sec. 4.12 (C-4)
    EXPECT_EQ(T.Fusion, FusionTable::A720);   // sec. 4.11 (C-5)
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a720");
    EXPECT_EQ(TC.STI->getSchedModel().MicroOpBufferSize, 192u);  // U-1 (not a SOG value)
    // Eight independent ADDs on the four I pipelines: 2 (the Mop bound is 8/5).
    auto R = a720Facile("add x1, x1, #1\nadd x2, x2, #1\nadd x3, x3, #1\nadd x4, x4, #1\n"
                        "add x5, x5, #1\nadd x6, x6, #1\nadd x7, x7, #1\nadd x8, x8, #1");
    EXPECT_NEAR(R.IssueBound, 8.0 / 5.0, kTol);
    EXPECT_NEAR(R.EstimatedCycles, 2.0, kTol);
    // "ALU, basic, flagset ... 1 4 I" (l.639-641): four per cycle, unlike the Cortex-A78's three.
    EXPECT_NEAR(a720Cycles("adds x1, x1, #1\nadds x2, x2, #1\nadds x3, x3, #1\nadds x4, x4, #1\n"
                           "adds x5, x5, #1\nadds x6, x6, #1\nadds x7, x7, #1\nadds x8, x8, #1"),
                2.0, kTol);
    // NOP and BTI: no pipeline (C-3); a NOP still takes a Mop: six instructions / 5 = 1.2.
    EXPECT_TRUE(a720Desc("nop").ResourceCycles.empty());
    EXPECT_TRUE(a720Desc("hint #34").ResourceCycles.empty());
    EXPECT_NEAR(a720Cycles("nop\nnop\nnop\nnop\nnop\nadd x1, x1, #1"), 1.2, kTol);
    // Integer store data has pipelines of its own (ID): two STR and four ADD in one cycle... per
    // pipeline; the Mop bound 6/5 decides.
    EXPECT_NEAR(a720Cycles("str x1, [x0]\nstr x2, [x0, #8]\nadd x3, x3, #1\nadd x4, x4, #1\n"
                           "add x5, x5, #1\nadd x6, x6, #1"),
                1.2, kTol);
}

// ---------------------------------------------------------------------------
// C-4 Zero-latency instructions (sec. 4.12, l.3183-3227); D-9 (MOVZ #imm[3:0]).
// ---------------------------------------------------------------------------
TEST(A720SogTest, ZeroLatencyInstructions) {
    for (const char *Asm : {"mov x1, #0", "mov w1, #15", "mov x1, xzr", "mov w1, wzr", "mov x1, x2",
                            "mov w1, w2", "fmov h0, wzr", "fmov h0, xzr", "fmov s0, wzr",
                            "fmov d0, xzr", "movi d0, #0", "movi v0.2d, #0", "fmov s0, s1",
                            "fmov d0, d1", "mov v0.16b, v1.16b", "mov v0.8b, v1.8b"}) {
        A720InstInfo I = a720Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 0u) << Asm;
        EXPECT_TRUE(I.ResourceCycles.empty()) << Asm;
    }
    // Not in the list: MOV #16, a shifted MOVZ, FMOV H (FP move, register 2 2 V, l.1040),
    // MOVI V.16B #0 (ASIMD move, integer immed 2 2 V, l.1546).
    EXPECT_EQ(a720Desc("mov x1, #16").MaxLatency, 1u);
    EXPECT_EQ(a720Desc("movz x1, #1, lsl #16").MaxLatency, 1u);
    EXPECT_EQ(a720Desc("fmov h0, h1").MaxLatency, 2u);
    EXPECT_EQ(a720Desc("movi v0.16b, #0").MaxLatency, 2u);
    // A chain through a register move costs nothing: add (1) + mov (0).
    EXPECT_NEAR(a720Cycles("add x1, x2, #1\nmov x2, x1"), 1.0, kTol);
}

// ---------------------------------------------------------------------------
// C-5 Instruction fusion (sec. 4.11, l.3149-3171): the second instruction of a pair takes no Mop,
// uOP or pipeline (D-11).  The table-vs-A78 tests are in facile_test.cpp.
// ---------------------------------------------------------------------------
TEST(A720SogTest, Fusion) {
    EXPECT_EQ(a720Facile("cmp x1, #3\nb.eq #8").TotalMicroOps, 1u);
    EXPECT_EQ(a720Facile("cmp x1, x2, lsl #2\nb.eq #8").TotalMicroOps, 1u);  // shifted forms (l.3169)
    EXPECT_EQ(a720Facile("cmp x1, x2\ncsel x3, x4, x5, eq").TotalMicroOps, 1u);
    EXPECT_EQ(a720Facile("tst x1, x2\nb.ne #8").TotalMicroOps, 1u);
    EXPECT_EQ(a720Facile("bics xzr, x1, x2\nb.eq #8").TotalMicroOps, 1u);
    // SHL + SRI, both vector or both scalar; FCMP + AXFLAG (l.3165-3166, [a720_fusion_simd]).
    EXPECT_EQ(a720Facile("shl v0.4s, v1.4s, #3\nsri v2.4s, v0.4s, #29").TotalMicroOps, 1u);
    EXPECT_EQ(a720Facile("shl d0, d1, #3\nsri d2, d0, #61").TotalMicroOps, 1u);
    EXPECT_EQ(a720Facile("shl d0, d1, #3\nsri v2.2d, v0.2d, #61").TotalMicroOps, 2u);
    EXPECT_EQ(a720Facile("fcmp d0, d1\naxflag").TotalMicroOps, 1u);
    // Not listed: NOP + any, CMN + CSEL, AESE + AESMC (D-11, U-6).
    EXPECT_EQ(a720Facile("nop\nadd x1, x2, x3").TotalMicroOps, 2u);
    EXPECT_EQ(a720Facile("cmn x1, #3\ncsel x3, x4, x5, eq").TotalMicroOps, 2u);
    EXPECT_EQ(a720Facile(".arch_extension aes\naese v0.16b, v1.16b\naesmc v0.16b, v0.16b").TotalMicroOps,
              2u);
}

// ---------------------------------------------------------------------------
// C-8 Branches (sec. 3.3, l.609-618): two branch pipelines; BL/BLR B, S.
// ---------------------------------------------------------------------------
TEST(A720SogTest, Branches) {
    EXPECT_EQ(a720Desc("b #8").ResourceCycles["A720UnitB"], 1u);
    EXPECT_EQ(a720Desc("ret").ResourceCycles["A720UnitB"], 1u);
    A720InstInfo Bl = a720Desc("bl #8");
    EXPECT_EQ(Bl.ResourceCycles["A720UnitB"], 1u);
    EXPECT_EQ(Bl.ResourceCycles["A720UnitS"], 1u);
    EXPECT_EQ(Bl.MaxLatency, 0u);  // X-2: MCA gives every call latency 0
    EXPECT_NEAR(a720Cycles("cbz x1, #8\ncbz x2, #8\ncbz x3, #8\ncbz x4, #8"), 2.0, kTol);
}

// ---------------------------------------------------------------------------
// C-9 Arithmetic and logical (sec. 3.4, l.631-663).
// ---------------------------------------------------------------------------
TEST(A720SogTest, ArithmeticAndLogical) {
    EXPECT_EQ(a720Desc("add x1, x2, x3, lsl #4").MaxLatency, 1u);
    A720InstInfo Lsl5 = a720Desc("add x1, x2, x3, lsl #5");
    EXPECT_EQ(Lsl5.MaxLatency, 2u);
    EXPECT_EQ(Lsl5.ResourceCycles["A720UnitM"], 1u);
    EXPECT_EQ(a720Desc("adds x1, x2, x3, asr #1").MaxLatency, 2u);
    // ALU, extend and shift: every extended form, also UXTB ([a720_ext]).
    for (const char *Asm : {"add x1, x2, w3, uxtw #2", "add x1, x2, w3, uxtb", "subs w1, w2, w3, sxtb"}) {
        A720InstInfo I = a720Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 2u) << Asm;
        EXPECT_EQ(I.ResourceCycles["A720UnitM"], 1u) << Asm;
    }
    // Logical, shift, flagset ANDS/BICS 2 2 M; no flagset 1 4 I; conditional ops 1 4 I.
    EXPECT_EQ(a720Desc("ands x1, x2, x3, lsl #2").MaxLatency, 2u);
    EXPECT_EQ(a720Desc("eor x1, x2, x3, ror #2").MaxLatency, 1u);
    EXPECT_EQ(a720Desc("ccmp x1, x2, #0, eq").ResourceCycles["A720UnitI"], 1u);
    EXPECT_EQ(a720Desc("axflag").MaxLatency, 1u);
}

// ---------------------------------------------------------------------------
// C-10 Divide and multiply (sec. 3.5, l.679-699); D-1, D-3; U-5.
// ---------------------------------------------------------------------------
TEST(A720SogTest, DivideAndMultiply) {
    for (const char *Asm : {"sdiv w1, w2, w3", "udiv x1, x2, x3"}) {
        A720InstInfo I = a720Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 5u) << Asm;                   // best case of "5 to 12" / "5 to 20"
        EXPECT_EQ(I.ResourceCycles["A720UnitM0"], 1u) << Asm;
        EXPECT_EQ(I.ResourceCycles["A720UnitIDiv"], 5u) << Asm;  // 1/5 per cycle
    }
    EXPECT_NEAR(a720Cycles("sdiv w1, w1, w2"), 5.0, kTol);
    // The divider does not block M0: udiv + two madd on M0 = 5 (divider), not 7.
    EXPECT_NEAR(a720Cycles("udiv x1, x2, x3\nmadd x4, x5, x6, x4\nmadd x7, x5, x6, x7"), 5.0, kTol);
    A720InstInfo Madd = a720Desc("madd x1, x2, x3, x4");
    EXPECT_EQ(Madd.MaxLatency, 2u);
    EXPECT_EQ(Madd.ResourceCycles["A720UnitM0"], 1u);
    EXPECT_NEAR(a720Cycles("madd x1, x2, x3, x1"), 1.0, kTol);      // 2(1)
    EXPECT_NEAR(a720Cycles("smaddl x1, w2, w3, x1"), 1.0, kTol);
    // Note 3: MUL / SMULL with a zero accumulator on M, 2 per cycle.
    A720InstInfo Mul = a720Desc("mul x1, x2, x3");
    EXPECT_EQ(Mul.ResourceCycles["A720UnitM"], 1u);
    EXPECT_EQ(Mul.ResourceCycles["A720UnitM0"], 0u);
    EXPECT_EQ(a720Desc("smulh x1, x2, x3").MaxLatency, 3u);
}

// ---------------------------------------------------------------------------
// C-11 Pointer authentication (sec. 3.6, Table 3-5, l.712-750).
// ---------------------------------------------------------------------------
TEST(A720SogTest, PointerAuthentication) {
    A720InstInfo Pac = a720Desc("paciasp");
    EXPECT_EQ(Pac.MaxLatency, 4u);
    EXPECT_EQ(Pac.ResourceCycles["A720UnitM"], 1u);
    A720InstInfo Aut = a720Desc("autiasp");
    EXPECT_EQ(Aut.MaxLatency, 1u);
    EXPECT_EQ(Aut.ResourceCycles["A720UnitM"], 1u);
    A720InstInfo Reta = a720Desc(".arch_extension pauth\nretaa");
    EXPECT_TRUE(Reta.Valid);
    EXPECT_EQ(Reta.ResourceCycles["A720UnitB"], 1u);
    EXPECT_EQ(Reta.ResourceCycles["A720UnitM"], 1u);
    A720InstInfo Ldraa = a720Desc(".arch_extension pauth\nldraa x1, [x2, #8]!");
    EXPECT_EQ(Ldraa.WriteLatencies, (std::vector<unsigned>{1, 5}));
    EXPECT_EQ(a720Desc(".arch_extension pauth\nxpaci x1").MaxLatency, 1u);
}

// ---------------------------------------------------------------------------
// C-12 Miscellaneous data processing (sec. 3.7, l.763-777).
// ---------------------------------------------------------------------------
TEST(A720SogTest, MiscDataProcessing) {
    A720InstInfo Adr = a720Desc("adrp x1, #4096");
    EXPECT_EQ(Adr.ResourceCycles["A720UnitS"], 1u);
    EXPECT_EQ(a720Desc("extr x1, x2, x3, #3").MaxLatency, 1u);
    EXPECT_EQ(a720Desc("bfi x1, x2, #3, #5").MaxLatency, 1u);
    EXPECT_EQ(a720Desc("rorv x1, x2, x3").MaxLatency, 1u);
    EXPECT_EQ(a720Desc("movk x1, #3").MaxLatency, 1u);
}

// ---------------------------------------------------------------------------
// C-13 Loads (sec. 3.8, l.782-853); D-10.
// ---------------------------------------------------------------------------
TEST(A720SogTest, Loads) {
    A720InstInfo Lit = a720Desc("ldr x1, .+8");
    EXPECT_EQ(Lit.MaxLatency, 5u);
    EXPECT_EQ(Lit.ResourceCycles["A720UnitS"], 1u);
    for (const char *Asm : {"ldur x1, [x2, #3]", "ldr x1, [x2, x3, lsl #3]", "ldrh w1, [x2, x3, lsl #1]",
                            "ldrsw x1, [x2, w3, sxtw #2]"}) {
        A720InstInfo I = a720Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 4u) << Asm;
        EXPECT_EQ(I.ResourceCycles["A720UnitI"], 0u) << Asm;
    }
    EXPECT_EQ(a720Desc("ldr x1, [x2], #8").WriteLatencies, (std::vector<unsigned>{1, 4}));
    A720InstInfo LdpW = a720Desc("ldp w1, w2, [x3]");
    EXPECT_EQ(LdpW.WriteLatencies, (std::vector<unsigned>{4, 4}));
    EXPECT_EQ(LdpW.NumMicroOps, 1u);                          // D-10
    EXPECT_EQ(a720Desc("ldp x1, x2, [x3]").ResourceCycles["A720UnitLS"], 2u);  // 3/2 per cycle
    A720InstInfo Ldpsw = a720Desc("ldpsw x1, x2, [x3]");
    EXPECT_EQ(Ldpsw.ResourceCycles["A720UnitLS"], 2u);
    EXPECT_EQ(Ldpsw.ResourceCycles["A720UnitI"], 1u);
}

// ---------------------------------------------------------------------------
// C-7, C-14 Stores (sec. 3.9, l.866-895): L01 + ID, two uOPs (address and data).
// ---------------------------------------------------------------------------
TEST(A720SogTest, Stores) {
    for (const char *Asm : {"str x1, [x2]", "stur w1, [x2, #3]", "strh w1, [x2, x3, lsl #1]",
                            "stp x1, x2, [x3]"}) {
        A720InstInfo I = a720Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 1u) << Asm;
        EXPECT_EQ(I.NumMicroOps, 2u) << Asm;
        EXPECT_EQ(I.ResourceCycles["A720UnitL01"], 1u) << Asm;
        EXPECT_EQ(I.ResourceCycles["A720UnitSD"], 1u) << Asm;
        EXPECT_EQ(I.ResourceCycles["A720UnitI"], 0u) << Asm;
    }
    A720InstInfo Post = a720Desc("str x1, [x2], #8");
    EXPECT_EQ(Post.ResourceCycles["A720UnitI"], 1u);
    EXPECT_EQ(Post.NumMicroOps, 3u);
    // Two stores per cycle (L01).
    EXPECT_NEAR(a720Cycles("str x1, [x0]\nstr x2, [x0, #8]\nstr x3, [x0, #16]\nstr x4, [x0, #24]"), 2.0,
                kTol);
}

// ---------------------------------------------------------------------------
// C-15, C-16 FP data processing and miscellaneous (sec. 3.12, 3.13, l.959-1048); D-4.
// ---------------------------------------------------------------------------
TEST(A720SogTest, FpScalar) {
    EXPECT_EQ(a720Desc("fadd d0, d1, d2").MaxLatency, 2u);
    A720InstInfo Fcmp = a720Desc("fcmp d0, d1");
    EXPECT_EQ(Fcmp.MaxLatency, 2u);
    EXPECT_EQ(Fcmp.ResourceCycles["A720UnitV"], 1u);
    struct Row { const char *Asm; unsigned Lat; };
    for (const Row &R : {Row{"fdiv h0, h1, h2", 5}, Row{"fdiv s0, s1, s2", 7}, Row{"fdiv d0, d1, d2", 12},
                         Row{"fsqrt d0, d1", 12}}) {
        A720InstInfo I = a720Desc(R.Asm);
        EXPECT_EQ(I.MaxLatency, R.Lat) << R.Asm;
        EXPECT_EQ(I.ResourceCycles["A720UnitV0"], 1u) << R.Asm;  // "fully pipelined", 1 per cycle
    }
    EXPECT_NEAR(a720Cycles("fmadd d0, d1, d2, d0"), 2.0, kTol);                // 4(2)
    EXPECT_NEAR(a720Cycles("fmul d3, d1, d2\nfmadd d0, d4, d5, d3\nfmov d1, d0"), 5.0, kTol);  // 1+4+0
    EXPECT_EQ(a720Desc("frinta d0, d1").ResourceCycles["A720UnitV0"], 1u);
    A720InstInfo Scvtf = a720Desc("scvtf d0, x1");
    EXPECT_EQ(Scvtf.MaxLatency, 3u);
    EXPECT_EQ(Scvtf.ResourceCycles["A720UnitM0"], 1u);
    EXPECT_EQ(a720Desc("fcvtzs x1, d0").ResourceCycles["A720UnitV0"], 1u);
    EXPECT_EQ(a720Desc("fcvt s0, d1").MaxLatency, 3u);
    EXPECT_EQ(a720Desc("fmov x1, d0").MaxLatency, 3u);
    EXPECT_EQ(a720Desc("fmov s0, w1").ResourceCycles["A720UnitM0"], 1u);
    EXPECT_EQ(a720Desc("fmov v0.d[1], x1").MaxLatency, 5u);
    // D-4: conversions with the operand and the result in FP registers, also H: 3 1 V0.
    for (const char *Asm : {"scvtf d0, d1", "fcvtzs s0, s1", "scvtf h0, h1", "fcvtzs h0, h1",
                            "ucvtf h0, h1, #3"}) {
        A720InstInfo I = a720Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 3u) << Asm;
        EXPECT_EQ(I.ResourceCycles["A720UnitV0"], 1u) << Asm;
    }
}

// ---------------------------------------------------------------------------
// C-17, C-18 FP loads and stores (sec. 3.14, 3.15, l.1052-1180; I-3).
// ---------------------------------------------------------------------------
TEST(A720SogTest, FpLoadsAndStores) {
    for (const char *Asm : {"ldr q0, [x1]", "ldr d0, [x1, x2, lsl #3]", "ldr q0, [x1, w2, uxtw #4]"}) {
        A720InstInfo I = a720Desc(Asm);
        EXPECT_EQ(I.MaxLatency, 6u) << Asm;
        EXPECT_EQ(I.ResourceCycles["A720UnitI"], 0u) << Asm;
    }
    A720InstInfo LdpQ = a720Desc("ldp q0, q1, [x1]");
    EXPECT_EQ(LdpQ.WriteLatencies, (std::vector<unsigned>{6, 6}));
    EXPECT_EQ(LdpQ.ResourceCycles["A720UnitLS"], 2u);
    EXPECT_EQ(a720Desc("ldp s0, s1, [x1], #8").ResourceCycles["A720UnitLS"], 2u);  // 3/2
    // STR Q register offset: I only when scaled ([a720_strq_ro]).
    EXPECT_EQ(a720Desc("str q0, [x1, x2]").ResourceCycles["A720UnitI"], 0u);
    EXPECT_EQ(a720Desc("str q0, [x1, w2, sxtw]").ResourceCycles["A720UnitI"], 0u);
    EXPECT_EQ(a720Desc("str q0, [x1, x2, lsl #4]").ResourceCycles["A720UnitI"], 1u);
    EXPECT_EQ(a720Desc("str q0, [x1, w2, sxtw #4]").ResourceCycles["A720UnitI"], 1u);
    EXPECT_EQ(a720Desc("str h0, [x1, x2, lsl #1]").ResourceCycles["A720UnitI"], 0u);
    // Pre-index B/H/S/D 3, post-index and Q 2 (I-3).
    EXPECT_EQ(a720Desc("str d0, [x1, #8]!").MaxLatency, 3u);
    EXPECT_EQ(a720Desc("str d0, [x1], #8").MaxLatency, 2u);
    EXPECT_EQ(a720Desc("str q0, [x1, #16]!").MaxLatency, 2u);
    A720InstInfo StpQ = a720Desc("stp q0, q1, [x1]");
    EXPECT_EQ(StpQ.NumMicroOps, 2u);
    EXPECT_EQ(StpQ.ResourceCycles["A720UnitV"], 1u);
    EXPECT_EQ(StpQ.ResourceCycles["A720UnitL01"], 1u);
}

// ---------------------------------------------------------------------------
// C-19 to C-22 ASIMD integer, FP, BF16 and miscellaneous (sec. 3.16-3.19, l.1185-1599); D-5.
// ---------------------------------------------------------------------------
TEST(A720SogTest, Asimd) {
    EXPECT_NEAR(a720Cycles("saba v0.16b, v1.16b, v2.16b"), 1.0, kTol);       // 4(1)
    EXPECT_NEAR(a720Cycles("mla v0.4s, v1.4s, v2.4s"), 1.0, kTol);           // 4(1)
    EXPECT_NEAR(a720Cycles("sqrdmlah v0.4s, v1.4s, v2.4s"), 2.0, kTol);      // 4(2)
    EXPECT_NEAR(a720Cycles("sdot v0.4s, v1.16b, v2.16b"), 1.0, kTol);        // 3(1)
    EXPECT_NEAR(a720Cycles("fmla v0.4s, v1.4s, v2.4s"), 2.0, kTol);          // 4(2)
    EXPECT_NEAR(a720Cycles("bfmmla v0.4s, v1.8h, v2.8h"), 3.0, kTol);        // 5(3)
    EXPECT_EQ(a720Desc("addv b0, v1.16b").ResourceCycles["A720UnitV1"], 2u);  // 6 1/2 V1
    EXPECT_EQ(a720Desc("addv h0, v1.8h").MaxLatency, 5u);
    EXPECT_EQ(a720Desc("pmull v0.8h, v1.8b, v2.8b").MaxLatency, 2u);
    EXPECT_EQ(a720Desc("smull v0.4s, v1.4h, v2.4h").ResourceCycles["A720UnitV0"], 1u);
    EXPECT_EQ(a720Desc("sqshl v0.4s, v1.4s, #3").MaxLatency, 4u);
    // D-5: scalar forms take the row of their mnemonic.
    EXPECT_EQ(a720Desc("shl d0, d1, #3").ResourceCycles["A720UnitV1"], 1u);
    EXPECT_EQ(a720Desc("sqshl d0, d1, #3").MaxLatency, 4u);
    EXPECT_NEAR(a720Cycles("ssra d0, d1, #3"), 1.0, kTol);                   // 4(1)
    EXPECT_EQ(a720Desc("sqdmlal s0, h1, h2").MaxLatency, 4u);
    EXPECT_EQ(a720Desc("fmulx d0, d1, d2").MaxLatency, 3u);
    // ASIMD FP.
    EXPECT_EQ(a720Desc("fcvtl v0.4s, v1.4h").ResourceCycles["A720UnitV0"], 2u);   // 4 1/2
    EXPECT_EQ(a720Desc("fcvtzs v0.8h, v1.8h").ResourceCycles["A720UnitV0"], 4u);  // 6 1/4
    EXPECT_EQ(a720Desc("fdiv v0.4s, v1.4s, v2.4s").MaxLatency, 10u);
    EXPECT_EQ(a720Desc("fdiv v0.4s, v1.4s, v2.4s").ResourceCycles["A720UnitV0"], 4u);  // I-2
    EXPECT_EQ(a720Desc("faddp v0.4s, v1.4s, v2.4s").MaxLatency, 3u);
    EXPECT_EQ(a720Desc("fminnmv h0, v1.8h").ResourceCycles["A720UnitV"], 3u);  // 6 2/3
    // Miscellaneous.
    A720InstInfo Dup = a720Desc("dup v0.4s, w1");
    EXPECT_EQ(Dup.MaxLatency, 3u);
    EXPECT_EQ(Dup.ResourceCycles["A720UnitM0"], 1u);
    EXPECT_EQ(a720Desc("umov w1, v0.s[1]").ResourceCycles["A720UnitV"], 2u);   // 2 1 V
    EXPECT_EQ(a720Desc("ins v0.s[1], w1").MaxLatency, 5u);
    EXPECT_EQ(a720Desc("tbx v0.16b, {v1.16b, v2.16b, v3.16b, v4.16b}, v5.16b").ResourceCycles["A720UnitV"],
              4u);  // 6 1/2
    // FRECPE scalar forms 3 1 V0 ([a720_frecpe_scalar]), Q-form F16 6 1/4.
    EXPECT_EQ(a720Desc("frecpe h0, h1").MaxLatency, 3u);
    EXPECT_EQ(a720Desc("frecpe v0.8h, v1.8h").ResourceCycles["A720UnitV0"], 4u);
    EXPECT_EQ(a720Desc("frecps v0.4s, v1.4s, v2.4s").MaxLatency, 4u);
}

// ---------------------------------------------------------------------------
// C-23, C-24 ASIMD loads and stores (sec. 3.20, 3.21, l.1612-1816); D-6.
// ---------------------------------------------------------------------------
TEST(A720SogTest, AsimdLoadsAndStores) {
    struct Row { const char *Asm; unsigned Lat; unsigned LS; };
    for (const Row &R : {Row{"ld1 {v0.16b}, [x0]", 6, 1}, Row{"ld1 {v0.16b, v1.16b, v2.16b}, [x0]", 6, 3},
                         Row{"ld1 {v0.8b, v1.8b, v2.8b, v3.8b}, [x0]", 7, 4},
                         Row{"ld1 {v0.s}[1], [x0]", 8, 1}, Row{"ld2 {v0.4s, v1.4s}, [x0]", 8, 2},
                         Row{"ld3 {v0.4s, v1.4s, v2.4s}, [x0]", 10, 1},
                         Row{"ld4 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0]", 8, 6}}) {
        A720InstInfo I = a720Desc(R.Asm);
        EXPECT_EQ(I.MaxLatency, R.Lat) << R.Asm;
        EXPECT_EQ(I.ResourceCycles["A720UnitLS"], R.LS) << R.Asm;
    }
    EXPECT_EQ(a720Desc("ld1 {v0.16b}, [x0], #16").WriteLatencies, (std::vector<unsigned>{1, 6}));
    struct SRow { const char *Asm; unsigned Lat; unsigned L01; };
    for (const SRow &R : {SRow{"st1 {v0.16b}, [x0]", 2, 1}, SRow{"st1 {v0.16b, v1.16b, v2.16b}, [x0]", 2, 2},
                          SRow{"st3 {v0.8b, v1.8b, v2.8b}, [x0]", 4, 2},
                          SRow{"st3 {v0.4s, v1.4s, v2.4s}, [x0]", 4, 3},
                          SRow{"st4 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0]", 4, 4},
                          SRow{"st4 {v0.2d, v1.2d, v2.2d, v3.2d}, [x0]", 2, 2}}) {
        A720InstInfo I = a720Desc(R.Asm);
        EXPECT_EQ(I.MaxLatency, R.Lat) << R.Asm;
        EXPECT_EQ(I.ResourceCycles["A720UnitL01"], R.L01) << R.Asm;
        EXPECT_EQ(I.NumMicroOps, 2u) << R.Asm;
    }
}

// ---------------------------------------------------------------------------
// C-25, C-26 Cryptography and CRC (sec. 3.22, 3.23, l.1820-1878).
// ---------------------------------------------------------------------------
TEST(A720SogTest, CryptoAndCrc) {
    const std::string Ext = ".arch_extension aes\n.arch_extension sha2\n.arch_extension sha3\n"
                            ".arch_extension sm4\n";
    A720InstInfo Aes = a720Desc(Ext + "aese v0.16b, v1.16b");
    EXPECT_EQ(Aes.MaxLatency, 2u);
    EXPECT_EQ(Aes.ResourceCycles["A720UnitV"], 1u);
    EXPECT_EQ(a720Desc(Ext + "pmull v0.1q, v1.1d, v2.1d").ResourceCycles["A720UnitV0"], 1u);
    EXPECT_EQ(a720Desc(Ext + "sha256h q0, q1, v2.4s").MaxLatency, 4u);
    EXPECT_EQ(a720Desc(Ext + "sha512h q0, q1, v2.2d").MaxLatency, 2u);
    EXPECT_EQ(a720Desc(Ext + "eor3 v0.16b, v1.16b, v2.16b, v3.16b").ResourceCycles["A720UnitV"], 1u);
    EXPECT_EQ(a720Desc(Ext + "sm4e v0.4s, v1.4s").MaxLatency, 4u);
    // CRC 2 1 M0, the consumer sees 1 (note 1, [a720_crc]).
    A720InstInfo Crc = a720Desc("crc32x w1, w2, x3");
    EXPECT_EQ(Crc.MaxLatency, 2u);
    EXPECT_EQ(Crc.ResourceCycles["A720UnitM0"], 1u);
    EXPECT_NEAR(a720Cycles("crc32x w1, w1, x3"), 1.0, kTol);
    EXPECT_NEAR(a720Cycles("crc32x w1, w1, x3\ncrc32x w2, w2, x3"), 2.0, kTol);  // M0: 1 per cycle
}

// ---------------------------------------------------------------------------
// D-12 System instructions: latency 1, no pipeline (no SOG row).
// ---------------------------------------------------------------------------
TEST(A720SogTest, SystemInstructions) {
    A720InstInfo Mrs = a720Desc("mrs x1, tpidr_el0");
    EXPECT_EQ(Mrs.MaxLatency, 1u);
    EXPECT_TRUE(Mrs.ResourceCycles.empty());
    EXPECT_TRUE(a720Desc("dmb ish").ResourceCycles.empty());
}

// ---------------------------------------------------------------------------
// B-1 Store data read one cycle late (NOT SOG-based; the A55/A520/A76/A78 decision).
// ---------------------------------------------------------------------------
TEST(A720SogTest, BorrowedStoreDataNeededLater) {
    // ldr -> str data: 4 - 1 (advance); str -> ldr memory edge: the store's latency 1 (D-7): 3 + 1.
    // FP: 6 - 1 + 2.
    EXPECT_NEAR(a720Cycles("ldr x1, [x0]\nstr x1, [x0]"), 4.0, kTol);
    EXPECT_NEAR(a720Cycles("ldr q1, [x0]\nstr q1, [x0]"), 7.0, kTol);
    // ST1-ST4 get no store -> load memory edge in Facile (shared code, docs/a720-sog.md X-3), so the
    // ReadST of their Vt shows only in the model data:
    EXPECT_EQ(a720Desc("st1 {v1.16b}, [x0]").ReadAdvances.front(), "0:1(w0)");
}
