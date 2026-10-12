#include "test_support.h"

TEST(MLPTest, DependencyNone) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC, "mov %eax, %ebx\nmovq (%rsi), %rax\nmovq (%rdi), %rbx");
    ASSERT_FALSE(instrs.empty());
    float ratio = 0.0f;
    X86MLPAnalyzer analyzer;
    float val1 = analyzer.compute_mlp(instrs, 2, DependencyKind::None, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio);
    EXPECT_NEAR(val1, 1.3333, 0.01);
    EXPECT_NEAR(ratio, 1.0, 0.01);
}


TEST(MLPTest, DependencyOOO) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC, "movq (%rdi), %rax\nmovq (%rax), %rbx");
    ASSERT_FALSE(instrs.empty());
    float ratio = 0.0f;
    X86MLPAnalyzer analyzer;
    float val1 = analyzer.compute_mlp(instrs, 2, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio);
    EXPECT_NEAR(val1, 1.0, 0.01);
    EXPECT_NEAR(ratio, 0.75, 0.01);
}


TEST(MLPTest, IOBarrier) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC,
        "movq (%rdi), %rax\n"
        "movq (%rsi), %rbx\n"
        "addq $1, %rcx\n"
        "addq %rax, %rdx\n"
        "movq (%r8), %r9"
    );
    ASSERT_FALSE(instrs.empty());
    float ratio = 0.0f;
    X86MLPAnalyzer analyzer;
    float val1 = analyzer.compute_mlp(instrs, 5, DependencyKind::IO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio);
    EXPECT_NEAR(val1, 1.5, 0.01);
    EXPECT_NEAR(ratio, 0.8888, 0.01);
}


TEST(MLPTest, MaxContainingNone) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC, "mov %eax, %ebx\nmovq (%rsi), %rax\nmovq (%rdi), %rbx");
    ASSERT_FALSE(instrs.empty());
    float ratio = 0.0f;
    X86MLPAnalyzer analyzer;
    float val1 = analyzer.compute_mlp(instrs, 2, DependencyKind::None, MLPWindowAssignmentKind::MaxContaining, *TC.STI, *TC.MCII, *TC.MRI, ratio);
    EXPECT_NEAR(val1, 2.0, 0.01);
}


TEST(MLPTest, DependencyMode) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC,
        "movq (%rdi), %rax\n"
        "addq $1, %rbx\n"
        "addq %rax, %rcx\n"
        "movq (%rsi), %rdx\n"
        "subq $1, %rdx"
    );
    ASSERT_FALSE(instrs.empty());
    float ratio = 0.0f;
    X86MLPAnalyzer analyzer;
    float val1 = analyzer.compute_mlp(instrs, 2, DependencyKind::Dependency, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio);
    EXPECT_NEAR(val1, 1.3333, 0.01);
}


TEST(MLPTest, WindowLoopNone) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC, "movq (%rsi), %rax\nmovq (%rdi), %rbx");
    ASSERT_FALSE(instrs.empty());
    float ratio = 0.0f;
    X86MLPAnalyzer analyzer;
    float val1 = analyzer.compute_mlp(instrs, 2, DependencyKind::None, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/true);
    EXPECT_NEAR(val1, 2.0, 0.01);
}


TEST(MLPTest, WindowLoopOOO) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC, "movq (%rdi), %rax\nmovq (%rax), %rbx");
    ASSERT_FALSE(instrs.empty());
    float ratio = 0.0f;
    X86MLPAnalyzer analyzer;
    float val1 = analyzer.compute_mlp(instrs, 2, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/true);
    EXPECT_NEAR(val1, 1.3333, 0.01);
    EXPECT_NEAR(ratio, 0.75, 0.01);
}


TEST(MLPTest, PointerChasingLoopOOO) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC, "movq (%rax), %rax");
    ASSERT_FALSE(instrs.empty());
    float ratio = 0.0f;
    X86MLPAnalyzer analyzer;
    float val1 = analyzer.compute_mlp(instrs, 2, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/true);
    EXPECT_NEAR(val1, 1.0, 0.01);
}


TEST(MLPTest, AArch64PointerChasingX0) {
    initLLVMAArch64();
    AArch64TestContext TC;
    auto instrs = parseAsm(TC, "ldr x0, [x0, #8]");
    ASSERT_FALSE(instrs.empty());
    float ratio = 0.0f;
    AArch64MLPAnalyzer analyzer;
    float val1 = analyzer.compute_mlp(instrs, 2, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/true);
    EXPECT_NEAR(val1, 1.0, 0.01);
}


TEST(MLPTest, AArch64WritebackPostIndex) {
    initLLVMAArch64();
    AArch64TestContext TC;
    auto instrs = parseAsm(TC, "ldr x0, [x1], #8");
    ASSERT_FALSE(instrs.empty());
    float ratio = 0.0f;
    AArch64MLPAnalyzer analyzer;
    float val1 = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/true);
    EXPECT_NEAR(val1, 2.0, 0.01);
}


TEST(MLPTest, AArch64WritebackPostIndexRegister) {
    initLLVMAArch64();
    AArch64TestContext TC;
    auto instrs = parseAsm(TC, "ld1 {v0.16b}, [x1], x2");
    ASSERT_FALSE(instrs.empty());
    float ratio = 0.0f;
    AArch64MLPAnalyzer analyzer;
    float val1 = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/true);
    EXPECT_NEAR(val1, 2.0, 0.01);
}


TEST(MLPTest, X86PushPopStackAccess) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC, "pushq %rax\npopq %rbx");
    ASSERT_FALSE(instrs.empty());
    X86MLPAnalyzer analyzer;
    size_t loads = analyzer.countPotentialMissLoads(instrs, *TC.STI, *TC.MCII, *TC.MRI);
    EXPECT_EQ(loads, 0u);
}


TEST(MLPTest, AArch64MixedDependencyProp) {
    initLLVMAArch64();
    AArch64TestContext TC;
    auto instrs = parseAsm(TC, "ldr x1, [x0, #8]\nldr x2, [x0, #16]\nldr x3, [x2, #8]");
    ASSERT_EQ(instrs.size(), 3u);
    float ratio = 0.0f;
    AArch64MLPAnalyzer analyzer;
    float val = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/true);
    EXPECT_NEAR(val, 1.6364, 0.01);
}


TEST(MLPTest, AArch64CacheHitBaseRegister) {
    initLLVMAArch64();
    AArch64TestContext TC;
    auto instrs = parseAsm(TC, "ldr x1, [x0, #8]\nldr x2, [x0, #16]");
    ASSERT_EQ(instrs.size(), 2u);
    float ratio = 0.0f;
    AArch64MLPAnalyzer analyzer;
    float val = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/true);
    EXPECT_NEAR(val, 2.0, 0.01);
}


TEST(MLPTest, AArch64CacheLineBoundary) {
    initLLVMAArch64();
    AArch64TestContext TC;
    // Offset 8 and 72 belong to different cache lines (8/64 = 0, 72/64 = 1).
    // Therefore, the second load should NOT be a cache hit and should be counted.
    auto instrs = parseAsm(TC, "ldr x1, [x0, #8]\nldr x2, [x0, #72]");
    ASSERT_EQ(instrs.size(), 2u);
    float ratio = 0.0f;
    AArch64MLPAnalyzer analyzer;
    float val = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/true);
    EXPECT_NEAR(val, 2.0, 0.01);
}


TEST(MLPTest, AArch64IndexRegisterExclusion) {
    initLLVMAArch64();
    AArch64TestContext TC;
    // Uses index register (x3). Since index register loads are predicted to cache-miss,
    // they are evaluated and not excluded from the evaluation queue.
    auto instrs = parseAsm(TC, "ldr x1, [x0, x3]\nldr x2, [x0, x3]");
    ASSERT_EQ(instrs.size(), 2u);
    float ratio = 0.0f;
    AArch64MLPAnalyzer analyzer;
    float val = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/true);
    EXPECT_NEAR(val, 2.0, 0.01);
}


TEST(MLPTest, AArch64CallInstructionClearsDependencies) {
    initLLVMAArch64();
    AArch64TestContext TC;
    // Test that 'bl' clears dependency on x1.
    // If dependency is cleared, both loads are independent.
    // ldr x1, [x0, #8]  (load A)
    // bl my_func        (clears dependency on x1)
    // ldr x2, [x1, #8]  (load B) - independent of A because of 'bl'
    auto instrs = parseAsm(TC, "ldr x1, [x0, #8]\nbl my_func\nldr x2, [x1, #8]");
    ASSERT_EQ(instrs.size(), 3u);
    float ratio = 0.0f;
    AArch64MLPAnalyzer analyzer;
    float val = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/false);
    EXPECT_NEAR(val, 1.3333, 0.01);
}


TEST(MLPTest, AArch64CallInstructionClearsSeenBaseRegs) {
    initLLVMAArch64();
    AArch64TestContext TC;
    auto instrs = parseAsm(TC, "ldr x1, [x0, #8]\nbl my_func\nldr x2, [x0, #8]");
    ASSERT_EQ(instrs.size(), 3u);
    float ratio = 0.0f;
    AArch64MLPAnalyzer analyzer;
    float val = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/false);
    EXPECT_NEAR(val, 1.3333, 0.01);
}


// === RISC-V Tests ===

TEST(MLPTest, RISCVBasicMLPLoadStore) {
    initLLVMRISCV();
    RISCVTestContext TC;
    if (!TC.TheTarget) return; // Skip if target not registered
    auto instrs = parseAsm(TC, "ld a0, 0(a1)\nld a2, 8(a1)");
    ASSERT_EQ(instrs.size(), 2u);
    float ratio = 0.0f;
    RISCVMLPAnalyzer analyzer;
    float val = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/true);
    EXPECT_NEAR(val, 2.4, 0.01);
}


TEST(MLPTest, RISCVZeroRegisterExclusion) {
    initLLVMRISCV();
    RISCVTestContext TC;
    if (!TC.TheTarget) return;
    auto instrs = parseAsm(TC, "ld x0, 0(a1)\nld a2, 0(a1)");
    ASSERT_EQ(instrs.size(), 2u);
    float ratio = 0.0f;
    RISCVMLPAnalyzer analyzer;
    float val = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/false);
    EXPECT_NEAR(val, 1.3333, 0.01);
}


TEST(MLPTest, RISCVCallClearsDependencies) {
    initLLVMRISCV();
    RISCVTestContext TC;
    if (!TC.TheTarget) return;
    auto instrs = parseAsm(TC, "ld a0, 0(a1)\njal ra, my_func\nld a2, 0(a0)");
    ASSERT_EQ(instrs.size(), 3u);
    float ratio = 0.0f;
    RISCVMLPAnalyzer analyzer;
    float val = analyzer.compute_mlp(instrs, 16, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/false);
    EXPECT_NEAR(val, 1.3333, 0.01);
}


// === x86 SIB & Complex Addressing Tests ===

TEST(MLPTest, X86SIBAddressingSeenBase) {
    initLLVMX86();
    TestContext TC;
    // Complex addressing: [base + index * scale + offset]
    // Both loads access the same base rdi and index rsi.
    auto instrs = parseAsm(TC, "movq 8(%rdi,%rsi,8), %rax\nmovq 16(%rdi,%rsi,8), %rbx");
    ASSERT_EQ(instrs.size(), 2u);
    float ratio = 0.0f;
    X86MLPAnalyzer analyzer;
    float val = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/true);
    EXPECT_NEAR(val, 4.0, 0.01);
}


TEST(MLPTest, X86CallClearsDependencies) {
    initLLVMX86();
    TestContext TC;
    // callq clears dependencies on volatile/return registers (rax, etc.)
    auto instrs = parseAsm(TC, "movq (%rdi), %rax\ncallq my_func\nmovq (%rax), %rbx");
    ASSERT_EQ(instrs.size(), 3u);
    float ratio = 0.0f;
    X86MLPAnalyzer analyzer;
    float val = analyzer.compute_mlp(instrs, 16, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/false);
    EXPECT_NEAR(val, 1.3333, 0.01);
}


// === Boundary Conditions & Extreme Configurations ===

TEST(MLPTest, BoundaryEmptySequence) {
    initLLVMX86();
    TestContext TC;
    std::vector<Instr> instrs;
    float ratio = 0.0f;
    X86MLPAnalyzer analyzer;
    float val = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio);
    EXPECT_EQ(val, 1.0f);
}


TEST(MLPTest, BoundaryNoLoads) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC, "addq $1, %rax\nsubq $1, %rbx\nxorq %rcx, %rcx");
    ASSERT_EQ(instrs.size(), 3u);
    float ratio = 0.0f;
    X86MLPAnalyzer analyzer;
    float val = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio);
    EXPECT_EQ(val, 1.0f);
}


TEST(MLPTest, ExtremeSmallWindow) {
    initLLVMX86();
    TestContext TC;
    // With W = 1, only one micro-op fits in the window at a time. Outstanding loads must be at most 1.
    auto instrs = parseAsm(TC, "movq (%rdi), %rax\nmovq (%rsi), %rbx");
    ASSERT_EQ(instrs.size(), 2u);
    float ratio = 0.0f;
    X86MLPAnalyzer analyzer;
    float val = analyzer.compute_mlp(instrs, 1, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio);
    EXPECT_NEAR(val, 1.0, 0.01);
}


TEST(MLPTest, ExtremeLargeWindow) {
    initLLVMX86();
    TestContext TC;
    // With large W = 100 and no dependencies, all independent loads fit in the window.
    auto instrs = parseAsm(TC, "movq (%rdi), %rax\nmovq (%rsi), %rbx\nmovq (%rcx), %rdx");
    ASSERT_EQ(instrs.size(), 3u);
    float ratio = 0.0f;
    X86MLPAnalyzer analyzer;
    float val = analyzer.compute_mlp(instrs, 100, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio);
    EXPECT_NEAR(val, 1.6364, 0.01);
}


TEST(MLPTest, DebugReturnRegisters) {
    initLLVMX86();
    TestContext TC;
    std::vector<unsigned> regs = getReturnRegisters(*TC.MRI, TC.STI->getTargetTriple().getArchName().str());
    std::cout << "--- X86 Return Registers (" << regs.size() << ") ---" << std::endl;
    for (unsigned r : regs) {
        std::cout << "  " << TC.MRI->getName(r) << std::endl;
    }
    EXPECT_GT(regs.size(), 0u);

    auto instrs = parseAsm(TC, "callq my_func");
    ASSERT_EQ(instrs.size(), 1u);
    const MCInstrDesc &Desc = TC.MCII->get(instrs[0].Inst.getOpcode());
    std::cout << "Opcode name: " << TC.MCII->getName(instrs[0].Inst.getOpcode()).str() << std::endl;
    std::cout << "isCall: " << Desc.isCall() << std::endl;
    std::cout << "isBranch: " << Desc.isBranch() << std::endl;
    std::cout << "isTerminator: " << Desc.isTerminator() << std::endl;

    X86MLPAnalyzer analyzer;
    std::vector<MLPInstInfo> infos = buildInstInfos(instrs, *TC.STI, *TC.MCII, *TC.MRI, &analyzer);
    ASSERT_EQ(infos.size(), 1u);
    std::cout << "  callq inputs:" << std::endl;
    for (unsigned reg : infos[0].io_regs.inputs) {
        std::cout << "    " << TC.MRI->getName(reg) << std::endl;
    }
    std::cout << "  callq outputs:" << std::endl;
    for (unsigned reg : infos[0].io_regs.outputs) {
        std::cout << "    " << TC.MRI->getName(reg) << std::endl;
    }

    // RISC-V Return Registers Debug
    initLLVMRISCV();
    RISCVTestContext RTC;
    if (RTC.TheTarget) {
        std::vector<unsigned> rregs = getReturnRegisters(*RTC.MRI, RTC.STI->getTargetTriple().getArchName().str());
        std::cout << "--- RISC-V Return Registers (" << rregs.size() << ") ---" << std::endl;
        for (unsigned r : rregs) {
            std::cout << "  " << RTC.MRI->getName(r) << std::endl;
        }
    }
}


TEST(MLPTest, AArch64OverrideLoadLatencyInfluence) {
    initLLVMAArch64();
    AArch64TestContext TC;
    // Load and immediate dependent use with loop-carried dependency:
    // ldr x0, [x0, #8]
    // add x0, x0, #1
    auto instrs = parseAsm(TC, "ldr x0, [x0, #8]\nadd x0, x0, #1");
    ASSERT_EQ(instrs.size(), 2u);

    float ratio = 0.0f;
    AArch64MLPAnalyzer analyzer;
    mca::PipelineOptions PO(0, 0, 0, 0, 0, 0, true);

    // Run without load latency override (-1, defaults to tablegen latency of ~4-5 cycles for ldr)
    auto default_result = analyzeMcaRegion(instrs, *TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), PO,
                                           100, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward,
                                           analyzer, /*ignoreLoopCarried=*/false,
                                           /*overrideLoadLatency=*/-1, /*mlpWindowLoop=*/false);

    // Run with load latency override set to 1 cycle
    auto overridden_result = analyzeMcaRegion(instrs, *TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), PO,
                                              100, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward,
                                              analyzer, /*ignoreLoopCarried=*/false,
                                              /*overrideLoadLatency=*/1, /*mlpWindowLoop=*/false);

    ASSERT_TRUE(default_result.Valid);
    ASSERT_TRUE(overridden_result.Valid);
    // Overriding load latency to 1 cycle should reduce the total execution cycle count
    EXPECT_LT(overridden_result.Cycles, default_result.Cycles);
}


TEST(MLPTest, A55FlagTransferPenalty) {
    initLLVMAArch64();
    AArch64TestContext TC;
    TC.STI.reset(TC.TheTarget->createMCSubtargetInfo(TC.TT, "cortex-a55", ""));

    // 1. Integer comparison with loop-carried dependency:
    // add x0, x0, #1
    // cmp x0, #1
    // csel x0, x1, x2, ne
    auto int_seq = parseAsm(TC, "add x0, x0, #1\ncmp x0, #1\ncsel x0, x1, x2, ne");
    ASSERT_EQ(int_seq.size(), 3u);

    // 2. FP comparison with loop-carried dependency:
    // fmov d0, x0
    // fcmp d0, d1
    // csel x0, x1, x2, ne
    auto fp_seq = parseAsm(TC, "fmov d0, x0\nfcmp d0, d1\ncsel x0, x1, x2, ne");
    ASSERT_EQ(fp_seq.size(), 3u);

    AArch64MLPAnalyzer analyzer;
    mca::PipelineOptions PO(0, 0, 0, 0, 0, 0, true);

    auto int_res = analyzeMcaRegion(int_seq, *TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), PO,
                                    100, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward,
                                    analyzer, /*ignoreLoopCarried=*/false, -1, /*mlpWindowLoop=*/false);

    auto fp_res = analyzeMcaRegion(fp_seq, *TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), PO,
                                   100, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward,
                                   analyzer, /*ignoreLoopCarried=*/false, -1, /*mlpWindowLoop=*/false);

    ASSERT_TRUE(int_res.Valid);
    ASSERT_TRUE(fp_res.Valid);

    // FP comparison sequence should take more cycles than the integer comparison sequence due to the penalty
    EXPECT_LT(int_res.Cycles, fp_res.Cycles);
}


TEST(MLPTest, A55PointerForwarding) {
    initLLVMAArch64();
    AArch64TestContext TC;
    TC.STI.reset(TC.TheTarget->createMCSubtargetInfo(TC.TT, "cortex-a55", ""));

    // 1. ADRP to LDR (should benefit from low latency pointer forwarding bypass, 0-cycle AGU dependency)
    auto adrp_seq = parseAsm(TC, "adrp x0, #0\nldr x1, [x0, #8]");
    ASSERT_EQ(adrp_seq.size(), 2u);

    // 2. ADD to LDR (no special bypass, standard 1-cycle latency data dependency stall, with loop-carried dependency)
    auto add_seq = parseAsm(TC, "add x0, x2, #8\nldr x2, [x0]");
    ASSERT_EQ(add_seq.size(), 2u);

    AArch64MLPAnalyzer analyzer;
    mca::PipelineOptions PO(0, 0, 0, 0, 0, 0, true);

    auto adrp_res = analyzeMcaRegion(adrp_seq, *TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), PO,
                                     100, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward,
                                     analyzer, /*ignoreLoopCarried=*/false, -1, /*mlpWindowLoop=*/false);

    auto add_res = analyzeMcaRegion(add_seq, *TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), PO,
                                    100, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward,
                                    analyzer, /*ignoreLoopCarried=*/false, -1, /*mlpWindowLoop=*/false);

    ASSERT_TRUE(adrp_res.Valid);
    ASSERT_TRUE(add_res.Valid);

    // ADRP sequence should take fewer cycles than the ADD sequence due to the pointer forwarding bypass
    EXPECT_LT(adrp_res.Cycles, add_res.Cycles);
}


TEST(MLPTest, StallOnUseCacheHitSpecification) {
    initLLVMAArch64();
    AArch64TestContext TC;
    
    // Case 1: ldr x1, [x2]; ldr x3, [x2, #4] (No user in between -> no cache hit)
    auto seq1 = parseAsm(TC, "ldr x1, [x2]\nldr x3, [x2, #4]");
    ASSERT_EQ(seq1.size(), 2u);
    
    // Case 2: ldr x1, [x2]; add x1, x1, #4; ldr x3, [x2, #4] (User 'add' in between -> cache hit)
    auto seq2 = parseAsm(TC, "ldr x1, [x2]\nadd x1, x1, #4\nldr x3, [x2, #4]");
    ASSERT_EQ(seq2.size(), 3u);
    
    AArch64MLPAnalyzer analyzer;
    
    float ratio1 = 0.0f;
    float val1 = analyzer.compute_mlp(seq1, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio1, false);
    
    float ratio2 = 0.0f;
    float val2 = analyzer.compute_mlp(seq2, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio2, false);
    
    // Case 1: load_indices has 2 entries.
    // For i=0: mlp_vals[0] = 2.0 (sees both loads).
    // For i=1: mlp_vals[1] = 1.0 (sees only itself because mlpWindowLoop is false).
    // total_mlp = 1.0/2.0 + 1.0/1.0 = 1.5. avg_mlp = 2 / 1.5 = 1.3333.
    EXPECT_NEAR(val1, 1.3333, 0.01);
    
    // Case 2: load_indices has 1 entry (the second load is skipped because of cache hit).
    // mlp_vals for the single load index will be 1.0.
    // total_mlp = 1.0/1.0 = 1.0. avg_mlp = 1.0 / 1 = 1.0.
    EXPECT_NEAR(val2, 1.0, 0.01);
}



TEST(MLPTest, AArch64ExplicitSPCacheHit) {
    initLLVMAArch64();
    AArch64TestContext TC;
    // Load instruction using SP as base register: ldr x1, [sp, #8]
    auto instrs = parseAsm(TC, "ldr x1, [sp, #8]");
    ASSERT_EQ(instrs.size(), 1u);

    AArch64MLPAnalyzer analyzer;
    size_t loads = analyzer.countPotentialMissLoads(instrs, *TC.STI, *TC.MCII, *TC.MRI);
    // Should be treated as guaranteed cache hit -> 0 potential miss loads
    EXPECT_EQ(loads, 0u);

    float ratio = 0.0f;
    float val = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, false);
    // Since there are 0 potential miss loads, MLP defaults to 1.0
    EXPECT_EQ(val, 1.0f);
}


// -----------------------------------------------------------------------------
// -line-reuse-in-order: extend the same-cache-line (line reuse) always-hit
// heuristic to the non-OOO code paths used by the in-order small core
// (cortex-a55, --dependency dependency). Off by default, so the default
// non-OOO behaviour (stack exclusion only) must be preserved.
// -----------------------------------------------------------------------------
namespace {
struct LineReuseInOrderFlagGuard {
    bool saved_line_reuse = opts::LineReuseInOrder;
    bool saved_disable = opts::DisableAlwaysHitLoadsHeuristic;
    ~LineReuseInOrderFlagGuard() {
        opts::LineReuseInOrder = saved_line_reuse;
        opts::DisableAlwaysHitLoadsHeuristic = saved_disable;
    }
};
}  // namespace


TEST(MLPTest, AArch64LineReuseInOrderLoadCount) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a55");
    LineReuseInOrderFlagGuard guard;
    // Two loads off the same base register in the same 64-byte line, with the
    // first load's result consumed in between (so the stall has actually
    // happened): the second load is then a guaranteed hit under the
    // stall-on-use in-order argument.
    auto instrs = parseAsm(TC, "ldr x1, [x0, #8]\nadd x5, x1, #1\nldr x2, [x0, #16]");
    ASSERT_EQ(instrs.size(), 3u);
    AArch64MLPAnalyzer analyzer;

    opts::LineReuseInOrder = false;
    EXPECT_EQ(analyzer.countPotentialMissLoads(instrs, *TC.STI, *TC.MCII, *TC.MRI,
                                               DependencyKind::Dependency, false), 2u);

    opts::LineReuseInOrder = true;
    EXPECT_EQ(analyzer.countPotentialMissLoads(instrs, *TC.STI, *TC.MCII, *TC.MRI,
                                               DependencyKind::Dependency, false), 1u);

    // The master opt-out still wins over the new flag.
    opts::DisableAlwaysHitLoadsHeuristic = true;
    EXPECT_EQ(analyzer.countPotentialMissLoads(instrs, *TC.STI, *TC.MCII, *TC.MRI,
                                               DependencyKind::Dependency, false), 2u);
}


TEST(MLPTest, AArch64LineReuseInOrderCacheLineBoundary) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a55");
    LineReuseInOrderFlagGuard guard;
    // Offsets 8 and 72 are in different cache lines, so nothing is excluded
    // even with the flag on.
    auto instrs = parseAsm(TC, "ldr x1, [x0, #8]\nldr x2, [x0, #72]");
    ASSERT_EQ(instrs.size(), 2u);
    AArch64MLPAnalyzer analyzer;

    opts::LineReuseInOrder = true;
    EXPECT_EQ(analyzer.countPotentialMissLoads(instrs, *TC.STI, *TC.MCII, *TC.MRI,
                                               DependencyKind::Dependency, false), 2u);
}


TEST(MLPTest, AArch64LineReuseInOrderComputeMlp) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a55");
    LineReuseInOrderFlagGuard guard;
    // ldr x1 is a potential miss with a stall distance of 3; ldr x2 reuses x0's
    // (now resident) line and is a guaranteed hit with a stall distance of 1,
    // which should drop out of the harmonic average once the flag is on
    // (1/((1/3+1/1)/2) = 1.5 -> 3.0).
    auto instrs = parseAsm(TC, "ldr x1, [x0, #8]\nnop\nnop\nadd x5, x1, #1\nldr x2, [x0, #16]\nadd x6, x2, #1");
    ASSERT_EQ(instrs.size(), 6u);
    AArch64MLPAnalyzer analyzer;
    float ratio = 0.0f;

    opts::LineReuseInOrder = false;
    float base = analyzer.compute_mlp(instrs, 8, DependencyKind::Dependency,
                                      MLPWindowAssignmentKind::Forward,
                                      *TC.STI, *TC.MCII, *TC.MRI, ratio, false);
    opts::LineReuseInOrder = true;
    float excl = analyzer.compute_mlp(instrs, 8, DependencyKind::Dependency,
                                      MLPWindowAssignmentKind::Forward,
                                      *TC.STI, *TC.MCII, *TC.MRI, ratio, false);
    // Dropping the short-distance guaranteed-hit load raises the average
    // stall distance over the remaining potential-miss loads.
    EXPECT_GT(excl, base);
}


TEST(MLPTest, AArch64LineReuseInOrderNoOpForOOO) {
    initLLVMAArch64();
    AArch64TestContext TC;
    LineReuseInOrderFlagGuard guard;
    auto instrs = parseAsm(TC, "ldr x1, [x0, #8]\nadd x5, x1, #1\nldr x2, [x0, #16]");
    ASSERT_EQ(instrs.size(), 3u);
    AArch64MLPAnalyzer analyzer;

    opts::LineReuseInOrder = false;
    size_t ooo_off = analyzer.countPotentialMissLoads(instrs, *TC.STI, *TC.MCII, *TC.MRI,
                                                      DependencyKind::OOO, false);
    opts::LineReuseInOrder = true;
    size_t ooo_on = analyzer.countPotentialMissLoads(instrs, *TC.STI, *TC.MCII, *TC.MRI,
                                                     DependencyKind::OOO, false);
    EXPECT_EQ(ooo_off, 1u);
    EXPECT_EQ(ooo_on, ooo_off);
}


TEST(MLPTest, AArch64LineReuseInOrderKeepsStackExclusion) {
    initLLVMAArch64();
    AArch64TestContext TC("cortex-a55");
    LineReuseInOrderFlagGuard guard;
    // Stack loads stay excluded in the new tracking path as well.
    auto instrs = parseAsm(TC, "ldr x1, [sp, #8]\nldr x2, [x0, #16]");
    ASSERT_EQ(instrs.size(), 2u);
    AArch64MLPAnalyzer analyzer;

    opts::LineReuseInOrder = true;
    EXPECT_EQ(analyzer.countPotentialMissLoads(instrs, *TC.STI, *TC.MCII, *TC.MRI,
                                               DependencyKind::Dependency, false), 1u);
}


TEST(MLPTest, RISCVExplicitSPCacheHit) {
    initLLVMRISCV();
    RISCVTestContext TC;
    // Load instruction using sp (x2) as base register: ld a0, 8(sp)
    auto instrs = parseAsm(TC, "ld a0, 8(sp)");
    ASSERT_EQ(instrs.size(), 1u);

    RISCVMLPAnalyzer analyzer;
    size_t loads = analyzer.countPotentialMissLoads(instrs, *TC.STI, *TC.MCII, *TC.MRI);
    // Should be treated as guaranteed cache hit -> 0 potential miss loads
    EXPECT_EQ(loads, 0u);

    float ratio = 0.0f;
    float val = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.STI, *TC.MCII, *TC.MRI, ratio, false);
    EXPECT_EQ(val, 1.0f);
}


TEST(MLPTest, FirestormMCASimulation) {
    initLLVMAArch64();
    AArch64TestContext TC("firestorm");
    auto instrs = parseAsm(TC, "add x0, x1, x2\nadd x3, x4, x5\n");
    ASSERT_FALSE(instrs.empty());

    AArch64MLPAnalyzer analyzer;
    mca::PipelineOptions PO(0, 0, 0, 0, 0, 0, true);
    McaMetrics M = analyzeMcaRegion(instrs, *TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), PO,
                                    10, 330, DependencyKind::OOO, MLPWindowAssignmentKind::Forward,
                                    analyzer, false, -1, false);
    EXPECT_TRUE(M.Valid);
    EXPECT_GT(M.Cycles, 0u);
}


// -----------------------------------------------------------------------------
// WindowResources (Firestorm's in-flight-limit window): per-instruction costs and
// the window ending at the first exceeded limit.
// -----------------------------------------------------------------------------
TEST(MLPTest, AArch64WindowCosts) {
    initLLVMAArch64();
    AArch64TestContext TC;
    AArch64MLPAnalyzer analyzer;
    // add: 1 rename; adds: result + NZCV = 2; str: 0 rename, 1 store;
    // ldp: 2 renames, 2 load slots; stp: 2 store slots; mov: 1 rename.
    auto instrs = parseAsm(TC,
        "add x0, x1, x2\nadds x3, x1, x2\nstr x4, [x5]\nldp x6, x7, [x8]\n"
        "stp x9, x10, [x11]\nmov x12, #1\nldr x13, [x14]");
    ASSERT_EQ(instrs.size(), 7u);
    std::vector<MLPInstInfo> v = buildInstInfos(instrs, *TC.STI, *TC.MCII, *TC.MRI, &analyzer);
    EXPECT_EQ(v[0].rename_entries, 1);
    EXPECT_EQ(v[1].rename_entries, 2);
    EXPECT_EQ(v[2].rename_entries, 0);
    EXPECT_EQ(v[2].store_slots, 1);
    EXPECT_EQ(v[3].rename_entries, 2);
    EXPECT_EQ(v[3].load_slots, 2);
    EXPECT_EQ(v[4].store_slots, 2);
    EXPECT_EQ(v[5].rename_entries, 1);
    EXPECT_EQ(v[6].load_slots, 1);
    for (const auto &i : v) EXPECT_EQ(i.branch_slots, 0);
}


TEST(MLPTest, AArch64WindowResourcesLoadSlots) {
    initLLVMAArch64();
    AArch64TestContext TC;
    AArch64MLPAnalyzer analyzer;
    // Four independent loads on distinct bases.  A 3-load-slot window sees 3 loads
    // from each of the first two starting loads, then 2 and 1.
    auto instrs = parseAsm(TC, "ldr x1, [x2]\nldr x3, [x4]\nldr x5, [x6]\nldr x7, [x8]");
    ASSERT_EQ(instrs.size(), 4u);
    float ratio = 0.0f;
    analyzer.setWindowResources({1000, 3, 1000, 1000, 100});
    float val = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward,
                                     *TC.STI, *TC.MCII, *TC.MRI, ratio, false);
    EXPECT_NEAR(val, 4.0 / (1.0 / 3 + 1.0 / 3 + 1.0 / 2 + 1.0), 0.01);

    // Without limits (plain width) the first load sees all four.
    analyzer.setWindowResources({});
    val = analyzer.compute_mlp(instrs, 100, DependencyKind::OOO, MLPWindowAssignmentKind::Forward,
                               *TC.STI, *TC.MCII, *TC.MRI, ratio, false);
    EXPECT_NEAR(val, 4.0 / (1.0 / 4 + 1.0 / 3 + 1.0 / 2 + 1.0), 0.01);
}


TEST(MLPTest, AArch64WindowResourcesLdpCountsTwo) {
    initLLVMAArch64();
    AArch64TestContext TC;
    AArch64MLPAnalyzer analyzer;
    // The LDP takes 2 of the 3 load slots, so only one more load fits behind it.
    auto instrs = parseAsm(TC, "ldp x1, x2, [x3]\nldr x4, [x5]\nldr x6, [x7]");
    ASSERT_EQ(instrs.size(), 3u);
    float ratio = 0.0f;
    analyzer.setWindowResources({1000, 3, 1000, 1000, 100});
    float val = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward,
                                     *TC.STI, *TC.MCII, *TC.MRI, ratio, false);
    // Windows: {ldp, ldr} = 2, {ldr, ldr} = 2, {ldr} = 1.
    EXPECT_NEAR(val, 3.0 / (0.5 + 0.5 + 1.0), 0.01);
}


TEST(MLPTest, AArch64WindowResourcesRenameEntries) {
    initLLVMAArch64();
    AArch64TestContext TC;
    AArch64MLPAnalyzer analyzer;
    // Each load (and add) writes one register; a 2-entry rename queue ends the
    // window after two of them.
    auto instrs = parseAsm(TC, "ldr x1, [x2]\nadd x9, x9, #1\nldr x3, [x4]");
    ASSERT_EQ(instrs.size(), 3u);
    float ratio = 0.0f;
    analyzer.setWindowResources({2, 1000, 1000, 1000, 100});
    float val = analyzer.compute_mlp(instrs, 4, DependencyKind::OOO, MLPWindowAssignmentKind::Forward,
                                     *TC.STI, *TC.MCII, *TC.MRI, ratio, false);
    // First load's window = {ldr, add}: 1 load.  Second load: 1.
    EXPECT_NEAR(val, 1.0, 0.01);
}
