#include <gtest/gtest.h>
#include "mca_common.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/MC/MCObjectFileInfo.h"

using namespace llvm;

struct TestContext {
    Triple TT;
    const Target* TheTarget;
    std::unique_ptr<MCRegisterInfo> MRI;
    std::unique_ptr<MCAsmInfo> MAI;
    std::unique_ptr<MCInstrInfo> MCII;
    std::unique_ptr<MCSubtargetInfo> STI;
    std::unique_ptr<MCInstrAnalysis> MCIA;

    TestContext() : TT("x86_64-unknown-linux-gnu") {
        std::string Error;
        TheTarget = TargetRegistry::lookupTarget(TT, Error);
        MRI.reset(TheTarget->createMCRegInfo(TT));
        MAI.reset(TheTarget->createMCAsmInfo(*MRI, TT, MCTargetOptions()));
        MCII.reset(TheTarget->createMCInstrInfo());
        STI.reset(TheTarget->createMCSubtargetInfo(TT, "haswell", ""));
        MCIA.reset(TheTarget->createMCInstrAnalysis(MCII.get()));
    }
};

static void initLLVMX86() {
    static bool initialized = false;
    if (!initialized) {
        LLVMInitializeX86TargetInfo();
        LLVMInitializeX86Target();
        LLVMInitializeX86TargetMC();
        LLVMInitializeX86AsmParser();
        initialized = true;
    }
}

template <typename T>
static std::vector<Instr> parseAsm(const T &TC, const std::string& asm_code) {
    SourceMgr SrcMgr;
    SrcMgr.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(asm_code), SMLoc());
    
    MCContext Ctx(TC.TT, TC.MAI.get(), TC.MRI.get(), TC.STI.get(), &SrcMgr);
    MCObjectFileInfo MOFI;
    MOFI.initMCObjectFileInfo(Ctx, /*PIC=*/false);
    Ctx.setObjectFileInfo(&MOFI);

    std::vector<Instr> instrs;
    struct TestStreamer : public MCStreamer {
        std::vector<Instr>& out;
        const MCInstrInfo &MCII;
        TestStreamer(MCContext& ctx, std::vector<Instr>& o, const MCInstrInfo &mcii) : MCStreamer(ctx), out(o), MCII(mcii) {}
        void emitInstruction(const MCInst& Inst, const MCSubtargetInfo& STI) override {
            Instr I;
            I.Inst = Inst;
            I.Addr = out.size() * 4;
            const MCInstrDesc &Desc = MCII.get(Inst.getOpcode());
            I.IsBranch = Desc.isBranch();
            I.IsReturn = Desc.isReturn();
            I.IsUnconditionalBranch = Desc.isUnconditionalBranch() || Desc.isIndirectBranch() || I.IsReturn;
            I.EndsBB = I.IsBranch || Desc.isTerminator();
            I.BranchTarget = 0;
            out.push_back(I);
        }
        bool emitSymbolAttribute(MCSymbol*, MCSymbolAttr) override { return true; }
        void emitCommonSymbol(MCSymbol*, uint64_t, Align) override {}
        void emitZerofill(MCSection*, MCSymbol*, uint64_t, Align, SMLoc) override {}
        void emitLabel(MCSymbol *Symbol, SMLoc Loc = SMLoc()) override {}
    };

    TestStreamer streamer(Ctx, instrs, *TC.MCII);
    std::unique_ptr<MCAsmParser> parser(createMCAsmParser(SrcMgr, Ctx, streamer, *TC.MAI));
    std::unique_ptr<MCTargetAsmParser> tap(TC.TheTarget->createMCAsmParser(*TC.STI, *parser, *TC.MCII, MCTargetOptions()));
    parser->setTargetParser(*tap);
    parser->Run(false);
    return instrs;
}

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

struct AArch64TestContext {
    Triple TT;
    const Target* TheTarget;
    std::unique_ptr<MCRegisterInfo> MRI;
    std::unique_ptr<MCAsmInfo> MAI;
    std::unique_ptr<MCInstrInfo> MCII;
    std::unique_ptr<MCSubtargetInfo> STI;
    std::unique_ptr<MCInstrAnalysis> MCIA;

    AArch64TestContext() : TT("aarch64-unknown-linux-gnu") {
        std::string Error;
        TheTarget = TargetRegistry::lookupTarget(TT, Error);
        MRI.reset(TheTarget->createMCRegInfo(TT));
        MAI.reset(TheTarget->createMCAsmInfo(*MRI, TT, MCTargetOptions()));
        MCII.reset(TheTarget->createMCInstrInfo());
        STI.reset(TheTarget->createMCSubtargetInfo(TT, "cortex-a76", ""));
        MCIA.reset(TheTarget->createMCInstrAnalysis(MCII.get()));
    }
};

static void initLLVMAArch64() {
    static bool initialized = false;
    if (!initialized) {
        LLVMInitializeAArch64TargetInfo();
        LLVMInitializeAArch64Target();
        LLVMInitializeAArch64TargetMC();
        LLVMInitializeAArch64AsmParser();
        initialized = true;
    }
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

// === RISC-V Test Setup ===

struct RISCVTestContext {
    Triple TT;
    const Target* TheTarget;
    std::unique_ptr<MCRegisterInfo> MRI;
    std::unique_ptr<MCAsmInfo> MAI;
    std::unique_ptr<MCInstrInfo> MCII;
    std::unique_ptr<MCSubtargetInfo> STI;

    RISCVTestContext() : TT("riscv64-unknown-elf") {
        std::string Error;
        TheTarget = TargetRegistry::lookupTarget(TT, Error);
        if (TheTarget) {
            MRI.reset(TheTarget->createMCRegInfo(TT));
            MAI.reset(TheTarget->createMCAsmInfo(*MRI, TT, MCTargetOptions()));
            MCII.reset(TheTarget->createMCInstrInfo());
            STI.reset(TheTarget->createMCSubtargetInfo(TT, "generic-rv64", ""));
        }
    }
};

static void initLLVMRISCV() {
    static bool initialized = false;
    if (!initialized) {
        LLVMInitializeRISCVTargetInfo();
        LLVMInitializeRISCVTarget();
        LLVMInitializeRISCVTargetMC();
        LLVMInitializeRISCVAsmParser();
        initialized = true;
    }
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

// === walkRegions Region Partitioning Tests ===

TEST(MLPTest, SplitterBasicBlockPartitioning) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC, "movq %rax, %rbx\naddq $1, %rcx\nsubq $1, %rdx");
    ASSERT_EQ(instrs.size(), 3u);
    
    FunctionBoundaries empty_bounds;
    std::vector<RegionSpan> bbs;
    std::vector<RegionSpan> loops;

    walkRegions(instrs, empty_bounds, 100, 100, 2, 2,
                [&](const RegionSpan &Span) { loops.push_back(Span); },
                [&](const RegionSpan &Span) { bbs.push_back(Span); });

    // There are no branches, so it should resolve as a single basic block of size 3.
    EXPECT_TRUE(loops.empty());
    ASSERT_EQ(bbs.size(), 1u);
    EXPECT_EQ(bbs[0].Start, 0u);
    EXPECT_EQ(bbs[0].Size, 3u);
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

TEST(MLPTest, SplitterBasicBlockSplittingLimit) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC, "movq %rax, %rbx\naddq $1, %rcx\nsubq $1, %rdx");
    ASSERT_EQ(instrs.size(), 3u);
    
    FunctionBoundaries empty_bounds;
    std::vector<RegionSpan> bbs;
    std::vector<RegionSpan> loops;

    // bbMaxInstrs = 2, so a block of size 3 should split into a block of size 2 and size 1.
    walkRegions(instrs, empty_bounds, 100, 2, 2, 2,
                [&](const RegionSpan &Span) { loops.push_back(Span); },
                [&](const RegionSpan &Span) { bbs.push_back(Span); });

    EXPECT_TRUE(loops.empty());
#if OPT_MERGE_BB
    ASSERT_EQ(bbs.size(), 1u);
    EXPECT_EQ(bbs[0].Start, 0u);
    EXPECT_EQ(bbs[0].Size, 3u);
#else
    ASSERT_EQ(bbs.size(), 2u);
    EXPECT_EQ(bbs[0].Start, 0u);
    EXPECT_EQ(bbs[0].Size, 2u);
    EXPECT_EQ(bbs[1].Start, 2u);
    EXPECT_EQ(bbs[1].Size, 1u);
#endif
}

TEST(MLPTest, SplitterNopInstructionCheck) {
    initLLVMX86();
    TestContext TC;
    // Test that standard nops are correctly identified
    auto instrs = parseAsm(TC, "nop\nnop\nnop");
    ASSERT_EQ(instrs.size(), 3u);
    EXPECT_TRUE(isAllNopRegion(instrs, *TC.MCII));

    auto mixed_instrs = parseAsm(TC, "nop\nmovq %rax, %rbx\nnop");
    ASSERT_EQ(mixed_instrs.size(), 3u);
    EXPECT_FALSE(isAllNopRegion(mixed_instrs, *TC.MCII));
}

TEST(MLPTest, SplitterNestedLoopNestingLimits) {
    initLLVMX86();
    TestContext TC;
    // Multi-level loop structure:
    // 0: nop (addr = 0)
    // 1: movq %rax, %rbx (addr = 4)  <-- Start of Outer Loop
    // 2: addq $1, %rcx   (addr = 8)  <-- Start of Inner Loop
    // 3: jne -8          (addr = 12) <-- Jump to 8 (addq)
    // 4: subq $1, %rdx   (addr = 16)
    // 5: jne -20         (addr = 20) <-- Jump to 4 (movq)
    auto instrs = parseAsm(TC, "nop\nmovq %rax, %rbx\naddq $1, %rcx\njne -8\nsubq $1, %rdx\njne -20");
    ASSERT_EQ(instrs.size(), 6u);

    // Resolve branch targets explicitly for the test framework parser
    for (size_t i = 0; i < instrs.size(); ++i) {
        instrs[i].Addr = i * 4;
    }
    instrs[3].BranchTarget = 8;
    instrs[5].BranchTarget = 4;

    FunctionBoundaries empty_bounds;

    // Test with nestLimitOuter = 1 (only allow the outermost loop: size 5)
    std::vector<RegionSpan> outer_loops;
    std::vector<RegionSpan> outer_bbs;
    walkRegions(instrs, empty_bounds, 100, 100, 1, 0,
                [&](const RegionSpan &Span) { outer_loops.push_back(Span); },
                [&](const RegionSpan &Span) { outer_bbs.push_back(Span); });

    ASSERT_EQ(outer_loops.size(), 2u);
    EXPECT_EQ(outer_loops[0].Start, 1u);
    EXPECT_EQ(outer_loops[0].Size, 5u);
    EXPECT_EQ(outer_loops[1].Start, 2u);
    EXPECT_EQ(outer_loops[1].Size, 2u);
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

TEST(MLPTest, SplitterPostDominatorLoopMerging) {
    initLLVMX86();
    TestContext TC;
    // 0: nop (addr = 0)
    // 1: addq $1, %rax (addr = 4)   <-- Loop pre-header (post-dominated by loop header 2)
    // 2: subq $1, %rbx (addr = 8)   <-- Loop Header
    // 3: jne -8 (addr = 12)         <-- Loop Latch (back-edge to 2)
    // 4: movq %rax, %rcx (addr = 16) <-- Post-loop BB (NOT post-dominated by loop)
    // 5: retq (addr = 20)           <-- Function EXIT
    auto instrs = parseAsm(TC, "nop\naddq $1, %rax\nsubq $1, %rbx\njne -8\nmovq %rax, %rcx\nretq");
    ASSERT_EQ(instrs.size(), 6u);

    // Normalize addresses and set target
    for (size_t i = 0; i < instrs.size(); ++i) {
        instrs[i].Addr = i * 4;
    }
    instrs[3].BranchTarget = 8; // target = 2 (subq)

    FunctionBoundaries empty_bounds;
    std::vector<RegionSpan> loops;
    std::vector<RegionSpan> bbs;

    walkRegions(instrs, empty_bounds, 100, 100, 0, 0,
                [&](const RegionSpan &Span) { loops.push_back(Span); },
                [&](const RegionSpan &Span) { bbs.push_back(Span); });

    // Loop should be detected (IID 2 to 3, size = 2)
    ASSERT_EQ(loops.size(), 1u);
    EXPECT_EQ(loops[0].Start, 2u);
    EXPECT_EQ(loops[0].Size, 2u);

    // IID 0 (nop) and IID 1 (pre-header) are post-dominated by loop header (IID 2)
    // because all paths from them must pass through the loop.  They are merged
    // into the loop region and NOT emitted as basic blocks.
    //
    // IID 4 (movq) and IID 5 (retq) are post-loop BBs.  Their post-dominator
    // chain goes directly to virtual_exit without passing through the loop header,
    // so they are NOT merged and ARE emitted as basic blocks.
    //
    // Expected: 1 BB spanning IID 4–5 (Start=4, Size=2).
#if OPT_MERGE_BB
    // With OPT_MERGE_BB = 1, we do not do post-dominator loop merging.
    // Instead, the non-loop interval [0, 2) is output as one basic block,
    // and [4, 6) is output as another basic block.
    ASSERT_EQ(bbs.size(), 2u);
    EXPECT_EQ(bbs[0].Start, 0u);
    EXPECT_EQ(bbs[0].Size, 2u);
    EXPECT_EQ(bbs[1].Start, 4u);
    EXPECT_EQ(bbs[1].Size, 2u);
#else
    // Expected: 1 BB spanning IID 4–5 (Start=4, Size=2).
    ASSERT_EQ(bbs.size(), 1u);
    EXPECT_EQ(bbs[0].Start, 4u);
    EXPECT_EQ(bbs[0].Size, 2u);
#endif
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


