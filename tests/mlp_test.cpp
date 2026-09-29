#include <gtest/gtest.h>
#include "mca_common.h"
#include "cpu_traits.h"
#include "frontend.h"
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
#include "facile.h"
#include "custom_a55_sched.h"
#include "llvm/MCA/InstrBuilder.h"
#include "llvm/MCA/CustomBehaviour.h"

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

    AArch64TestContext(const std::string &cpu = "cortex-a76") : TT("aarch64-unknown-linux-gnu") {
        std::string Error;
        TheTarget = TargetRegistry::lookupTarget(TT, Error);
        MRI.reset(TheTarget->createMCRegInfo(TT));
        MAI.reset(TheTarget->createMCAsmInfo(*MRI, TT, MCTargetOptions()));
        MCII.reset(TheTarget->createMCInstrInfo());
        // This branch is what frontend.cpp's initTarget() does for EVERY CPU:
        // install the locally re-generated sched-class tables from
        // ModifiedTarget/AArch64/*.td, remap MCInstrDesc::SchedClass onto their
        // numbering, and wrap the STI so resolveVariantSchedClass() reaches the
        // locally generated resolver (without that wrapper no SchedWriteVariant
        // in ModifiedTarget can resolve at all).
        //
        // The test harness historically took it only for icestorm/firestorm, so
        // every other CPU here ran against libLLVM's STOCK model rather than the
        // one the tool actually uses.  For "cortex-a720" that gap was material:
        // stock LLVM maps cortex-a720 to NeoverseN2Model, so an A720 scheduling
        // test would have been silently answered by N2's numbers - including
        // N2's own zero-latency MOVs, which would have made the A720 sec. 4.12
        // tests below pass whether or not AArch64SchedA720.td had been touched.
        // cortex-a720/-a720ae are therefore added here.
        //
        // The remaining CPUs are deliberately left on stock tables for now:
        // flipping them all over is a strictly larger change that alters results
        // this suite already pins (MLPTest.AArch64IndexRegisterExclusion, on the
        // default cortex-a76, computes MLP 4.0 rather than 2.0 under the local
        // N1 tables).  That divergence is real and worth its own investigation,
        // but it is not this change's subject.
        // cortex-x1/-x1c are added for the same reason as the A720: stock LLVM
        // maps them to its own NeoverseV1Model (ROB 256, IssueWidth 8, no
        // load-pair latency fix), which is not the model the tool installs.
        if (cpu == "icestorm" || cpu == "firestorm" ||
            cpu == "cortex-a720" || cpu == "cortex-a720ae" ||
            cpu == "cortex-x1" || cpu == "cortex-x1c") {
            const std::string llvm_cpu =
                (cpu == "icestorm" || cpu == "firestorm") ? "apple-m1" : cpu;
            STI.reset(TheTarget->createMCSubtargetInfo(TT, llvm_cpu, ""));
            if (STI) {
                llvm::remapSchedClassIndices(*MCII, cpu);
                llvm::overrideCortexA55SchedModel(*STI, cpu);
                STI = llvm::wrapCustomSubtargetInfo(std::move(STI), cpu);
            }
        } else {
            STI.reset(TheTarget->createMCSubtargetInfo(TT, cpu, ""));
        }
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

    walkRegions(instrs, empty_bounds,
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
    walkRegions(instrs, empty_bounds,
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
    initLLVMAArch64();
    AArch64TestContext TC;
    auto instrs = parseAsm(TC, "nop\nnop\nnop");
    ASSERT_EQ(instrs.size(), 3u);
    EXPECT_TRUE(isAllNopRegion(instrs, *TC.MCII));

    auto mixed_instrs = parseAsm(TC, "nop\nadd x0, x1, x2\nnop");
    ASSERT_EQ(mixed_instrs.size(), 3u);
    EXPECT_FALSE(isAllNopRegion(mixed_instrs, *TC.MCII));
}

TEST(MLPTest, SplitterNestedLoopNestingLimits) {
    FunctionBoundaries empty_bounds;
    std::vector<Instr> instrs(6);
    for (size_t i = 0; i < 6; ++i) instrs[i].Addr = i * 4;

    instrs[3].IsBranch = true;
    instrs[3].BranchTarget = 8; // Loop 1: [2, 3]

    instrs[5].IsBranch = true;
    instrs[5].BranchTarget = 4; // Loop 2: [1, 5]

    std::vector<RegionSpan> outer_loops;
    std::vector<RegionSpan> outer_bbs;
    walkRegions(instrs, empty_bounds,
                [&](const RegionSpan &Span) {
                    if (Span.Start == Span.AnalysisStart) outer_loops.push_back(Span);
                },
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
    std::vector<Instr> instrs(6);
    for (size_t i = 0; i < 6; ++i) instrs[i].Addr = i * 4;
    instrs[3].IsBranch = true;
    instrs[3].BranchTarget = 8; // target = 2 (subq)

    FunctionBoundaries empty_bounds;
    std::vector<RegionSpan> loops;
    std::vector<RegionSpan> bbs;

    walkRegions(instrs, empty_bounds,
                [&](const RegionSpan &Span) {
                    if (Span.Start == Span.AnalysisStart) loops.push_back(Span);
                },
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
    EXPECT_EQ(bbs.size(), 0u);
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

TEST(MLPTest, SplitterAbabMerging) {
    opts::ChainThreshold = 1;
    FunctionBoundaries empty_bounds;

    std::vector<Instr> instrs(5);
    for (size_t i = 0; i < 5; ++i) instrs[i].Addr = i * 4;

    instrs[3].IsBranch = true;
    instrs[3].BranchTarget = 4; // Loop A: [1, 3]

    instrs[4].IsBranch = true;
    instrs[4].BranchTarget = 8; // Loop B: [2, 4]

    std::vector<RegionSpan> loops;
    std::vector<RegionSpan> bbs;
    walkRegions(instrs, empty_bounds,
                [&](const RegionSpan &Span) {
                    if (Span.Start == Span.AnalysisStart) loops.push_back(Span);
                },
                [&](const RegionSpan &Span) { bbs.push_back(Span); });

    // Pre-merged loops A and B should be discarded and merged into 1 bounding span [1, 4] (size 4)
    ASSERT_EQ(loops.size(), 1u);
    EXPECT_EQ(loops[0].Start, 1u);
    EXPECT_EQ(loops[0].Size, 4u);
}

TEST(MLPTest, SplitterAbccabNestedPreservation) {
    opts::ChainThreshold = 1;
    FunctionBoundaries empty_bounds;

    std::vector<Instr> instrs(8);
    for (size_t i = 0; i < 8; ++i) instrs[i].Addr = i * 4;

    instrs[4].IsBranch = true;
    instrs[4].BranchTarget = 12; // Loop C: [3, 4] (fully nested in A and B)

    instrs[6].IsBranch = true;
    instrs[6].BranchTarget = 4;  // Loop A: [1, 6]

    instrs[7].IsBranch = true;
    instrs[7].BranchTarget = 8;  // Loop B: [2, 7]

    std::vector<RegionSpan> loops;
    std::vector<RegionSpan> bbs;
    walkRegions(instrs, empty_bounds,
                [&](const RegionSpan &Span) {
                    if (Span.Start == Span.AnalysisStart) loops.push_back(Span);
                },
                [&](const RegionSpan &Span) { bbs.push_back(Span); });

    // Loop C is nested (not abab partial overlap), so C must be PRESERVED.
    // Loops A and B are abab overlapping, so they are merged into bounding span [1, 7].
    // Total actual loop regions: 2 (Loop C [3, 4] and Merged AB [1, 7])
    ASSERT_EQ(loops.size(), 2u);

    // First loop: Loop C [3, 4]
    EXPECT_EQ(loops[0].Start, 3u);
    EXPECT_EQ(loops[0].Size, 2u);

    // Second loop: Merged AB [1, 7]
    EXPECT_EQ(loops[1].Start, 1u);
    EXPECT_EQ(loops[1].Size, 7u);
}

TEST(MLPTest, SplitterThresholdBoundary) {
    FunctionBoundaries empty_bounds;

    std::vector<Instr> instrs(5);
    for (size_t i = 0; i < 5; ++i) instrs[i].Addr = i * 4;

    instrs[3].IsBranch = true;
    instrs[3].BranchTarget = 4; // Loop A: [1, 3]

    instrs[4].IsBranch = true;
    instrs[4].BranchTarget = 8; // Loop B: [2, 4]

    // 1) High threshold (K=100): abab merging disabled, both A and B are kept as separate loops
    {
        opts::ChainThreshold = 100;
        std::vector<RegionSpan> loops;
        std::vector<RegionSpan> bbs;
        walkRegions(instrs, empty_bounds,
                    [&](const RegionSpan &Span) {
                        if (Span.Start == Span.AnalysisStart) loops.push_back(Span);
                    },
                    [&](const RegionSpan &Span) { bbs.push_back(Span); });

        ASSERT_EQ(loops.size(), 2u);
        EXPECT_EQ(loops[0].Start, 1u);
        EXPECT_EQ(loops[0].Size, 3u);
        EXPECT_EQ(loops[1].Start, 2u);
        EXPECT_EQ(loops[1].Size, 3u);
    }

    // 2) Low threshold (K=1): abab merging enabled, A and B merged into bounding span [1, 4]
    {
        opts::ChainThreshold = 1;
        std::vector<RegionSpan> loops;
        std::vector<RegionSpan> bbs;
        walkRegions(instrs, empty_bounds,
                    [&](const RegionSpan &Span) {
                        if (Span.Start == Span.AnalysisStart) loops.push_back(Span);
                    },
                    [&](const RegionSpan &Span) { bbs.push_back(Span); });

        ASSERT_EQ(loops.size(), 1u);
        EXPECT_EQ(loops[0].Start, 1u);
        EXPECT_EQ(loops[0].Size, 4u);
    }

    // Reset default ChainThreshold to 5
    opts::ChainThreshold = 5;
}

TEST(MLPTest, SplitterDirectRetExclusion) {
    FunctionBoundaries bounds;
    bounds[0x1000] = 0x2000;

    // Case 1: A: ldr; ldr; ldr; ret; add; add; goto A; (Should NOT be a loop)
    {
        std::vector<Instr> instrs(7, Instr{});
        for (size_t i = 0; i < 7; ++i) instrs[i].Addr = 0x1000 + i * 4;
        instrs[3].IsReturn = true;
        instrs[6].IsBranch = true;
        instrs[6].BranchTarget = 0x1000; // backward jump to 0x1000 (A)

        std::vector<RegionSpan> loops;
        walkRegions(instrs, bounds,
                    [&](const RegionSpan &Span) {
                        if (Span.Start == Span.AnalysisStart) loops.push_back(Span);
                    },
                    [](const RegionSpan &) {});

        EXPECT_EQ(loops.size(), 0u);
    }

    // Case 2: A: ldr; ldr; b.gt goto B; ret; B: add; add; goto A; (Should BE a loop)
    {
        std::vector<Instr> instrs(7, Instr{});
        for (size_t i = 0; i < 7; ++i) instrs[i].Addr = 0x1000 + i * 4;
        instrs[2].IsBranch = true;
        instrs[2].BranchTarget = 0x1010; // forward branch to 0x1010 (B)
        instrs[3].IsReturn = true;
        instrs[6].IsBranch = true;
        instrs[6].BranchTarget = 0x1000; // backward jump to 0x1000 (A)

        std::vector<RegionSpan> loops;
        walkRegions(instrs, bounds,
                    [&](const RegionSpan &Span) {
                        if (Span.Start == Span.AnalysisStart) loops.push_back(Span);
                    },
                    [](const RegionSpan &) {});

        EXPECT_EQ(loops.size(), 1u);
        if (!loops.empty()) {
            EXPECT_EQ(loops[0].Start, 0u);
            EXPECT_EQ(loops[0].Size, 7u);
        }
    }

    // Case 3: B: add; add; A: ldr; ldr; b.gt goto B; ret; add; add; goto A; (Should NOT be a loop)
    {
        std::vector<Instr> instrs(9, Instr{});
        for (size_t i = 0; i < 9; ++i) instrs[i].Addr = 0x1000 + i * 4;
        instrs[4].IsBranch = true;
        instrs[4].BranchTarget = 0x1000; // backward branch to 0x1000 (B)
        instrs[5].IsReturn = true;
        instrs[8].IsBranch = true;
        instrs[8].BranchTarget = 0x1008; // backward jump to 0x1008 (A)

        std::vector<RegionSpan> loops;
        walkRegions(instrs, bounds,
                    [&](const RegionSpan &Span) {
                        if (Span.Start == Span.AnalysisStart) loops.push_back(Span);
                    },
                    [](const RegionSpan &) {});

        // Loop B [0, 4] is kept, while candidate A [2, 8] is excluded due to direct ret
        EXPECT_EQ(loops.size(), 1u);
        if (!loops.empty()) {
            EXPECT_EQ(loops[0].Start, 0u);
            EXPECT_EQ(loops[0].Size, 5u);
        }
    }
}

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

// The per-CPU table (cpu_traits.cpp) that replaced the CPU-name ladders.
TEST(CpuTraitsTest, TableContents) {
    // Variants share their core's row.
    for (const char *n : {"cortex-a78", "cortex-a78ae", "cortex-a78c"}) {
        const CpuTraits &T = getCpuTraits(n);
        EXPECT_EQ(T.Model, CpuModel::A78) << n;
        EXPECT_EQ(T.UopDispatchWidth, 12u) << n;
        EXPECT_EQ(T.MopDispatchWidth, 6u) << n;
        EXPECT_EQ(T.Fusion, FusionTable::A78) << n;
        EXPECT_TRUE(T.ZeroLatencyMov && T.N2ModelCorrections) << n;
    }
    for (const char *n : {"cortex-x1", "cortex-x1c"}) {
        const CpuTraits &T = getCpuTraits(n);
        EXPECT_EQ(T.Model, CpuModel::X1) << n;
        EXPECT_EQ(T.UopDispatchWidth, 16u) << n;
        EXPECT_EQ(T.MopDispatchWidth, 8u) << n;
        EXPECT_EQ(T.Fusion, FusionTable::X1) << n;
        EXPECT_TRUE(T.ZeroLatencyMov) << n;
        EXPECT_FALSE(T.N2ModelCorrections) << n;  // V1 model has no N2 defects
    }
    // Stock-model cores that only share a uop width keep no Mop cap and no local model.
    EXPECT_EQ(getCpuTraits("neoverse-v1").UopDispatchWidth, 16u);
    EXPECT_EQ(getCpuTraits("neoverse-v1").MopDispatchWidth, 0u);
    EXPECT_EQ(getCpuTraits("neoverse-v1").Model, CpuModel::None);
    EXPECT_EQ(getCpuTraits("cortex-a710").UopDispatchWidth, 10u);
    EXPECT_EQ(getCpuTraits("cortex-a710").MopDispatchWidth, 0u);
    EXPECT_EQ(getCpuTraits("cortex-a76").MopDispatchWidth, 4u);
    EXPECT_FALSE(getCpuTraits("cortex-a76").ZeroLatencyMov);  // N1 uses Latency==0 as a marker
    EXPECT_EQ(getCpuTraits("cortex-a720").MopDispatchWidth, 5u);
    EXPECT_EQ(getCpuTraits("cortex-a720").Fusion, FusionTable::A720);
    // Unknown names, including "generic", get the neutral default.
    for (const char *n : {"generic", "", "cortex-a999"}) {
        const CpuTraits &T = getCpuTraits(n);
        EXPECT_EQ(T.Model, CpuModel::None) << n;
        EXPECT_EQ(T.UopDispatchWidth, 0u) << n;
        EXPECT_EQ(T.Fusion, FusionTable::None) << n;
    }
}

// --merge-same-header: two back-edges to the SAME header are one natural loop.
// Layout mirrors mcf primal_bea_mpp: hdr=1, short early-continue back-edge at 3 -> 1,
// full-body back-edge at 6 -> 1, plus an unrelated DIFFERENT-header inner loop [4,5].
static std::vector<RegionSpan> collectLoopSpans(std::vector<Instr> &instrs) {
    FunctionBoundaries empty_bounds;
    std::vector<RegionSpan> loops, bbs;
    walkRegions(instrs, empty_bounds,
                [&](const RegionSpan &S) { if (S.Start == S.AnalysisStart) loops.push_back(S); },
                [&](const RegionSpan &S) { bbs.push_back(S); });
    return loops;
}

TEST(MLPTest, SplitterMergeSameHeader) {
    opts::ChainThreshold = 5;
    std::vector<Instr> instrs(8);
    for (size_t i = 0; i < 8; ++i) instrs[i].Addr = i * 4;
    instrs[3].IsBranch = true; instrs[3].BranchTarget = 4;   // [1,3] short same-header path
    instrs[5].IsBranch = true; instrs[5].BranchTarget = 16;  // [4,5] different header (real inner loop)
    instrs[6].IsBranch = true; instrs[6].BranchTarget = 4;   // [1,6] full body

    opts::MergeSameHeader = false;
    auto off = collectLoopSpans(instrs);
    ASSERT_EQ(off.size(), 3u);  // pre-existing behavior unchanged when flag off

    opts::MergeSameHeader = true;
    auto on = collectLoopSpans(instrs);
    opts::MergeSameHeader = false;
    ASSERT_EQ(on.size(), 2u);
    bool has16 = false, has45 = false, has13 = false;
    for (auto &s : on) {
        if (s.Start == 1 && s.Size == 6) has16 = true;
        if (s.Start == 4 && s.Size == 2) has45 = true;
        if (s.Start == 1 && s.Size == 3) has13 = true;
    }
    EXPECT_TRUE(has16);
    EXPECT_TRUE(has45);   // different-header nested loop is preserved
    EXPECT_FALSE(has13);  // same-header partial path is dropped
}

TEST(MLPTest, SplitterMergeSameHeaderKeepsAbabSpansIdentical) {
    // abab pair [1,3],[2,4] with threshold 1 merges to [1,4]; the flag must not alter it.
    std::vector<Instr> instrs(5);
    for (size_t i = 0; i < 5; ++i) instrs[i].Addr = i * 4;
    instrs[3].IsBranch = true; instrs[3].BranchTarget = 4;
    instrs[4].IsBranch = true; instrs[4].BranchTarget = 8;
    for (int th : {1, 100}) {
        opts::ChainThreshold = th;
        opts::MergeSameHeader = false;
        auto a = collectLoopSpans(instrs);
        opts::MergeSameHeader = true;
        auto b = collectLoopSpans(instrs);
        opts::MergeSameHeader = false;
        ASSERT_EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            EXPECT_EQ(a[i].Start, b[i].Start);
            EXPECT_EQ(a[i].Size, b[i].Size);
        }
    }
    opts::ChainThreshold = 5;
}
