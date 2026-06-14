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

    TestContext() : TT("x86_64-unknown-linux-gnu") {
        std::string Error;
        TheTarget = TargetRegistry::lookupTarget(TT, Error);
        MRI.reset(TheTarget->createMCRegInfo(TT));
        MAI.reset(TheTarget->createMCAsmInfo(*MRI, TT, MCTargetOptions()));
        MCII.reset(TheTarget->createMCInstrInfo());
        STI.reset(TheTarget->createMCSubtargetInfo(TT, "haswell", ""));
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

static std::vector<Instr> parseAsm(const TestContext &TC, const std::string& asm_code) {
    SourceMgr SrcMgr;
    SrcMgr.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(asm_code), SMLoc());
    
    MCContext Ctx(TC.TT, *TC.MAI, *TC.MRI, *TC.STI, &SrcMgr);
    MCObjectFileInfo MOFI;
    MOFI.initMCObjectFileInfo(Ctx, /*PIC=*/false);
    Ctx.setObjectFileInfo(&MOFI);

    std::vector<Instr> instrs;
    struct TestStreamer : public MCStreamer {
        std::vector<Instr>& out;
        TestStreamer(MCContext& ctx, std::vector<Instr>& o) : MCStreamer(ctx), out(o) {}
        void emitInstruction(const MCInst& Inst, const MCSubtargetInfo& STI) override {
            Instr I; I.Inst = Inst; I.Addr = out.size() * 4;
            out.push_back(I);
        }
        bool emitSymbolAttribute(MCSymbol*, MCSymbolAttr) override { return true; }
        void emitCommonSymbol(MCSymbol*, uint64_t, Align) override {}
        void emitZerofill(MCSection*, MCSymbol*, uint64_t, Align, SMLoc) override {}
        void emitLabel(MCSymbol *Symbol, SMLoc Loc = SMLoc()) override {}
    };

    TestStreamer streamer(Ctx, instrs);
    std::unique_ptr<MCAsmParser> parser(createMCAsmParser(SrcMgr, Ctx, streamer, *TC.MAI));
    std::unique_ptr<MCTargetAsmParser> tap(TC.TheTarget->createMCAsmParser(*TC.STI, *parser, *TC.MCII));
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
    float val1 = compute_mlp(instrs, 2, DependencyKind::None, MLPWindowAssignmentKind::Forward, *TC.MCII, *TC.MRI, ratio);
    EXPECT_NEAR(val1, 1.5, 0.01);
    EXPECT_NEAR(ratio, 1.0, 0.01);
}

TEST(MLPTest, DependencyOOO) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC, "movq (%rdi), %rax\nmovq (%rax), %rbx");
    ASSERT_FALSE(instrs.empty());
    float ratio = 0.0f;
    float val1 = compute_mlp(instrs, 2, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.MCII, *TC.MRI, ratio);
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
    float val1 = compute_mlp(instrs, 5, DependencyKind::IO, MLPWindowAssignmentKind::Forward, *TC.MCII, *TC.MRI, ratio);
    EXPECT_NEAR(val1, 1.666, 0.01);
    EXPECT_NEAR(ratio, 0.8888, 0.01);
}

TEST(MLPTest, MaxContainingNone) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC, "mov %eax, %ebx\nmovq (%rsi), %rax\nmovq (%rdi), %rbx");
    ASSERT_FALSE(instrs.empty());
    float ratio = 0.0f;
    float val1 = compute_mlp(instrs, 2, DependencyKind::None, MLPWindowAssignmentKind::MaxContaining, *TC.MCII, *TC.MRI, ratio);
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
    float val1 = compute_mlp(instrs, 2, DependencyKind::Dependency, MLPWindowAssignmentKind::Forward, *TC.MCII, *TC.MRI, ratio);
    EXPECT_NEAR(val1, 1.5, 0.01);
}

TEST(MLPTest, WindowLoopNone) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC, "movq (%rsi), %rax\nmovq (%rdi), %rbx");
    ASSERT_FALSE(instrs.empty());
    float ratio = 0.0f;
    float val1 = compute_mlp(instrs, 2, DependencyKind::None, MLPWindowAssignmentKind::Forward, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/true);
    EXPECT_NEAR(val1, 2.0, 0.01);
}

TEST(MLPTest, WindowLoopOOO) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC, "movq (%rdi), %rax\nmovq (%rax), %rbx");
    ASSERT_FALSE(instrs.empty());
    float ratio = 0.0f;
    float val1 = compute_mlp(instrs, 2, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *TC.MCII, *TC.MRI, ratio, /*mlpWindowLoop=*/true);
    EXPECT_NEAR(val1, 1.5, 0.01);
    EXPECT_NEAR(ratio, 0.75, 0.01);
}
