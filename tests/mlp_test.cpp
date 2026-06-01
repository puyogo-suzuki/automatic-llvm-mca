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

class MLPTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        LLVMInitializeX86TargetInfo();
        LLVMInitializeX86Target();
        LLVMInitializeX86TargetMC();
        LLVMInitializeX86AsmParser();
    }

    std::unique_ptr<MCInstrInfo> MCII;
    std::unique_ptr<MCRegisterInfo> MRI;
    std::unique_ptr<MCSubtargetInfo> STI;
    std::unique_ptr<MCAsmInfo> MAI;
    const Target* TheTarget;
    Triple TT;

    MLPTest() : TT("x86_64-unknown-linux-gnu") {
        std::string Error;
        TheTarget = TargetRegistry::lookupTarget(TT, Error);
        MRI.reset(TheTarget->createMCRegInfo(TT));
        MAI.reset(TheTarget->createMCAsmInfo(*MRI, TT, MCTargetOptions()));
        MCII.reset(TheTarget->createMCInstrInfo());
        STI.reset(TheTarget->createMCSubtargetInfo(TT, "haswell", ""));
    }

    std::vector<Instr> parseAsm(const std::string& asm_code) {
        SourceMgr SrcMgr;
        SrcMgr.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(asm_code), SMLoc());
        
        MCContext Ctx(TT, *MAI, *MRI, *STI, &SrcMgr);
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
        std::unique_ptr<MCAsmParser> parser(createMCAsmParser(SrcMgr, Ctx, streamer, *MAI));
        std::unique_ptr<MCTargetAsmParser> tap(TheTarget->createMCAsmParser(*STI, *parser, *MCII));
        parser->setTargetParser(*tap);
        parser->Run(false);
        return instrs;
    }
};

TEST_F(MLPTest, DependencyNone) {
    auto instrs = parseAsm("mov %eax, %ebx\nmovq (%rsi), %rax\nmovq (%rdi), %rbx");
    ASSERT_FALSE(instrs.empty());
    EXPECT_NEAR(compute_mlp(instrs, 2, DependencyKind::None, MLPWindowAssignmentKind::Forward, *MCII, *MRI), 1.5, 0.01);
}

TEST_F(MLPTest, DependencyOOO) {
    auto instrs = parseAsm("movq (%rdi), %rax\nmovq (%rax), %rbx");
    ASSERT_FALSE(instrs.empty());
    EXPECT_NEAR(compute_mlp(instrs, 2, DependencyKind::OOO, MLPWindowAssignmentKind::Forward, *MCII, *MRI), 1.0, 0.01);
}

TEST_F(MLPTest, IOBarrier) {
    auto instrs = parseAsm(
        "movq (%rdi), %rax\n"
        "movq (%rsi), %rbx\n"
        "addq $1, %rcx\n"
        "addq %rax, %rdx\n"
        "movq (%r8), %r9"
    );
    ASSERT_FALSE(instrs.empty());
    EXPECT_NEAR(compute_mlp(instrs, 5, DependencyKind::IO, MLPWindowAssignmentKind::Forward, *MCII, *MRI), 1.666, 0.01);
}

TEST_F(MLPTest, MaxContainingNone) {
    auto instrs = parseAsm("mov %eax, %ebx\nmovq (%rsi), %rax\nmovq (%rdi), %rbx");
    ASSERT_FALSE(instrs.empty());
    EXPECT_NEAR(compute_mlp(instrs, 2, DependencyKind::None, MLPWindowAssignmentKind::MaxContaining, *MCII, *MRI), 2.0, 0.01);
}

TEST_F(MLPTest, DependencyMode) {
    auto instrs = parseAsm(
        "movq (%rdi), %rax\n"
        "addq $1, %rbx\n"
        "addq %rax, %rcx\n"
        "movq (%rsi), %rdx\n"
        "subq $1, %rdx"
    );
    ASSERT_FALSE(instrs.empty());
    EXPECT_NEAR(compute_mlp(instrs, 2, DependencyKind::Dependency, MLPWindowAssignmentKind::Forward, *MCII, *MRI), 1.5, 0.01);
}

