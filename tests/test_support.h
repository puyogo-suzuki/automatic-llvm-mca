#pragma once

// Shared fixtures for the unit tests: per-ISA MC contexts and an assembler helper.

#include <gtest/gtest.h>
#include "core/region_splitter.h"
#include "mca/mca_simulation.h"
#include "mlp/mlp_analyzer.h"
#include "cpu/cpu_traits.h"
#include "cpu/sched_model.h"
#include "frontend/frontend.h"
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
#include "facile/facile.h"
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

inline void initLLVMX86() {
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
inline std::vector<Instr> parseAsm(const T &TC, const std::string& asm_code) {
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
        void emitInstruction(const MCInst& Inst, const MCSubtargetInfo&) override {
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
        void emitLabel(MCSymbol *, SMLoc = SMLoc()) override {}
    };

    TestStreamer streamer(Ctx, instrs, *TC.MCII);
    std::unique_ptr<MCAsmParser> parser(createMCAsmParser(SrcMgr, Ctx, streamer, *TC.MAI));
    std::unique_ptr<MCTargetAsmParser> tap(TC.TheTarget->createMCAsmParser(*TC.STI, *parser, *TC.MCII, MCTargetOptions()));
    parser->setTargetParser(*tap);
    parser->Run(false);
    return instrs;
}

// The CPUs (canonical names, canonicalCpuName()) for which AArch64TestContext installs the scheduling model mca_tool uses (what
// initializeFrontend() in frontend.cpp does for every CPU: install the sched-class tables generated from
// ModifiedTarget/AArch64/*.td, remap MCInstrDesc::SchedClass onto their numbering, and wrap the STI
// so that resolveVariantSchedClass() reaches the local resolver).  Stock LLVM maps cortex-a720 to
// NeoverseN2Model and cortex-x1 to its own NeoverseV1Model, and its CortexA55Model lacks the SOG
// values, so these CPUs need the local model to be tested at all; the two MLPTest.A55* tests that
// want the stock A55 model reset TC.STI explicitly.
// The other CPUs (cortex-a76, the default, and cortex-a78) stay on libLLVM's stock tables here:
// moving them changes results this suite pins (MLPTest.AArch64IndexRegisterExclusion, on the
// default cortex-a76, computes MLP 4.0 rather than 2.0 under the local N1 tables).  The SOG tests
// use LocalModelContext (tests/sog_test_support.h), which installs the local model for any CPU.
inline bool testContextInstallsLocalModel(const std::string &cpu) {
    return cpu == "icestorm" || cpu == "firestorm" ||
           cpu == "cortex-a720" || cpu == "cortex-a720ae" ||
           cpu == "cortex-x1" || cpu == "cortex-x1c" ||
           cpu == "cortex-a55" || cpu == "cortex-a520";
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
        // `cpu` may be an alias ("ice", "fire"); resolved as initializeFrontend() does.
        const std::string canon = canonicalCpuName(cpu).str();
        if (testContextInstallsLocalModel(canon)) {
            STI = llvm::createLocalSubtargetInfo(*TheTarget, TT, canon, *MCII);
        } else {
            STI.reset(TheTarget->createMCSubtargetInfo(TT, cpu, ""));
        }
        MCIA.reset(TheTarget->createMCInstrAnalysis(MCII.get()));
    }
};

inline void initLLVMAArch64() {
    static bool initialized = false;
    if (!initialized) {
        LLVMInitializeAArch64TargetInfo();
        LLVMInitializeAArch64Target();
        LLVMInitializeAArch64TargetMC();
        LLVMInitializeAArch64AsmParser();
        initialized = true;
    }
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

inline void initLLVMRISCV() {
    static bool initialized = false;
    if (!initialized) {
        LLVMInitializeRISCVTargetInfo();
        LLVMInitializeRISCVTarget();
        LLVMInitializeRISCVTargetMC();
        LLVMInitializeRISCVAsmParser();
        initialized = true;
    }
}
