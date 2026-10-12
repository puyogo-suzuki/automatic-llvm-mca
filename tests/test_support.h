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
#include "cpu/sched_model.h"
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
                llvm::installLocalSchedModel(*STI, cpu);
                STI = llvm::wrapCustomSubtargetInfo(std::move(STI), cpu);
            }
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
