#include "mca_common.h"
#include <algorithm>
#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/MCA/Context.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/MCA/HWEventListener.h"
#include "llvm/MCA/InstrBuilder.h"
#include "llvm/MCA/Pipeline.h"
#include "llvm/MCA/SourceMgr.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

namespace {

uint64_t getELFSymbolSizeImpl(const ObjectFile &Obj, SymbolRef Sym) {
    if (const auto *Elf32LE = dyn_cast<ELF32LEObjectFile>(&Obj)) {
        if (auto SymOrErr = Elf32LE->getSymbol(Sym.getRawDataRefImpl())) return (*SymOrErr)->st_size;
    } else if (const auto *Elf64LE = dyn_cast<ELF64LEObjectFile>(&Obj)) {
        if (auto SymOrErr = Elf64LE->getSymbol(Sym.getRawDataRefImpl())) return (*SymOrErr)->st_size;
    } else if (const auto *Elf32BE = dyn_cast<ELF32BEObjectFile>(&Obj)) {
        if (auto SymOrErr = Elf32BE->getSymbol(Sym.getRawDataRefImpl())) return (*SymOrErr)->st_size;
    } else if (const auto *Elf64BE = dyn_cast<ELF64BEObjectFile>(&Obj)) {
        if (auto SymOrErr = Elf64BE->getSymbol(Sym.getRawDataRefImpl())) return (*SymOrErr)->st_size;
    }
    return 0;
}

struct SteadyStateTracker : mca::HWEventListener {
    unsigned WarmupRetiredLimit;
    unsigned TotalRetired = 0;
    unsigned SteadyRetired = 0;
    unsigned CurrentCycle = 0;
    unsigned SteadyStartCycle = 0;
    unsigned SteadyCycles = 0;
    bool WarmupComplete = false;

    explicit SteadyStateTracker(unsigned WarmupRetiredLimit) : WarmupRetiredLimit(WarmupRetiredLimit) {}

    void onCycleBegin() override { ++CurrentCycle; }

    void onCycleEnd() override {
        if (WarmupComplete && CurrentCycle > SteadyStartCycle) ++SteadyCycles;
    }

    void onEvent(const mca::HWInstructionEvent &Event) override {
        if (Event.Type != mca::HWInstructionEvent::Retired) return;
        ++TotalRetired;
        if (WarmupComplete) {
            ++SteadyRetired;
            return;
        }
        if (TotalRetired >= WarmupRetiredLimit) {
            WarmupComplete = true;
            SteadyStartCycle = CurrentCycle;
        }
    }
};

unsigned getWarmupWindowSize(const MCSubtargetInfo &STI) {
    const auto &SM = STI.getSchedModel();
    if (SM.isOutOfOrder()) return std::max(SM.MicroOpBufferSize, SM.LoopMicroOpBufferSize);
    return std::max(1u, SM.IssueWidth);
}

unsigned computeWarmupIterations(const MCSubtargetInfo &STI, size_t regionInstrCount) {
    if (regionInstrCount == 0) return 1;
    const unsigned WindowSize = std::max(1u, getWarmupWindowSize(STI));
    return std::max(1u, (WindowSize + static_cast<unsigned>(regionInstrCount) - 1) /
                            static_cast<unsigned>(regionInstrCount));
}

} // namespace

void initializeTargets() {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86AsmParser();
    LLVMInitializeX86Disassembler();

    LLVMInitializeAArch64TargetInfo();
    LLVMInitializeAArch64Target();
    LLVMInitializeAArch64TargetMC();
    LLVMInitializeAArch64AsmParser();
    LLVMInitializeAArch64Disassembler();

    LLVMInitializeARMTargetInfo();
    LLVMInitializeARMTarget();
    LLVMInitializeARMTargetMC();
    LLVMInitializeARMAsmParser();
    LLVMInitializeARMDisassembler();

    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVTarget();
    LLVMInitializeRISCVTargetMC();
    LLVMInitializeRISCVAsmParser();
    LLVMInitializeRISCVDisassembler();
}

uint64_t getELFSymbolSize(const ObjectFile &Obj, SymbolRef Sym) {
    return getELFSymbolSizeImpl(Obj, Sym);
}

FunctionBoundaries collectFunctionBoundaries(const ObjectFile &Obj) {
    FunctionBoundaries boundaries;
    for (const auto &Sym : Obj.symbols()) {
        auto TypeOrErr = Sym.getType();
        if (!TypeOrErr || *TypeOrErr != SymbolRef::ST_Function) continue;

        auto AddrOrErr = Sym.getAddress();
        if (!AddrOrErr) continue;

        const uint64_t Addr = *AddrOrErr;
        const uint64_t Size = getELFSymbolSizeImpl(Obj, Sym);
        if (Size > 0) boundaries[Addr] = Addr + Size;
    }
    return boundaries;
}

std::vector<Instr> disassembleTextSection(const SectionRef &Section, const MCDisassembler &DisAsm,
                                          const MCInstrInfo &MCII, const MCInstrAnalysis *MCIA) {
    std::vector<Instr> SectionInstrs;
    auto ContentsOrErr = Section.getContents();
    if (!ContentsOrErr) return SectionInstrs;

    StringRef Contents = *ContentsOrErr;
    ArrayRef<uint8_t> Data(reinterpret_cast<const uint8_t *>(Contents.data()), Contents.size());
    SectionInstrs.reserve(Contents.size() / 4);

    for (uint64_t Index = 0; Index < Data.size();) {
        MCInst Inst;
        uint64_t Size = 0;
        auto Status = DisAsm.getInstruction(Inst, Size, Data.slice(Index), Section.getAddress() + Index, nulls());
        if (Status != MCDisassembler::Success) {
            Index++;
            continue;
        }

        Instr I;
        I.Addr = Section.getAddress() + Index;
        I.Inst = Inst;
        I.IsBranch = MCII.get(Inst.getOpcode()).isBranch();
        I.EndsBB = I.IsBranch || MCII.get(Inst.getOpcode()).isTerminator();
        I.BranchTarget = 0;
        if (I.IsBranch && MCIA) MCIA->evaluateBranch(Inst, I.Addr, Size, I.BranchTarget);
        SectionInstrs.push_back(I);
        Index += Size;
    }
    return SectionInstrs;
}

McaMetrics analyzeMcaRegion(ArrayRef<Instr> instrs, const MCSubtargetInfo &STI, const MCInstrInfo &MCII,
                            const MCRegisterInfo &MRI, const MCInstrAnalysis *MCIA, const mca::PipelineOptions &PO,
                            int iterations, int windowWidth, DependencyKind depKind,
                            MLPWindowAssignmentKind assignKind) {
    if (instrs.empty()) return {};

    mca::Context MCAContext(MRI, STI);
    mca::InstrumentManager IM(STI, MCII);
    mca::InstrBuilder IB(STI, MCII, MRI, MCIA, IM, 0);
    std::vector<mca::SourceMgr::UniqueInst> Sequence;
    SmallVector<mca::Instrument *> IVec;
    for (const auto &I : instrs) {
        auto ExpectedInst = IB.createInstruction(I.Inst, IVec);
        if (!ExpectedInst) {
            consumeError(ExpectedInst.takeError());
            return {};
        }
        Sequence.push_back(std::move(*ExpectedInst));
    }

    const unsigned WarmupIterations = computeWarmupIterations(STI, instrs.size());
    const unsigned SteadyIterations = std::max(1u, static_cast<unsigned>(iterations) * WarmupIterations);
    mca::CircularSourceMgr CSM(Sequence, WarmupIterations + SteadyIterations);
    mca::CustomBehaviour CB(STI, CSM, MCII);
    std::unique_ptr<mca::Pipeline> P;
    if (STI.getSchedModel().isOutOfOrder()) P = MCAContext.createDefaultPipeline(PO, CSM, CB);
    else P = MCAContext.createInOrderPipeline(PO, CSM, CB);

    SteadyStateTracker Tracker(instrs.size() * WarmupIterations);
    P->addEventListener(&Tracker);

    auto ExpectedCycles = P->run();
    if (!ExpectedCycles) {
        consumeError(ExpectedCycles.takeError());
        return {};
    }

    McaMetrics M;
    M.RetiredInstructions = Tracker.SteadyRetired;
    M.LoadInstructions = 0;
    for (const auto &I : instrs) if (MCII.get(I.Inst.getOpcode()).mayLoad()) ++M.LoadInstructions;
    M.LoadInstructions *= static_cast<size_t>(SteadyIterations);
    M.Cycles = Tracker.SteadyCycles;
    M.MLP = compute_mlp(instrs, windowWidth, depKind, assignKind, MCII, MRI);
    if (M.RetiredInstructions > 0) {
        M.BaseCPI = static_cast<double>(M.Cycles) / static_cast<double>(M.RetiredInstructions);
    }
    M.Valid = true;
    return M;
}
