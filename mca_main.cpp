#include "mca_common.h"
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <optional>
#include <unistd.h>
#include <fcntl.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/MCA/Context.h"
#include "llvm/MCA/InstrBuilder.h"
#include "llvm/MCA/Pipeline.h"
#include "llvm/MCA/SourceMgr.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace llvm::object;

namespace {

using FunctionBoundaries = std::map<uint64_t, uint64_t>;

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

FunctionBoundaries collectFunctionBoundaries(const ObjectFile &Obj) {
    FunctionBoundaries boundaries;
    for (const auto &Sym : Obj.symbols()) {
        auto TypeOrErr = Sym.getType();
        if (!TypeOrErr || *TypeOrErr != SymbolRef::ST_Function) continue;

        auto AddrOrErr = Sym.getAddress();
        if (!AddrOrErr) continue;

        const uint64_t Addr = *AddrOrErr;
        const uint64_t Size = getELFSymbolSize(Obj, Sym);
        if (Size > 0) boundaries[Addr] = Addr + Size;
    }
    return boundaries;
}

struct LoopSpan {
    size_t Start;
    size_t Size;
};

}  // namespace

// Command line options
static cl::opt<std::string> InputBinary(cl::Positional, cl::desc("<input binary>"), cl::Required);
static cl::opt<std::string> MTriple("mtriple", cl::desc("Target triple"));
static cl::opt<std::string> MCPU("mcpu", cl::desc("Target CPU"));
static cl::opt<int> WindowWidth("window-width", cl::desc("MLP window width"), cl::init(4));

static cl::opt<DependencyKind> DepKind("dependency", cl::desc("Dependency mode"),
    cl::values(
        clEnumValN(DependencyKind::None, "none", "No dependency tracking"),
        clEnumValN(DependencyKind::IO, "io", "In-order dependency"),
        clEnumValN(DependencyKind::OOO, "ooo", "Out-of-order dependency")
    ), cl::init(DependencyKind::None));

static cl::opt<MLPWindowAssignmentKind> AssignKind("mlp-window-assignment", cl::desc("Per-load MLP assignment mode"),
    cl::values(
        clEnumValN(MLPWindowAssignmentKind::Forward, "forward", "Forward window"),
        clEnumValN(MLPWindowAssignmentKind::MaxContaining, "max-containing", "Max MLP of containing windows")
    ), cl::init(MLPWindowAssignmentKind::MaxContaining));

static cl::opt<int> Iterations("iterations", cl::desc("Number of MCA iterations"), cl::init(100));
static cl::opt<int> LoopMaxInstrs("loop-max-instrs", cl::desc("Maximum instructions in a loop to analyze"), cl::init(100));
static cl::opt<int> BBMaxInstrs("bb-max-instrs", cl::desc("Maximum instructions in a basic block to analyze"), cl::init(100));

struct ScopedSilence {
    int devNull;
    int oldStderr;
    bool active;
    ScopedSilence() : devNull(-1), oldStderr(-1), active(false) {
        devNull = open("/dev/null", O_WRONLY);
        if (devNull != -1) {
            oldStderr = dup(STDERR_FILENO);
            if (dup2(devNull, STDERR_FILENO) != -1) active = true;
        }
    }
    ~ScopedSilence() {
        if (active) {
            errs().flush();
            dup2(oldStderr, STDERR_FILENO);
        }
        if (oldStderr != -1) close(oldStderr);
        if (devNull != -1) close(devNull);
    }
};

void run_mca_print(llvm::ArrayRef<Instr> instrs, const MCSubtargetInfo& STI, const MCInstrInfo& MCII,
                   const MCRegisterInfo& MRI, const MCInstrAnalysis* MCIA, const mca::PipelineOptions& PO) {
    if (instrs.empty()) return;
    mca::Context MCAContext(MRI, STI);
    mca::InstrumentManager IM(STI, MCII);
    mca::InstrBuilder IB(STI, MCII, MRI, MCIA, IM, 0);
    std::vector<mca::SourceMgr::UniqueInst> Sequence;
    SmallVector<mca::Instrument *> IVec;
    for (const auto &I : instrs) {
        auto ExpectedInst = IB.createInstruction(I.Inst, IVec);
        if (!ExpectedInst) { consumeError(ExpectedInst.takeError()); return; }
        Sequence.push_back(std::move(*ExpectedInst));
    }
    mca::CircularSourceMgr CSM(Sequence, Iterations);
    mca::CustomBehaviour CB(STI, CSM, MCII);
    std::unique_ptr<mca::Pipeline> P;
    if (STI.getSchedModel().isOutOfOrder()) P = MCAContext.createDefaultPipeline(PO, CSM, CB);
    else P = MCAContext.createInOrderPipeline(PO, CSM, CB);

    auto ExpectedCycles = P->run();
    if (!ExpectedCycles) { consumeError(ExpectedCycles.takeError()); return; }
    float mlp = compute_mlp(instrs, WindowWidth, DepKind, AssignKind, MCII, MRI);
    uint64_t load_count = 0;
    for (const auto& I : instrs) if (MCII.get(I.Inst.getOpcode()).mayLoad()) load_count++;
    
    printf("0x%lx,0x%lx,%zu,%lu,%u,%.2f\n",
           instrs.front().Addr, instrs.back().Addr, instrs.size() * Iterations,
           load_count * Iterations, (unsigned)*ExpectedCycles, mlp);
}

int main(int argc, char **argv) {
    InitLLVM X(argc, argv);
    initializeTargets();

    cl::ParseCommandLineOptions(argc, argv, "automatic-llvm-mca optimized C++ tool\n");

    auto BinaryOrErr = ObjectFile::createObjectFile(InputBinary);
    if (!BinaryOrErr) {
        WithColor::error() << "Failed to open binary: " << toString(BinaryOrErr.takeError()) << "\n";
        return 1;
    }
    ObjectFile &Obj = *BinaryOrErr.get().getBinary();

    Triple TT = Obj.makeTriple();
    if (!MTriple.empty()) TT = Triple(MTriple);
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
    if (!TheTarget) {
        WithColor::error() << "No target for " << TT.str() << ": " << Error << "\n";
        return 1;
    }

    std::string CPU = MCPU.empty() ? "generic" : std::string(MCPU);
    std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TT));
    MCTargetOptions MCOPT;
    std::unique_ptr<MCAsmInfo> MAI(TheTarget->createMCAsmInfo(*MRI, TT, MCOPT));
    std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
    std::unique_ptr<MCSubtargetInfo> STI(TheTarget->createMCSubtargetInfo(TT, CPU, ""));
    MCContext Ctx(TT, *MAI, *MRI, *STI);
    std::unique_ptr<MCDisassembler> DisAsm(TheTarget->createMCDisassembler(*STI, Ctx));
    std::unique_ptr<MCInstrAnalysis> MCIA(TheTarget->createMCInstrAnalysis(MCII.get()));

    if (!MRI || !MAI || !MCII || !STI || !DisAsm || !MCIA) {
        WithColor::error() << "Failed to initialize LLVM components\n";
        return 1;
    }

    mca::PipelineOptions PO(0, 0, 0, 0, 0, 0, true);
    const MCSchedModel &SM = STI->getSchedModel();
    PO.MicroOpQueueSize = SM.MicroOpBufferSize;
    PO.DispatchWidth = SM.IssueWidth;
    PO.AssumeNoAlias = true;

    printf("start_address,end_address,retired_instructions,load_instructions,cycles,mlp\n");

    FunctionBoundaries FunctionRanges = collectFunctionBoundaries(Obj);

    for (const SectionRef &Section : Obj.sections()) {
        if (!Section.isText() || Section.getSize() == 0) continue;
        uint64_t SAddr = Section.getAddress();
        auto ContentsOrErr = Section.getContents();
        if (!ContentsOrErr) continue;
        StringRef Contents = *ContentsOrErr;

        std::vector<Instr> SectionInstrs;
        SectionInstrs.reserve(Contents.size() / 4);
        ArrayRef<uint8_t> Data(reinterpret_cast<const uint8_t*>(Contents.data()), Contents.size());
        for (uint64_t Index = 0; Index < Data.size(); ) {
            MCInst Inst;
            uint64_t Size;
            auto Status = DisAsm->getInstruction(Inst, Size, Data.slice(Index), SAddr + Index, nulls());
            if (Status != MCDisassembler::Success) { Index++; continue; }
            Instr I;
            I.Addr = SAddr + Index; I.Inst = Inst;
            I.IsBranch = MCII->get(Inst.getOpcode()).isBranch();
            I.EndsBB = I.IsBranch || MCII->get(Inst.getOpcode()).isTerminator();
            I.BranchTarget = 0;
            if (I.IsBranch) MCIA->evaluateBranch(Inst, I.Addr, Size, I.BranchTarget);
            SectionInstrs.push_back(I);
            Index += Size;
        }
        if (SectionInstrs.empty()) continue;

        auto find_idx = [&](uint64_t addr) -> int64_t {
            auto it = std::lower_bound(SectionInstrs.begin(), SectionInstrs.end(), addr,
                                       [](const Instr& a, uint64_t val) { return a.Addr < val; });
            if (it != SectionInstrs.end() && it->Addr == addr)
                return std::distance(SectionInstrs.begin(), it);
            return -1;
        };

        ScopedSilence silence;
        std::vector<bool> InLoop(SectionInstrs.size(), false);
        const bool hasSymbols = !FunctionRanges.empty();
        std::vector<uint64_t> FuncStartAt(SectionInstrs.size(), 0);

        if (hasSymbols) {
            auto FuncIt = FunctionRanges.begin();
            for (size_t i = 0; i < SectionInstrs.size(); ++i) {
                const uint64_t Addr = SectionInstrs[i].Addr;
                while (FuncIt != FunctionRanges.end() && Addr >= FuncIt->second) ++FuncIt;
                if (FuncIt != FunctionRanges.end() && Addr >= FuncIt->first && Addr < FuncIt->second) {
                    FuncStartAt[i] = FuncIt->first;
                }
            }
        }

        auto run_slice = [&](size_t start, size_t size, int max_instrs) {
            if (size == 0 || max_instrs <= 0) return;
            for (size_t offset = 0; offset < size; offset += max_instrs) {
                size_t chunk_size = std::min(size - offset, static_cast<size_t>(max_instrs));
                run_mca_print(ArrayRef<Instr>(SectionInstrs).slice(start + offset, chunk_size), *STI, *MCII, *MRI, MCIA.get(), PO);
            }
        };

        auto is_same_function = [&](size_t idx, uint64_t branch_target) {
            return !hasSymbols || FuncStartAt[idx] == 0 || branch_target >= FuncStartAt[idx];
        };

        auto get_loop_span = [&](size_t idx) -> std::optional<LoopSpan> {
            const auto &I = SectionInstrs[idx];
            if (!I.IsBranch || I.BranchTarget == 0 || I.BranchTarget >= I.Addr) return std::nullopt;

            int64_t start_idx = find_idx(I.BranchTarget);
            if (start_idx == -1) return std::nullopt;

            const size_t loop_start = static_cast<size_t>(start_idx);
            const size_t loop_size = idx - loop_start + 1;
            if (loop_size > static_cast<size_t>(LoopMaxInstrs) || !is_same_function(idx, I.BranchTarget)) {
                return std::nullopt;
            }
            return LoopSpan{loop_start, loop_size};
        };

        // Pass 1: Pre-detect all loops and mark InLoop
        for (size_t i = 0; i < SectionInstrs.size(); ++i) {
            if (auto loop = get_loop_span(i)) {
                for (size_t j = loop->Start; j < loop->Start + loop->Size; ++j) InLoop[j] = true;
            }
        }

        // Pass 2: Iterate and analyze on-the-fly (Natural end-address order)
        size_t bb_start = 0;
        bool in_bb = false;
        for (size_t i = 0; i < SectionInstrs.size(); ++i) {
            const auto& I = SectionInstrs[i];

            // A. Check for loop end at this instruction
            if (auto loop = get_loop_span(i)) {
                run_slice(loop->Start, loop->Size, LoopMaxInstrs);
            }

            // B. Basic Block handling
            if (InLoop[i]) {
                if (in_bb) {
                    size_t bb_size = i - bb_start;
                    run_slice(bb_start, bb_size, BBMaxInstrs);
                    in_bb = false;
                }
                continue;
            }

            if (!in_bb) { bb_start = i; in_bb = true; }

            if (I.EndsBB) {
                size_t bb_size = i - bb_start + 1;
                run_slice(bb_start, bb_size, BBMaxInstrs);
                in_bb = false;
            }
        }
        if (in_bb) {
            size_t bb_size = SectionInstrs.size() - bb_start;
            run_slice(bb_start, bb_size, BBMaxInstrs);
        }
    }
    return 0;
}
