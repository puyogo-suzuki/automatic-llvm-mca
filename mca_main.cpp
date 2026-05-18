#include "mca_common.h"
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
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

void run_mca(const std::vector<Instr>& instrs, const MCSubtargetInfo& STI, const MCInstrInfo& MCII,
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
    float mlp = compute_mlp(instrs, WindowWidth, DepKind, AssignKind, MCII);
    uint64_t load_count = 0;
    for (const auto& I : instrs) if (MCII.get(I.Inst.getOpcode()).mayLoad()) load_count++;
    printf("0x%lx,0x%lx,%zu,%lu,%u,%.2f\n",
           instrs.front().Addr, instrs.back().Addr, instrs.size() * Iterations,
           load_count * Iterations, (unsigned)*ExpectedCycles, mlp);
}

int main(int argc, char **argv) {
    InitLLVM X(argc, argv);
    LLVMInitializeX86TargetInfo(); LLVMInitializeX86Target(); LLVMInitializeX86TargetMC(); LLVMInitializeX86AsmParser(); LLVMInitializeX86Disassembler();
    LLVMInitializeAArch64TargetInfo(); LLVMInitializeAArch64Target(); LLVMInitializeAArch64TargetMC(); LLVMInitializeAArch64AsmParser(); LLVMInitializeAArch64Disassembler();
    LLVMInitializeARMTargetInfo(); LLVMInitializeARMTarget(); LLVMInitializeARMTargetMC(); LLVMInitializeARMAsmParser(); LLVMInitializeARMDisassembler();
    LLVMInitializeRISCVTargetInfo(); LLVMInitializeRISCVTarget(); LLVMInitializeRISCVTargetMC(); LLVMInitializeRISCVAsmParser(); LLVMInitializeRISCVDisassembler();

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

    for (const SectionRef &Section : Obj.sections()) {
        if (!Section.isText() || Section.getSize() == 0) continue;
        uint64_t SAddr = Section.getAddress();
        auto ContentsOrErr = Section.getContents();
        if (!ContentsOrErr) continue;
        StringRef Contents = *ContentsOrErr;

        std::vector<Instr> SectionInstrs;
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

        ScopedSilence silence;
        std::map<uint64_t, size_t> AddrToIndex;
        for (size_t i = 0; i < SectionInstrs.size(); ++i) AddrToIndex[SectionInstrs[i].Addr] = i;
        
        std::set<uint64_t> InLoop;
        for (const auto& I : SectionInstrs) {
            if (I.IsBranch && I.BranchTarget != 0 && I.BranchTarget < I.Addr && AddrToIndex.count(I.BranchTarget)) {
                std::vector<Instr> Loop;
                for (size_t j = AddrToIndex[I.BranchTarget]; j <= AddrToIndex[I.Addr]; ++j) {
                    Loop.push_back(SectionInstrs[j]);
                    InLoop.insert(SectionInstrs[j].Addr);
                }
                run_mca(Loop, *STI, *MCII, *MRI, MCIA.get(), PO);
            }
        }
        std::vector<Instr> BB;
        for (const auto& I : SectionInstrs) {
            if (InLoop.count(I.Addr)) { 
                if (!BB.empty()) run_mca(BB, *STI, *MCII, *MRI, MCIA.get(), PO); 
                BB.clear(); continue; 
            }
            BB.push_back(I);
            if (I.EndsBB) { run_mca(BB, *STI, *MCII, *MRI, MCIA.get(), PO); BB.clear(); }
        }
        if (!BB.empty()) run_mca(BB, *STI, *MCII, *MRI, MCIA.get(), PO);
    }
    return 0;
}
