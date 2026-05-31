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
#include <iomanip>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace llvm::object;

static cl::opt<std::string> InputBinary(cl::Positional, cl::desc("<input binary>"), cl::Required);

struct Result {
    double cpi;
    float mlp;
};

Result run_mca_varied(llvm::ArrayRef<Instr> instrs, int iterations, const MCSubtargetInfo& STI, const MCInstrInfo& MCII,
                     const MCRegisterInfo& MRI, const MCInstrAnalysis* MCIA, const mca::PipelineOptions& PO) {
    mca::Context MCAContext(MRI, STI);
    mca::InstrumentManager IM(STI, MCII);
    mca::InstrBuilder IB(STI, MCII, MRI, MCIA, IM, 0);
    std::vector<mca::SourceMgr::UniqueInst> Sequence;
    SmallVector<mca::Instrument *> IVec;
    for (const auto &I : instrs) {
        auto ExpectedInst = IB.createInstruction(I.Inst, IVec);
        if (!ExpectedInst) { consumeError(ExpectedInst.takeError()); return {0, 0}; }
        Sequence.push_back(std::move(*ExpectedInst));
    }
    mca::CircularSourceMgr CSM(Sequence, iterations);
    mca::CustomBehaviour CB(STI, CSM, MCII);
    std::unique_ptr<mca::Pipeline> P;
    if (STI.getSchedModel().isOutOfOrder()) P = MCAContext.createDefaultPipeline(PO, CSM, CB);
    else P = MCAContext.createInOrderPipeline(PO, CSM, CB);

    auto ExpectedCycles = P->run();
    if (!ExpectedCycles) { consumeError(ExpectedCycles.takeError()); return {0, 0}; }
    
    double total_instrs = (double)instrs.size() * iterations;
    double cpi = (double)*ExpectedCycles / total_instrs;
    float mlp = compute_mlp(instrs, 4, DependencyKind::None, MLPWindowAssignmentKind::MaxContaining, MCII, MRI);
    return {cpi, mlp};
}

int main(int argc, char **argv) {
    InitLLVM X(argc, argv);
    LLVMInitializeX86TargetInfo(); LLVMInitializeX86Target(); LLVMInitializeX86TargetMC(); LLVMInitializeX86AsmParser(); LLVMInitializeX86Disassembler();
    LLVMInitializeAArch64TargetInfo(); LLVMInitializeAArch64Target(); LLVMInitializeAArch64TargetMC(); LLVMInitializeAArch64AsmParser(); LLVMInitializeAArch64Disassembler();
    LLVMInitializeARMTargetInfo(); LLVMInitializeARMTarget(); LLVMInitializeARMTargetMC(); LLVMInitializeARMAsmParser(); LLVMInitializeARMDisassembler();
    LLVMInitializeRISCVTargetInfo(); LLVMInitializeRISCVTarget(); LLVMInitializeRISCVTargetMC(); LLVMInitializeRISCVAsmParser(); LLVMInitializeRISCVDisassembler();
    cl::ParseCommandLineOptions(argc, argv, "MCA Convergence Tester\n");

    auto BinaryOrErr = ObjectFile::createObjectFile(InputBinary);
    if (!BinaryOrErr) {
        errs() << "Failed to open binary: " << toString(BinaryOrErr.takeError()) << "\n";
        return 1;
    }
    ObjectFile &Obj = *BinaryOrErr.get().getBinary();
    Triple TT = Obj.makeTriple();
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
    if (!TheTarget) {
        errs() << "No target for " << TT.str() << ": " << Error << "\n";
        return 1;
    }
    std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TT));
    MCTargetOptions MCOPT;
    std::unique_ptr<MCAsmInfo> MAI(TheTarget->createMCAsmInfo(*MRI, TT, MCOPT));
    std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
    std::unique_ptr<MCSubtargetInfo> STI(TheTarget->createMCSubtargetInfo(TT, "generic", ""));
    MCContext Ctx(TT, *MAI, *MRI, *STI);
    std::unique_ptr<MCDisassembler> DisAsm(TheTarget->createMCDisassembler(*STI, Ctx));
    std::unique_ptr<MCInstrAnalysis> MCIA(TheTarget->createMCInstrAnalysis(MCII.get()));

    mca::PipelineOptions PO(0, 0, 0, 0, 0, 0, true);
    const MCSchedModel &SM = STI->getSchedModel();
    PO.MicroOpQueueSize = SM.MicroOpBufferSize;
    PO.DispatchWidth = SM.IssueWidth;
    if (STI->getCPU() == "cortex-a76" || STI->getCPU() == "cortex-a76ae" || STI->getCPU() == "neoverse-n1") {
        PO.DispatchWidth = 8;
    }
    PO.AssumeNoAlias = true;

    // Find 3 representative loops
    std::vector<std::vector<Instr>> samples;
    for (const SectionRef &Section : Obj.sections()) {
        if (!Section.isText() || Section.getSize() == 0) continue;
        uint64_t SAddr = Section.getAddress();
        StringRef Contents = *Section.getContents();
        std::vector<Instr> SectionInstrs;
        ArrayRef<uint8_t> Data(reinterpret_cast<const uint8_t*>(Contents.data()), Contents.size());
        for (uint64_t Index = 0; Index < Data.size(); ) {
            MCInst Inst; uint64_t Size;
            if (DisAsm->getInstruction(Inst, Size, Data.slice(Index), SAddr + Index, nulls()) != MCDisassembler::Success) { Index++; continue; }
            Instr I; I.Addr = SAddr + Index; I.Inst = Inst; I.IsBranch = MCII->get(Inst.getOpcode()).isBranch();
            if (I.IsBranch) MCIA->evaluateBranch(Inst, I.Addr, Size, I.BranchTarget);
            SectionInstrs.push_back(I); Index += Size;
        }
        std::map<uint64_t, size_t> AddrToIndex;
        for (size_t i = 0; i < SectionInstrs.size(); ++i) AddrToIndex[SectionInstrs[i].Addr] = i;
        for (const auto& I : SectionInstrs) {
            if (I.IsBranch && I.BranchTarget != 0 && I.BranchTarget < I.Addr && AddrToIndex.count(I.BranchTarget)) {
                size_t sz = AddrToIndex[I.Addr] - AddrToIndex[I.BranchTarget] + 1;
                if (samples.size() == 0 && sz < 15) samples.push_back({SectionInstrs.begin() + AddrToIndex[I.BranchTarget], SectionInstrs.begin() + AddrToIndex[I.Addr] + 1});
                if (samples.size() == 1 && sz > 40 && sz < 60) samples.push_back({SectionInstrs.begin() + AddrToIndex[I.BranchTarget], SectionInstrs.begin() + AddrToIndex[I.Addr] + 1});
                if (samples.size() == 2 && sz > 90 && sz <= 100) samples.push_back({SectionInstrs.begin() + AddrToIndex[I.BranchTarget], SectionInstrs.begin() + AddrToIndex[I.Addr] + 1});
            }
            if (samples.size() >= 3) break;
        }
        if (samples.size() >= 3) break;
    }

    std::vector<int> iter_counts = {1, 5, 10, 50, 100, 200, 500};
    std::cout << "=== MCA Convergence Investigation (CPI values) ===\n";
    std::cout << std::left << std::setw(15) << "Iterations";
    for (size_t i = 0; i < samples.size(); ++i) std::cout << "Loop " << i+1 << " (sz=" << samples[i].size() << ")  ";
    std::cout << "\n" << std::string(60, '-') << "\n";

    for (int iters : iter_counts) {
        std::cout << std::left << std::setw(15) << iters;
        for (const auto& sample : samples) {
            Result r = run_mca_varied(sample, iters, *STI, *MCII, *MRI, MCIA.get(), PO);
            std::cout << std::fixed << std::setprecision(4) << r.cpi << "          ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\nNote: MLP is computed statically and does not depend on iterations.\n";

    return 0;
}
