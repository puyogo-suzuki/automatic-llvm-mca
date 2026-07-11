#ifndef FRONTEND_H
#define FRONTEND_H

#include "mca_common.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MCA/Context.h"
#include "llvm/MCA/Pipeline.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/MC/TargetRegistry.h"

struct TargetInfo {
    std::string TripleName;
    std::string CPU;
    const llvm::Target *TheTarget = nullptr;
    std::unique_ptr<llvm::MCRegisterInfo> MRI;
    std::unique_ptr<llvm::MCAsmInfo> MAI;
    std::unique_ptr<llvm::MCInstrInfo> MCII;
    std::unique_ptr<llvm::MCSubtargetInfo> STI;
    std::unique_ptr<llvm::MCContext> Ctx;
    std::unique_ptr<llvm::MCDisassembler> DisAsm;
    std::unique_ptr<llvm::MCInstrAnalysis> MCIA;
    std::unique_ptr<llvm::mca::Context> MCAContext;
    llvm::mca::PipelineOptions PO;
    int WindowWidthVal = 4;
    uint64_t TargetAddress = 0;
    std::unique_ptr<MLPAnalyzer> Analyzer;
    
    TargetInfo() : PO(0, 0, 0, 0, 0, 0, true) {}
};

namespace opts {
    extern llvm::cl::opt<std::string> InputBinary;
    extern llvm::cl::opt<std::string> MTriple;
    extern llvm::cl::opt<std::string> MCPU;
    extern llvm::cl::opt<int> WindowWidth;
    extern llvm::cl::opt<DependencyKind> DepKind;
    extern llvm::cl::opt<MLPWindowAssignmentKind> AssignKind;
    extern llvm::cl::opt<int> Iterations;
    extern llvm::cl::opt<int> LoopMaxInstrs;
    extern llvm::cl::opt<int> BBMaxInstrs;
    extern llvm::cl::opt<int> NestLimitOuter;
    extern llvm::cl::opt<int> NestLimitInner;
    extern llvm::cl::opt<IgnoreLoopCarriedMode> IgnoreLoopCarried;
    extern llvm::cl::opt<int> OverrideLoadLatency;
    extern llvm::cl::opt<MlpWindowLoopMode> MlpWindowLoop;
    extern llvm::cl::opt<std::string> TargetAddressStr;
    
    // Specific to mlp_update.cpp but declared globally for simplicity
    extern llvm::cl::opt<std::string> InputCsv;
    extern llvm::cl::opt<std::string> OutputCsv;
}

bool initializeFrontend(int argc, char **argv, const char *Overview,
                        std::unique_ptr<llvm::object::ObjectFile> &Obj,
                        TargetInfo &TI);

#endif
