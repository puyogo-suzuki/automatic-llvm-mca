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

#include "llvm/Support/MemoryBuffer.h"

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
    // SOG-documented Macro-op (Mop) decode/dispatch cap, a SECOND front-end
    // constraint independent of PO.DispatchWidth (which this tool sets to the
    // uop cap - see initializeFrontend()). Neither llvm::MCSchedModel nor
    // TargetSchedule.td has any field for this: TargetSchedule.td's own
    // comment on IssueWidth is "Max micro-ops that may be scheduled per
    // cycle" - LLVM's scheduling model has no concept of a macro-op stage at
    // all, so this cannot be expressed via the .td tables and must be
    // supplied out-of-band, the same way isA78FusionCandidate() (facile.cpp)
    // and overrideCortexA55SchedModel() (custom_a55_sched.cpp) already key
    // CPU-specific behavior off STI->getCPU() rather than any .td field.
    // 0 = not known for this CPU; calculateIssueBound() then skips the bound.
    unsigned MopDispatchWidth = 0;
    int WindowWidthVal = 4;
    uint64_t TargetAddress = 0;
    std::unique_ptr<MLPAnalyzer> Analyzer;
    std::unique_ptr<llvm::MemoryBuffer> BinaryBuffer;
    
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
    extern llvm::cl::opt<IgnoreLoopCarriedMode> IgnoreLoopCarried;
    extern llvm::cl::opt<int> OverrideLoadLatency;
    extern llvm::cl::opt<MlpWindowLoopMode> MlpWindowLoop;
    extern llvm::cl::opt<std::string> TargetAddressStr;
    
    extern llvm::cl::opt<std::string> UpdateMlp;
    extern llvm::cl::opt<bool> Facile;
    extern llvm::cl::opt<bool> FacileReason;
    extern llvm::cl::opt<bool> NoFacileMemoryDeps;
    extern llvm::cl::opt<int> ChainThreshold;
    extern llvm::cl::opt<bool> MergeSameHeader;
    extern llvm::cl::opt<bool> CountOnly;
    extern llvm::cl::opt<bool> DisableAlwaysHitLoadsHeuristic;
    extern llvm::cl::opt<bool> LineReuseInOrder;
    extern llvm::cl::opt<bool> StackOnlyMissLoadCount;
    extern llvm::cl::opt<bool> StackConstOffsetOnly;
    extern llvm::cl::opt<bool> StackSpillOnly;
    extern llvm::cl::opt<bool> StackLoopResident;
    extern llvm::cl::opt<bool> NoStackExclusion;
}

bool initializeFrontend(int argc, char **argv, const char *Overview,
                        std::unique_ptr<llvm::object::ObjectFile> &Obj,
                        TargetInfo &TI);

struct ScopedSilence {
    int devNull = -1;
    int oldStderr = -1;
    bool active = false;
    ScopedSilence();
    ~ScopedSilence();
};

#endif
