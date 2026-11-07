#ifndef FRONTEND_OPTIONS_H
#define FRONTEND_OPTIONS_H

// Command-line options shared by the tools and read by the analysis code.

#include <string>
#include "core/types.h"
#include "llvm/Support/CommandLine.h"

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
    extern llvm::cl::opt<bool> FacileMemoryDeps;
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

#endif // FRONTEND_OPTIONS_H
