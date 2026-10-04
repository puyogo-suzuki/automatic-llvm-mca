#ifndef CORE_TYPES_H
#define CORE_TYPES_H

// Plain data types shared by the whole pipeline: an analysed instruction, a
// region of instructions, and the metrics computed for it.

#ifndef OPT_MERGE_BB
#define OPT_MERGE_BB 1
#endif

#include <cstdint>
#include <map>
#include <string>
#include "llvm/ADT/BitVector.h"
#include "llvm/MC/MCInst.h"

enum class DependencyKind { None, IO, OOO, Dependency };
enum class MLPWindowAssignmentKind { Forward, MaxContaining };
enum class IgnoreLoopCarriedMode { Default, Force, Disable };
enum class MlpWindowLoopMode { Default, Force, Disable };

struct Instr {
    uint64_t Addr;
    llvm::MCInst Inst;
    bool IsBranch;
    bool IsCall;                // function call instruction (e.g. BL/BLR)
    bool EndsBB;
    bool IsUnconditionalBranch; // unconditional branch or indirect jump (cannot fall through)
    bool IsReturn;              // return instruction
    uint64_t BranchTarget;
};

using RegSet = llvm::BitVector;
using FunctionBoundaries = std::map<uint64_t, uint64_t>;

struct RegionSpan {
    size_t Start;
    size_t Size;
    size_t AnalysisStart;
    size_t AnalysisSize;
    bool IsAbabMerged = false;
};

struct McaMetrics {
    uint64_t RetiredInstructions = 0;
    uint64_t LoadInstructions = 0;
    uint64_t Cycles = 0;
    float MLP = 0.0f;
    float MLP_R = 0.0f;
    double BaseCPI = 0.0;
    std::string FacileReason;
    bool Valid = false;
};

struct McaRegion {
    size_t Start;
    size_t Size;
    size_t AnalysisStart;
    size_t AnalysisSize;
    size_t SimulatedSize;
    bool IsLoop;
    bool IsAbabMerged = false;
    McaMetrics Metrics;
    bool Valid = false;
    uint64_t StartAddr;
    uint64_t EndAddr;
    uint64_t AnalysisStartAddr;
    uint64_t AnalysisEndAddr;
};

#endif // CORE_TYPES_H
