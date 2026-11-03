#ifndef FACILE_H
#define FACILE_H

#include "mlp/mlp_analyzer.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MCA/Instruction.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <vector>

namespace facile {

struct FacileResult {
    double IssueBound = 0.0;
    double PortBound = 0.0;
    double PrecedenceBound = 0.0;
    double EstimatedCycles = 0.0;
    double EstimatedCPI = 0.0;
    std::string DominantBottleneck;
    std::string PortBottleneckName;
    std::string FacileReason; // "inst", "exec", or "prec"
    unsigned TotalInstructions = 0;
    unsigned TotalMicroOps = 0;
};

// `DispatchWidth` should be the CPU's real dispatch/issue width in uops/cycle
// (e.g. the SOG-sourced PipelineOptions::DispatchWidth already computed for
// known CPUs in frontend.cpp: 8 for cortex-a76, 12 for cortex-a78, etc.) --
// NOT necessarily STI.getSchedModel().IssueWidth, which for CPUs modeled by
// borrowing a related upstream scheduling model (e.g. cortex-a78 borrowing
// NeoverseN2Model, cortex-a76 borrowing NeoverseN1Model) reflects that
// *other* core's own decode/issue width, not the modeled CPU's. Pass 0 to
// fall back to STI.getSchedModel().IssueWidth.
//
// `MemInfos`, when non-empty, must be index-aligned with `SimInstrs` /
// `MCInsts` and supplies the per-instruction memory-access descriptor
// (MLPAnalyzer::getMemAccessInfo) used to add store->load memory RAW edges to
// the precedence-constraint graph. Pass an empty ArrayRef for a graph with
// register dependences only (--no-facile-memory-deps). See the long note above
// addMemoryDependencies() in facile.cpp for why these edges are needed.
//
// `MopWidth`, when nonzero, is the CPU's SOG-documented macro-op (Mop)
// decode/dispatch cap in Mops/cycle - a SECOND, independent front-end bound
// from `DispatchWidth` (which is the uop cap). See frontend.h's
// TargetInfo::MopDispatchWidth comment and calculateIssueBound()'s comment in
// facile.cpp for why this can't be derived from DispatchWidth or from
// STI.getSchedModel() and must be passed in explicitly. 0 = not known for
// this CPU, or not applicable (Apple's coalesced-ROB cores) - skips the bound.
//
// `LoopCarried` says whether the region repeats as a loop.  When false, the
// dependence graph has only intra-iteration edges (registers and memory): a
// register written at the end of the region is NOT taken to feed a read at the
// start of the next "iteration", which for a function body or other straight-line
// region is no recurrence at all (the return value of one call is not the
// argument of the next).  The precedence bound is then 0 and the estimate is the
// issue / port bound.  mca_tool passes `!ignore`, where `ignore` is the
// --ignore-loop-carried decision the MCA path already uses (default: a region
// that is not a loop ignores loop-carried dependences).
FacileResult computeFacilePrediction(const llvm::MCSubtargetInfo &STI,
                                     const llvm::MCInstrInfo &MCII,
                                     const llvm::MCRegisterInfo &MRI,
                                     llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                                     llvm::ArrayRef<const llvm::MCInst *> MCInsts = {},
                                     unsigned DispatchWidth = 0,
                                     llvm::ArrayRef<MemAccessInfo> MemInfos = {},
                                     unsigned MopWidth = 0,
                                     bool LoopCarried = true);

void printFacileResult(const FacileResult &Res, llvm::StringRef CPUName, llvm::raw_ostream &OS);

} // namespace facile

#endif // FACILE_H
