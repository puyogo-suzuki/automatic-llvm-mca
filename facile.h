#ifndef FACILE_H
#define FACILE_H

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
FacileResult computeFacilePrediction(const llvm::MCSubtargetInfo &STI,
                                     const llvm::MCInstrInfo &MCII,
                                     const llvm::MCRegisterInfo &MRI,
                                     llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                                     llvm::ArrayRef<const llvm::MCInst *> MCInsts = {},
                                     unsigned DispatchWidth = 0);

void printFacileResult(const FacileResult &Res, llvm::StringRef CPUName, llvm::raw_ostream &OS);

} // namespace facile

#endif // FACILE_H
