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
    unsigned TotalInstructions = 0;
    unsigned TotalMicroOps = 0;
};

FacileResult computeFacilePrediction(const llvm::MCSubtargetInfo &STI,
                                     const llvm::MCInstrInfo &MCII,
                                     const llvm::MCRegisterInfo &MRI,
                                     llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                                     llvm::ArrayRef<const llvm::MCInst *> MCInsts = {});

void printFacileResult(const FacileResult &Res, llvm::StringRef CPUName, llvm::raw_ostream &OS);

} // namespace facile

#endif // FACILE_H
