#ifndef CUSTOM_A55_SCHED_H
#define CUSTOM_A55_SCHED_H

#include <memory>
#include "llvm/ADT/StringRef.h"

namespace llvm {
class MCSubtargetInfo;
class MCInstrInfo;
void overrideCortexA55SchedModel(llvm::MCSubtargetInfo &STI, llvm::StringRef CPUName);
// Must be called on every MCInstrInfo used alongside
// overrideCortexA55SchedModel(); see the comment on the definition.
void remapSchedClassIndices(llvm::MCInstrInfo &MCII, llvm::StringRef CPUName);
std::unique_ptr<llvm::MCSubtargetInfo> wrapCustomSubtargetInfo(std::unique_ptr<llvm::MCSubtargetInfo> STI, llvm::StringRef CPUName);
}

#endif
