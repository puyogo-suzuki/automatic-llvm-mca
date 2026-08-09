#ifndef CUSTOM_A55_SCHED_H
#define CUSTOM_A55_SCHED_H

#include <memory>
#include "llvm/ADT/StringRef.h"

namespace llvm {
class MCSubtargetInfo;
void overrideCortexA55SchedModel(llvm::MCSubtargetInfo &STI, llvm::StringRef CPUName);
std::unique_ptr<llvm::MCSubtargetInfo> wrapCustomSubtargetInfo(std::unique_ptr<llvm::MCSubtargetInfo> STI, llvm::StringRef CPUName);
}

#endif
