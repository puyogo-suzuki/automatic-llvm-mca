#ifndef CPU_SCHED_MODEL_H
#define CPU_SCHED_MODEL_H

// Installs the locally generated scheduling models (ModifiedTarget/AArch64/*.td)
// into an MCSubtargetInfo.  Which CPU gets which model is CpuTraits::Model
// (cpu_traits.h).

#include <memory>
#include "llvm/ADT/StringRef.h"

namespace llvm {
class MCSubtargetInfo;
class MCInstrInfo;

// Replaces the subtarget's sched model and tables with the local ones for
// `CPUName`; leaves CPUs without a local model (e.g. "generic") untouched.
void installLocalSchedModel(llvm::MCSubtargetInfo &STI, llvm::StringRef CPUName);
// Must be called on every MCInstrInfo used alongside
// installLocalSchedModel(); see docs.md section 5.A.
void remapSchedClassIndices(llvm::MCInstrInfo &MCII, llvm::StringRef CPUName);
// Wraps `STI` so variant sched classes resolve through the local tables.
std::unique_ptr<llvm::MCSubtargetInfo> wrapCustomSubtargetInfo(std::unique_ptr<llvm::MCSubtargetInfo> STI, llvm::StringRef CPUName);
}

#endif // CPU_SCHED_MODEL_H
