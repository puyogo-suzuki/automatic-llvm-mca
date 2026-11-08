#ifndef CPU_SCHED_MODEL_H
#define CPU_SCHED_MODEL_H

// Installs the locally generated scheduling models (ModifiedTarget/AArch64/*.td)
// into an MCSubtargetInfo.  Which CPU gets which model is CpuTraits::Model
// (cpu_traits.h).

#include <memory>
#include "llvm/ADT/StringRef.h"

struct CpuTraits;

namespace llvm {
class MCSubtargetInfo;
class MCInstrInfo;
class Target;
class Triple;

// `CPUName` below is always the canonical name (canonicalCpuName() in
// cpu_traits.h); an alias such as "ice" gets no local model.

// Replaces the subtarget's sched model and tables with the local ones for
// `CPUName`; leaves CPUs without a local model (e.g. "generic") untouched.
void installLocalSchedModel(llvm::MCSubtargetInfo &STI, llvm::StringRef CPUName);
// Must be called on every MCInstrInfo used alongside
// installLocalSchedModel(); see docs/docs.md section 5.A.
void remapSchedClassIndices(llvm::MCInstrInfo &MCII, llvm::StringRef CPUName);
// The whole subtarget set-up the tools use for the canonical CPU name `CPUName`:
// remapSchedClassIndices() on `MCII`, createMCSubtargetInfo() with
// llvmCpuName(CPUName), installLocalSchedModel() and wrapCustomSubtargetInfo().
// Returns null when LLVM cannot create the subtarget.
std::unique_ptr<llvm::MCSubtargetInfo> createLocalSubtargetInfo(const llvm::Target &T, const llvm::Triple &TT,
                                                                llvm::StringRef CPUName, llvm::MCInstrInfo &MCII);
// True when `STI` runs Apple Firestorm's or Icestorm's sched model.  Apple's cores
// coalesce several uops into one ROB entry, so every instruction counts as a
// single uop there (NumMicroOps is forced to 1 by the users of this).  This looks
// at the installed model, not at MCSubtargetInfo::getCPU(): LLVM reports
// "apple-a14" for both Apple cores (llvmCpuName() in cpu_traits.h).
bool isCoalescedROBSubtarget(const llvm::MCSubtargetInfo &STI);
// Wraps `STI` so variant sched classes resolve through the local tables.
std::unique_ptr<llvm::MCSubtargetInfo> wrapCustomSubtargetInfo(std::unique_ptr<llvm::MCSubtargetInfo> STI, llvm::StringRef CPUName);
}

// Overloads of the CPU-name functions of cpu_traits.h for code that only has the
// subtarget (Facile, the MCA driver).  The canonical CPU name of a subtarget set
// up by llvm::createLocalSubtargetInfo(): "firestorm" / "icestorm" when that local
// model is installed (getCPU() is "apple-a14" for both), MCSubtargetInfo::getCPU()
// otherwise - which, for every other CPU, is the canonical name the subtarget was
// created with (llvmCpuName() passes it unchanged).
llvm::StringRef canonicalCpuName(const llvm::MCSubtargetInfo &STI);
// getCpuTraits(canonicalCpuName(STI)).
const CpuTraits &getCpuTraits(const llvm::MCSubtargetInfo &STI);

#endif // CPU_SCHED_MODEL_H
