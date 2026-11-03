#ifndef MCA_MCA_SIMULATION_H
#define MCA_MCA_SIMULATION_H

// Cycle-accurate estimate of a region with LLVM MCA.

#include "core/types.h"
#include "mlp/mlp_analyzer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MCA/Context.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MCA/Instruction.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

class ZeroAwareInstrBuilder;

unsigned computeSteadyIterations(const llvm::MCSubtargetInfo &STI, size_t regionInstrCount, unsigned iterations);
unsigned computeWarmupIterations(const llvm::MCSubtargetInfo &STI, size_t regionInstrCount);

McaMetrics analyzeMcaRegion(llvm::ArrayRef<Instr> instrs, const llvm::MCSubtargetInfo &STI,
                            const llvm::MCInstrInfo &MCII, const llvm::MCRegisterInfo &MRI,
                            const llvm::MCInstrAnalysis *MCIA, const llvm::mca::PipelineOptions &PO,
                            int iterations, int windowWidth, DependencyKind depKind,
                            MLPWindowAssignmentKind assignKind, const MLPAnalyzer &analyzer,
                            bool ignoreLoopCarriedDep = false,
                            int overrideLoadLatency = -1, bool mlpWindowLoop = false);

// Builds the mca::Instruction of each of `instrs` with `IB`, in order, and hands
// it to `Accept` together with its source instruction.  An instruction that
// InstrBuilder rejects is recorded with noteDroppedInstruction() and skipped.
// On the coalesced-ROB subtargets (isCoalescedROBSubtarget(), cpu/sched_model.h)
// the instruction's NumMicroOps is set to 1 before it is handed over; this writes
// the InstrDesc, which InstrBuilder shares between instructions.
void buildRegionInstructions(
    ZeroAwareInstrBuilder &IB, const llvm::MCSubtargetInfo &STI, const llvm::MCInstrInfo &MCII,
    llvm::ArrayRef<Instr> instrs,
    llvm::function_ref<void(const Instr &, std::unique_ptr<llvm::mca::Instruction>)> Accept);

// Instructions that llvm::mca::InstrBuilder refuses to model (typically because
// the scheduling model marks their SchedWrite `Unsupported`) are left out of the
// simulated sequence.  Each such drop is counted here per opcode name instead of
// being discarded silently; the tools print the tally to stderr once the
// (stderr-silenced) analysis is over.  The CSV output is unaffected.
void noteDroppedInstruction(llvm::StringRef OpcodeName);
// Prints "[WARN] ..." lines summarising the drops recorded so far to `OS` and
// clears the tally.  Prints nothing if nothing was dropped.
void reportDroppedInstructions(llvm::raw_ostream &OS);

#endif // MCA_MCA_SIMULATION_H
