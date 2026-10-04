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

unsigned computeSteadyIterations(const llvm::MCSubtargetInfo &STI, size_t regionInstrCount, unsigned iterations);
unsigned computeWarmupIterations(const llvm::MCSubtargetInfo &STI, size_t regionInstrCount);

McaMetrics analyzeMcaRegion(llvm::ArrayRef<Instr> instrs, const llvm::MCSubtargetInfo &STI,
                            const llvm::MCInstrInfo &MCII, const llvm::MCRegisterInfo &MRI,
                            const llvm::MCInstrAnalysis *MCIA, const llvm::mca::PipelineOptions &PO,
                            int iterations, int windowWidth, DependencyKind depKind,
                            MLPWindowAssignmentKind assignKind, const MLPAnalyzer &analyzer,
                            bool ignoreLoopCarriedDep = false,
                            int overrideLoadLatency = -1, bool mlpWindowLoop = false);

#endif // MCA_MCA_SIMULATION_H
