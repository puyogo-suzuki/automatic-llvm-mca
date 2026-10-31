#ifndef A55_A55_ADAPTATIONS_H
#define A55_A55_ADAPTATIONS_H

// Cortex-A55 corrections applied on top of the InstrBuilder output and the
// simulated cycle count (see docs/a55-sog.md, items D-5 and C-12).

#include <cstddef>
#include <cstdint>
#include "llvm/MC/MCInst.h"
#include "llvm/MCA/Instruction.h"

// Adjusts the instruction descriptor built from `MCI`: shift-less register-shifted
// ALU ops take one cycle (redundant with the WriteISReg variant of
// AArch64SchedA55.td, kept as a no-op safety net).  The former "branches are free"
// override (zero latency, no resources for B/B.cc) is disabled: no SOG basis
// (SOG 4.2: B 1, throughput 1).
void applyA55InstructionAdaptations(llvm::mca::Instruction &Inst, const llvm::MCInst &MCI);

// Formerly subtracted 0.5 cycle per steady-state iteration for an assumed
// branch-predictor overlap.  Disabled (no SOG basis): returns `Cycles` unchanged.
uint64_t applyA55BranchOverlapCorrection(uint64_t Cycles, uint64_t SteadyRetired, size_t RegionInstrs);

#endif // A55_A55_ADAPTATIONS_H
