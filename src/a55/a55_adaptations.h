#ifndef A55_A55_ADAPTATIONS_H
#define A55_A55_ADAPTATIONS_H

// Cortex-A55 corrections applied on top of the InstrBuilder output and the
// simulated cycle count (see docs.md section 1.B).

#include <cstddef>
#include <cstdint>
#include "llvm/MC/MCInst.h"
#include "llvm/MCA/Instruction.h"

// Adjusts the instruction descriptor built from `MCI`: shift-less register-shifted
// ALU ops take one cycle, and (unconditional or conditional) branches are free.
void applyA55InstructionAdaptations(llvm::mca::Instruction &Inst, const llvm::MCInst &MCI);

// Cortex-A55's non-blocking branch predictor overlaps about half of the branch
// penalty/bubble under dual-issue (approx. 0.5 cycles per iteration).  Returns
// the corrected cycle count for a region of `RegionInstrs` instructions of which
// `SteadyRetired` retired in steady state.
uint64_t applyA55BranchOverlapCorrection(uint64_t Cycles, uint64_t SteadyRetired, size_t RegionInstrs);

#endif // A55_A55_ADAPTATIONS_H
