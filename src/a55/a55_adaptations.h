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
// AArch64SchedA55.td, kept as a no-op safety net).  Branches are left as the
// model gives them (SOG 4.2: B 1, throughput 1; docs/a55-sog.md C-12).
void applyA55InstructionAdaptations(llvm::mca::Instruction &Inst, const llvm::MCInst &MCI);

// Returns `Cycles` unchanged: no branch-overlap correction is applied (the SOG
// gives none; docs/a55-sog.md C-12).
uint64_t applyA55BranchOverlapCorrection(uint64_t Cycles, uint64_t SteadyRetired, size_t RegionInstrs);

#endif // A55_A55_ADAPTATIONS_H
