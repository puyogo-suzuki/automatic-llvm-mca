#ifndef FACILE_N2_CORRECTIONS_H
#define FACILE_N2_CORRECTIONS_H

// Corrections applied on top of NeoverseN2Model for the Cortex-A78, where the
// borrowed model disagrees with the A78 SOG (docs.md section 4.D).  Enabled per
// CPU by CpuTraits::N2ModelCorrections.

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MCA/Instruction.h"

namespace facile {

enum class A78ResOverride { None, OccupancyCap1, LDPSW, DivCap5 };

A78ResOverride classifyA78ResOverride(const llvm::MCSubtargetInfo &STI, const llvm::MCInstrInfo &MCII,
                                      const llvm::mca::Instruction &Inst);

// True for integer register-offset loads/stores that use no integer (I) pipe on the A78.
bool isA78RegOffsetNoAlu(const llvm::MCSubtargetInfo &STI, const llvm::MCInstrInfo &MCII,
                         const llvm::mca::Instruction &Inst, const llvm::MCInst *MCI);

} // namespace facile

#endif // FACILE_N2_CORRECTIONS_H
