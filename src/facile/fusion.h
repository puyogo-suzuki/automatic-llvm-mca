#ifndef FACILE_FUSION_H
#define FACILE_FUSION_H

// Adjacent-pair macro-op fusion (docs/docs.md section 4.C).  Each core has its own SOG
// table, selected by CpuTraits::Fusion.

#include <memory>
#include <vector>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MCA/Instruction.h"

namespace facile {

// One entry per instruction: true if that instruction is the half of a fused pair
// that disappears (and must be excluded from issue-width / port accounting).
std::vector<bool> computeFusedMask(const llvm::MCSubtargetInfo &STI,
                                   const llvm::MCInstrInfo &MCII,
                                   llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                                   llvm::ArrayRef<const llvm::MCInst *> MCInsts);

} // namespace facile

#endif // FACILE_FUSION_H
