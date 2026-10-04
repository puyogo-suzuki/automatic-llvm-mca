#ifndef CORE_REGION_SPLITTER_H
#define CORE_REGION_SPLITTER_H

// Splits a function's instructions into loop and basic-block regions.

#include <functional>
#include "core/types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"

void walkRegions(llvm::ArrayRef<Instr> instrs, const FunctionBoundaries &boundaries,
                 const std::function<void(const RegionSpan &)> &onLoop,
                 const std::function<void(const RegionSpan &)> &onBasicBlock);

bool isNopInstruction(const llvm::MCInst &Inst, const llvm::MCInstrInfo &MCII);
bool isAllNopRegion(llvm::ArrayRef<Instr> instrs, const llvm::MCInstrInfo &MCII);

#endif // CORE_REGION_SPLITTER_H
