#ifndef MCA_IN_ORDER_ISSUE_HELPERS_H
#define MCA_IN_ORDER_ISSUE_HELPERS_H

// Helpers shared by the project's in-order issue stages (A55DecoupledIssueStage,
// src/a55/a55_issue_stage.cpp, and A520InOrderIssueStage,
// src/a520/a520_issue_stage.cpp).  They are the file-local helpers of
// llvm::mca::InOrderIssueStage of LLVM 22.1.8
// (llvm/lib/MCA/Stages/InOrderIssueStage.cpp; part of the LLVM Project, under
// the Apache License v2.0 with LLVM Exceptions, see https://llvm.org/LICENSE.txt),
// unchanged.

#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MCA/HardwareUnits/RegisterFile.h"
#include "llvm/MCA/HardwareUnits/ResourceManager.h"
#include "llvm/MCA/Instruction.h"
#include <algorithm>

namespace llvm {
namespace mca {
namespace in_order_issue {

// True if a resource the instruction needs is not available this cycle.
inline bool hasResourceHazard(const ResourceManager &RM, const InstRef &IR) {
  return RM.checkAvailability(IR.getInstruction()->getDesc()) != 0;
}

// Cycles until the instruction's first register write-back (its latency if it
// writes no register).
inline unsigned findFirstWriteBackCycle(const InstRef &IR) {
  unsigned FirstWBCycle = IR.getInstruction()->getLatency();
  for (const WriteState &WS : IR.getInstruction()->getDefs()) {
    int CyclesLeft = WS.getCyclesLeft();
    if (CyclesLeft == UNKNOWN_CYCLES)
      CyclesLeft = WS.getLatency();
    if (CyclesLeft < 0)
      CyclesLeft = 0;
    FirstWBCycle = std::min(FirstWBCycle, (unsigned)CyclesLeft);
  }
  return FirstWBCycle;
}

// Registers the instruction's reads and writes with the register file.
inline void addRegisterReadWrite(RegisterFile &PRF, Instruction &IS, unsigned SourceIndex,
                                 const MCSubtargetInfo &STI,
                                 SmallVectorImpl<unsigned> &UsedRegs) {
  for (ReadState &RS : IS.getUses())
    PRF.addRegisterRead(RS, STI);

  for (WriteState &WS : IS.getDefs())
    PRF.addRegisterWrite(WriteRef(SourceIndex, &WS), UsedRegs);
}

} // namespace in_order_issue
} // namespace mca
} // namespace llvm

#endif // MCA_IN_ORDER_ISSUE_HELPERS_H
