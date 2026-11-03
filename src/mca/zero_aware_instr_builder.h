#ifndef MCA_ZERO_AWARE_INSTR_BUILDER_H
#define MCA_ZERO_AWARE_INSTR_BUILDER_H

// llvm::mca::InstrBuilder caches the InstrDesc of a non-variant scheduling class by
// (opcode, scheduling class id).  A descriptor built from an instruction whose source
// operand is a constant register (WZR / XZR) has no Read for that operand, and every later
// instruction of the same key built by the same InstrBuilder inherits that: `mul w1, w1, w2`
// is `madd w1, w1, w2, wzr`, so a following `madd w1, w3, w4, w1` lost its accumulator read
// (and `str wzr, [x0]` followed by `str w9, [x0]` lost the data read).  Which instruction came
// first in the region then decided the dependences.
//
// This wrapper keeps one InstrBuilder per pattern of constant-register operands, so an
// instruction is always built by a builder whose cache only holds descriptors made from
// instructions with the same pattern.  The builders live as long as the wrapper: an
// mca::Instruction refers to its InstrDesc, which the builder owns.  Create the wrapper before
// the instructions built with it, so that it is destroyed after them.

#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/MCA/InstrBuilder.h"
#include "llvm/MCA/Instruction.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <memory>

class ZeroAwareInstrBuilder {
public:
  ZeroAwareInstrBuilder(const llvm::MCSubtargetInfo &STI, const llvm::MCInstrInfo &MCII,
                        const llvm::MCRegisterInfo &MRI, const llvm::MCInstrAnalysis *MCIA,
                        llvm::mca::InstrumentManager &IM, unsigned CallLatency)
      : STI(STI), MCII(MCII), MRI(MRI), MCIA(MCIA), IM(IM), CallLatency(CallLatency) {}

  llvm::Expected<std::unique_ptr<llvm::mca::Instruction>>
  createInstruction(const llvm::MCInst &MCI, const llvm::SmallVector<llvm::mca::Instrument *> &IVec) {
    return builderFor(MCI).createInstruction(MCI, IVec);
  }

private:
  // Bit I is set when operand I is a constant register.
  uint64_t constantRegisterMask(const llvm::MCInst &MCI) const {
    uint64_t Mask = 0;
    for (unsigned I = 0, E = MCI.getNumOperands(); I < E && I < 64; ++I) {
      const llvm::MCOperand &Op = MCI.getOperand(I);
      if (Op.isReg() && Op.getReg() && MRI.isConstant(Op.getReg()))
        Mask |= uint64_t(1) << I;
    }
    return Mask;
  }

  llvm::mca::InstrBuilder &builderFor(const llvm::MCInst &MCI) {
    std::unique_ptr<llvm::mca::InstrBuilder> &B = Builders[constantRegisterMask(MCI)];
    if (!B)
      B = std::make_unique<llvm::mca::InstrBuilder>(STI, MCII, MRI, MCIA, IM, CallLatency);
    return *B;
  }

  const llvm::MCSubtargetInfo &STI;
  const llvm::MCInstrInfo &MCII;
  const llvm::MCRegisterInfo &MRI;
  const llvm::MCInstrAnalysis *MCIA;
  llvm::mca::InstrumentManager &IM;
  unsigned CallLatency;
  std::map<uint64_t, std::unique_ptr<llvm::mca::InstrBuilder>> Builders;
};

#endif // MCA_ZERO_AWARE_INSTR_BUILDER_H
