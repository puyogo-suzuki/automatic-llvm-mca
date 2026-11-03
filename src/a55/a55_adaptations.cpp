#include "a55/a55_adaptations.h"
#include "llvm-source/llvm/lib/Target/AArch64/MCTargetDesc/AArch64AddressingModes.h"
#include "llvm-source/llvm/lib/Target/AArch64/MCTargetDesc/AArch64MCTargetDesc.h"

void applyA55InstructionAdaptations(llvm::mca::Instruction &Inst, const llvm::MCInst &MCI) {
    unsigned opc = MCI.getOpcode();
    if (opc == llvm::AArch64::ADDXrs || opc == llvm::AArch64::ADDWrs ||
        opc == llvm::AArch64::SUBXrs || opc == llvm::AArch64::SUBWrs ||
        opc == llvm::AArch64::SUBSXrs || opc == llvm::AArch64::SUBSWrs ||
        opc == llvm::AArch64::ADDSXrs || opc == llvm::AArch64::ADDSWrs) {

        unsigned shiftVal = 0;
        if (MCI.getNumOperands() > 3 && MCI.getOperand(3).isImm()) {
            shiftVal = llvm::AArch64_AM::getShiftValue(MCI.getOperand(3).getImm());
        }

        if (shiftVal == 0) {
            llvm::mca::InstrDesc &MutableDesc = const_cast<llvm::mca::InstrDesc &>(Inst.getDesc());
            for (auto &W : MutableDesc.Writes) {
                if (W.Latency > 1) {
                    W.Latency = 1;
                }
            }
            MutableDesc.MaxLatency = 1;
        }
    }
    // [a55_td_fix U] Branches (B, B.cond) keep the latency and resources the model
    // gives them: the SOG gives them no special treatment (SOG 4.2: B 1, 1/cycle).
}

uint64_t applyA55BranchOverlapCorrection(uint64_t Cycles, uint64_t /*SteadyRetired*/,
                                         size_t /*RegionInstrs*/) {
    return Cycles;  // [a55_td_fix U] no -0.5 cycle/iteration correction (no SOG basis)
}
