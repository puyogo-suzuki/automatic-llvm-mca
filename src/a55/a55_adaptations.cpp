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
    if (false && (opc == llvm::AArch64::Bcc || opc == llvm::AArch64::B)) {  // [a55_td_fix U] disabled: no SOG basis (SOG 4.2: B 1, 1/cycle)
        llvm::mca::InstrDesc &MutableDesc = const_cast<llvm::mca::InstrDesc &>(Inst.getDesc());
        for (auto &W : MutableDesc.Writes) {
            W.Latency = 0;
        }
        MutableDesc.MaxLatency = 0;
        MutableDesc.Resources.clear();
        MutableDesc.UsedProcResUnits = 0;
        MutableDesc.UsedProcResGroups = 0;
        MutableDesc.UsedBuffers = 0;
    }
}

uint64_t applyA55BranchOverlapCorrection(uint64_t Cycles, uint64_t SteadyRetired, size_t RegionInstrs) {
    return Cycles;  // [a55_td_fix U] -0.5 cycle/iteration correction disabled (no SOG basis)
    if (RegionInstrs == 0)
        return Cycles;
    unsigned NumSteadyIterations = SteadyRetired / RegionInstrs;
    if (NumSteadyIterations > 0) {
        double correctedCycles = static_cast<double>(Cycles) - (static_cast<double>(NumSteadyIterations) * 0.5);
        if (correctedCycles < 1.0) correctedCycles = 1.0;
        return static_cast<unsigned>(correctedCycles + 0.5);
    }
    return Cycles;
}
