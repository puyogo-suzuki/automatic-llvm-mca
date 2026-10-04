#include "mlp/mlp_analyzer.h"
#include <algorithm>
#include <bitset>
#include <cmath>
#include <iostream>
#include <string>
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

bool RISCVMLPAnalyzer::isZeroRegister(unsigned reg, const llvm::MCRegisterInfo &MRI) const {
    if (reg == 0) return true;
    if (const char* name = MRI.getName(reg)) {
        std::string s(name);
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return (s == "zero" || s == "x0");
    }
    return false;
}

MemAccessInfo RISCVMLPAnalyzer::getMemAccessInfo(const MCInst &Inst, const MCInstrDesc &MCID, const MCRegisterInfo &MRI, const MCInstrInfo &MCII) const {
    MemAccessInfo info;
    if (!MCID.mayLoad() && !MCID.mayStore()) return info;
    unsigned num_ops = Inst.getNumOperands();
    if (num_ops == 0) return info;

    for (unsigned i = 0; i < num_ops; ++i) {
        if (Inst.getOperand(i).isExpr()) {
            info.set_valid(true);
            info.set_is_pc_relative(true);
            finalizeMemAccessInfo(info, MCID);
            return info;
        }
    }

    if (num_ops >= 3 && Inst.getOperand(0).isReg() && 
        Inst.getOperand(1).isReg() && Inst.getOperand(2).isImm()) {
        info.base_reg = Inst.getOperand(1).getReg();
        info.offset = Inst.getOperand(2).getImm();
        info.set_valid(true);
        info.set_is_constant_offset(true);
    }

    if (info.valid() && info.base_reg != 0) {
        if (const char* reg_name = MRI.getName(info.base_reg)) {
            std::string name(reg_name);
            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            // RISC-V Stack pointer names
            if (name == "sp" || name == "x2") {
                info.set_is_stack_access(true);
            }
        }
    }

    finalizeMemAccessInfo(info, MCID);
    return info;
}
