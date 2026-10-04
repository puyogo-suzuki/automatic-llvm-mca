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

bool X86MLPAnalyzer::isZeroRegister(unsigned reg, const llvm::MCRegisterInfo &MRI) const {
    return (reg == 0);
}

MemAccessInfo X86MLPAnalyzer::getMemAccessInfo(const MCInst &Inst, const MCInstrDesc &MCID, const MCRegisterInfo &MRI, const MCInstrInfo &MCII) const {
    MemAccessInfo info;
    if (!MCID.mayLoad() && !MCID.mayStore()) return info;
    unsigned num_ops = Inst.getNumOperands();
    if (num_ops == 0) return info;

    std::string Name = std::string(MCII.getName(Inst.getOpcode()));
    std::transform(Name.begin(), Name.end(), Name.begin(), ::tolower);
    if (Name.rfind("pop", 0) == 0 || Name.rfind("push", 0) == 0) {
        info.set_valid(true);
        info.set_is_stack_access(true);
        info.set_is_constant_offset(true);
        // Find RSP/ESP/SP register using MRI
        unsigned sp_reg = 0;
        for (unsigned r = 1; r < MRI.getNumRegs(); ++r) {
            if (const char* rname = MRI.getName(r)) {
                std::string rn(rname);
                std::transform(rn.begin(), rn.end(), rn.begin(), ::tolower);
                if (rn == "rsp" || rn == "esp" || rn == "sp") {
                    sp_reg = r;
                    break;
                }
            }
        }
        info.base_reg = sp_reg;
        finalizeMemAccessInfo(info, MCID);
        return info;
    }

    for (unsigned i = 0; i < num_ops; ++i) {
        if (Inst.getOperand(i).isExpr()) {
            info.set_valid(true);
            info.set_is_pc_relative(true);
            info.set_is_constant_offset(true);
            finalizeMemAccessInfo(info, MCID);
            return info;
        }
    }

    for (unsigned i = 0; i + 4 < num_ops; ++i) {
        if (Inst.getOperand(i).isReg() && 
            Inst.getOperand(i+1).isImm() && 
            Inst.getOperand(i+2).isReg() && 
            (Inst.getOperand(i+3).isImm() || Inst.getOperand(i+3).isExpr()) && 
            Inst.getOperand(i+4).isReg()) {
            
            info.base_reg = Inst.getOperand(i).getReg();
            unsigned index_reg = Inst.getOperand(i+2).getReg();
            if (Inst.getOperand(i+3).isImm()) {
                info.offset = Inst.getOperand(i+3).getImm();
                info.set_is_constant_offset(true);
            } else {
                info.set_is_pc_relative(true);
                info.set_is_constant_offset(true);
            }
            info.set_valid(true);
            break;
        }
    }

    if (info.valid() && info.base_reg != 0) {
        if (const char* reg_name = MRI.getName(info.base_reg)) {
            std::string name(reg_name);
            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            // x86 Stack & Frame pointer names
            if (name == "rsp" || name == "esp" || name == "sp" || 
                name == "rbp" || name == "ebp" || name == "bp") {
                info.set_is_stack_access(true);
            }
        }
    }

    finalizeMemAccessInfo(info, MCID);
    return info;
}
