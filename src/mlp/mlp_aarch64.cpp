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

bool AArch64MLPAnalyzer::isZeroRegister(unsigned reg, const llvm::MCRegisterInfo &MRI) const {
    if (reg == 0) return true;
    if (const char* name = MRI.getName(reg)) {
        std::string s(name);
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return (s == "xzr" || s == "wzr" || s == "zero");
    }
    return false;
}

static int64_t getAArch64Scale(const std::string &Name) {
    // Check pairs
    if (Name.find("ldpx") != std::string::npos || Name.find("stpx") != std::string::npos ||
        Name.find("ldnpx") != std::string::npos || Name.find("stnpx") != std::string::npos) {
        return 8;
    }
    if (Name.find("ldpw") != std::string::npos || Name.find("stpw") != std::string::npos ||
        Name.find("ldnpw") != std::string::npos || Name.find("stnpw") != std::string::npos) {
        return 4;
    }
    if (Name.find("ldpd") != std::string::npos || Name.find("stpd") != std::string::npos) return 8;
    if (Name.find("ldps") != std::string::npos || Name.find("stps") != std::string::npos) return 4;
    if (Name.find("ldpq") != std::string::npos || Name.find("stpq") != std::string::npos) return 16;

    // Check singles
    if (Name.find("ldrx") != std::string::npos || Name.find("strx") != std::string::npos) return 8;
    if (Name.find("ldrw") != std::string::npos || Name.find("strw") != std::string::npos || Name.find("ldrsw") != std::string::npos) return 4;
    if (Name.find("ldrh") != std::string::npos || Name.find("strh") != std::string::npos || Name.find("ldrsh") != std::string::npos) return 2;
    if (Name.find("ldrb") != std::string::npos || Name.find("strb") != std::string::npos || Name.find("ldrsb") != std::string::npos) return 1;
    
    // Floating point / SIMD singles
    if (Name.find("ldrd") != std::string::npos || Name.find("strd") != std::string::npos) return 8;
    if (Name.find("ldrs") != std::string::npos || Name.find("strs") != std::string::npos) return 4;
    if (Name.find("ldrh") != std::string::npos || Name.find("strh") != std::string::npos) return 2;
    if (Name.find("ldrb") != std::string::npos || Name.find("strb") != std::string::npos) return 1;
    if (Name.find("ldrq") != std::string::npos || Name.find("strq") != std::string::npos) return 16;

    return 1; // Default fallback
}

MemAccessInfo AArch64MLPAnalyzer::getMemAccessInfo(const MCInst &Inst, const MCInstrDesc &MCID, const MCRegisterInfo &MRI, const MCInstrInfo &MCII) const {
    MemAccessInfo info;
    if (!MCID.mayLoad() && !MCID.mayStore()) return info;
    unsigned num_ops = Inst.getNumOperands();
    if (num_ops == 0) return info;

    for (unsigned i = 0; i < num_ops; ++i) {
        if (Inst.getOperand(i).isExpr()) {
            info.set_valid(true);
            info.set_is_pc_relative(true);
            info.set_is_constant_offset(true);
            finalizeMemAccessInfo(info, MCID);
            return info;
        }
    }

    std::string Name = std::string(MCII.getName(Inst.getOpcode()));
    std::transform(Name.begin(), Name.end(), Name.begin(), ::tolower);

    // 1. Literal/PC-relative single load
    if (Name.size() >= 4 && Name.rfind("ldr", 0) == 0 && Name.back() == 'l') {
        info.set_valid(true);
        info.set_is_pc_relative(true);
        info.set_is_constant_offset(true);
        finalizeMemAccessInfo(info, MCID);
        return info;
    }

    // 2. Exclusive, load-acquire, store-release (no offset, base is the last operand)
    bool is_exclusive = (Name.rfind("ldx", 0) == 0 || Name.rfind("ldax", 0) == 0 ||
                         Name.rfind("ldar", 0) == 0 || Name.rfind("stx", 0) == 0 ||
                         Name.rfind("stlx", 0) == 0 || Name.rfind("stlr", 0) == 0);
    if (is_exclusive) {
        if (num_ops > 0 && Inst.getOperand(num_ops - 1).isReg()) {
            info.base_reg = Inst.getOperand(num_ops - 1).getReg();
            info.offset = 0;
            info.set_valid(true);
            info.set_is_constant_offset(true);
            finalizeMemAccessInfo(info, MCID);
            return info;
        }
    }

    // 3. Pair Loads/Stores (LDP/STP/LDNP/STNP)
    bool is_pair = (Name.rfind("ldp", 0) == 0 || Name.rfind("stp", 0) == 0 ||
                    Name.rfind("ldnp", 0) == 0 || Name.rfind("stnp", 0) == 0);
    if (is_pair) {
        bool has_wb = (Name.find("post") != std::string::npos ||
                       Name.find("pre") != std::string::npos ||
                       Name.find("writeback") != std::string::npos);
        if (has_wb) {
            info.set_is_writeback(true);
            if (num_ops >= 5 && Inst.getOperand(3).isReg() && 
                (Inst.getOperand(4).isImm() || Inst.getOperand(4).isReg())) {
                info.base_reg = Inst.getOperand(3).getReg();
                bool is_imm = Inst.getOperand(4).isImm();
                info.offset = is_imm ? Inst.getOperand(4).getImm() * getAArch64Scale(Name) : 0;
                info.set_valid(true);
                if (is_imm) {
                    info.set_is_constant_offset(true);
                }
            }
        } else {
            if (num_ops >= 4 && Inst.getOperand(2).isReg() && Inst.getOperand(3).isImm()) {
                info.base_reg = Inst.getOperand(2).getReg();
                info.offset = Inst.getOperand(3).getImm() * getAArch64Scale(Name);
                info.set_valid(true);
                info.set_is_constant_offset(true);
            }
        }
    }
    // 4. Single Loads/Stores (LDR/STR etc.)
    else {
        bool is_reg_offset = (Name.find("row") != std::string::npos || Name.find("rox") != std::string::npos);
        if (is_reg_offset) {
            if (num_ops >= 2 && Inst.getOperand(1).isReg()) {
                info.base_reg = Inst.getOperand(1).getReg();
                info.offset = 0;
                info.set_valid(true);
                info.set_is_constant_offset(false);
            }
        } else {
            bool has_wb = (Name.find("post") != std::string::npos ||
                           Name.find("pre") != std::string::npos ||
                           Name.find("writeback") != std::string::npos);
            if (has_wb) {
                info.set_is_writeback(true);
                if (num_ops >= 4 && Inst.getOperand(2).isReg() && 
                    (Inst.getOperand(3).isImm() || Inst.getOperand(3).isReg())) {
                    info.base_reg = Inst.getOperand(2).getReg();
                    bool is_imm = Inst.getOperand(3).isImm();
                    info.offset = is_imm ? Inst.getOperand(3).getImm() * getAArch64Scale(Name) : 0;
                    info.set_valid(true);
                    if (is_imm) {
                        info.set_is_constant_offset(true);
                    }
                }
            } else {
                if (num_ops >= 3 && Inst.getOperand(1).isReg() && Inst.getOperand(2).isImm()) {
                    info.base_reg = Inst.getOperand(1).getReg();
                    info.offset = Inst.getOperand(2).getImm() * getAArch64Scale(Name);
                    info.set_valid(true);
                    info.set_is_constant_offset(true);
                }
            }
        }
    }

    if (info.valid() && info.base_reg != 0) {
        if (const char* reg_name = MRI.getName(info.base_reg)) {
            std::string name(reg_name);
            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            // ARM/AArch64 Stack and Frame pointer names
            if (name == "sp" || name == "wsp" || name == "x29" || name == "w29" || name == "fp") {
                info.set_is_stack_access(true);
            }
        }
    }

    finalizeMemAccessInfo(info, MCID);
    return info;
}
