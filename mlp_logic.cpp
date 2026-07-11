#include "mca_common.h"
#include <algorithm>
#include <bitset>
#include <cmath>
#include <string>
#include <iostream>
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

namespace {

static void finalizeMemAccessInfo(MemAccessInfo &info, const MCInstrDesc &MCID) {
    info.set_is_load(MCID.mayLoad());
    info.set_is_store(MCID.mayStore());
}

static bool has_intersection(const llvm::SmallVectorImpl<unsigned>& regs, const RegSet& mask) {
    for (unsigned r : regs) {
        if (r < mask.size() && mask.test(r)) return true;
    }
    return false;
}

static void set_reg(RegSet &mask, unsigned reg, const llvm::MCRegisterInfo &MRI) {
    if (reg == 0) return;
    for (llvm::MCRegAliasIterator AI(reg, &MRI, true); AI.isValid(); ++AI) {
        unsigned alias = *AI;
        if (alias < mask.size()) mask.set(alias);
    }
}

static void reset_reg(RegSet &mask, unsigned reg, const llvm::MCRegisterInfo &MRI) {
    if (reg == 0) return;
    for (llvm::MCRegAliasIterator AI(reg, &MRI, true); AI.isValid(); ++AI) {
        unsigned alias = *AI;
        if (alias < mask.size()) mask.reset(alias);
    }
}

int window_end(int i, int n, int width) {
    return std::min(n, i + width);
}

static int getNumMicroOps(const llvm::MCInst &Inst, const llvm::MCSubtargetInfo &STI, const llvm::MCInstrInfo &MCII) {
    const llvm::MCSchedModel &SM = STI.getSchedModel();
    const llvm::MCInstrDesc &Desc = MCII.get(Inst.getOpcode());
    unsigned SchedClass = Desc.getSchedClass();
    unsigned ResolvedSchedClass = SchedClass;

    if (SchedClass < SM.NumSchedClasses) {
        const llvm::MCSchedClassDesc *SCDesc = SM.getSchedClassDesc(ResolvedSchedClass);
        unsigned PrevSchedClass = ResolvedSchedClass;
        while (SCDesc && SCDesc->isVariant()) {
            ResolvedSchedClass = STI.resolveVariantSchedClass(ResolvedSchedClass, &Inst, &MCII, SM.getProcessorID());
            if (ResolvedSchedClass == PrevSchedClass) {
                break;
            }
            PrevSchedClass = ResolvedSchedClass;
            SCDesc = SM.getSchedClassDesc(ResolvedSchedClass);
        }
        if (SCDesc && SCDesc->isValid()) {
            return SCDesc->NumMicroOps;
        }
    }
    return 1;
}

int count_loads_no_dependency(const std::vector<MLPInstInfo> &inst_infos, int i, int n, int width, bool mlpWindowLoop) {
    int count = 0;
    int uops_sum = 0;
    int step_limit = mlpWindowLoop ? width : (n - i);
    for (int step = 0; step < step_limit; ++step) {
        int j = (i + step) % n;
        int inst_uops = std::max(1, static_cast<int>(inst_infos[j].num_uops));
        if (uops_sum + inst_uops > width) {
            if (step > 0) {
                break;
            }
        }
        uops_sum += inst_uops;
        if (inst_infos[j].is_load()) ++count;
    }
    return count;
}

int count_loads_ooo(const std::vector<MLPInstInfo> &inst_infos, int i, int n,
                    int width, RegSet &load_dep_regs, float &ratio, bool mlpWindowLoop,
                    const llvm::MCRegisterInfo &MRI, const std::vector<unsigned> &return_regs) {
    load_dep_regs.reset();
    SeenBaseRegs seen_base_regs;

    // Helper: update seen_base_regs and load_dep_regs after processing instruction j.
    auto update_registers = [&](int j, bool is_valid_load, unsigned base_reg, bool is_dep) {
        updateSeenBaseRegs(inst_infos[j], seen_base_regs, MRI);
        for (unsigned reg : inst_infos[j].io_regs.outputs) {
            if (is_dep)
                set_reg(load_dep_regs, reg, MRI);
            else if (is_valid_load) {
                if (reg == base_reg && inst_infos[j].mem_info.is_writeback())
                    reset_reg(load_dep_regs, reg, MRI);
                else
                    set_reg(load_dep_regs, reg, MRI);
            } else
                reset_reg(load_dep_regs, reg, MRI);
        }
        if (inst_infos[j].is_call()) {
            for (unsigned ret_reg : return_regs) {
                seen_base_regs.reset(ret_reg, MRI);
                reset_reg(load_dep_regs, ret_reg, MRI);
            }
        }
    };

    // Scan the instruction window.
    // seen_base_regs tracks base registers used by loads within the current window;
    // a load whose base was already seen in the same window is treated as a cache hit
    // and not counted.
    int count_indep = 0;
    int count_dep = 0;
    int uops_sum = 0;

    for (int step = 0;; ++step) {
        if (!mlpWindowLoop && (i + step) >= n) break;
        int j = (i + step) % n;
        uops_sum += std::max(1, static_cast<int>(inst_infos[j].num_uops));
        // The instruction window is filled; stop (but always process at least one instruction).
        if (uops_sum > width && step > 0) break;

        bool is_dep = has_intersection(inst_infos[j].io_regs.inputs, load_dep_regs);
        bool is_load = inst_infos[j].is_load() && inst_infos[j].mem_info.valid();
        unsigned base_reg = is_load ? inst_infos[j].mem_info.base_reg : 0;
        


        bool is_hit = false;
        if (is_load) {
            if (base_reg != 0 && inst_infos[j].mem_info.is_constant_offset()) {
                int64_t cache_line = inst_infos[j].mem_info.offset / 64;
                is_hit = seen_base_regs.test(base_reg, cache_line);
            }
        }

        if (is_load) {
            if (is_hit) {
                // Cache hit on a base register already seen in this window: skip.
            } else {
                if (is_dep) {
                    ++count_dep;
                } else {
                    ++count_indep;
                }
            }
        }

        update_registers(j, is_load, base_reg, is_dep);
    }
    int total = count_indep + count_dep;
    ratio = (total == 0) ? -1.0f : static_cast<float>(count_indep) / total;
    return count_indep;
}

int count_loads_io(const std::vector<MLPInstInfo> &inst_infos, int i, int n,
                   int width, RegSet &load_dep_regs, float &ratio, bool mlpWindowLoop,
                   const llvm::MCRegisterInfo &MRI, const std::vector<unsigned> &return_regs) {
    load_dep_regs.reset();

    int count_indep = 0;
    int count_dep = 0;
    bool barrier_hit = false;

    if (inst_infos[i].is_load()) {
        count_indep = 1;
        for (unsigned reg : inst_infos[i].io_regs.outputs) set_reg(load_dep_regs, reg, MRI);
    }

    int uops_sum = std::max(1, static_cast<int>(inst_infos[i].num_uops));
    int step_limit = mlpWindowLoop ? width : (n - i);
    for (int step = 1; step < step_limit; ++step) {
        int j = (i + step) % n;
        int inst_uops = std::max(1, static_cast<int>(inst_infos[j].num_uops));
        if (uops_sum + inst_uops > width) {
            break;
        }
        uops_sum += inst_uops;

        if (inst_infos[j].is_call()) {
            for (unsigned ret_reg : return_regs) {
                reset_reg(load_dep_regs, ret_reg, MRI);
            }
        }

        if (barrier_hit) {
            if (inst_infos[j].is_load()) {
                ++count_dep;
            }
        } else {
            if (has_intersection(inst_infos[j].io_regs.inputs, load_dep_regs)) {
                barrier_hit = true;
            }
            if (inst_infos[j].is_load()) {
                ++count_indep;
                for (unsigned reg : inst_infos[j].io_regs.outputs) set_reg(load_dep_regs, reg, MRI);
            } else {
                if (has_intersection(inst_infos[j].io_regs.inputs, load_dep_regs)) {
                    for (unsigned reg : inst_infos[j].io_regs.outputs) set_reg(load_dep_regs, reg, MRI);
                } else {
                    for (unsigned reg : inst_infos[j].io_regs.outputs) reset_reg(load_dep_regs, reg, MRI);
                }
            }
        }
    }

    int total = count_indep + count_dep;
    ratio = (total == 0) ? -1.0f : static_cast<float>(count_indep) / total;
    return count_indep;
}

int count_loads_dependency(const std::vector<MLPInstInfo> &inst_infos, int i, int n, int width,
                           RegSet &load_dep_regs, bool mlpWindowLoop,
                           const llvm::MCRegisterInfo &MRI, const std::vector<unsigned> &return_regs) {
    load_dep_regs.reset();
    for (unsigned reg : inst_infos[i].io_regs.outputs) set_reg(load_dep_regs, reg, MRI);

    int count = 1;
    int limit = mlpWindowLoop ? n : (n - i);
    bool used = false;
    for (int step = 1; step < limit; ++step) {
        int j = (i + step) % n;
        if (has_intersection(inst_infos[j].io_regs.inputs, load_dep_regs)) {
            used = true;
            break;
        }
        ++count;
        for (unsigned reg : inst_infos[j].io_regs.outputs) {
            reset_reg(load_dep_regs, reg, MRI);
        }
        if (inst_infos[j].is_call()) {
            for (unsigned ret_reg : return_regs) {
                reset_reg(load_dep_regs, ret_reg, MRI);
            }
        }
        if (load_dep_regs.none()) break;
    }
    if (!used) return -1;
    return count;
}

void assign_mlp_score(std::vector<float> &mlp_vals, const std::vector<MLPInstInfo> &inst_infos, int i, int n, int width,
                      float score, MLPWindowAssignmentKind assign_kind, bool mlpWindowLoop) {
    if (assign_kind == MLPWindowAssignmentKind::Forward) {
        mlp_vals[i] = score;
        return;
    }

    int uops_sum = 0;
    int step_limit = mlpWindowLoop ? width : (n - i);
    for (int step = 0; step < step_limit; ++step) {
        int j = (i + step) % n;
        int inst_uops = std::max(1, static_cast<int>(inst_infos[j].num_uops));
        if (uops_sum + inst_uops > width && step > 0) break;
        uops_sum += inst_uops;
        if (inst_infos[j].is_load()) mlp_vals[j] = std::max(mlp_vals[j], score);
    }
}

}  // namespace

bool SeenBaseRegs::test(unsigned reg, int64_t cache_line) const {
    for (const auto &pair : data) {
        if (pair.first == reg && pair.second == cache_line) {
            return true;
        }
    }
    return false;
}

void SeenBaseRegs::set(unsigned reg, int64_t cache_line, const llvm::MCRegisterInfo &MRI) {
    if (reg == 0) return;
    for (llvm::MCRegAliasIterator AI(reg, &MRI, true); AI.isValid(); ++AI) {
        unsigned alias = *AI;
        bool found = false;
        for (const auto &pair : data) {
            if (pair.first == alias && pair.second == cache_line) {
                found = true;
                break;
            }
        }
        if (!found) {
            data.push_back({alias, cache_line});
        }
    }
}

void SeenBaseRegs::reset(unsigned reg, const llvm::MCRegisterInfo &MRI) {
    if (reg == 0) return;
    for (llvm::MCRegAliasIterator AI(reg, &MRI, true); AI.isValid(); ++AI) {
        unsigned alias = *AI;
        data.erase(std::remove_if(data.begin(), data.end(),
                                  [alias](const std::pair<unsigned, int64_t> &p) {
                                      return p.first == alias;
                                  }),
                   data.end());
    }
}

void updateSeenBaseRegs(const MLPInstInfo &inst_info, SeenBaseRegs &seen_base_regs, const llvm::MCRegisterInfo &MRI) {
    bool is_valid_load = inst_info.is_load() && inst_info.mem_info.valid();
    unsigned base_reg = is_valid_load ? inst_info.mem_info.base_reg : 0;
    if (is_valid_load && base_reg != 0 && inst_info.mem_info.is_constant_offset()) {
        int64_t cache_line = inst_info.mem_info.offset / 64;
        for (unsigned reg : inst_info.io_regs.outputs) {
            for (llvm::MCRegAliasIterator AI(base_reg, &MRI, true); AI.isValid(); ++AI) {
                if (*AI == reg) goto end_base_regs_set;
            }
        }
        seen_base_regs.set(base_reg, cache_line, MRI);
    }
  end_base_regs_set:
    for (unsigned reg : inst_info.io_regs.outputs) {
        seen_base_regs.reset(reg, MRI);
    }
}

std::vector<unsigned> getReturnRegisters(const llvm::MCRegisterInfo &MRI, const std::string &ArchName) {
    std::vector<unsigned> reg_ids;
    StringRef arch = ArchName;
    if (arch.contains_insensitive("aarch64")) {
        unsigned num_regs = MRI.getNumRegs();
        for (unsigned r = 1; r < num_regs; ++r) {
            StringRef name = MRI.getName(r);
            if (name.empty()) continue;
            char first = name[0];
            if (first >= 'A' && first <= 'Z') {
                first = first - 'A' + 'a';
            }
            if (first == 'x' || first == 'w' || first == 'v' || first == 's' || first == 'd' || first == 'h' || first == 'b' || first == 'q') {
                StringRef num_part = name.drop_front(1);
                unsigned val;
                if (!num_part.getAsInteger(10, val) && val <= 7) {
                    reg_ids.push_back(r);
                }
            }
        }
    } else if (arch.contains_insensitive("x86")) {
        unsigned num_regs = MRI.getNumRegs();
        for (unsigned r = 1; r < num_regs; ++r) {
            StringRef name = MRI.getName(r);
            if (name.empty()) continue;
            if (name.equals_insensitive("rax") || name.equals_insensitive("eax") || name.equals_insensitive("ax") || name.equals_insensitive("al") || name.equals_insensitive("ah") ||
                name.equals_insensitive("rdx") || name.equals_insensitive("edx") || name.equals_insensitive("dx") || name.equals_insensitive("dl") || name.equals_insensitive("dh") ||
                name.equals_insensitive("rcx") || name.equals_insensitive("ecx") || name.equals_insensitive("cx") || name.equals_insensitive("cl") || name.equals_insensitive("ch")) {
                reg_ids.push_back(r);
            } else if (name.starts_with_insensitive("xmm") || name.starts_with_insensitive("ymm") || name.starts_with_insensitive("zmm")) {
                StringRef num_part = name.drop_front(3);
                unsigned val;
                if (!num_part.getAsInteger(10, val) && val <= 7) {
                    reg_ids.push_back(r);
                }
            }
        }
    } else if (arch.contains_insensitive("riscv")) {
        unsigned num_regs = MRI.getNumRegs();
        for (unsigned r = 1; r < num_regs; ++r) {
            StringRef name = MRI.getName(r);
            if (name.empty()) continue;
            if (name.equals_insensitive("a0") || name.equals_insensitive("x10") ||
                name.equals_insensitive("a1") || name.equals_insensitive("x11") ||
                name.equals_insensitive("fa0") || name.equals_insensitive("f10") ||
                name.equals_insensitive("fa1") || name.equals_insensitive("f11")) {
                reg_ids.push_back(r);
            }
        }
    }
    return reg_ids;
}

std::vector<MLPInstInfo> buildInstInfos(
    llvm::ArrayRef<Instr> instrs,
    const llvm::MCSubtargetInfo& STI,
    const llvm::MCInstrInfo& MCII,
    const llvm::MCRegisterInfo& MRI,
    const MLPAnalyzer* Analyzer) {
    int n = instrs.size();
    std::vector<MLPInstInfo> inst_infos(n);
    for (int i = 0; i < n; ++i) {
        const MCInst& Inst = instrs[i].Inst;
        const MCInstrDesc& MCID = MCII.get(Inst.getOpcode());
        MemAccessInfo mem_info = Analyzer->getMemAccessInfo(Inst, MCID, MRI, MCII);
        inst_infos[i].set_is_load(mem_info.is_load());
        inst_infos[i].set_is_store(mem_info.is_store());
        bool is_call = MCID.isCall();
        if (!is_call) {
            StringRef name = MCII.getName(Inst.getOpcode());
            if (name.equals_insensitive("JAL") || name.equals_insensitive("JALR")) {
                if (Inst.getNumOperands() > 0 && Inst.getOperand(0).isReg()) {
                    unsigned rd = Inst.getOperand(0).getReg();
                    if (const char* rd_name = MRI.getName(rd)) {
                        StringRef rdn(rd_name);
                        if (rdn.equals_insensitive("x1") || rdn.equals_insensitive("x5") ||
                            rdn.equals_insensitive("ra") || rdn.equals_insensitive("t0")) {
                            is_call = true;
                        }
                    }
                }
            }
        }
        inst_infos[i].set_is_call(is_call);

        inst_infos[i].num_uops = static_cast<short>(getNumMicroOps(Inst, STI, MCII));
        inst_infos[i].mem_info = mem_info;
        for (unsigned j = 0; j < Inst.getNumOperands(); ++j) {
            const MCOperand& Op = Inst.getOperand(j);
            if (Op.isReg() && Op.getReg() != 0) {
                unsigned reg = Op.getReg();
                if (Analyzer->isZeroRegister(reg, MRI)) continue;
                if (j < MCID.getNumDefs()) inst_infos[i].io_regs.outputs.push_back(reg);
                else inst_infos[i].io_regs.inputs.push_back(reg);
            }
        }
        for (MCPhysReg R : MCID.implicit_defs()) {
            if (!Analyzer->isZeroRegister(R, MRI)) inst_infos[i].io_regs.outputs.push_back(R);
        }
        for (MCPhysReg R : MCID.implicit_uses()) {
            if (!Analyzer->isZeroRegister(R, MRI)) inst_infos[i].io_regs.inputs.push_back(R);
        }
    }
    return inst_infos;
}

bool MLPAnalyzer::isZeroRegister(unsigned reg, const llvm::MCRegisterInfo &MRI) const {
    if (reg == 0) return true;
    if (const char* name = MRI.getName(reg)) {
        std::string s(name);
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return (s == "zero");
    }
    return false;
}

bool RISCVMLPAnalyzer::isZeroRegister(unsigned reg, const llvm::MCRegisterInfo &MRI) const {
    if (reg == 0) return true;
    if (const char* name = MRI.getName(reg)) {
        std::string s(name);
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return (s == "zero" || s == "x0");
    }
    return false;
}

bool X86MLPAnalyzer::isZeroRegister(unsigned reg, const llvm::MCRegisterInfo &MRI) const {
    return (reg == 0);
}

bool AArch64MLPAnalyzer::isZeroRegister(unsigned reg, const llvm::MCRegisterInfo &MRI) const {
    if (reg == 0) return true;
    if (const char* name = MRI.getName(reg)) {
        std::string s(name);
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return (s == "xzr" || s == "wzr" || s == "zero");
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

float MLPAnalyzer::compute_mlp(llvm::ArrayRef<Instr> instrs, int width, 
                              DependencyKind DepKind, 
                              MLPWindowAssignmentKind AssignKind, 
                              const llvm::MCSubtargetInfo& STI,
                              const llvm::MCInstrInfo& MCII,
                              const llvm::MCRegisterInfo& MRI,
                              float &mlp_r,
                              bool mlpWindowLoop) const {

    int n = instrs.size();
    if (n == 0) {
        mlp_r = 1.0f;
        return 1.0f;
    }

    const unsigned reg_count = MRI.getNumRegs() + 1;
    std::vector<MLPInstInfo> inst_infos = buildInstInfos(instrs, STI, MCII, MRI, this);
    std::vector<unsigned> return_regs = getReturnRegisters(MRI, STI.getTargetTriple().getArchName().str());
    std::vector<int> load_indices;
    SeenBaseRegs global_seen_base_regs;
    for (int i = 0; i < n; ++i) {
        bool is_hit = false;
        if (inst_infos[i].is_load()) {
            if (inst_infos[i].mem_info.is_stack_access()) {
                continue;
            }
            if (DepKind == DependencyKind::OOO) {
                unsigned base_reg = inst_infos[i].mem_info.valid() ? inst_infos[i].mem_info.base_reg : 0;
                if (base_reg != 0 && inst_infos[i].mem_info.is_constant_offset()) {
                    int64_t cache_line = inst_infos[i].mem_info.offset / 64;
                    is_hit = global_seen_base_regs.test(base_reg, cache_line);
                }
            }
            if (!is_hit) {
                load_indices.push_back(i);
            }
        }
        updateSeenBaseRegs(inst_infos[i], global_seen_base_regs, MRI);
        if (inst_infos[i].is_call()) {
            for (unsigned ret_reg : return_regs) {
                global_seen_base_regs.reset(ret_reg, MRI);
            }
        }
    }
    bool actual_window_loop = mlpWindowLoop;
    if (load_indices.empty()) {
        mlp_r = 1.0f;
        return 1.0f;
    }

    std::vector<float> mlp_vals(n, 1.0f);
    std::vector<float> mlp_r_vals(n, -1.0f);
    RegSet load_dep_regs(reg_count);

    switch (DepKind) {
        case DependencyKind::None:
            for (int i : load_indices) {
                float score = static_cast<float>(count_loads_no_dependency(inst_infos, i, n, width, actual_window_loop));
                assign_mlp_score(mlp_vals, inst_infos, i, n, width, score, AssignKind, actual_window_loop);
                assign_mlp_score(mlp_r_vals, inst_infos, i, n, width, 1.0f, AssignKind, actual_window_loop);
            }
            break;
        case DependencyKind::OOO:
            for (int i : load_indices) {
                float score_r = 0.0f;
                int score = count_loads_ooo(inst_infos, i, n, width, load_dep_regs, score_r, actual_window_loop, MRI, return_regs);
                assign_mlp_score(mlp_vals, inst_infos, i, n, width, static_cast<float>(score), AssignKind, actual_window_loop);
                assign_mlp_score(mlp_r_vals, inst_infos, i, n, width, score_r, AssignKind, actual_window_loop);
            }
            break;
        case DependencyKind::IO:
            for (int i : load_indices) {
                float score_r = 0.0f;
                int score = count_loads_io(inst_infos, i, n, width, load_dep_regs, score_r, actual_window_loop, MRI, return_regs);
                assign_mlp_score(mlp_vals, inst_infos, i, n, width, static_cast<float>(score), AssignKind, actual_window_loop);
                assign_mlp_score(mlp_r_vals, inst_infos, i, n, width, score_r, AssignKind, actual_window_loop);
            }
            break;
        case DependencyKind::Dependency:
        {
            std::vector<int> valid_load_indices;
            for (int i : load_indices) {
                int score_val = count_loads_dependency(inst_infos, i, n, width, load_dep_regs, actual_window_loop, MRI, return_regs);
                if (score_val >= 0) {
                    float score = static_cast<float>(score_val);
                    assign_mlp_score(mlp_vals, inst_infos, i, n, width, score, AssignKind, actual_window_loop);
                    assign_mlp_score(mlp_r_vals, inst_infos, i, n, width, 1.0f, AssignKind, actual_window_loop);
                    valid_load_indices.push_back(i);
                }
            }
            load_indices = valid_load_indices;
            break;
        }
    }
    if (load_indices.empty()) {
        mlp_r = 1.0f;
        return 1.0f;
    }
    double total_mlp = 0;
    for (int i : load_indices) total_mlp += mlp_vals[i];
    float avg_mlp = (float)(total_mlp / load_indices.size());

    double total_mlp_r = 0;
    int count_r = 0;
    for (int i : load_indices) {
        if (mlp_r_vals[i] >= 0.0f) {
            total_mlp_r += mlp_r_vals[i];
            count_r++;
        }
    }
    mlp_r = (count_r == 0) ? 0.0f : (float)(total_mlp_r / count_r);

    return avg_mlp;
}

size_t MLPAnalyzer::countPotentialMissLoads(llvm::ArrayRef<Instr> instrs,
                                           const llvm::MCSubtargetInfo& STI,
                                           const llvm::MCInstrInfo& MCII,
                                           const llvm::MCRegisterInfo& MRI,
                                           DependencyKind depKind) const {
    if (depKind == DependencyKind::OOO) {
        int n = instrs.size();
        std::vector<MLPInstInfo> inst_infos = buildInstInfos(instrs, STI, MCII, MRI, this);
        std::vector<unsigned> return_regs = getReturnRegisters(MRI, STI.getTargetTriple().getArchName().str());

        SeenBaseRegs seen_base_regs;
        size_t non_hit_count = 0;

        for (int j = 0; j < n; ++j) {
            bool is_load = inst_infos[j].is_load() && inst_infos[j].mem_info.valid();
            unsigned base_reg = is_load ? inst_infos[j].mem_info.base_reg : 0;
            
            bool is_hit = false;
            if (is_load) {
                if (base_reg != 0 && inst_infos[j].mem_info.is_constant_offset()) {
                    int64_t cache_line = inst_infos[j].mem_info.offset / 64;
                    is_hit = seen_base_regs.test(base_reg, cache_line);
                }
            }

            if (is_load) {
                if (is_hit) {
                    // Cache hit on same base register: do not count in BB_LOADS
                } else {
                    non_hit_count++;
                }
            }

            updateSeenBaseRegs(inst_infos[j], seen_base_regs, MRI);
            if (inst_infos[j].is_call()) {
                for (unsigned ret_reg : return_regs) {
                    seen_base_regs.reset(ret_reg, MRI);
                }
            }
        }
        return non_hit_count;
    }

    size_t count = 0;
    for (const auto &I : instrs) {
        const MCInst& Inst = I.Inst;
        const MCInstrDesc& MCID = MCII.get(Inst.getOpcode());
        MemAccessInfo mem_info = getMemAccessInfo(Inst, MCID, MRI, MCII);
        if (mem_info.is_load()) {
            count++;
        }
    }
    return count;
}

std::unique_ptr<MLPAnalyzer> MLPAnalyzer::create(const llvm::MCSubtargetInfo &STI) {
    std::string Arch = STI.getTargetTriple().getArchName().str();
    std::transform(Arch.begin(), Arch.end(), Arch.begin(), ::tolower);
    if (Arch.rfind("riscv", 0) == 0) {
        return std::make_unique<RISCVMLPAnalyzer>();
    } else if (Arch.rfind("x86", 0) == 0 || Arch == "i386") {
        return std::make_unique<X86MLPAnalyzer>();
    } else {
        return std::make_unique<AArch64MLPAnalyzer>();
    }
}
