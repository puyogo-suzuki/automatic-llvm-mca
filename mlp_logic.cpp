#include "mca_common.h"
#include <algorithm>
#include <bitset>
#include <cmath>
#include <string>
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

namespace {

struct RegDeps {
    llvm::SmallVector<unsigned, 4> inputs;
    llvm::SmallVector<unsigned, 4> outputs;
};

struct MemAccessInfo {
    bool valid = false;
    unsigned base_reg = 0;
    int64_t offset = 0;
    bool is_pc_relative = false;
    bool is_stack_access = false;
    bool is_load = false;
    bool is_store = false;
};

struct MLPInstInfo {
    std::bitset<2> flags; // bit 0: is_load, bit 1: is_store
    short num_uops = 1;
    RegDeps io_regs;
    MemAccessInfo mem_info;

    bool is_load() const { return flags.test(0); }
    bool is_store() const { return flags.test(1); }
    void set_is_load(bool val) { flags.set(0, val); }
    void set_is_store(bool val) { flags.set(1, val); }
};

using MemAccessDecoderFn = MemAccessInfo (*)(const MCInst &Inst, const MCInstrDesc &MCID, const MCRegisterInfo &MRI);

static void finalizeMemAccessInfo(MemAccessInfo &info, const MCInstrDesc &MCID) {
    info.is_load = MCID.mayLoad();
    info.is_store = MCID.mayStore();

    // Ignore stack pointer-based memory operations entirely as they won't miss in cache
    if (info.is_stack_access) {
        info.is_load = false;
        info.is_store = false;
    }
}

static MemAccessInfo getMemAccessInfoRISCV(const MCInst &Inst, const MCInstrDesc &MCID, const MCRegisterInfo &MRI) {
    MemAccessInfo info;
    if (!MCID.mayLoad() && !MCID.mayStore()) return info;
    unsigned num_ops = Inst.getNumOperands();
    if (num_ops == 0) return info;

    for (unsigned i = 0; i < num_ops; ++i) {
        if (Inst.getOperand(i).isExpr()) {
            info.valid = true;
            info.is_pc_relative = true;
            finalizeMemAccessInfo(info, MCID);
            return info;
        }
    }

    if (num_ops >= 3 && Inst.getOperand(0).isReg() && 
        Inst.getOperand(1).isReg() && Inst.getOperand(2).isImm()) {
        info.base_reg = Inst.getOperand(1).getReg();
        info.offset = Inst.getOperand(2).getImm();
        info.valid = true;
    }

    if (info.valid && info.base_reg != 0) {
        if (const char* reg_name = MRI.getName(info.base_reg)) {
            std::string name(reg_name);
            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            // RISC-V Stack pointer names
            if (name == "sp" || name == "x2") {
                info.is_stack_access = true;
            }
        }
    }

    finalizeMemAccessInfo(info, MCID);
    return info;
}

static MemAccessInfo getMemAccessInfoX86(const MCInst &Inst, const MCInstrDesc &MCID, const MCRegisterInfo &MRI) {
    MemAccessInfo info;
    if (!MCID.mayLoad() && !MCID.mayStore()) return info;
    unsigned num_ops = Inst.getNumOperands();
    if (num_ops == 0) return info;

    for (unsigned i = 0; i < num_ops; ++i) {
        if (Inst.getOperand(i).isExpr()) {
            info.valid = true;
            info.is_pc_relative = true;
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
            if (Inst.getOperand(i+3).isImm()) {
                info.offset = Inst.getOperand(i+3).getImm();
            } else {
                info.is_pc_relative = true;
            }
            info.valid = true;
            break;
        }
    }

    if (info.valid && info.base_reg != 0) {
        if (const char* reg_name = MRI.getName(info.base_reg)) {
            std::string name(reg_name);
            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            // x86 Stack & Frame pointer names
            if (name == "rsp" || name == "esp" || name == "sp" || 
                name == "rbp" || name == "ebp" || name == "bp") {
                info.is_stack_access = true;
            }
        }
    }

    finalizeMemAccessInfo(info, MCID);
    return info;
}

static MemAccessInfo getMemAccessInfoAArch64(const MCInst &Inst, const MCInstrDesc &MCID, const MCRegisterInfo &MRI) {
    MemAccessInfo info;
    if (!MCID.mayLoad() && !MCID.mayStore()) return info;
    unsigned num_ops = Inst.getNumOperands();
    if (num_ops == 0) return info;

    for (unsigned i = 0; i < num_ops; ++i) {
        if (Inst.getOperand(i).isExpr()) {
            info.valid = true;
            info.is_pc_relative = true;
            finalizeMemAccessInfo(info, MCID);
            return info;
        }
    }

    // Pattern A: LDP/STP (Pair)
    if (num_ops >= 4 && Inst.getOperand(0).isReg() && Inst.getOperand(1).isReg() && 
        Inst.getOperand(2).isReg() && Inst.getOperand(3).isImm()) {
        info.base_reg = Inst.getOperand(2).getReg();
        info.offset = Inst.getOperand(3).getImm();
        info.valid = true;
    }
    // Pattern B: LDR/STR (Single)
    else if (num_ops >= 3 && Inst.getOperand(0).isReg() && 
             Inst.getOperand(1).isReg() && Inst.getOperand(2).isImm()) {
        info.base_reg = Inst.getOperand(1).getReg();
        info.offset = Inst.getOperand(2).getImm();
        info.valid = true;
    }
    // Pattern C: LDR/STR post/pre-index
    else if (num_ops >= 4 && Inst.getOperand(0).isReg() && Inst.getOperand(1).isReg() && 
             Inst.getOperand(2).isReg() && Inst.getOperand(3).isImm()) {
        info.base_reg = Inst.getOperand(2).getReg();
        info.offset = Inst.getOperand(3).getImm();
        info.valid = true;
    }

    if (info.valid && info.base_reg != 0) {
        if (const char* reg_name = MRI.getName(info.base_reg)) {
            std::string name(reg_name);
            std::transform(name.begin(), name.end(), name.begin(), ::tolower);
            // ARM/AArch64 Stack pointer names
            if (name == "sp" || name == "wsp") {
                info.is_stack_access = true;
            }
        }
    }

    finalizeMemAccessInfo(info, MCID);
    return info;
}

static bool isIndependentMemoryAccess(const MemAccessInfo &load_info, const MemAccessInfo &store_info) {
    if (!load_info.valid || !store_info.valid) {
        return false;
    }
    // Pattern 3: Stack access vs Heap/Global access separation.
    if (load_info.is_stack_access != store_info.is_stack_access) {
        return true;
    }
    // Pattern 1: Same base register but different offsets.
    if (load_info.base_reg == store_info.base_reg &&
        !store_info.is_pc_relative &&
        load_info.offset != store_info.offset) {
        return true;
    }
    return false;
}

static bool hasStoreDependency(const MemAccessInfo &load_info, const std::vector<MemAccessInfo> &seen_stores) {
    // Check is_pc_relative once before looping over all stores
    if (load_info.valid && load_info.is_pc_relative) {
        return false;
    }
    for (const auto &store_info : seen_stores) {
        if (!isIndependentMemoryAccess(load_info, store_info)) {
            return true;
        }
    }
    return false;
}

static bool has_intersection(const llvm::SmallVectorImpl<unsigned>& regs, const RegSet& mask) {
    for (unsigned r : regs) {
        if (r < mask.size() && mask.test(r)) return true;
    }
    return false;
}

static void set_reg(RegSet &mask, unsigned reg) {
    if (reg < mask.size()) mask.set(reg);
}

static void reset_reg(RegSet &mask, unsigned reg) {
    if (reg < mask.size()) mask.reset(reg);
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
                    int width, RegSet &load_dep_regs, float &ratio, bool mlpWindowLoop) {
    load_dep_regs.reset();

    int count_indep = 0;
    int count_dep = 0;
    int uops_sum = 0;
    int step_limit = mlpWindowLoop ? width : (n - i);
    std::vector<MemAccessInfo> seen_stores;

    for (int step = 0; step < step_limit; ++step) {
        int j = (i + step) % n;
        int inst_uops = std::max(1, static_cast<int>(inst_infos[j].num_uops));
        if (uops_sum + inst_uops > width) {
            if (step > 0) {
                break;
            }
        }
        uops_sum += inst_uops;

        if (inst_infos[j].is_store()) {
            seen_stores.push_back(inst_infos[j].mem_info);
        }

        bool is_dep = has_intersection(inst_infos[j].io_regs.inputs, load_dep_regs);
        if (inst_infos[j].is_load() && hasStoreDependency(inst_infos[j].mem_info, seen_stores)) {
            is_dep = true;
        }

        if (inst_infos[j].is_load()) {
            if (is_dep) {
                ++count_dep;
            } else {
                ++count_indep;
            }
            for (unsigned reg : inst_infos[j].io_regs.outputs) set_reg(load_dep_regs, reg);
        } else if (!inst_infos[j].io_regs.outputs.empty()) {
            if (is_dep) {
                for (unsigned reg : inst_infos[j].io_regs.outputs) set_reg(load_dep_regs, reg);
            } else {
                for (unsigned reg : inst_infos[j].io_regs.outputs) reset_reg(load_dep_regs, reg);
            }
        }
    }
    int total = count_indep + count_dep;
    ratio = (total == 0) ? -1.0f : static_cast<float>(count_indep) / total;
    return count_indep;
}

int count_loads_io(const std::vector<MLPInstInfo> &inst_infos, int i, int n,
                   int width, RegSet &load_dep_regs, float &ratio, bool mlpWindowLoop) {
    load_dep_regs.reset();

    int count_indep = 0;
    int count_dep = 0;
    bool barrier_hit = false;

    if (inst_infos[i].is_load()) {
        count_indep = 1;
        for (unsigned reg : inst_infos[i].io_regs.outputs) set_reg(load_dep_regs, reg);
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

        if (!barrier_hit) {
            if (has_intersection(inst_infos[j].io_regs.inputs, load_dep_regs)) {
                barrier_hit = true;
            }
        }

        if (barrier_hit) {
            if (inst_infos[j].is_load()) {
                ++count_dep;
            }
        } else {
            if (inst_infos[j].is_load()) {
                ++count_indep;
                for (unsigned reg : inst_infos[j].io_regs.outputs) set_reg(load_dep_regs, reg);
            } else {
                if (has_intersection(inst_infos[j].io_regs.inputs, load_dep_regs)) {
                    for (unsigned reg : inst_infos[j].io_regs.outputs) set_reg(load_dep_regs, reg);
                } else {
                    for (unsigned reg : inst_infos[j].io_regs.outputs) reset_reg(load_dep_regs, reg);
                }
            }
        }
    }

    int total = count_indep + count_dep;
    ratio = (total == 0) ? -1.0f : static_cast<float>(count_indep) / total;
    return count_indep;
}

int count_loads_dependency(const std::vector<MLPInstInfo> &inst_infos, int i, int n, int width,
                           RegSet &load_dep_regs, bool mlpWindowLoop) {
    load_dep_regs.reset();
    for (unsigned reg : inst_infos[i].io_regs.outputs) set_reg(load_dep_regs, reg);

    int count = 1;
    int limit = mlpWindowLoop ? n : (n - i);
    for (int step = 1; step < limit; ++step) {
        int j = (i + step) % n;
        if (has_intersection(inst_infos[j].io_regs.inputs, load_dep_regs)) break;
        ++count;
        if (inst_infos[j].is_load()) {
            for (unsigned reg : inst_infos[j].io_regs.outputs) set_reg(load_dep_regs, reg);
        } else {
            for (unsigned reg : inst_infos[j].io_regs.outputs) reset_reg(load_dep_regs, reg);
        }
    }
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
        if (uops_sum + inst_uops > width) {
            if (step > 0) {
                break;
            }
        }
        uops_sum += inst_uops;
        if (inst_infos[j].is_load()) mlp_vals[j] = std::max(mlp_vals[j], score);
    }
}

}  // namespace

float compute_mlp(llvm::ArrayRef<Instr> instrs, int width, 
                  DependencyKind DepKind, 
                  MLPWindowAssignmentKind AssignKind, 
                  const llvm::MCSubtargetInfo& STI,
                  const llvm::MCInstrInfo& MCII,
                  const llvm::MCRegisterInfo& MRI,
                  float &mlp_r,
                  bool mlpWindowLoop) {

    int n = instrs.size();
    if (n == 0) {
        mlp_r = 1.0f;
        return 1.0f;
    }

    // Resolve target triple and select the appropriate decoder function pointer
    std::string arch = STI.getTargetTriple().getArchName().str();
    std::transform(arch.begin(), arch.end(), arch.begin(), ::tolower);

    MemAccessDecoderFn decoder = nullptr;
    if (arch.rfind("riscv", 0) == 0) {
        decoder = getMemAccessInfoRISCV;
    } else if (arch.rfind("x86", 0) == 0 || arch == "i386") {
        decoder = getMemAccessInfoX86;
    } else {
        decoder = getMemAccessInfoAArch64;
    }

    const unsigned reg_count = MRI.getNumRegs() + 1;
    std::vector<MLPInstInfo> inst_infos(n);
    std::vector<int> load_indices;
    for (int i = 0; i < n; ++i) {
        const MCInst& Inst = instrs[i].Inst;
        const MCInstrDesc& MCID = MCII.get(Inst.getOpcode());
        MemAccessInfo mem_info = decoder(Inst, MCID, MRI);
        
        inst_infos[i].set_is_load(mem_info.is_load);
        inst_infos[i].set_is_store(mem_info.is_store);
        inst_infos[i].num_uops = static_cast<short>(getNumMicroOps(Inst, STI, MCII));
        inst_infos[i].mem_info = mem_info;
        if (inst_infos[i].is_load()) load_indices.push_back(i);
        for (unsigned j = 0; j < Inst.getNumOperands(); ++j) {
            const MCOperand& Op = Inst.getOperand(j);
            if (Op.isReg() && Op.getReg() != 0) {
                if (j < MCID.getNumDefs()) inst_infos[i].io_regs.outputs.push_back(Op.getReg());
                else inst_infos[i].io_regs.inputs.push_back(Op.getReg());
            }
        }
        for (MCPhysReg R : MCID.implicit_defs()) inst_infos[i].io_regs.outputs.push_back(R);
        for (MCPhysReg R : MCID.implicit_uses()) inst_infos[i].io_regs.inputs.push_back(R);
    }
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
                float score = static_cast<float>(count_loads_no_dependency(inst_infos, i, n, width, mlpWindowLoop));
                assign_mlp_score(mlp_vals, inst_infos, i, n, width, score, AssignKind, mlpWindowLoop);
                assign_mlp_score(mlp_r_vals, inst_infos, i, n, width, 1.0f, AssignKind, mlpWindowLoop);
            }
            break;
        case DependencyKind::OOO:
            for (int i : load_indices) {
                float score_r = 0.0f;
                int score = count_loads_ooo(inst_infos, i, n, width, load_dep_regs, score_r, mlpWindowLoop);
                assign_mlp_score(mlp_vals, inst_infos, i, n, width, static_cast<float>(score), AssignKind, mlpWindowLoop);
                assign_mlp_score(mlp_r_vals, inst_infos, i, n, width, score_r, AssignKind, mlpWindowLoop);
            }
            break;
        case DependencyKind::IO:
            for (int i : load_indices) {
                float score_r = 0.0f;
                int score = count_loads_io(inst_infos, i, n, width, load_dep_regs, score_r, mlpWindowLoop);
                assign_mlp_score(mlp_vals, inst_infos, i, n, width, static_cast<float>(score), AssignKind, mlpWindowLoop);
                assign_mlp_score(mlp_r_vals, inst_infos, i, n, width, score_r, AssignKind, mlpWindowLoop);
            }
            break;
        case DependencyKind::Dependency:
            for (int i : load_indices) {
                float score = static_cast<float>(count_loads_dependency(inst_infos, i, n, width, load_dep_regs, mlpWindowLoop));
                assign_mlp_score(mlp_vals, inst_infos, i, n, width, score, AssignKind, mlpWindowLoop);
                assign_mlp_score(mlp_r_vals, inst_infos, i, n, width, 1.0f, AssignKind, mlpWindowLoop);
            }
            break;
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

size_t countNonStackLoads(llvm::ArrayRef<Instr> instrs,
                          const llvm::MCSubtargetInfo& STI,
                          const llvm::MCInstrInfo& MCII,
                          const llvm::MCRegisterInfo& MRI) {
    std::string arch = STI.getTargetTriple().getArchName().str();
    std::transform(arch.begin(), arch.end(), arch.begin(), ::tolower);

    MemAccessDecoderFn decoder = nullptr;
    if (arch.rfind("riscv", 0) == 0) {
        decoder = getMemAccessInfoRISCV;
    } else if (arch.rfind("x86", 0) == 0 || arch == "i386") {
        decoder = getMemAccessInfoX86;
    } else {
        decoder = getMemAccessInfoAArch64;
    }

    size_t count = 0;
    for (const auto &I : instrs) {
        const MCInst& Inst = I.Inst;
        const MCInstrDesc& MCID = MCII.get(Inst.getOpcode());
        MemAccessInfo mem_info = decoder(Inst, MCID, MRI);
        if (mem_info.is_load) {
            count++;
        }
    }
    return count;
}

