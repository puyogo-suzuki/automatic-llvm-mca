#include "mca_common.h"
#include <algorithm>
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/MCInstrDesc.h"

using namespace llvm;

namespace {

struct RegDeps {
    llvm::SmallVector<unsigned, 4> inputs;
    llvm::SmallVector<unsigned, 4> outputs;
};

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

int count_loads_no_dependency(const std::vector<bool> &is_load, const std::vector<int> &num_uops, int i, int n, int width, bool mlpWindowLoop) {
    int count = 0;
    int uops_sum = 0;
    int step_limit = mlpWindowLoop ? width : (n - i);
    for (int step = 0; step < step_limit; ++step) {
        int j = (i + step) % n;
        int inst_uops = std::max(1, num_uops[j]);
        if (uops_sum + inst_uops > width) {
            if (step > 0) {
                break;
            }
        }
        uops_sum += inst_uops;
        if (is_load[j]) ++count;
    }
    return count;
}

int count_loads_ooo(const std::vector<bool> &is_load, const std::vector<RegDeps> &io_regs, const std::vector<int> &num_uops, int i, int n,
                    int width, RegSet &load_dep_regs, float &ratio, bool mlpWindowLoop) {
    load_dep_regs.reset();

    int count_indep = 0;
    int count_dep = 0;
    int uops_sum = 0;
    int step_limit = mlpWindowLoop ? width : (n - i);
    for (int step = 0; step < step_limit; ++step) {
        int j = (i + step) % n;
        int inst_uops = std::max(1, num_uops[j]);
        if (uops_sum + inst_uops > width) {
            if (step > 0) {
                break;
            }
        }
        uops_sum += inst_uops;

        const bool is_dep = has_intersection(io_regs[j].inputs, load_dep_regs);
        if (is_load[j]) {
            if (is_dep) {
                ++count_dep;
            } else {
                ++count_indep;
            }
            for (unsigned reg : io_regs[j].outputs) set_reg(load_dep_regs, reg);
        } else if (!io_regs[j].outputs.empty()) {
            if (is_dep) {
                for (unsigned reg : io_regs[j].outputs) set_reg(load_dep_regs, reg);
            } else {
                for (unsigned reg : io_regs[j].outputs) reset_reg(load_dep_regs, reg);
            }
        }
    }
    int total = count_indep + count_dep;
    ratio = (total == 0) ? -1.0f : static_cast<float>(count_indep) / total;
    return count_indep;
}

int count_loads_io(const std::vector<bool> &is_load, const std::vector<RegDeps> &io_regs, const std::vector<int> &num_uops, int i, int n,
                   int width, RegSet &load_dep_regs, float &ratio, bool mlpWindowLoop) {
    load_dep_regs.reset();

    int count_indep = 0;
    int count_dep = 0;
    bool barrier_hit = false;

    if (is_load[i]) {
        count_indep = 1;
        for (unsigned reg : io_regs[i].outputs) set_reg(load_dep_regs, reg);
    }

    int uops_sum = std::max(1, num_uops[i]);
    int step_limit = mlpWindowLoop ? width : (n - i);
    for (int step = 1; step < step_limit; ++step) {
        int j = (i + step) % n;
        int inst_uops = std::max(1, num_uops[j]);
        if (uops_sum + inst_uops > width) {
            break;
        }
        uops_sum += inst_uops;

        if (!barrier_hit) {
            if (has_intersection(io_regs[j].inputs, load_dep_regs)) {
                barrier_hit = true;
            }
        }

        if (barrier_hit) {
            if (is_load[j]) {
                ++count_dep;
            }
        } else {
            if (is_load[j]) {
                ++count_indep;
                for (unsigned reg : io_regs[j].outputs) set_reg(load_dep_regs, reg);
            } else {
                if (has_intersection(io_regs[j].inputs, load_dep_regs)) {
                    for (unsigned reg : io_regs[j].outputs) set_reg(load_dep_regs, reg);
                } else {
                    for (unsigned reg : io_regs[j].outputs) reset_reg(load_dep_regs, reg);
                }
            }
        }
    }

    int total = count_indep + count_dep;
    ratio = (total == 0) ? -1.0f : static_cast<float>(count_indep) / total;
    return count_indep;
}

int count_loads_dependency(const std::vector<bool> &is_load, const std::vector<RegDeps> &io_regs, int i, int n, int width,
                           RegSet &load_dep_regs, bool mlpWindowLoop) {
    load_dep_regs.reset();
    for (unsigned reg : io_regs[i].outputs) set_reg(load_dep_regs, reg);

    int count = 1;
    int limit = mlpWindowLoop ? n : (n - i);
    for (int step = 1; step < limit; ++step) {
        int j = (i + step) % n;
        if (has_intersection(io_regs[j].inputs, load_dep_regs)) break;
        ++count;
        if (is_load[j]) {
            for (unsigned reg : io_regs[j].outputs) set_reg(load_dep_regs, reg);
        } else {
            for (unsigned reg : io_regs[j].outputs) reset_reg(load_dep_regs, reg);
        }
    }
    return count;
}

void assign_mlp_score(std::vector<float> &mlp_vals, const std::vector<bool> &is_load, const std::vector<int> &num_uops, int i, int n, int width,
                      float score, MLPWindowAssignmentKind assign_kind, bool mlpWindowLoop) {
    if (assign_kind == MLPWindowAssignmentKind::Forward) {
        mlp_vals[i] = score;
        return;
    }

    int uops_sum = 0;
    int step_limit = mlpWindowLoop ? width : (n - i);
    for (int step = 0; step < step_limit; ++step) {
        int j = (i + step) % n;
        int inst_uops = std::max(1, num_uops[j]);
        if (uops_sum + inst_uops > width) {
            if (step > 0) {
                break;
            }
        }
        uops_sum += inst_uops;
        if (is_load[j]) mlp_vals[j] = std::max(mlp_vals[j], score);
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
    const unsigned reg_count = MRI.getNumRegs() + 1;
    std::vector<bool> is_load(n, false);
    std::vector<int> num_uops(n, 1);
    std::vector<RegDeps> io_regs(n);
    std::vector<int> load_indices;
    for (int i = 0; i < n; ++i) {
        const MCInst& Inst = instrs[i].Inst;
        const MCInstrDesc& MCID = MCII.get(Inst.getOpcode());
        is_load[i] = MCID.mayLoad();
        num_uops[i] = getNumMicroOps(Inst, STI, MCII);
        if (is_load[i]) load_indices.push_back(i);
        for (unsigned j = 0; j < Inst.getNumOperands(); ++j) {
            const MCOperand& Op = Inst.getOperand(j);
            if (Op.isReg() && Op.getReg() != 0) {
                if (j < MCID.getNumDefs()) io_regs[i].outputs.push_back(Op.getReg());
                else io_regs[i].inputs.push_back(Op.getReg());
            }
        }
        for (MCPhysReg R : MCID.implicit_defs()) io_regs[i].outputs.push_back(R);
        for (MCPhysReg R : MCID.implicit_uses()) io_regs[i].inputs.push_back(R);
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
                float score = static_cast<float>(count_loads_no_dependency(is_load, num_uops, i, n, width, mlpWindowLoop));
                assign_mlp_score(mlp_vals, is_load, num_uops, i, n, width, score, AssignKind, mlpWindowLoop);
                assign_mlp_score(mlp_r_vals, is_load, num_uops, i, n, width, 1.0f, AssignKind, mlpWindowLoop);
            }
            break;
        case DependencyKind::OOO:
            for (int i : load_indices) {
                float score_r = 0.0f;
                int score = count_loads_ooo(is_load, io_regs, num_uops, i, n, width, load_dep_regs, score_r, mlpWindowLoop);
                assign_mlp_score(mlp_vals, is_load, num_uops, i, n, width, static_cast<float>(score), AssignKind, mlpWindowLoop);
                assign_mlp_score(mlp_r_vals, is_load, num_uops, i, n, width, score_r, AssignKind, mlpWindowLoop);
            }
            break;
        case DependencyKind::IO:
            for (int i : load_indices) {
                float score_r = 0.0f;
                int score = count_loads_io(is_load, io_regs, num_uops, i, n, width, load_dep_regs, score_r, mlpWindowLoop);
                assign_mlp_score(mlp_vals, is_load, num_uops, i, n, width, static_cast<float>(score), AssignKind, mlpWindowLoop);
                assign_mlp_score(mlp_r_vals, is_load, num_uops, i, n, width, score_r, AssignKind, mlpWindowLoop);
            }
            break;
        case DependencyKind::Dependency:
            for (int i : load_indices) {
                float score = static_cast<float>(count_loads_dependency(is_load, io_regs, i, n, width, load_dep_regs, mlpWindowLoop));
                assign_mlp_score(mlp_vals, is_load, num_uops, i, n, width, score, AssignKind, mlpWindowLoop);
                assign_mlp_score(mlp_r_vals, is_load, num_uops, i, n, width, 1.0f, AssignKind, mlpWindowLoop);
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
