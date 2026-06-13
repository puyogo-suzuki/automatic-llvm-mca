#include "mca_common.h"
#include <algorithm>
#include "llvm/ADT/SmallVector.h"

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

int count_loads_no_dependency(const std::vector<bool> &is_load, int i, int n, int width) {
    int count = 0;
    for (int j = i; j < window_end(i, n, width); ++j) {
        if (is_load[j]) ++count;
    }
    return count;
}

int count_loads_ooo(const std::vector<bool> &is_load, const std::vector<RegDeps> &io_regs, int i, int n,
                    int width, RegSet &load_dep_regs) {
    load_dep_regs.reset();

    int count = 0;
    for (int j = i; j < window_end(i, n, width); ++j) {
        const bool is_dep = has_intersection(io_regs[j].inputs, load_dep_regs);
        if (is_load[j]) {
            if (!is_dep) ++count;
            for (unsigned reg : io_regs[j].outputs) set_reg(load_dep_regs, reg);
        } else if (!io_regs[j].outputs.empty()) {
            if (is_dep) {
                for (unsigned reg : io_regs[j].outputs) set_reg(load_dep_regs, reg);
            } else {
                for (unsigned reg : io_regs[j].outputs) reset_reg(load_dep_regs, reg);
            }
        }
    }
    return count;
}

int count_loads_io(const std::vector<bool> &is_load, const std::vector<RegDeps> &io_regs, int i, int n, int width,
                   RegSet &load_dep_regs) {
    load_dep_regs.reset();
    for (unsigned reg : io_regs[i].outputs) set_reg(load_dep_regs, reg);

    int count = 1;
    for (int j = i + 1; j < window_end(i, n, width); ++j) {
        if (has_intersection(io_regs[j].inputs, load_dep_regs)) break;
        if (is_load[j]) {
            ++count;
            for (unsigned reg : io_regs[j].outputs) set_reg(load_dep_regs, reg);
        } else {
            if (has_intersection(io_regs[j].inputs, load_dep_regs)) {
                for (unsigned reg : io_regs[j].outputs) set_reg(load_dep_regs, reg);
            } else {
                for (unsigned reg : io_regs[j].outputs) reset_reg(load_dep_regs, reg);
            }
        }
    }
    return count;
}

int count_loads_dependency(const std::vector<bool> &is_load, const std::vector<RegDeps> &io_regs, int i, int n, int width,
                           RegSet &load_dep_regs) {
    load_dep_regs.reset();
    for (unsigned reg : io_regs[i].outputs) set_reg(load_dep_regs, reg);

    int count = 1;
    for (int j = i + 1; j < window_end(i, n, width); ++j) {
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

float count_loads_ooo_r(const std::vector<bool> &is_load, const std::vector<RegDeps> &io_regs, int i, int n,
                        int width, RegSet &load_dep_regs) {
    load_dep_regs.reset();

    int count_dep = 0;
    int count_indep = 0;
    for (int j = i; j < window_end(i, n, width); ++j) {
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
    if (total == 0) return -1.0f;
    return static_cast<float>(count_indep) / total;
}

float count_loads_io_r(const std::vector<bool> &is_load, const std::vector<RegDeps> &io_regs, int i, int n,
                       int width, RegSet &load_dep_regs) {
    load_dep_regs.reset();

    int count_indep = 0;
    int count_dep = 0;
    bool barrier_hit = false;

    if (is_load[i]) {
        count_indep = 1;
        for (unsigned reg : io_regs[i].outputs) set_reg(load_dep_regs, reg);
    }

    for (int j = i + 1; j < window_end(i, n, width); ++j) {
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
    if (total == 0) return -1.0f;
    return static_cast<float>(count_indep) / total;
}

void assign_mlp_score(std::vector<float> &mlp_vals, const std::vector<bool> &is_load, int i, int n, int width,
                      float score, MLPWindowAssignmentKind assign_kind) {
    if (assign_kind == MLPWindowAssignmentKind::Forward) {
        mlp_vals[i] = score;
        return;
    }

    for (int j = i; j < window_end(i, n, width); ++j) {
        if (is_load[j]) mlp_vals[j] = std::max(mlp_vals[j], score);
    }
}

}  // namespace

float compute_mlp(llvm::ArrayRef<Instr> instrs, int width, 
                  DependencyKind DepKind, 
                  MLPWindowAssignmentKind AssignKind, 
                  const llvm::MCInstrInfo& MCII,
                  const llvm::MCRegisterInfo& MRI,
                  float &mlp_r) {

    int n = instrs.size();
    if (n == 0) {
        mlp_r = 1.0f;
        return 1.0f;
    }
    const unsigned reg_count = MRI.getNumRegs() + 1;
    std::vector<bool> is_load(n, false);
    std::vector<RegDeps> io_regs(n);
    std::vector<int> load_indices;
    for (int i = 0; i < n; ++i) {
        const MCInst& Inst = instrs[i].Inst;
        const MCInstrDesc& MCID = MCII.get(Inst.getOpcode());
        is_load[i] = MCID.mayLoad();
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
                float score = static_cast<float>(count_loads_no_dependency(is_load, i, n, width));
                assign_mlp_score(mlp_vals, is_load, i, n, width, score, AssignKind);
                assign_mlp_score(mlp_r_vals, is_load, i, n, width, 1.0f, AssignKind);
            }
            break;
        case DependencyKind::OOO:
            for (int i : load_indices) {
                float score = static_cast<float>(count_loads_ooo(is_load, io_regs, i, n, width, load_dep_regs));
                assign_mlp_score(mlp_vals, is_load, i, n, width, score, AssignKind);
                float score_r = count_loads_ooo_r(is_load, io_regs, i, n, width, load_dep_regs);
                assign_mlp_score(mlp_r_vals, is_load, i, n, width, score_r, AssignKind);
            }
            break;
        case DependencyKind::IO:
            for (int i : load_indices) {
                float score = static_cast<float>(count_loads_io(is_load, io_regs, i, n, width, load_dep_regs));
                assign_mlp_score(mlp_vals, is_load, i, n, width, score, AssignKind);
                float score_r = count_loads_io_r(is_load, io_regs, i, n, width, load_dep_regs);
                assign_mlp_score(mlp_r_vals, is_load, i, n, width, score_r, AssignKind);
            }
            break;
        case DependencyKind::Dependency:
            for (int i : load_indices) {
                float score = static_cast<float>(count_loads_dependency(is_load, io_regs, i, n, width, load_dep_regs));
                assign_mlp_score(mlp_vals, is_load, i, n, width, score, AssignKind);
                assign_mlp_score(mlp_r_vals, is_load, i, n, width, 1.0f, AssignKind);
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
