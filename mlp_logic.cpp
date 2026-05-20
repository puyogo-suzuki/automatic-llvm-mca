#include "mca_common.h"
#include <algorithm>

using namespace llvm;

namespace {

struct RegDeps {
    std::vector<unsigned> inputs;
    std::vector<unsigned> outputs;
};

static bool has_intersection(const std::vector<unsigned>& regs, const RegSet& mask) {
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
                    int width, unsigned reg_count) {
    RegSet load_dep_regs(reg_count);

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
                   unsigned reg_count) {
    RegSet load_dep_regs(reg_count);
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
                  const llvm::MCRegisterInfo& MRI) {

    int n = instrs.size();
    if (n == 0) return 1.0;
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
    if (load_indices.empty()) return 1.0;

    std::vector<float> mlp_vals(n, 1.0);

    switch (DepKind) {
        case DependencyKind::None:
            for (int i : load_indices) {
                const float score = static_cast<float>(count_loads_no_dependency(is_load, i, n, width));
                assign_mlp_score(mlp_vals, is_load, i, n, width, score, AssignKind);
            }
            break;
        case DependencyKind::OOO:
            for (int i : load_indices) {
                const float score = static_cast<float>(count_loads_ooo(is_load, io_regs, i, n, width, reg_count));
                assign_mlp_score(mlp_vals, is_load, i, n, width, score, AssignKind);
            }
            break;
        case DependencyKind::IO:
            for (int i : load_indices) {
                const float score = static_cast<float>(count_loads_io(is_load, io_regs, i, n, width, reg_count));
                assign_mlp_score(mlp_vals, is_load, i, n, width, score, AssignKind);
            }
            break;
    }
    double total_mlp = 0;
    for (int i : load_indices) total_mlp += mlp_vals[i];
    return (float)(total_mlp / load_indices.size());
}
