#include "mca_common.h"
#include <algorithm>
#include <set>

using namespace llvm;

static bool has_intersection(const std::vector<unsigned>& regs, const RegSet& mask) {
    for (unsigned r : regs) if (mask.test(r % 1024)) return true;
    return false;
}

float compute_mlp(llvm::ArrayRef<Instr> instrs, int width, 
                  DependencyKind DepKind, 
                  MLPWindowAssignmentKind AssignKind, 
                  const llvm::MCInstrInfo& MCII) {

    int n = instrs.size();
    if (n == 0) return 1.0;
    std::vector<bool> is_load(n, false);
    struct RegDeps { std::vector<unsigned> inputs, outputs; };
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
    RegSet load_dep_regs;

    for (int i : load_indices) {
        int count = 0;
        switch (DepKind) {
            case DependencyKind::None:
                for (int j = i; j < std::min(n, i + width); ++j) if (is_load[j]) count++;
                break;
            case DependencyKind::OOO:
                load_dep_regs.reset();
                for (int j = i; j < std::min(n, i + width); ++j) {
                    bool is_dep = has_intersection(io_regs[j].inputs, load_dep_regs);
                    if (is_load[j]) {
                        if (!is_dep) count++;
                        for (unsigned reg : io_regs[j].outputs) load_dep_regs.set(reg % 1024);
                    } else if (!io_regs[j].outputs.empty()) {
                        if (is_dep) for (unsigned reg : io_regs[j].outputs) load_dep_regs.set(reg % 1024);
                        else for (unsigned reg : io_regs[j].outputs) load_dep_regs.reset(reg % 1024);
                    }
                }
                break;
            case DependencyKind::IO:
                load_dep_regs.reset();
                for (unsigned reg : io_regs[i].outputs) load_dep_regs.set(reg % 1024);
                count = 1;
                for (int j = i + 1; j < std::min(n, i + width); ++j) {
                    if (has_intersection(io_regs[j].inputs, load_dep_regs)) break;
                    if (is_load[j]) {
                        count++;
                        for (unsigned reg : io_regs[j].outputs) load_dep_regs.set(reg % 1024);
                    } else {
                        if (has_intersection(io_regs[j].inputs, load_dep_regs)) {
                            for (unsigned reg : io_regs[j].outputs) load_dep_regs.set(reg % 1024);
                        } else {
                            for (unsigned reg : io_regs[j].outputs) load_dep_regs.reset(reg % 1024);
                        }
                    }
                }
                break;
        }
        
        float score = (float)count;
        if (AssignKind == MLPWindowAssignmentKind::Forward) {
            mlp_vals[i] = score;
        } else { // MaxContaining
            for (int j = i; j < std::min(n, i + width); ++j) {
                if (is_load[j]) mlp_vals[j] = std::max(mlp_vals[j], score);
            }
        }
    }
    double total_mlp = 0;
    for (int i : load_indices) total_mlp += mlp_vals[i];
    return (float)(total_mlp / load_indices.size());
}
