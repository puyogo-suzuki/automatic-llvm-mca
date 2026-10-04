#include "mlp/mlp_analyzer.h"
#include "frontend/options.h"
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

namespace {

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
        if (is_load && !opts::DisableAlwaysHitLoadsHeuristic) {
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


// ---------------------------------------------------------------------------
// Always-hit heuristic diagnostic/ablation flags -- status as of 2026-09-08
// ---------------------------------------------------------------------------
// countPotentialMissLoads (the load_instructions column) and compute_mlp (the
// mlp column) both need to decide, per load, whether to treat it as an
// unconditional cache hit ("always-hit") and therefore exclude it from their
// respective counts/averages. There are two independent always-hit rules,
// each unconditional by default with its own dedicated opt-out flag:
//
//   - stack/frame-pointer access (sp/wsp/x29/w29/fp base register): always
//     excluded by default; -no-stack-exclusion turns this off.
//   - line reuse (a load repeating the same base-register+cache-line already
//     seen in the analyzed window): also excluded by default as of
//     2026-09-08; -disable-always-hit-loads-heuristic turns this off. NOTE: line reuse is
//     enabled by default only in the OOO code paths (--dependency ooo). The
//     non-OOO / in-order paths (--dependency dependency|io|none, i.e. the
//     cortex-a55 small-core configuration) apply only the stack rule unless
//     -line-reuse-in-order is passed; see that flag's description.
//
// Other diagnostic/ablation flags around this decision, kept for future
// ablation experiments but never a production default:
//
//   -stack-only-miss-load-count (+ -stack-const-offset-only,
//   -stack-spill-only, -stack-loop-resident)
//       countPotentialMissLoads-only diagnostic family that decomposes the
//       stack-exclusion and line-reuse-exclusion rules so each can be
//       studied in isolation. Diagnostic only, never a production default.
//
//   -no-stack-exclusion
//       Removes stack/frame-pointer always-hit treatment entirely, in both
//       countPotentialMissLoads and compute_mlp. NOT ADOPTED; kept for ablation.
//
// Both functions share the same "plain mode" decision -- i.e. the case where
// -stack-only-miss-load-count (which only exists for countPotentialMissLoads)
// is not in effect: a stack/frame-pointer access is unconditionally
// always-hit (unless -no-stack-exclusion), and a same-cache-line repeat
// access is always-hit unless the caller opts out via
// -disable-always-hit-loads-heuristic (see each call site below for how that
// flag is threaded through). classifyPlainModeAlwaysHit() centralizes that
// shared decision so it isn't duplicated between the two functions.
enum class PlainModeAlwaysHit {
    Miss,        // not an always-hit: counts as a potential-miss / MLP load
    StackAccess, // unconditional stack/frame-pointer always-hit
    LineReuse,   // same-cache-line repeat access (only when enabled by caller)
};

static PlainModeAlwaysHit classifyPlainModeAlwaysHit(const MLPInstInfo &inst_info,
                                                     unsigned base_reg,
                                                     const SeenBaseRegs &seen_base_regs,
                                                     bool lineReuseExclusionEnabled) {
    if (!opts::NoStackExclusion && inst_info.mem_info.is_stack_access()) {
        return PlainModeAlwaysHit::StackAccess;
    }
    if (lineReuseExclusionEnabled && base_reg != 0 && inst_info.mem_info.is_constant_offset()) {
        int64_t cache_line = inst_info.mem_info.offset / 64;
        if (seen_base_regs.test(base_reg, cache_line)) {
            return PlainModeAlwaysHit::LineReuse;
        }
    }
    return PlainModeAlwaysHit::Miss;
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
            // Plain mode: see classifyPlainModeAlwaysHit's doc comment above
            // compute_mlp for the full rationale.
            unsigned base_reg = inst_infos[i].mem_info.valid() ? inst_infos[i].mem_info.base_reg : 0;
            bool lineReuseExclusionEnabled = !opts::DisableAlwaysHitLoadsHeuristic &&
                                             (DepKind == DependencyKind::OOO || opts::LineReuseInOrder);
            PlainModeAlwaysHit hit = classifyPlainModeAlwaysHit(
                inst_infos[i], base_reg, global_seen_base_regs, lineReuseExclusionEnabled);
            if (hit == PlainModeAlwaysHit::StackAccess) {
                // Skip this load entirely: it is not added to load_indices, and
                // (via this `continue`) it is also never recorded into
                // global_seen_base_regs below, unlike countPotentialMissLoads'
                // plain mode which still records stack-access loads.
                continue;
            }
            is_hit = (hit != PlainModeAlwaysHit::Miss);
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
    for (int i : load_indices) total_mlp += 1.0 / mlp_vals[i];
    float avg_mlp = (float)(1 / (total_mlp / load_indices.size()));

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
                                           DependencyKind depKind,
                                           bool mlpWindowLoop) const {
    if (depKind == DependencyKind::OOO) {
        int n = instrs.size();
        std::vector<MLPInstInfo> inst_infos = buildInstInfos(instrs, STI, MCII, MRI, this);
        std::vector<unsigned> return_regs = getReturnRegisters(MRI, STI.getTargetTriple().getArchName().str());

        SeenBaseRegs seen_base_regs;
        SeenStoreAddrs seen_store_addrs;
        size_t non_hit_count = 0;

        // Prime the loop-carried state: with -stack-spill-only, a store near
        // the end of a loop's steady-state body can be recognized as a
        // spill/reload pair with a load near the start of the next iteration.
        bool storeAddrTrackingNeeded = opts::StackOnlyMissLoadCount && opts::StackSpillOnly;
        if (storeAddrTrackingNeeded && mlpWindowLoop) {
            for (int j = 0; j < n; ++j) {
                updateSeenBaseRegs(inst_infos[j], seen_base_regs, MRI);
                updateSeenStoreAddrs(inst_infos[j], seen_store_addrs, MRI);
                if (inst_infos[j].is_call()) {
                    for (unsigned ret_reg : return_regs) {
                        seen_base_regs.reset(ret_reg, MRI);
                        seen_store_addrs.reset(ret_reg, MRI);
                    }
                }
            }
        }

        for (int j = 0; j < n; ++j) {
            bool is_load = inst_infos[j].is_load() && inst_infos[j].mem_info.valid();
            unsigned base_reg = is_load ? inst_infos[j].mem_info.base_reg : 0;

            bool is_hit = false;
            if (is_load && opts::StackOnlyMissLoadCount && !opts::DisableAlwaysHitLoadsHeuristic) {
                // Diagnostic mode: keep only the stack/frame-pointer always-hit
                // rule and drop the same-cache-line repeat-access rule, so the two
                // components of the load_instructions filter can be told apart.
                is_hit = inst_infos[j].mem_info.is_stack_access() &&
                         (!opts::StackConstOffsetOnly ||
                          inst_infos[j].mem_info.is_constant_offset()) &&
                         (!opts::StackSpillOnly ||
                          (inst_infos[j].mem_info.is_constant_offset() &&
                           (seen_store_addrs.test(base_reg, inst_infos[j].mem_info.offset) ||
                            (opts::StackLoopResident && mlpWindowLoop))));
            } else if (is_load && !opts::StackOnlyMissLoadCount) {
                // Plain/default mode: see classifyPlainModeAlwaysHit's doc comment
                // above compute_mlp for the full rationale. This mirrors compute_mlp's
                // own plain-mode branch, except a stack-access always-hit here still
                // falls through to record the load into seen_base_regs below (via the
                // unconditional updateSeenBaseRegs call further down), whereas
                // compute_mlp skips that recording for stack accesses via `continue`.
                bool lineReuseExclusionEnabled = !opts::DisableAlwaysHitLoadsHeuristic;
                is_hit = classifyPlainModeAlwaysHit(inst_infos[j], base_reg, seen_base_regs,
                                                     lineReuseExclusionEnabled) != PlainModeAlwaysHit::Miss;
            }

            if (is_load) {
                if (is_hit) {
                    // Cache hit on same base register: do not count in BB_LOADS
                } else {
                    non_hit_count++;
                }
            }

            updateSeenBaseRegs(inst_infos[j], seen_base_regs, MRI);
            if (storeAddrTrackingNeeded) {
                updateSeenStoreAddrs(inst_infos[j], seen_store_addrs, MRI);
            }
            if (inst_infos[j].is_call()) {
                for (unsigned ret_reg : return_regs) {
                    seen_base_regs.reset(ret_reg, MRI);
                    seen_store_addrs.reset(ret_reg, MRI);
                }
            }
        }
        return non_hit_count;
    }

    // Non-OOO (in-order) path. By default this only applies the stack/frame-pointer
    // always-hit rule and walks the instructions without any cache-line tracking
    // state. With -line-reuse-in-order it additionally applies the same
    // line-reuse always-hit rule as the OOO path above, reusing the same
    // SeenBaseRegs machinery and the same classifyPlainModeAlwaysHit() decision so
    // the two paths cannot drift apart. The load set itself (which instructions are
    // considered loads at all) is deliberately identical between the two variants,
    // so -line-reuse-in-order changes nothing except the line-reuse exclusion.
    if (opts::LineReuseInOrder && !opts::DisableAlwaysHitLoadsHeuristic) {
        int n = instrs.size();
        std::vector<MLPInstInfo> inst_infos = buildInstInfos(instrs, STI, MCII, MRI, this);
        std::vector<unsigned> return_regs = getReturnRegisters(MRI, STI.getTargetTriple().getArchName().str());

        SeenBaseRegs seen_base_regs;
        size_t non_hit_count = 0;
        for (int j = 0; j < n; ++j) {
            if (inst_infos[j].mem_info.is_load()) {
                unsigned base_reg = inst_infos[j].mem_info.valid() ? inst_infos[j].mem_info.base_reg : 0;
                if (classifyPlainModeAlwaysHit(inst_infos[j], base_reg, seen_base_regs,
                                               /*lineReuseExclusionEnabled=*/true) ==
                    PlainModeAlwaysHit::Miss) {
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
        if (mem_info.is_load() && (opts::NoStackExclusion || !mem_info.is_stack_access())) {
            count++;
        }
    }
    return count;
}
