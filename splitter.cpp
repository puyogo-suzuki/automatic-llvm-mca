#include "mca_common.h"
#include "frontend.h"
#include <algorithm>
#include <vector>
#include <set>
#include <functional>
#include <cstdlib>
#include "llvm/ADT/ArrayRef.h"

using namespace llvm;

namespace {

struct SimpleLoop {
    size_t h_idx;
    size_t l_idx;
    size_t size;
};

// Step 1: Collect function start indices from binary function boundaries
std::vector<size_t> collectFunctionStartIndices(ArrayRef<Instr> instrs,
                                                const FunctionBoundaries &boundaries) {
    std::set<size_t> fn_starts;
    fn_starts.insert(0);

    for (size_t i = 0; i < instrs.size(); ++i) {
        uint64_t addr = instrs[i].Addr;
        auto it = boundaries.upper_bound(addr);
        if (it != boundaries.begin()) {
            --it;
            if (addr == it->first) {
                fn_starts.insert(i);
            }
        }
    }

    std::vector<size_t> fn_list(fn_starts.begin(), fn_starts.end());
    fn_list.push_back(instrs.size());
    return fn_list;
}

// Step 2: Detect basic backward-branch loops within a single function
std::vector<SimpleLoop> detectBackwardBranchLoops(ArrayRef<Instr> funcInstrs) {
    std::vector<SimpleLoop> loops;
    size_t f_size = funcInstrs.size();
    if (f_size == 0) return loops;

    uint64_t f_base_addr = funcInstrs[0].Addr;
    for (size_t i = 0; i < f_size; ++i) {
        const auto &I = funcInstrs[i];
        if (I.IsBranch && !I.IsCall && I.BranchTarget != 0 && I.BranchTarget <= I.Addr) {
            if (I.BranchTarget >= f_base_addr) {
                uint64_t offset = I.BranchTarget - f_base_addr;
                if (offset % 4 == 0) {
                    size_t h_idx = offset / 4;
                    if (h_idx <= i && h_idx < f_size) {
                        loops.push_back({h_idx, i, i - h_idx + 1});
                    }
                }
            }
        }
    }
    return loops;
}

// Step 3: Detect abab interlocking chains and perform adaptive merging
std::vector<SimpleLoop> detectAndMergeAbabChains(std::vector<SimpleLoop> loops, int threshold) {
    size_t n_loops = loops.size();
    if (n_loops == 0) return {};

    std::sort(loops.begin(), loops.end(), [](const SimpleLoop &a, const SimpleLoop &b) {
        if (a.h_idx != b.h_idx) return a.h_idx < b.h_idx;
        return a.l_idx > b.l_idx;
    });

    std::vector<bool> is_abab(n_loops, false);
    std::vector<int> dp(n_loops, 1);
    int max_chain = 1;

    // Fast sweep-line to mark abab interlocking pairs and compute max chain depth
    std::vector<size_t> active_indices;
    for (size_t i = 0; i < n_loops; ++i) {
        size_t h = loops[i].h_idx;
        size_t l = loops[i].l_idx;

        size_t write_pos = 0;
        for (size_t k = 0; k < active_indices.size(); ++k) {
            size_t idx = active_indices[k];
            if (loops[idx].l_idx > h) {
                active_indices[write_pos++] = idx;
                if (loops[idx].l_idx < l) {
                    is_abab[i] = true;
                    is_abab[idx] = true;
                }
            }
        }
        active_indices.resize(write_pos);
        active_indices.push_back(i);

        int limit = std::max(0, static_cast<int>(i) - 50);
        for (int j = static_cast<int>(i) - 1; j >= limit; --j) {
            if (loops[j].h_idx < h && h < loops[j].l_idx && loops[j].l_idx < l) {
                if (dp[j] + 1 > dp[i]) dp[i] = dp[j] + 1;
            }
        }
        if (dp[i] > max_chain) max_chain = dp[i];
    }

    if (max_chain < threshold) {
        return loops;
    }

    // Merge abab interlocking loops into outer bounding spans and discard pre-merged loops
    std::vector<SimpleLoop> selected_loops;
    for (size_t i = 0; i < n_loops; ++i) {
        if (!is_abab[i]) {
            selected_loops.push_back(loops[i]);
        }
    }

    std::vector<SimpleLoop> merged_abab;
    for (size_t i = 0; i < n_loops; ++i) {
        if (!is_abab[i]) continue;
        if (merged_abab.empty()) {
            merged_abab.push_back(loops[i]);
        } else {
            auto &prev = merged_abab.back();
            if (loops[i].h_idx <= prev.l_idx) {
                prev.l_idx = std::max(prev.l_idx, loops[i].l_idx);
                prev.size = prev.l_idx - prev.h_idx + 1;
            } else {
                merged_abab.push_back(loops[i]);
            }
        }
    }

    for (const auto &ml : merged_abab) {
        selected_loops.push_back(ml);
    }

    return selected_loops;
}

} // namespace

void walkRegions(ArrayRef<Instr> instrs, const FunctionBoundaries &boundaries,
                 const std::function<void(const RegionSpan &)> &onLoop,
                 const std::function<void(const RegionSpan &)> &onBasicBlock) {
    if (instrs.empty()) return;

    std::vector<size_t> fn_list = collectFunctionStartIndices(instrs, boundaries);

    for (size_t f = 0; f < fn_list.size() - 1; ++f) {
        size_t f_start = fn_list[f];
        size_t f_end = fn_list[f + 1];
        size_t f_size = f_end - f_start;
        if (f_size == 0) continue;

        auto funcInstrs = instrs.slice(f_start, f_size);

        std::vector<SimpleLoop> raw_loops = detectBackwardBranchLoops(funcInstrs);
        std::vector<SimpleLoop> selected_loops = detectAndMergeAbabChains(raw_loops, opts::ChainThreshold);

        std::vector<bool> in_loop(f_size, false);
        for (const auto &l : selected_loops) {
            onLoop(RegionSpan{
                f_start + l.h_idx,
                l.size,
                f_start + l.h_idx,
                l.size
            });
            for (size_t i = l.h_idx; i <= l.l_idx; ++i) {
                in_loop[i] = true;
            }
        }

        size_t g_idx = 0;
        while (g_idx < f_size) {
            if (in_loop[g_idx]) {
                g_idx++;
                continue;
            }
            size_t g_end = g_idx;
            while (g_end < f_size && !in_loop[g_end]) {
                g_end++;
            }
            size_t g_size = g_end - g_idx;

            SimpleLoop const *following_loop = nullptr;
            auto it = std::lower_bound(raw_loops.begin(), raw_loops.end(), g_end, [](const SimpleLoop &l, size_t val) {
                return l.h_idx < val;
            });
            if (it != raw_loops.end()) {
                following_loop = &(*it);
            }

            if (following_loop) {
                onLoop(RegionSpan{
                    f_start + g_idx,
                    g_size,
                    f_start + following_loop->h_idx,
                    following_loop->size
                });
            } else if (!raw_loops.empty()) {
                const auto &last_l = raw_loops.back();
                onLoop(RegionSpan{
                    f_start + g_idx,
                    g_size,
                    f_start + last_l.h_idx,
                    last_l.size
                });
            } else {
                onBasicBlock(RegionSpan{
                    f_start + g_idx,
                    g_size,
                    f_start + g_idx,
                    g_size
                });
            }
            g_idx = g_end;
        }
    }
}

bool isNopInstruction(const llvm::MCInst &Inst, const llvm::MCInstrInfo &MCII) {
    unsigned Opcode = Inst.getOpcode();
    llvm::StringRef Name = MCII.getName(Opcode);
    if (Name.contains_insensitive("NOP") || Name == "HINT") {
        return true;
    }
    return false;
}

bool isAllNopRegion(llvm::ArrayRef<Instr> instrs, const llvm::MCInstrInfo &MCII) {
    if (instrs.empty()) return true;
    for (const auto &I : instrs) {
        if (!isNopInstruction(I.Inst, MCII)) {
            return false;
        }
    }
    return true;
}
