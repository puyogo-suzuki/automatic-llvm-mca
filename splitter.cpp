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

// NOTE on ordering invariants: several helpers below (and walkRegions() itself) depend on
// vectors of SimpleLoop being sorted by (h_idx asc, l_idx desc). A prior refactor (ee9a922)
// once broke this by having a helper sort a by-value copy while the caller kept using its
// own unsorted vector for a std::lower_bound gap lookup - silently wrong results with no
// crash. When touching sort/ordering here, double-check which vector instance downstream
// code actually reads.
struct SimpleLoop {
    size_t h_idx;
    size_t l_idx;
    size_t size;
    bool is_abab_merged = false;
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

// AArch64 A64 instructions are fixed-width: every instruction advances the program
// counter by exactly this many bytes, so (target_addr - func_base_addr) / kInstrSizeBytes
// converts a branch target address into an instruction index within the function.
constexpr uint64_t kInstrSizeBytes = 4;

// Step 2: Detect basic backward-branch loops within a single function
std::vector<SimpleLoop> detectBackwardBranchLoops(ArrayRef<Instr> funcInstrs) {
    std::vector<SimpleLoop> loops;
    size_t f_size = funcInstrs.size();
    if (f_size == 0) return loops;

    auto isForwardBranch = [](const Instr &J) {
        return J.IsBranch && !J.IsCall && J.BranchTarget != 0 && J.BranchTarget > J.Addr;
    };

    // Prefix count of forward branches/jumps, so "is there a forward branch in [a, b)"
    // becomes the O(1) check fwdBranchCount[b] - fwdBranchCount[a] > 0 below, instead of
    // rescanning that range from scratch for every return found under every candidate
    // loop header (which made this function up to O(f_size^3) on return-heavy functions).
    std::vector<size_t> fwdBranchCount(f_size + 1, 0);
    for (size_t t = 0; t < f_size; ++t) {
        fwdBranchCount[t + 1] = fwdBranchCount[t] + (isForwardBranch(funcInstrs[t]) ? 1 : 0);
    }

    uint64_t f_base_addr = funcInstrs[0].Addr;
    for (size_t i = 0; i < f_size; ++i) {
        const auto &I = funcInstrs[i];
        if (I.IsBranch && !I.IsCall && I.BranchTarget != 0 && I.BranchTarget <= I.Addr) {
            if (I.BranchTarget >= f_base_addr) {
                uint64_t offset = I.BranchTarget - f_base_addr;
                if (offset % kInstrSizeBytes == 0) {
                    size_t h_idx = offset / kInstrSizeBytes;
                    if (h_idx <= i && h_idx < f_size) {
                        // A candidate loop [h_idx, i] is invalid if it contains a return
                        // that is reachable without passing through any forward branch/jump
                        // first (i.e. an unconditional early exit rather than a genuine,
                        // skippable loop body).
                        bool invalid_loop = false;
                        for (size_t k = h_idx; k <= i; ++k) {
                            if (funcInstrs[k].IsReturn && fwdBranchCount[k] == fwdBranchCount[h_idx]) {
                                invalid_loop = true;
                                break;
                            }
                        }
                        if (!invalid_loop) {
                            loops.push_back({h_idx, i, i - h_idx + 1});
                        }
                    }
                }
            }
        }
    }
    return loops;
}

// Maximum look-back window (in sorted-loop position, not instructions) for the abab
// chain-depth DP below. Bounding it keeps detectAndMergeAbabChains near-linear even when
// a function contains many interlocking loops; a chain longer than this is vanishingly
// rare in practice and would still be caught as max_chain >= threshold well before the
// window is exhausted (threshold is typically single-digit, see opts::ChainThreshold).
constexpr int kChainLookbackWindow = 50;

// Step 3: Detect abab interlocking chains and perform adaptive merging.
// Precondition: `loops` must already be sorted by (h_idx asc, l_idx desc) - the same
// order walkRegions() sorts raw_loops into before calling this, so it is not re-sorted
// here (see the ee9a922 history note at the top of this file for why that invariant
// matters). Takes `loops` by const reference since this function only reads it.
std::vector<SimpleLoop> detectAndMergeAbabChains(const std::vector<SimpleLoop> &loops, int threshold) {
    size_t n_loops = loops.size();
    if (n_loops == 0) return {};

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

        int limit = std::max(0, static_cast<int>(i) - kChainLookbackWindow);
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
            SimpleLoop merged_loop = loops[i];
            merged_loop.is_abab_merged = true;
            merged_abab.push_back(merged_loop);
        } else {
            auto &prev = merged_abab.back();
            if (loops[i].h_idx <= prev.l_idx) {
                prev.l_idx = std::max(prev.l_idx, loops[i].l_idx);
                prev.size = prev.l_idx - prev.h_idx + 1;
                prev.is_abab_merged = true;
            } else {
                SimpleLoop merged_loop = loops[i];
                merged_loop.is_abab_merged = true;
                merged_abab.push_back(merged_loop);
            }
        }
    }

    selected_loops.insert(selected_loops.end(), merged_abab.begin(), merged_abab.end());

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
        std::sort(raw_loops.begin(), raw_loops.end(), [](const SimpleLoop &a, const SimpleLoop &b) {
            if (a.h_idx != b.h_idx) return a.h_idx < b.h_idx;
            return a.l_idx > b.l_idx;
        });
        std::vector<SimpleLoop> selected_loops = detectAndMergeAbabChains(raw_loops, opts::ChainThreshold);

        std::vector<bool> in_loop(f_size, false);
        for (const auto &l : selected_loops) {
            onLoop(RegionSpan{
                f_start + l.h_idx,
                l.size,
                f_start + l.h_idx,
                l.size,
                l.is_abab_merged
            });
            for (size_t i = l.h_idx; i <= l.l_idx; ++i) {
                in_loop[i] = true;
            }
        }

        // Walk the non-loop "gaps" left between (and around) the selected loops. Each gap
        // is reported either standalone (onBasicBlock, when the function has no loops at
        // all) or attached to a loop for combined MCA analysis (onLoop): the loop chosen
        // is whichever selected loop starts right after the gap, since raw_loops is sorted
        // by h_idx (see the precondition comment on detectAndMergeAbabChains above) and
        // lower_bound finds it in O(log n); if no such loop exists (the gap is after the
        // last loop), fall back to the last loop in the function.
        size_t gap_start = 0;
        while (gap_start < f_size) {
            if (in_loop[gap_start]) {
                gap_start++;
                continue;
            }
            size_t gap_end = gap_start;
            while (gap_end < f_size && !in_loop[gap_end]) {
                gap_end++;
            }
            size_t gap_size = gap_end - gap_start;

            // Nearby loop to attach this gap's analysis to: the first loop starting at or
            // after gap_end, or (if none starts after the gap) the last loop in the
            // function. Null only when the whole function has no loops at all.
            SimpleLoop const *nearby_loop = nullptr;
            if (!raw_loops.empty()) {
                auto it = std::lower_bound(raw_loops.begin(), raw_loops.end(), gap_end, [](const SimpleLoop &l, size_t val) {
                    return l.h_idx < val;
                });
                nearby_loop = (it != raw_loops.end()) ? &(*it) : &raw_loops.back();
            }

            if (nearby_loop) {
                onLoop(RegionSpan{
                    f_start + gap_start,
                    gap_size,
                    f_start + nearby_loop->h_idx,
                    nearby_loop->size
                });
            } else {
                onBasicBlock(RegionSpan{
                    f_start + gap_start,
                    gap_size,
                    f_start + gap_start,
                    gap_size
                });
            }
            gap_start = gap_end;
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
