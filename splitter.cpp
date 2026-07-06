#include "mca_common.h"
#include <algorithm>
#include <optional>
#include <vector>

#include "llvm/ADT/ArrayRef.h"

using namespace llvm;

namespace {

bool containsAddress(const FunctionBoundaries &Boundaries, uint64_t Addr) {
    auto It = Boundaries.upper_bound(Addr);
    if (It == Boundaries.begin()) return false;
    --It;
    return Addr >= It->first && Addr < It->second;
}

bool sameFunction(const FunctionBoundaries &Boundaries, uint64_t A, uint64_t B) {
    if (Boundaries.empty()) return true;
    auto ItA = Boundaries.upper_bound(A);
    if (ItA == Boundaries.begin()) return false;
    --ItA;
    if (A < ItA->first || A >= ItA->second) return false;

    auto ItB = Boundaries.upper_bound(B);
    if (ItB == Boundaries.begin()) return false;
    --ItB;
    if (B < ItB->first || B >= ItB->second) return false;

    return ItA->first == ItB->first;
}

int64_t findIndex(ArrayRef<Instr> instrs, uint64_t addr) {
    auto it = std::lower_bound(instrs.begin(), instrs.end(), addr,
                               [](const Instr &a, uint64_t val) { return a.Addr < val; });
    if (it != instrs.end() && it->Addr == addr) return std::distance(instrs.begin(), it);
    return -1;
}

} // namespace

void walkRegions(ArrayRef<Instr> instrs, const FunctionBoundaries &boundaries, int loopMaxInstrs, int bbMaxInstrs, int nestLimitOuter, int nestLimitInner,
                 const std::function<void(const RegionSpan &)> &onLoop,
                 const std::function<void(const RegionSpan &)> &onBasicBlock) {
    if (instrs.empty()) return;

    // Precompute which function boundary each instruction belongs to using 2-pointer scan
    std::vector<uint64_t> instrFuncEntry(instrs.size(), 0);
    if (!boundaries.empty()) {
        std::vector<std::pair<uint64_t, uint64_t>> bounds(boundaries.begin(), boundaries.end());
        size_t b_idx = 0;
        for (size_t i = 0; i < instrs.size(); ++i) {
            uint64_t addr = instrs[i].Addr;
            while (b_idx < bounds.size() && addr >= bounds[b_idx].second) {
                b_idx++;
            }
            if (b_idx < bounds.size() && addr >= bounds[b_idx].first && addr < bounds[b_idx].second) {
                instrFuncEntry[i] = bounds[b_idx].first;
            } else {
                instrFuncEntry[i] = 0;
            }
        }
    }

    auto sameFuncIdx = [&](size_t idxA, size_t idxB) {
        if (boundaries.empty()) return true;
        uint64_t fA = instrFuncEntry[idxA];
        uint64_t fB = instrFuncEntry[idxB];
        return fA != 0 && fA == fB;
    };

    // --- Precompute and memoize getLoopSpan for each instruction ---
    std::vector<std::optional<RegionSpan>> loopSpans(instrs.size());
    auto getLoopSpanMemoized = [&](size_t idx) -> std::optional<RegionSpan> {
        const auto &I = instrs[idx];
        if (!I.IsBranch || I.BranchTarget == 0 || I.BranchTarget >= I.Addr) return std::nullopt;

        int64_t startIdx = findIndex(instrs, I.BranchTarget);
        if (startIdx == -1) return std::nullopt;

        const size_t start = static_cast<size_t>(startIdx);
        const size_t size = idx - start + 1;
        if (loopMaxInstrs > 0 && size > static_cast<size_t>(loopMaxInstrs)) return std::nullopt;
        if (!sameFuncIdx(idx, start)) return std::nullopt;
        return RegionSpan{start, size};
    };

    for (size_t i = 0; i < instrs.size(); ++i) {
        loopSpans[i] = getLoopSpanMemoized(i);
    }

    // Deduplicate: same start address -> keep only largest
    std::map<size_t, size_t> largestLoopByStart;
    for (size_t i = 0; i < instrs.size(); ++i) {
        if (const auto &loop = loopSpans[i]) {
            auto it = largestLoopByStart.find(loop->Start);
            if (it == largestLoopByStart.end() || loop->Size > it->second) {
                largestLoopByStart[loop->Start] = loop->Size;
            }
        }
    }

    // Calculate nesting depth (from outer to inner) and height (from inner to outer)
    // Candidates are sorted by Start asc (keys of std::map)
    std::vector<RegionSpan> candidates;
    for (const auto &pair : largestLoopByStart) {
        candidates.push_back({pair.first, pair.second});
    }

    std::vector<int> nestingDepth(candidates.size(), 0);
    std::vector<int> parent(candidates.size(), -1);
    std::vector<size_t> active_stack; // stack of candidate indices
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto &curr = candidates[i];
        while (!active_stack.empty()) {
            const auto &top = candidates[active_stack.back()];
            // Since candidates are sorted by Start ascending, top.Start <= curr.Start.
            // curr is nested inside top if curr.Start + curr.Size <= top.Start + top.Size.
            if (curr.Start + curr.Size <= top.Start + top.Size) {
                nestingDepth[i] = active_stack.size();
                parent[i] = active_stack.back();
                break;
            } else {
                active_stack.pop_back();
            }
        }
        active_stack.push_back(i);
    }

    // Calculate nesting height bottom-up (height = max path from leaf loops)
    std::vector<int> nestingHeight(candidates.size(), 0);
    for (int i = (int)candidates.size() - 1; i >= 0; --i) {
        int p = parent[i];
        if (p != -1) {
            nestingHeight[p] = std::max(nestingHeight[p], nestingHeight[i] + 1);
        }
    }

    // Build the final loop map: start_idx -> size, keeping those allowed by
    // either nestLimitOuter (depth < nestLimitOuter) or nestLimitInner (height < nestLimitInner)
    std::map<size_t, size_t> finalLoops;
    for (size_t i = 0; i < candidates.size(); ++i) {
        bool allowed_outer = (nestLimitOuter <= 0 || nestingDepth[i] < nestLimitOuter);
        bool allowed_inner = (nestLimitInner <= 0 || nestingHeight[i] < nestLimitInner);
        if (allowed_outer || allowed_inner) {
            finalLoops[candidates[i].Start] = candidates[i].Size;
        }
    }

    // --- Pass 1: mark which instructions belong to an allowed loop region ---
    std::vector<bool> inLoop(instrs.size(), false);
    for (size_t i = 0; i < instrs.size(); ++i) {
        if (const auto &loop = loopSpans[i]) {
            auto it = finalLoops.find(loop->Start);
            if (it != finalLoops.end() && loop->Size == it->second) {
                for (size_t j = loop->Start; j < loop->Start + loop->Size; ++j) {
                    inLoop[j] = true;
                }
            }
        }
    }

    // --- Pass 2: emit loops ---
    for (size_t i = 0; i < instrs.size(); ++i) {
        if (const auto &loop = loopSpans[i]) {
            auto it = finalLoops.find(loop->Start);
            if (it != finalLoops.end() && loop->Size == it->second) {
                if (onLoop) onLoop(*loop);
                // Emit once
                finalLoops.erase(it);
            }
        }
    }

    // --- Emit basic blocks ---
    size_t bbStart = 0;
    bool inBB = false;

    for (size_t i = 0; i < instrs.size(); ++i) {
        const auto &I = instrs[i];

        if (inLoop[i]) {
            if (inBB) {
                size_t size = i - bbStart;
                if (size > 0 && (bbMaxInstrs <= 0 || size <= static_cast<size_t>(bbMaxInstrs))) {
                    if (onBasicBlock) onBasicBlock(RegionSpan{bbStart, size});
                }
                inBB = false;
            }
            continue;
        }

        if (!inBB) {
            bbStart = i;
            inBB = true;
        }

        bool endBB = I.EndsBB;
        if (i + 1 < instrs.size()) {
            if (inLoop[i + 1]) {
                endBB = true;
            } else if (!boundaries.empty() && !sameFuncIdx(i, i + 1)) {
                endBB = true;
            }
        } else {
            endBB = true;
        }

        if (endBB) {
            size_t size = i - bbStart + 1;
            if (bbMaxInstrs <= 0 || size <= static_cast<size_t>(bbMaxInstrs)) {
                if (onBasicBlock) onBasicBlock(RegionSpan{bbStart, size});
            }
            inBB = false;
        }
    }
}

void computePostDominatorOverwrites(llvm::ArrayRef<Instr> SectionInstrs,
                                    const FunctionBoundaries &Boundaries,
                                    std::vector<McaRegion> &regions,
                                    std::map<size_t, size_t> &overwrite_map) {
    if (regions.size() < 2) return;

    std::map<uint64_t, std::vector<size_t>> func_groups;
    for (size_t i = 0; i < regions.size(); ++i) {
        uint64_t startAddr = regions[i].StartAddr;
        uint64_t func_entry = 0;
        auto It = Boundaries.upper_bound(startAddr);
        if (It != Boundaries.end() && It != Boundaries.begin()) {
            --It;
            if (startAddr >= It->first && startAddr < It->second) {
                func_entry = It->first;
            }
        }
        func_groups[func_entry].push_back(i);
    }

    for (auto &pair : func_groups) {
        auto &group = pair.second;
        if (group.size() < 2) continue;

        std::sort(group.begin(), group.end(), [&](size_t a, size_t b) {
            return regions[a].StartAddr < regions[b].StartAddr;
        });

        std::map<uint64_t, size_t> addr_to_node;
        for (size_t u = 0; u < group.size(); ++u) {
            addr_to_node[regions[group[u]].StartAddr] = u;
        }

        size_t num_nodes = group.size();
        std::vector<std::vector<size_t>> successors(num_nodes);
        std::vector<std::vector<size_t>> predecessors(num_nodes);

        for (size_t u = 0; u < num_nodes; ++u) {
            size_t r_idx = group[u];
            const auto &r = regions[r_idx];
            const auto &last_instr = SectionInstrs[r.Start + r.Size - 1];
            std::vector<uint64_t> targets;
            
            if (last_instr.IsBranch) {
                if (last_instr.BranchTarget != 0) {
                    targets.push_back(last_instr.BranchTarget);
                }
            }
            if (!last_instr.IsUnconditionalBranch) {
                if (r.Start + r.Size < SectionInstrs.size()) {
                    targets.push_back(SectionInstrs[r.Start + r.Size].Addr);
                }
            }

            for (uint64_t t : targets) {
                auto it = addr_to_node.find(t);
                if (it != addr_to_node.end()) {
                    successors[u].push_back(it->second);
                } else {
                    // Binary search in group for v where regions[group[v]].StartAddr <= t < regions[group[v]].EndAddr
                    auto it_bound = std::upper_bound(group.begin(), group.end(), t,
                        [&](uint64_t val, size_t idx) {
                            return val < regions[idx].StartAddr;
                        });
                    if (it_bound != group.begin()) {
                        auto prev_it = std::prev(it_bound);
                        size_t v = std::distance(group.begin(), prev_it);
                        const auto &node = regions[*prev_it];
                        if (t >= node.StartAddr && t < node.EndAddr) {
                            successors[u].push_back(v);
                        }
                    }
                }
            }
        }

        for (size_t u = 0; u < num_nodes; ++u) {
            for (size_t v : successors[u]) {
                predecessors[v].push_back(u);
            }
        }

        size_t num_nodes_extended = num_nodes + 1;
        size_t virtual_exit = num_nodes;

        std::vector<std::vector<size_t>> successors_rev(num_nodes_extended);
        std::vector<std::vector<size_t>> predecessors_rev(num_nodes_extended);

        for (size_t u = 0; u < num_nodes; ++u) {
            for (size_t v : successors[u]) {
                successors_rev[v].push_back(u);
                predecessors_rev[u].push_back(v);
            }
            
            const auto &r = regions[group[u]];
            const auto &last_instr = SectionInstrs[r.Start + r.Size - 1];
            bool is_exit = successors[u].empty() || last_instr.IsReturn;
            
            if (is_exit) {
                successors_rev[virtual_exit].push_back(u);
                predecessors_rev[u].push_back(virtual_exit);
            }
        }

        std::vector<int> dfnum(num_nodes_extended, -1);
        std::vector<int> vertex(num_nodes_extended, -1);
        std::vector<int> parent(num_nodes_extended, -1);
        std::vector<int> semi(num_nodes_extended, -1);
        std::vector<int> dom(num_nodes_extended, -1);
        std::vector<int> ancestor(num_nodes_extended, -1);
        std::vector<int> label(num_nodes_extended, -1);
        std::vector<std::vector<int>> bucket(num_nodes_extended);

        int dfs_count = 0;

        std::function<void(int)> dfs = [&](int u) {
            dfnum[u] = dfs_count;
            vertex[dfs_count] = u;
            semi[u] = dfs_count;
            label[u] = u;
            dfs_count++;

            for (size_t v : successors_rev[u]) {
                if (dfnum[v] == -1) {
                    parent[v] = u;
                    dfs(v);
                }
            }
        };

        dfs(virtual_exit);

        std::function<void(int)> compress = [&](int v) {
            int anc = ancestor[v];
            if (ancestor[anc] != -1) {
                compress(anc);
                if (semi[label[anc]] < semi[label[v]]) {
                    label[v] = label[anc];
                }
                ancestor[v] = ancestor[anc];
            }
        };

        auto eval = [&](int v) -> int {
            if (ancestor[v] == -1) {
                return v;
            }
            compress(v);
            return label[v];
        };

        auto link = [&](int u, int v) {
            ancestor[v] = u;
        };

        for (int i = dfs_count - 1; i >= 1; --i) {
            int w = vertex[i];
            for (size_t v : predecessors_rev[w]) {
                if (dfnum[v] == -1) continue;
                int u = eval(v);
                if (semi[u] < semi[w]) {
                    semi[w] = semi[u];
                }
            }
            bucket[vertex[semi[w]]].push_back(w);
            link(parent[w], w);
            int p = parent[w];
            for (int v : bucket[p]) {
                int u = eval(v);
                dom[v] = (semi[u] < semi[v]) ? u : p;
            }
            bucket[p].clear();
        }

        for (int i = 1; i < dfs_count; ++i) {
            int w = vertex[i];
            if (dom[w] != vertex[semi[w]]) {
                dom[w] = dom[dom[w]];
            }
        }

        for (size_t u = 0; u < num_nodes; ++u) {
            size_t r_idx = group[u];
            if (!regions[r_idx].IsLoop) {
                std::vector<size_t> pdom_loops;
                int curr = dom[u];
                while (curr != -1 && curr != (int)virtual_exit) {
                    size_t ancestor_r_idx = group[curr];
                    if (regions[ancestor_r_idx].IsLoop) {
                        pdom_loops.push_back(curr);
                    }
                    curr = dom[curr];
                }

                if (!pdom_loops.empty()) {
                    std::vector<size_t> shallow_loops;
                    for (size_t l : pdom_loops) {
                        const auto &loop_l = regions[group[l]];
                        bool is_contained = false;
                        for (size_t other : pdom_loops) {
                            if (other == l) continue;
                            const auto &loop_o = regions[group[other]];
                            if (loop_o.StartAddr <= loop_l.StartAddr && loop_l.EndAddr <= loop_o.EndAddr) {
                                is_contained = true;
                                break;
                            }
                        }
                        if (!is_contained) {
                            shallow_loops.push_back(l);
                        }
                    }
                    if (shallow_loops.size() == 1) {
                        overwrite_map[r_idx] = group[shallow_loops[0]];
                    }
                }
            }
        }
    }
}
