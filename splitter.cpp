#include "mca_common.h"
#include <algorithm>
#include <optional>
#include <vector>
#include <set>
#include <map>
#include <functional>
#include "llvm/ADT/ArrayRef.h"

using namespace llvm;

namespace {

// コントロールフローエッジ用の構造体
struct CFGNode {
    size_t id;
    size_t start_idx;
    size_t size;
    uint64_t start_addr;
    uint64_t end_addr;
    std::vector<size_t> succs;
    std::vector<size_t> preds;
};

// ループ情報用の構造体
struct LoopInfo {
    size_t header;
    size_t latch;
    std::vector<size_t> member_nodes;
    size_t total_instrs;
    size_t min_idx;
    size_t max_idx;
    size_t analysis_min_idx;
    size_t analysis_max_idx;
    bool valid;
    int depth = 0;
    int height = 0;
};

// Lengauer-Tarjan アルゴリズムによる支配木 (Dominator Tree) の計算
std::vector<int> computeLengauerTarjan(size_t num_nodes, size_t root,
                                       const std::vector<std::vector<size_t>> &succs,
                                       const std::vector<std::vector<size_t>> &preds) {
    std::vector<int> dfnum(num_nodes, -1);
    std::vector<int> vertex(num_nodes, -1);
    std::vector<int> parent(num_nodes, -1);
    std::vector<int> semi(num_nodes, -1);
    std::vector<int> dom(num_nodes, -1);
    std::vector<int> ancestor(num_nodes, -1);
    std::vector<int> label(num_nodes, -1);
    std::vector<std::vector<int>> bucket(num_nodes);
    int dfs_count = 0;

    std::function<void(int)> dfs = [&](int u) {
        dfnum[u] = dfs_count;
        vertex[dfs_count] = u;
        semi[u] = dfs_count;
        label[u] = u;
        dfs_count++;

        for (size_t v : succs[u]) {
            if (dfnum[v] == -1) {
                parent[v] = u;
                dfs(v);
            }
        }
    };

    dfs(static_cast<int>(root));

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
        if (ancestor[v] == -1) return v;
        compress(v);
        return label[v];
    };

    auto link = [&](int u, int v) {
        ancestor[v] = u;
    };

    for (int i = dfs_count - 1; i >= 1; --i) {
        int w = vertex[i];
        for (size_t v : preds[w]) {
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

    return dom;
}

// 1. 各命令がBBの境界であるかを判定し、CFGNode を作成する
std::vector<CFGNode> buildCFGNodes(ArrayRef<Instr> funcInstrs, int bbMaxInstrs) {
    size_t n = funcInstrs.size();
    std::vector<bool> cuts(n, false);
    cuts[0] = true; // 関数の開始は必ずBBの開始

    // 分岐ターゲットアドレスの集合を特定
    std::set<uint64_t> targets;
    for (size_t i = 0; i < n; ++i) {
        const auto &I = funcInstrs[i];
        if (I.IsBranch && I.BranchTarget != 0) {
            targets.insert(I.BranchTarget);
        }
    }

    // ターゲットに一致する命令インデックスをカットポイントにする
    for (size_t i = 0; i < n; ++i) {
        if (targets.count(funcInstrs[i].Addr)) {
            cuts[i] = true;
        }
        // 分岐、コール、リターン命令の直後はBBの切れ目
        if (funcInstrs[i].IsBranch || funcInstrs[i].IsReturn || funcInstrs[i].EndsBB) {
            if (i + 1 < n) {
                cuts[i + 1] = true;
            }
        }
    }

    std::vector<CFGNode> nodes;
    auto addNode = [&](size_t s, size_t sz) {
        CFGNode node;
        node.id = nodes.size();
        node.start_idx = s;
        node.size = sz;
        node.start_addr = funcInstrs[s].Addr;
        node.end_addr = funcInstrs[s + sz - 1].Addr + 4;
        nodes.push_back(node);
    };

    size_t start = 0;
    for (size_t i = 1; i < n; ++i) {
        if (cuts[i]) {
            size_t sz = i - start;
            if (bbMaxInstrs > 0 && sz > static_cast<size_t>(bbMaxInstrs)) {
                size_t rem = sz;
                size_t curr = start;
                while (rem > 0) {
                    size_t chunk = std::min(rem, static_cast<size_t>(bbMaxInstrs));
                    addNode(curr, chunk);
                    curr += chunk;
                    rem -= chunk;
                }
            } else {
                addNode(start, sz);
            }
            start = i;
        }
    }

    size_t sz = n - start;
    if (bbMaxInstrs > 0 && sz > static_cast<size_t>(bbMaxInstrs)) {
        size_t rem = sz;
        size_t curr = start;
        while (rem > 0) {
            size_t chunk = std::min(rem, static_cast<size_t>(bbMaxInstrs));
            addNode(curr, chunk);
            curr += chunk;
            rem -= chunk;
        }
    } else {
        addNode(start, sz);
    }

    return nodes;
}

// 2. CFGエッジの構築
void buildCFGEdges(ArrayRef<Instr> funcInstrs, std::vector<CFGNode> &nodes) {
    size_t num_nodes = nodes.size();
    std::map<uint64_t, size_t> addr_to_node;
    for (size_t i = 0; i < num_nodes; ++i) {
        addr_to_node[nodes[i].start_addr] = i;
    }

    auto get_node_by_addr = [&](uint64_t addr) -> std::optional<size_t> {
        auto it = addr_to_node.find(addr);
        if (it != addr_to_node.end()) return it->second;
        auto it_upper = addr_to_node.upper_bound(addr);
        if (it_upper != addr_to_node.begin()) {
            --it_upper;
            size_t idx = it_upper->second;
            if (addr >= nodes[idx].start_addr && addr < nodes[idx].end_addr) {
                return idx;
            }
        }
        return std::nullopt;
    };

    for (size_t u = 0; u < num_nodes; ++u) {
        const auto &last_instr = funcInstrs[nodes[u].start_idx + nodes[u].size - 1];
        std::vector<size_t> succs;
        if (last_instr.IsBranch) {
            if (last_instr.BranchTarget != 0) {
                if (auto target_id = get_node_by_addr(last_instr.BranchTarget)) {
                    succs.push_back(*target_id);
                }
            }
        }
        if (!last_instr.IsUnconditionalBranch && !last_instr.IsReturn) {
            if (u + 1 < num_nodes) {
                succs.push_back(u + 1);
            }
        }
        std::sort(succs.begin(), succs.end());
        succs.erase(std::unique(succs.begin(), succs.end()), succs.end());
        nodes[u].succs = succs;
        for (size_t v : succs) {
            nodes[v].preds.push_back(u);
        }
    }
}

// 3. バックエッジ (latch -> header, header dom latch) から Natural Loop を検出
std::vector<LoopInfo> detectNaturalLoops(const std::vector<CFGNode> &nodes,
                                         const std::vector<int> &dom_fwd,
                                         int loopMaxInstrs) {
    size_t num_nodes = nodes.size();
    auto dominates = [&](int a, int b) -> bool {
        int curr = b;
        while (curr != -1) {
            if (curr == a) return true;
            curr = dom_fwd[curr];
        }
        return false;
    };

    std::vector<std::pair<size_t, size_t>> back_edges; // (latch, header)
    std::set<std::pair<size_t, size_t>> seen_edges;
    for (size_t u = 0; u < num_nodes; ++u) {
        for (size_t v : nodes[u].succs) {
            if (dominates(v, u)) {
                if (seen_edges.find({u, v}) == seen_edges.end()) {
                    back_edges.push_back({u, v});
                    seen_edges.insert({u, v});
                }
            }
        }
    }

    std::vector<LoopInfo> loops;
    for (const auto &edge : back_edges) {
        size_t latch = edge.first;
        size_t header = edge.second;

        std::vector<size_t> loop_nodes;
        std::vector<size_t> stack;
        std::set<size_t> visited;

        loop_nodes.push_back(header);
        visited.insert(header);

        if (visited.find(latch) == visited.end()) {
            stack.push_back(latch);
            loop_nodes.push_back(latch);
            visited.insert(latch);
        }

        while (!stack.empty()) {
            size_t curr = stack.back();
            stack.pop_back();
            if (curr == header) continue;
            for (size_t p : nodes[curr].preds) {
                if (visited.find(p) == visited.end()) {
                    visited.insert(p);
                    loop_nodes.push_back(p);
                    stack.push_back(p);
                }
            }
        }

        size_t min_idx = SIZE_MAX;
        size_t max_idx = 0;
        size_t total_instrs = 0;
        for (size_t m : loop_nodes) {
            size_t m_start = nodes[m].start_idx;
            size_t m_end   = nodes[m].start_idx + nodes[m].size - 1;
            if (m_start < min_idx) min_idx = m_start;
            if (m_end   > max_idx) max_idx = m_end;
            total_instrs += nodes[m].size;
        }

        size_t analysis_min_idx = nodes[header].start_idx;
        size_t analysis_max_idx = nodes[latch].start_idx + nodes[latch].size - 1;

        // header が latch より後のアドレスにある場合は不正（安全策としてフォールバック）
        if (analysis_min_idx > analysis_max_idx) {
            analysis_min_idx = min_idx;
            analysis_max_idx = max_idx;
        }

        bool valid = true;
        if (loopMaxInstrs > 0 && total_instrs > static_cast<size_t>(loopMaxInstrs)) {
            valid = false;
        }

        LoopInfo loop;
        loop.header = header;
        loop.latch = latch;
        loop.member_nodes = loop_nodes;
        loop.total_instrs = total_instrs;
        loop.min_idx = min_idx;
        loop.max_idx = max_idx;
        loop.analysis_min_idx = analysis_min_idx;
        loop.analysis_max_idx = analysis_max_idx;
        loop.valid = valid;
        loops.push_back(loop);
    }

    return loops;
}

// 4. 同一 header の最大ループ選択と包含関係に基づく Loop 木構造・深さ・高さの計算
std::vector<LoopInfo> buildLoopHierarchy(std::vector<LoopInfo> &loops) {
    std::map<size_t, size_t> header_to_max_loop;
    for (size_t i = 0; i < loops.size(); ++i) {
        if (!loops[i].valid) continue;
        size_t h = loops[i].header;
        if (header_to_max_loop.find(h) == header_to_max_loop.end() || loops[i].total_instrs > loops[header_to_max_loop[h]].total_instrs) {
            header_to_max_loop[h] = i;
        }
    }

    std::vector<LoopInfo> valid_loops;
    for (const auto &pair : header_to_max_loop) {
        valid_loops.push_back(loops[pair.second]);
    }

    size_t num_loops = valid_loops.size();
    std::vector<int> parent(num_loops, -1);
    std::vector<std::vector<size_t>> children(num_loops);

    auto is_subset = [](const std::vector<size_t> &a, const std::vector<size_t> &b) {
        for (size_t x : a) {
            if (std::find(b.begin(), b.end(), x) == b.end()) return false;
        }
        return true;
    };

    for (size_t i = 0; i < num_loops; ++i) {
        int best_p = -1;
        size_t best_p_size = SIZE_MAX;
        for (size_t j = 0; j < num_loops; ++j) {
            if (i == j) continue;
            if (is_subset(valid_loops[i].member_nodes, valid_loops[j].member_nodes)) {
                if (valid_loops[j].member_nodes.size() < best_p_size) {
                    best_p = j;
                    best_p_size = valid_loops[j].member_nodes.size();
                }
            }
        }
        parent[i] = best_p;
        if (best_p != -1) {
            children[best_p].push_back(i);
        }
    }

    std::function<void(size_t, int)> calc_depth = [&](size_t idx, int d) {
        valid_loops[idx].depth = d;
        for (size_t child : children[idx]) {
            calc_depth(child, d + 1);
        }
    };
    for (size_t i = 0; i < num_loops; ++i) {
        if (parent[i] == -1) {
            calc_depth(i, 0);
        }
    }

    std::function<int(size_t)> calc_height = [&](size_t idx) {
        if (children[idx].empty()) {
            valid_loops[idx].height = 0;
            return 0;
        }
        int max_child_h = calc_height(children[idx][0]);
        for (size_t i = 1; i < children[idx].size(); ++i) {
            max_child_h = std::max(max_child_h, calc_height(children[idx][i]));
        }
        int h = max_child_h + 1;
        valid_loops[idx].height = h;
        return h;
    };
    for (size_t i = 0; i < num_loops; ++i) {
        if (parent[i] == -1) {
            calc_height(i);
        }
    }

    return valid_loops;
}

// 5. Stage 1: ループ抽出 (Phase A: Inner-limit, Phase B: Outer-limit)
void extractLoopRegions(size_t n, size_t globalOffset,
                        const std::vector<LoopInfo> &valid_loops,
                        int nestLimitOuter, int nestLimitInner,
                        std::vector<int> &inner_counts,
                        std::vector<int> &outer_counts,
                        const std::function<void(const RegionSpan &)> &onLoop) {
    size_t num_loops = valid_loops.size();
    int limit_inner = nestLimitInner;
    int limit_outer = nestLimitOuter;
    if (limit_inner == 0 && limit_outer == 0) {
        limit_inner = 1;
    }

    if (limit_inner > 0) {
        // Phase A: Inner Loops (Height 昇順 - 最内側優先)
        std::vector<size_t> inner_order(num_loops);
        for (size_t i = 0; i < num_loops; ++i) inner_order[i] = i;
        std::sort(inner_order.begin(), inner_order.end(), [&](size_t a, size_t b) {
            if (valid_loops[a].height != valid_loops[b].height)
                return valid_loops[a].height < valid_loops[b].height;
            return valid_loops[a].total_instrs < valid_loops[b].total_instrs;
        });

        for (size_t idx : inner_order) {
            const auto &loop = valid_loops[idx];
            if (!loop.valid) continue;
            bool can_extract = true;
            for (size_t i = loop.min_idx; i <= loop.max_idx; ++i) {
                if (inner_counts[i] >= limit_inner) {
                    can_extract = false;
                    break;
                }
            }
            if (can_extract) {
                onLoop(RegionSpan{
                    globalOffset + loop.min_idx,
                    loop.max_idx - loop.min_idx + 1,
                    globalOffset + loop.analysis_min_idx,
                    loop.analysis_max_idx - loop.analysis_min_idx + 1
                });
                for (size_t i = loop.min_idx; i <= loop.max_idx; ++i) {
                    inner_counts[i]++;
                }
            }
        }
    }

    if (limit_outer > 0) {
        // Phase B: Outer Loops (Depth 昇順 - 最外側優先)
        std::vector<size_t> outer_order(num_loops);
        for (size_t i = 0; i < num_loops; ++i) outer_order[i] = i;
        std::sort(outer_order.begin(), outer_order.end(), [&](size_t a, size_t b) {
            if (valid_loops[a].depth != valid_loops[b].depth)
                return valid_loops[a].depth < valid_loops[b].depth;
            return valid_loops[a].total_instrs > valid_loops[b].total_instrs;
        });

        for (size_t idx : outer_order) {
            const auto &loop = valid_loops[idx];
            if (!loop.valid) continue;
            bool can_extract = true;
            for (size_t i = loop.min_idx; i <= loop.max_idx; ++i) {
                if (outer_counts[i] >= limit_outer) {
                    can_extract = false;
                    break;
                }
            }
            if (can_extract) {
                onLoop(RegionSpan{
                    globalOffset + loop.min_idx,
                    loop.max_idx - loop.min_idx + 1,
                    globalOffset + loop.analysis_min_idx,
                    loop.analysis_max_idx - loop.analysis_min_idx + 1
                });
                for (size_t i = loop.min_idx; i <= loop.max_idx; ++i) {
                    outer_counts[i]++;
                }
            }
        }
    }
}

// 6. Stage 2: Post-Dominator Tree による未解析基本ブロックの集約
void aggregateBasicBlocks(ArrayRef<Instr> funcInstrs, size_t globalOffset,
                          const std::vector<CFGNode> &nodes,
                          const std::vector<int> &inner_counts,
                          const std::vector<int> &outer_counts,
                          const std::function<void(const RegionSpan &)> &onBasicBlock) {
    size_t num_nodes = nodes.size();
    size_t virtual_exit = num_nodes;
    std::vector<std::vector<size_t>> rev_succs(num_nodes + 1);
    std::vector<std::vector<size_t>> rev_preds(num_nodes + 1);

    for (size_t u = 0; u < num_nodes; ++u) {
        for (size_t v : nodes[u].succs) {
            rev_succs[v].push_back(u);
            rev_preds[u].push_back(v);
        }
        const auto &last_instr = funcInstrs[nodes[u].start_idx + nodes[u].size - 1];
        if (nodes[u].succs.empty() || last_instr.IsReturn) {
            rev_succs[virtual_exit].push_back(u);
            rev_preds[u].push_back(virtual_exit);
        }
    }

    std::vector<int> dom_rev = computeLengauerTarjan(num_nodes + 1, virtual_exit, rev_succs, rev_preds);

    // 各ノードの命令が一つでも未解析 (inner_count == 0 && outer_count == 0) かどうか判定
    std::vector<bool> node_has_unanalyzed(num_nodes, false);
    for (size_t u = 0; u < num_nodes; ++u) {
        for (size_t i = nodes[u].start_idx; i < nodes[u].start_idx + nodes[u].size; ++i) {
            if (inner_counts[i] == 0 && outer_counts[i] == 0) {
                node_has_unanalyzed[u] = true;
                break;
            }
        }
    }

    // 未解析の基本ブロックをグループ化（CFG 制御フローに沿った Post-Dominance / 直列パスをマージ）
    std::vector<int> bb_group(num_nodes, -1);
    for (size_t u = 0; u < num_nodes; ++u) {
        if (!node_has_unanalyzed[u]) continue;
        if (bb_group[u] != -1) continue;

        size_t g_start = nodes[u].start_idx;
        size_t g_size = nodes[u].size;
        bb_group[u] = u;

        size_t curr = u;
        while (curr + 1 < num_nodes && node_has_unanalyzed[curr + 1]) {
            bool has_edge = (std::find(nodes[curr].succs.begin(), nodes[curr].succs.end(), curr + 1) != nodes[curr].succs.end());
            if (has_edge && (nodes[curr].succs.size() == 1 || dom_rev[curr] == static_cast<int>(curr + 1))) {
                curr++;
                g_size += nodes[curr].size;
                bb_group[curr] = u;
            } else {
                break;
            }
        }
        onBasicBlock(RegionSpan{
            globalOffset + g_start,
            g_size,
            globalOffset + g_start,
            g_size
        });
    }
}

void processFunction(ArrayRef<Instr> funcInstrs, size_t globalOffset, int loopMaxInstrs, int bbMaxInstrs, int nestLimitOuter, int nestLimitInner,
                     const std::function<void(const RegionSpan &)> &onLoop,
                     const std::function<void(const RegionSpan &)> &onBasicBlock) {
    size_t n = funcInstrs.size();
    if (n == 0) return;

    // 1. CFG ノードの構築
    std::vector<CFGNode> nodes = buildCFGNodes(funcInstrs, bbMaxInstrs);
    if (nodes.empty()) return;

    // 2. CFG エッジの構築
    buildCFGEdges(funcInstrs, nodes);

    // 3. 支配木 (Forward Dominator Tree) の計算
    size_t num_nodes = nodes.size();
    std::vector<std::vector<size_t>> succs(num_nodes), preds(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
        succs[i] = nodes[i].succs;
        preds[i] = nodes[i].preds;
    }
    std::vector<int> dom_fwd = computeLengauerTarjan(num_nodes, 0, succs, preds);

    // 4. ループ検出と階層ツリーの構築
    std::vector<LoopInfo> raw_loops = detectNaturalLoops(nodes, dom_fwd, loopMaxInstrs);
    std::vector<LoopInfo> valid_loops = buildLoopHierarchy(raw_loops);

    // 5. Stage 1: ループ領域の抽出
    std::vector<int> inner_counts(n, 0);
    std::vector<int> outer_counts(n, 0);
    extractLoopRegions(n, globalOffset, valid_loops, nestLimitOuter, nestLimitInner, inner_counts, outer_counts, onLoop);

    // 6. Stage 2: 未解析基本ブロックの集約
    aggregateBasicBlocks(funcInstrs, globalOffset, nodes, inner_counts, outer_counts, onBasicBlock);
}

} // namespace

void walkRegions(ArrayRef<Instr> instrs, const FunctionBoundaries &boundaries, int loopMaxInstrs, int bbMaxInstrs, int nestLimitOuter, int nestLimitInner,
                 const std::function<void(const RegionSpan &)> &onLoop,
                 const std::function<void(const RegionSpan &)> &onBasicBlock) {
    if (instrs.empty()) return;

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
            }
        }
    }

    std::vector<std::pair<size_t, size_t>> funcSpans;
    if (boundaries.empty()) {
        funcSpans.push_back({0, instrs.size()});
    } else {
        size_t start = 0;
        for (size_t i = 1; i < instrs.size(); ++i) {
            if (instrFuncEntry[i] != instrFuncEntry[start]) {
                if (instrFuncEntry[start] != 0) {
                    funcSpans.push_back({start, i - start});
                }
                start = i;
            }
        }
        if (instrFuncEntry[start] != 0) {
            funcSpans.push_back({start, instrs.size() - start});
        }
    }

    for (const auto &span : funcSpans) {
        size_t f_start = span.first;
        size_t f_size = span.second;
        auto funcInstrs = instrs.slice(f_start, f_size);
        processFunction(funcInstrs, f_start, loopMaxInstrs, bbMaxInstrs, nestLimitOuter, nestLimitInner, onLoop, onBasicBlock);
    }
}

bool isNopInstruction(const llvm::MCInst &Inst, const llvm::MCInstrInfo &MCII) {
    StringRef Name = MCII.getName(Inst.getOpcode());
    if (Name.contains_insensitive("nop") || Name.contains_insensitive("noop")) return true;
    if (Name.equals_insensitive("hint") && Inst.getNumOperands() > 0 && Inst.getOperand(0).isImm() && Inst.getOperand(0).getImm() == 0) return true;
    return false;
}

bool isAllNopRegion(llvm::ArrayRef<Instr> instrs, const llvm::MCInstrInfo &MCII) {
    if (instrs.empty()) return false;
    for (const auto &I : instrs) {
        if (!isNopInstruction(I.Inst, MCII)) return false;
    }
    return true;
}
