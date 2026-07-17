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

void processFunction(ArrayRef<Instr> funcInstrs, size_t globalOffset, int loopMaxInstrs, int bbMaxInstrs, int nestLimitOuter, int nestLimitInner,
                     const std::function<void(const RegionSpan &)> &onLoop,
                     const std::function<void(const RegionSpan &)> &onBasicBlock) {
    size_t n = funcInstrs.size();
    if (n == 0) return;

    // 1. 各命令がBBの境界（カットポイント）であるかを判定する
    std::vector<bool> cuts(n, false);
    cuts[0] = true; // 関数の開始は必ずBBの開始

    // 分岐ターゲットアドレスの集合を特定する
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

    // BBの切り出しと、必要に応じた bbMaxInstrs 分割
    std::vector<CFGNode> nodes;
    size_t start = 0;
    auto addNode = [&](size_t s, size_t sz) {
        CFGNode node;
        node.id = nodes.size();
        node.start_idx = s;
        node.size = sz;
        node.start_addr = funcInstrs[s].Addr;
        node.end_addr = funcInstrs[s + sz - 1].Addr + 4;
        nodes.push_back(node);
    };

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
    // 最後のBB
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

    size_t num_nodes = nodes.size();
    if (num_nodes == 0) return;

    // 2. CFGエッジの構築
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

    // 3. EXITノードの作成 (ID = num_nodes)
    // 逆CFGを構築する（post-dominator 算出のため）。
    // 元CFGの辺 u→v に対して、逆CFGでは v→u の辺となる。
    //   rev_succs[v]  : 逆CFGにおける v の後続ノード群（元CFGで v へ入ってくる辺の始点）
    //   rev_preds[u]  : 逆CFGにおける u の先行ノード群（元CFGで u から出ていく辺の終点）
    // Lengauer-Tarjan アルゴリズムは rev_succs を DFS で辿り、
    // semi-dominator の計算で rev_preds を参照する。
    size_t virtual_exit = num_nodes;
    std::vector<std::vector<size_t>> rev_succs(num_nodes + 1);  // 逆CFGの後続（= 元CFGの前駆）
    std::vector<std::vector<size_t>> rev_preds(num_nodes + 1);  // 逆CFGの先行（= 元CFGの後続）

    for (size_t u = 0; u < num_nodes; ++u) {
        for (size_t v : nodes[u].succs) {
            // 元CFG: u→v  ⇒  逆CFG: v→u
            rev_succs[v].push_back(u);   // 逆CFGで v の後続に u を追加
            rev_preds[u].push_back(v);   // 逆CFGで u の先行に v を追加
        }
        const auto &last_instr = funcInstrs[nodes[u].start_idx + nodes[u].size - 1];
        if (nodes[u].succs.empty() || last_instr.IsReturn) {
            // u は virtual_exit へ繋がる（リターンまたは後続なし）
            // 逆CFG: virtual_exit→u
            rev_succs[virtual_exit].push_back(u);
            rev_preds[u].push_back(virtual_exit);
        }
    }

    // 4. Natural Loop の検出 (DFS)
    std::vector<std::pair<size_t, size_t>> back_edges; // (latch, header)
    std::vector<int> dfnum(num_nodes + 1, -1);
    std::vector<bool> active(num_nodes + 1, false);
    int dfs_count = 0;

    std::function<void(size_t)> find_loops_dfs = [&](size_t u) {
        dfnum[u] = dfs_count++;
        active[u] = true;
        for (size_t v : nodes[u].succs) {
            if (dfnum[v] == -1) {
                find_loops_dfs(v);
            } else if (active[v]) {
                back_edges.push_back({u, v});
            }
        }
        active[u] = false;
    };

    find_loops_dfs(0);

    struct LoopInfo {
        size_t header;
        std::vector<size_t> member_nodes;
        size_t total_instrs;
        size_t min_idx;
        size_t max_idx;
        bool valid;
        int depth;
        int height;
    };
    std::vector<LoopInfo> loops;

    for (const auto &edge : back_edges) {
        size_t latch = edge.first;
        size_t header = edge.second;

        std::vector<size_t> loop_nodes;
        std::vector<size_t> stack;
        std::set<size_t> visited;

        loop_nodes.push_back(header);
        visited.insert(header);

        stack.push_back(latch);
        loop_nodes.push_back(latch);
        visited.insert(latch);

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


        // min_idx/max_idx/total_instrs を member_nodes 全体から算出する。
        // header と latch のみでは、ループボディ内の全ノードを網羅できない。
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

        bool valid = true;
        if (loopMaxInstrs > 0 && total_instrs > static_cast<size_t>(loopMaxInstrs)) {
            valid = false;
        }

        LoopInfo loop;
        loop.header = header;
        loop.member_nodes = loop_nodes;
        loop.total_instrs = total_instrs;
        loop.min_idx = min_idx;
        loop.max_idx = max_idx;
        loop.valid = valid;
        loop.depth = 0;
        loop.height = 0;
        loops.push_back(loop);
    }

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
        size_t best_p_size = -1;
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
        int min_child_h = calc_height(children[idx][0]);
        for (size_t i = 1; i < children[idx].size(); ++i) {
            min_child_h = std::min(min_child_h, calc_height(children[idx][i]));
        }
        int h = min_child_h + 1;
        valid_loops[idx].height = h;
        return h;
    };
    for (size_t i = 0; i < num_loops; ++i) {
        if (parent[i] == -1) {
            calc_height(i);
        }
    }

    for (size_t i = 0; i < num_loops; ++i) {
        if (nestLimitOuter > 0 && valid_loops[i].depth >= nestLimitOuter) {
            valid_loops[i].valid = false;
        }
        if (nestLimitInner > 0 && valid_loops[i].height >= nestLimitInner) {
            valid_loops[i].valid = false;
        }
    }

    std::vector<int> node_merged_to_loop(num_nodes, -1);

    bool has_valid_loops = false;
    for (size_t l = 0; l < num_loops; ++l) {
        if (valid_loops[l].valid) {
            has_valid_loops = true;
            break;
        }
    }

    if (has_valid_loops) {
        // 5. Post-dominator の算出 (Lengauer-Tarjan on reverse CFG)
        std::vector<int> dfnum_rev(num_nodes + 1, -1);
        std::vector<int> vertex_rev(num_nodes + 1, -1);
        std::vector<int> parent_rev(num_nodes + 1, -1);
        std::vector<int> semi_rev(num_nodes + 1, -1);
        std::vector<int> dom_rev(num_nodes + 1, -1);
        std::vector<int> ancestor_rev(num_nodes + 1, -1);
        std::vector<int> label_rev(num_nodes + 1, -1);
        std::vector<std::vector<int>> bucket_rev(num_nodes + 1);
        int dfs_count_rev = 0;

        std::function<void(int)> dfs_rev = [&](int u) {
            dfnum_rev[u] = dfs_count_rev;
            vertex_rev[dfs_count_rev] = u;
            semi_rev[u] = dfs_count_rev;
            label_rev[u] = u;
            dfs_count_rev++;

            for (size_t v : rev_succs[u]) {  // 逆CFGの後続（= 元CFGの前駆）を辿る
                if (dfnum_rev[v] == -1) {
                    parent_rev[v] = u;
                    dfs_rev(v);
                }
            }
        };

        dfs_rev(virtual_exit);

        std::function<void(int)> compress_rev = [&](int v) {
            int anc = ancestor_rev[v];
            if (ancestor_rev[anc] != -1) {
                compress_rev(anc);
                if (semi_rev[label_rev[anc]] < semi_rev[label_rev[v]]) {
                    label_rev[v] = label_rev[anc];
                }
                ancestor_rev[v] = ancestor_rev[anc];
            }
        };

        auto eval_rev = [&](int v) -> int {
            if (ancestor_rev[v] == -1) {
                return v;
            }
            compress_rev(v);
            return label_rev[v];
        };

        auto link_rev = [&](int u, int v) {
            ancestor_rev[v] = u;
        };

        for (int i = dfs_count_rev - 1; i >= 1; --i) {
            int w = vertex_rev[i];
            for (size_t v : rev_preds[w]) {  // 逆CFGの先行（= 元CFGの後続）を参照
                if (dfnum_rev[v] == -1) continue;
                int u = eval_rev(v);
                if (semi_rev[u] < semi_rev[w]) {
                    semi_rev[w] = semi_rev[u];
                }
            }
            bucket_rev[vertex_rev[semi_rev[w]]].push_back(w);
            link_rev(parent_rev[w], w);
            int p = parent_rev[w];
            for (int v : bucket_rev[p]) {
                int u = eval_rev(v);
                dom_rev[v] = (semi_rev[u] < semi_rev[v]) ? u : p;
            }
            bucket_rev[p].clear();
        }

        for (int i = 1; i < dfs_count_rev; ++i) {
            int w = vertex_rev[i];
            if (dom_rev[w] != vertex_rev[semi_rev[w]]) {
                dom_rev[w] = dom_rev[dom_rev[w]];
            }
        }

        // 6. 各基本ブロックがループに後支配（pdom）されてマージされるかの判定
        // ループヘッダからループインデックスへの高速ルックアップテーブルを構築
        std::vector<int> header_to_loop_idx(num_nodes, -1);
        for (size_t l = 0; l < num_loops; ++l) {
            if (valid_loops[l].valid) {
                header_to_loop_idx[valid_loops[l].header] = l;
            }
        }

        for (size_t u = 0; u < num_nodes; ++u) {
            bool directly_in_loop = false;
            for (size_t l = 0; l < num_loops; ++l) {
                if (!valid_loops[l].valid) continue;
                if (nodes[u].start_idx >= valid_loops[l].min_idx && 
                    (nodes[u].start_idx + nodes[u].size - 1) <= valid_loops[l].max_idx) {
                    directly_in_loop = true;
                    break;
                }
            }
            if (directly_in_loop) {
                node_merged_to_loop[u] = -2;
                continue;
            }

            int curr = dom_rev[u];
            std::vector<size_t> pdom_loop_headers;
            while (curr != -1 && curr != (int)virtual_exit) {
                if (curr < (int)num_nodes && header_to_loop_idx[curr] != -1) {
                    pdom_loop_headers.push_back(curr);
                }
                curr = dom_rev[curr];
            }

            if (!pdom_loop_headers.empty()) {
                int best_header = -1;
                size_t min_loop_nodes_size = SIZE_MAX;
                for (size_t h : pdom_loop_headers) {
                    size_t l = header_to_loop_idx[h];
                    // ループの前方・後方を問わず post-dominate されているBBをマージする。
                    // 最小サイズのループ（最も内側）を選ぶ。
                    if (valid_loops[l].member_nodes.size() < min_loop_nodes_size) {
                        min_loop_nodes_size = valid_loops[l].member_nodes.size();
                        best_header = h;
                    }
                }
                node_merged_to_loop[u] = best_header;
            }
        }
    }

    // 7. 出力 (onLoop と onBasicBlock)
    for (size_t i = 0; i < num_loops; ++i) {
        if (!valid_loops[i].valid) continue;
        onLoop(RegionSpan{globalOffset + valid_loops[i].min_idx, valid_loops[i].max_idx - valid_loops[i].min_idx + 1});
    }

    for (size_t u = 0; u < num_nodes; ++u) {
        if (node_merged_to_loop[u] == -1) {
            onBasicBlock(RegionSpan{globalOffset + nodes[u].start_idx, nodes[u].size});
        }
    }
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
