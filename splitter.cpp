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
    return Boundaries.empty() || (containsAddress(Boundaries, A) && containsAddress(Boundaries, B));
}

int64_t findIndex(ArrayRef<Instr> instrs, uint64_t addr) {
    auto it = std::lower_bound(instrs.begin(), instrs.end(), addr,
                               [](const Instr &a, uint64_t val) { return a.Addr < val; });
    if (it != instrs.end() && it->Addr == addr) return std::distance(instrs.begin(), it);
    return -1;
}

} // namespace

void walkRegions(ArrayRef<Instr> instrs, const FunctionBoundaries &boundaries, int loopMaxInstrs, int bbMaxInstrs,
                 const std::function<void(const RegionSpan &)> &onLoop,
                 const std::function<void(const RegionSpan &)> &onBasicBlock) {
    if (instrs.empty()) return;

    std::vector<bool> inLoop(instrs.size(), false);

    auto getLoopSpan = [&](size_t idx) -> std::optional<RegionSpan> {
        const auto &I = instrs[idx];
        if (!I.IsBranch || I.BranchTarget == 0 || I.BranchTarget >= I.Addr) return std::nullopt;

        int64_t startIdx = findIndex(instrs, I.BranchTarget);
        if (startIdx == -1) return std::nullopt;

        const size_t start = static_cast<size_t>(startIdx);
        const size_t size = idx - start + 1;
        if (loopMaxInstrs > 0 && size > static_cast<size_t>(loopMaxInstrs)) return std::nullopt;
        if (!sameFunction(boundaries, I.Addr, I.BranchTarget)) return std::nullopt;
        return RegionSpan{start, size};
    };

    for (size_t i = 0; i < instrs.size(); ++i) {
        if (auto loop = getLoopSpan(i)) {
            for (size_t j = loop->Start; j < loop->Start + loop->Size; ++j) inLoop[j] = true;
        }
    }

    size_t bbStart = 0;
    bool inBB = false;
    for (size_t i = 0; i < instrs.size(); ++i) {
        const auto &I = instrs[i];

        if (auto loop = getLoopSpan(i)) {
            if (onLoop) onLoop(*loop);
        }

        if (inLoop[i]) {
            if (inBB) {
                const size_t bbSize = i - bbStart;
                if (bbMaxInstrs <= 0 || bbSize <= static_cast<size_t>(bbMaxInstrs)) {
                    if (onBasicBlock) onBasicBlock(RegionSpan{bbStart, bbSize});
                }
                inBB = false;
            }
            continue;
        }

        if (!inBB) {
            bbStart = i;
            inBB = true;
        }

        if (I.EndsBB) {
            const size_t bbSize = i - bbStart + 1;
            if (bbMaxInstrs <= 0 || bbSize <= static_cast<size_t>(bbMaxInstrs)) {
                if (onBasicBlock) onBasicBlock(RegionSpan{bbStart, bbSize});
            }
            inBB = false;
        }
    }

    if (inBB) {
        const size_t bbSize = instrs.size() - bbStart;
        if (bbMaxInstrs <= 0 || bbSize <= static_cast<size_t>(bbMaxInstrs)) {
            if (onBasicBlock) onBasicBlock(RegionSpan{bbStart, bbSize});
        }
    }
}
