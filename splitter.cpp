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
                 int minBBSize,
                 const std::function<void(const RegionSpan &)> &onLoop,
                 const std::function<void(const RegionSpan &)> &onBasicBlock) {
    if (instrs.empty()) return;

    // --- Pass 1: mark which instructions belong to a loop region ---
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

    // --- Pass 2: emit loops and merged basic blocks ---
    // A "hard boundary" for BB merging is any instruction where:
    //   - it's in a loop region, OR
    //   - it ends with an unconditional branch / return (cannot fall through)
    //
    // We accumulate consecutive non-loop BB segments.
    // When we hit a hard boundary after emitting the current BB instruction,
    // we flush the pending region.
    //
    // A conditional branch (EndsBB but NOT IsUnconditionalBranch) is a "soft boundary":
    // we still end the current atomic BB at that point, but if the accumulated
    // region is still below minBBSize we continue accumulating into the next BB.

    // Helper: flush the pending accumulated region if non-empty
    size_t pendingStart = 0;
    size_t pendingSize  = 0;
    bool   hasPending   = false;

    auto flushPending = [&]() {
        if (!hasPending || pendingSize == 0) return;
        if (bbMaxInstrs <= 0 || pendingSize <= static_cast<size_t>(bbMaxInstrs)) {
            if (onBasicBlock) onBasicBlock(RegionSpan{pendingStart, pendingSize});
        }
        hasPending = false;
        pendingSize = 0;
    };

    size_t atomicStart = 0;  // start of current atomic BB (since last hard/soft boundary)
    bool   inAtomicBB  = false;

    for (size_t i = 0; i < instrs.size(); ++i) {
        const auto &I = instrs[i];

        // Emit loops as before
        if (auto loop = getLoopSpan(i)) {
            if (onLoop) onLoop(*loop);
        }

        if (inLoop[i]) {
            // We've entered a loop region — flush whatever we had and stop accumulating
            if (inAtomicBB) {
                // Flush the partial atomic BB accumulated so far (may be below minBBSize — emit as-is)
                if (hasPending) {
                    // Merge the partial segment into pending
                    size_t segSize = i - atomicStart;
                    if (segSize > 0) pendingSize += segSize;
                }
                flushPending();
                inAtomicBB = false;
            } else {
                flushPending();
            }
            continue;
        }

        // Non-loop instruction
        if (!inAtomicBB) {
            // Check function boundary: if we have a pending region from a different function, flush first
            if (hasPending) {
                const auto &pendingFirst = instrs[pendingStart];
                if (!boundaries.empty() && !sameFunction(boundaries, pendingFirst.Addr, I.Addr)) {
                    flushPending();
                }
            }
            atomicStart = i;
            inAtomicBB  = true;
        }

        if (I.EndsBB) {
            // End of current atomic BB
            size_t atomicSize = i - atomicStart + 1;
            bool isHard = I.IsUnconditionalBranch; // return or unconditional/indirect branch

            // Function boundary check: if the last instr of next potential BB
            // would cross a function boundary, treat as hard
            // (we'll check at the start of the next iteration instead)

            if (!hasPending) {
                pendingStart = atomicStart;
                pendingSize  = 0;
                hasPending   = true;
            }
            pendingSize += atomicSize;
            inAtomicBB = false;

            if (isHard || pendingSize >= static_cast<size_t>(minBBSize <= 0 ? 1 : minBBSize)) {
                // Hard boundary or reached min size: flush
                flushPending();
            }
            // else: soft boundary (conditional branch) — keep accumulating
        }
    }

    // Flush any remaining pending region at end of section
    if (inAtomicBB) {
        // Unterminated trailing instructions
        size_t segSize = instrs.size() - atomicStart;
        if (hasPending) {
            pendingSize += segSize;
        } else {
            pendingStart = atomicStart;
            pendingSize  = segSize;
            hasPending   = true;
        }
    }
    flushPending();
}
