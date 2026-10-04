#include "core/object_loader.h"
#include "core/types.h"
#include "frontend/frontend.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "llvm/Support/InitLLVM.h"

using namespace llvm;
using namespace llvm::object;

struct Stats {
    size_t count = 0;
    size_t max_size = 0;
    double total_instrs = 0;
};

void print_stats(const std::string& label, const Stats& s) {
    std::cout << label << ":\n";
    std::cout << "  Count:      " << s.count << "\n";
    std::cout << "  Max Size:   " << s.max_size << "\n";
    std::cout << "  Avg Size:   " << (s.count > 0 ? s.total_instrs / s.count : 0) << "\n";
}

int main(int argc, char **argv) {
    InitLLVM X(argc, argv);

    std::unique_ptr<ObjectFile> Obj;
    TargetInfo TI;
    if (!initializeFrontend(argc, argv, "Analysis Strategy Comparator\n", Obj, TI)) return 1;

    FunctionBoundaries FunctionRanges = collectFunctionBoundaries(*Obj);

    Stats optLoop;
    std::map<size_t, std::map<size_t, size_t>> dist2D; // [InstrBucket100][BranchCount] -> Freq

    for (const SectionRef &Section : Obj->sections()) {
        if (!Section.isText() || Section.getSize() == 0) continue;

        std::vector<Instr> SectionInstrs = disassembleTextSection(Section, *TI.DisAsm, *TI.MCII, TI.MCIA.get());

        auto find_idx = [&](uint64_t addr) -> int64_t {
            auto it = std::lower_bound(SectionInstrs.begin(), SectionInstrs.end(), addr,
                                       [](const Instr& a, uint64_t val) { return a.Addr < val; });
            if (it != SectionInstrs.end() && it->Addr == addr)
                return std::distance(SectionInstrs.begin(), it);
            return -1;
        };
        
        for (size_t i = 0; i < SectionInstrs.size(); ++i) {
            const auto& I = SectionInstrs[i];
            if (I.IsBranch && I.BranchTarget != 0 && I.BranchTarget < I.Addr) {
                int64_t start_idx = find_idx(I.BranchTarget);
                if (start_idx != -1) {
                    size_t sz = i - (size_t)start_idx + 1;
                    bool same_func = false;
                    auto it = FunctionRanges.upper_bound(I.Addr);
                    if (it != FunctionRanges.begin()) {
                        --it;
                        if (I.Addr >= it->first && I.Addr < it->second && I.BranchTarget >= it->first && I.BranchTarget < it->second) same_func = true;
                    }
                    if (same_func && sz <= 1000) {
                        optLoop.count++;
                        optLoop.total_instrs += sz;
                        if (sz > optLoop.max_size) optLoop.max_size = sz;

                        size_t branchCount = 0;
                        for (size_t j = (size_t)start_idx; j < i; ++j) {
                            if (SectionInstrs[j].IsBranch) branchCount++;
                        }
                        size_t instrBucket = (sz / 100) * 100;
                        dist2D[instrBucket][branchCount]++;
                    }
                }
            }
        }
    }

    std::cout << "=== 2D Distribution: Instr Count (Buckets of 100) vs Internal Branch Count ===\n";
    std::cout << "InstrRange, BranchCount, Frequency\n";
    for (auto const& [instrBucket, branchMap] : dist2D) {
        for (auto const& [brCount, freq] : branchMap) {
            std::cout << instrBucket << "-" << (instrBucket + 99) << ", " << brCount << ", " << freq << "\n";
        }
    }

    return 0;
}
