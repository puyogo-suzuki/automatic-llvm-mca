#include "mca_common.h"
#include "frontend.h"
#include "custom_a55_sched.h"
#include <cstdio>
#include <algorithm>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <unistd.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

struct ScopedSilence {
    int devNull = -1;
    int oldStderr = -1;
    bool active = false;
    ScopedSilence() {
        devNull = open("/dev/null", O_WRONLY);
        if (devNull != -1) {
            oldStderr = dup(STDERR_FILENO);
            if (dup2(devNull, STDERR_FILENO) != -1) active = true;
        }
    }
    ~ScopedSilence() {
        if (active) {
            errs().flush();
            dup2(oldStderr, STDERR_FILENO);
        }
        if (oldStderr != -1) close(oldStderr);
        if (devNull != -1) close(devNull);
    }
};

static void printResultCsv(const Instr &First, const Instr &Last, size_t Length, bool isLoop, const McaMetrics &M) {
    std::printf("0x%lx,0x%lx,%lu,%d,%lu,%lu,%u,%.2f,%.2f\n", First.Addr, Last.Addr,
                static_cast<unsigned long>(Length),
                isLoop ? 1 : 0,
                static_cast<unsigned long>(M.RetiredInstructions),
                static_cast<unsigned long>(M.LoadInstructions),
                static_cast<unsigned>(M.Cycles), M.MLP, M.MLP_R);
}

int main(int argc, char **argv) {
    InitLLVM X(argc, argv);
    std::unique_ptr<ObjectFile> Obj;
    TargetInfo TI;
    if (!initializeFrontend(argc, argv, "automatic-llvm-mca optimized C++ tool\n", Obj, TI)) {
        return 1;
    }

    std::printf("start_address,end_address,length,loop,retired_instructions,load_instructions,cycles,mlp,mlp_r\n");
    FunctionBoundaries FunctionRanges = collectFunctionBoundaries(*Obj);

    for (const SectionRef &Section : Obj->sections()) {
        if (!Section.isText() || Section.getSize() == 0) continue;

        if (auto NameOrErr = Section.getName()) {
            StringRef Name = *NameOrErr;
            if (Name == ".plt" || Name == ".init" || Name == ".fini" || 
                Name.starts_with(".plt.") || Name == ".plt.got" || Name == ".plt.sec") {
                continue;
            }
        }

        auto SectionInstrs = disassembleTextSection(Section, *TI.DisAsm, *TI.MCII, TI.MCIA.get());
        if (SectionInstrs.empty()) continue;

        std::vector<McaRegion> regions;

        walkRegions(SectionInstrs, FunctionRanges, opts::LoopMaxInstrs, opts::BBMaxInstrs, opts::NestLimitOuter, opts::NestLimitInner,
                    [&](const RegionSpan &Span) {
                        McaRegion r;
                        r.Start = Span.Start;
                        r.Size = Span.Size;
                        r.SimulatedSize = Span.Size;
                        r.IsLoop = true;
                        r.StartAddr = SectionInstrs[Span.Start].Addr;
                        r.EndAddr = SectionInstrs[Span.Start + Span.Size - 1].Addr + 4;
                        regions.push_back(r);
                    },
                    [&](const RegionSpan &Span) {
                        McaRegion r;
                        r.Start = Span.Start;
                        r.Size = Span.Size;
                        r.SimulatedSize = Span.Size;
                        r.IsLoop = false;
                        r.StartAddr = SectionInstrs[Span.Start].Addr;
                        r.EndAddr = SectionInstrs[Span.Start + Span.Size - 1].Addr + 4;
                        regions.push_back(r);
                    });

        // Map from region index to overwrite target loop index
        std::map<size_t, size_t> overwrite_map;
        computePostDominatorOverwrites(SectionInstrs, FunctionRanges, regions, overwrite_map);

        auto runMca = [&](McaRegion &r) {
            uint64_t regionAddr = r.StartAddr;
            if (TI.TargetAddress != 0 && regionAddr != TI.TargetAddress) return;
            
            auto region_instrs = ArrayRef<Instr>(SectionInstrs).slice(r.Start, r.Size);
            if (isAllNopRegion(region_instrs, *TI.MCII)) return;

            bool ignore = false;
            if (opts::IgnoreLoopCarried == IgnoreLoopCarriedMode::Force) {
                ignore = true;
            } else if (opts::IgnoreLoopCarried == IgnoreLoopCarriedMode::Default) {
                ignore = !r.IsLoop;
            } else if (opts::IgnoreLoopCarried == IgnoreLoopCarriedMode::Disable) {
                ignore = false;
            }
            bool mlpLoop = false;
            if (opts::MlpWindowLoop == MlpWindowLoopMode::Force) {
                mlpLoop = true;
            } else if (opts::MlpWindowLoop == MlpWindowLoopMode::Default) {
                mlpLoop = r.IsLoop;
            } else if (opts::MlpWindowLoop == MlpWindowLoopMode::Disable) {
                mlpLoop = false;
            }
            r.Metrics = analyzeMcaRegion(ArrayRef<Instr>(SectionInstrs).slice(r.Start, r.Size), *TI.STI, *TI.MCII,
                                         *TI.MRI, TI.MCIA.get(), TI.PO, opts::Iterations, TI.WindowWidthVal, opts::DepKind, opts::AssignKind,
                                         *TI.Analyzer, ignore, opts::OverrideLoadLatency, mlpLoop);
            r.Valid = r.Metrics.Valid;
        };

        ScopedSilence silence;

        // Pass 1: Run MCA for all loops and non-overwritten BBs
        for (size_t i = 0; i < regions.size(); ++i) {
            if (regions[i].IsLoop || overwrite_map.find(i) == overwrite_map.end()) {
                runMca(regions[i]);
            }
        }

        // Pass 2: Process overwritten BBs (copy loop results, or fallback to BB run if loop was invalid)
        for (size_t i = 0; i < regions.size(); ++i) {
            if (!regions[i].IsLoop && overwrite_map.find(i) != overwrite_map.end()) {
                size_t loop_idx = overwrite_map[i];
                if (regions[loop_idx].Valid) {
                    regions[i].Metrics = regions[loop_idx].Metrics;
                    regions[i].SimulatedSize = regions[loop_idx].SimulatedSize;
                    regions[i].Valid = true;
                } else {
                    runMca(regions[i]);
                }
            }
        }

        // Sort regions by StartAddr so the output is ordered by start_address
        std::vector<McaRegion> valid_regions;
        for (const auto &r : regions) {
            if (r.Valid) {
                valid_regions.push_back(r);
            }
        }
        std::sort(valid_regions.begin(), valid_regions.end(), [](const McaRegion &a, const McaRegion &b) {
            return a.EndAddr != b.EndAddr ? a.EndAddr < b.EndAddr : a.StartAddr < b.StartAddr;
        });

        // Output all valid results
        for (const auto &r : valid_regions) {
            printResultCsv(SectionInstrs[r.Start], SectionInstrs[r.Start + r.Size - 1], r.SimulatedSize, r.IsLoop, r.Metrics);
        }
    }

    return 0;
}
