#include "mca_common.h"
#include "frontend.h"
#include "custom_a55_sched.h"
#include <cstdio>
#include <algorithm>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <unistd.h>

#include <fstream>
#include <sstream>
#include <map>
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


    std::map<uint64_t, McaMetrics> csv_cache;
    if (!opts::UpdateMlp.empty()) {
        std::ifstream infile(opts::UpdateMlp.c_str());
        if (!infile.is_open()) {
            WithColor::error() << "Failed to open --update-mlp CSV: " << opts::UpdateMlp << "\n";
            return 1;
        }
        std::string line;
        if (std::getline(infile, line)) {
            // skip header
        }
        while (std::getline(infile, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;
            while (std::getline(ss, cell, ',')) {
                row.push_back(cell);
            }
            if (row.size() < 7) continue;
            try {
                uint64_t start_addr = std::stoull(row[0], nullptr, 16);
                McaMetrics M;
                M.RetiredInstructions = std::stoull(row[4]);
                M.Cycles = std::stoul(row[6]);
                M.Valid = true;
                csv_cache[start_addr] = M;
            } catch (...) {
                // ignore parse errors
            }
        }
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
            bool reused = false;
            if (!opts::UpdateMlp.empty()) {
                auto it = csv_cache.find(regionAddr);
                if (it != csv_cache.end()) {
                    McaMetrics M;
                    M.RetiredInstructions = it->second.RetiredInstructions;
                    M.Cycles = it->second.Cycles;
                    M.LoadInstructions = TI.Analyzer->countPotentialMissLoads(region_instrs, *TI.STI, *TI.MCII, *TI.MRI, opts::DepKind);
                    float mlp_r = 0.0f;
                    M.MLP = TI.Analyzer->compute_mlp(region_instrs, TI.WindowWidthVal, opts::DepKind, opts::AssignKind, *TI.STI, *TI.MCII, *TI.MRI, mlp_r, mlpLoop);
                    M.MLP_R = mlp_r;
                    if (M.RetiredInstructions > 0) {
                        M.BaseCPI = static_cast<double>(M.Cycles) / static_cast<double>(M.RetiredInstructions);
                    }
                    M.Valid = true;
                    r.Metrics = M;
                    reused = true;
                }
            }

            if (!reused) {
                r.Metrics = analyzeMcaRegion(ArrayRef<Instr>(SectionInstrs).slice(r.Start, r.Size), *TI.STI, *TI.MCII,
                                             *TI.MRI, TI.MCIA.get(), TI.PO, opts::Iterations, TI.WindowWidthVal, opts::DepKind, opts::AssignKind,
                                             *TI.Analyzer, ignore, opts::OverrideLoadLatency, mlpLoop);
            }
            r.Valid = r.Metrics.Valid;
        };

        ScopedSilence silence;

        // Run MCA for all regions
        for (size_t i = 0; i < regions.size(); ++i) {
            runMca(regions[i]);
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
