#include "core/object_loader.h"
#include "core/region_splitter.h"
#include "facile/facile.h"
#include "frontend/frontend.h"
#include "mca/mca_simulation.h"
#include "mca/zero_aware_instr_builder.h"
#include "mlp/mlp_types.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

// EXPERIMENTAL (opt-in, default off): print the facile cycle estimate unrounded.
// M.Cycles is std::round(EstimatedCycles); for short regions on a wide core
// (e.g. an 8-instr region at ~1.6 cycles on the A78) that quantisation alone is
// a +-0.5 cycle (up to ~30%) error in BB_CPI. The unrounded value is
// BaseCPI * RetiredInstructions = EstimatedCycles exactly.
static cl::opt<bool> FractionalCycles("fractional-cycles",
    cl::desc("EXPERIMENTAL: print facile cycles unrounded (%.3f) instead of rounded to an integer"),
    cl::init(false));

static void printResultCsv(uint64_t StartAddr, uint64_t EndAddr, uint64_t AnalysisStartAddr, uint64_t AnalysisEndAddr,
                           size_t Length, bool isLoop, const McaMetrics &M) {
    if (FractionalCycles && M.BaseCPI > 0.0) {
        std::printf("0x%lx,0x%lx,0x%lx,0x%lx,%lu,%d,%lu,%lu,%.3f,%.2f,%.2f",
                    StartAddr, EndAddr, AnalysisStartAddr, AnalysisEndAddr,
                    static_cast<unsigned long>(Length),
                    isLoop ? 1 : 0,
                    static_cast<unsigned long>(M.RetiredInstructions),
                    static_cast<unsigned long>(M.LoadInstructions),
                    M.BaseCPI * static_cast<double>(M.RetiredInstructions), M.MLP, M.MLP_R);
    } else {
        std::printf("0x%lx,0x%lx,0x%lx,0x%lx,%lu,%d,%lu,%lu,%u,%.2f,%.2f",
                    StartAddr, EndAddr, AnalysisStartAddr, AnalysisEndAddr,
                    static_cast<unsigned long>(Length),
                    isLoop ? 1 : 0,
                    static_cast<unsigned long>(M.RetiredInstructions),
                    static_cast<unsigned long>(M.LoadInstructions),
                    static_cast<unsigned>(M.Cycles), M.MLP, M.MLP_R);
    }
    if (opts::FacileReason) {
        std::printf(",%s", M.FacileReason.c_str());
    }
    std::printf("\n");
}

static void printCsvHeader() {
    std::printf("start_address,end_address,analysis_start,analysis_end,length,loop,retired_instructions,load_instructions,cycles,mlp,mlp_r%s\n",
                opts::FacileReason ? ",facile_reason" : "");
}

// --update-mlp: reads the retired-instruction and cycle columns of a previous
// mca_tool CSV, keyed by start address.  Rows that do not parse are skipped.
// Returns false if the file cannot be opened.
static bool loadPreviousResults(const std::string &Path, std::map<uint64_t, McaMetrics> &Cache) {
    std::ifstream infile(Path.c_str());
    if (!infile.is_open()) {
        WithColor::error() << "Failed to open --update-mlp CSV: " << Path << "\n";
        return false;
    }
    std::string line;
    std::getline(infile, line);  // header
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        // start_address .. cycles (column 8) must be present.
        if (row.size() < 9) continue;
        try {
            uint64_t start_addr = std::stoull(row[0], nullptr, 16);
            McaMetrics M;
            M.RetiredInstructions = std::stoull(row[6]);
            M.Cycles = std::stoul(row[8]);
            M.Valid = true;
            Cache[start_addr] = M;
        } catch (...) {
            // ignore parse errors
        }
    }
    return true;
}

// Text sections that hold no analysable code (PLT stubs, init/fini glue).
static bool isSkippedTextSection(StringRef Name) {
    return Name == ".plt" || Name == ".init" || Name == ".fini" ||
           Name.starts_with(".plt.") || Name == ".plt.got" || Name == ".plt.sec";
}

static McaRegion makeRegion(const RegionSpan &Span, bool IsLoop, ArrayRef<Instr> SectionInstrs) {
    // Every AArch64 instruction is 4 bytes: the end address is one past the last one.
    constexpr uint64_t InstrBytes = 4;
    McaRegion r;
    r.Start = Span.Start;
    r.Size = Span.Size;
    r.AnalysisStart = Span.AnalysisStart;
    r.AnalysisSize = Span.AnalysisSize;
    r.SimulatedSize = Span.AnalysisSize;
    r.IsLoop = IsLoop;
    r.IsAbabMerged = Span.IsAbabMerged;
    r.StartAddr = SectionInstrs[Span.Start].Addr;
    r.EndAddr = SectionInstrs[Span.Start + Span.Size - 1].Addr + InstrBytes;
    r.AnalysisStartAddr = SectionInstrs[Span.AnalysisStart].Addr;
    r.AnalysisEndAddr = SectionInstrs[Span.AnalysisStart + Span.AnalysisSize - 1].Addr + InstrBytes;
    return r;
}

// --ignore-loop-carried: by default only regions that are not loops ignore
// loop-carried register dependences.
static bool shouldIgnoreLoopCarried(bool IsLoop) {
    switch (opts::IgnoreLoopCarried) {
    case IgnoreLoopCarriedMode::Force:   return true;
    case IgnoreLoopCarriedMode::Default: return !IsLoop;
    case IgnoreLoopCarriedMode::Disable: return false;
    }
    return false;
}

// --mlp-window-loop: by default only loops wrap the MLP window around.
static bool shouldWrapMlpWindow(bool IsLoop) {
    switch (opts::MlpWindowLoop) {
    case MlpWindowLoopMode::Force:   return true;
    case MlpWindowLoopMode::Default: return IsLoop;
    case MlpWindowLoopMode::Disable: return false;
    }
    return false;
}

// --facile: the analytical estimate of computeFacilePrediction() plus the MLP
// metrics, for one region.
static McaMetrics analyzeFacileRegion(const TargetInfo &TI, ArrayRef<Instr> region_instrs,
                                      bool ignoreLoopCarried, bool mlpLoop) {
    mca::InstrumentManager IM(*TI.STI, *TI.MCII);
    ZeroAwareInstrBuilder IB(*TI.STI, *TI.MCII, *TI.MRI, TI.MCIA.get(), IM, 0);
    std::vector<std::unique_ptr<mca::Instruction>> SimInstrs;
    std::vector<const MCInst *> MCInsts;
    // Index-aligned with SimInstrs/MCInsts: the memory-access
    // descriptor facile uses for store->load precedence edges.
    std::vector<MemAccessInfo> MemInfos;
    buildRegionInstructions(IB, *TI.STI, *TI.MCII, region_instrs,
                            [&](const Instr &I, std::unique_ptr<mca::Instruction> Inst) {
        SimInstrs.push_back(std::move(Inst));
        MCInsts.push_back(&I.Inst);
        if (!opts::NoFacileMemoryDeps) {
            const MCInstrDesc &MCID = TI.MCII->get(I.Inst.getOpcode());
            MemInfos.push_back(TI.Analyzer->getMemAccessInfo(I.Inst, MCID, *TI.MRI, *TI.MCII));
        }
    });

    facile::FacileResult Res = facile::computeFacilePrediction(*TI.STI, *TI.MCII, *TI.MRI, SimInstrs, MCInsts, TI.PO.DispatchWidth, MemInfos, TI.MopDispatchWidth, !ignoreLoopCarried);

    McaMetrics M;
    M.RetiredInstructions = static_cast<uint64_t>(region_instrs.size());
    M.LoadInstructions = TI.Analyzer->countPotentialMissLoads(region_instrs, *TI.STI, *TI.MCII, *TI.MRI, opts::DepKind, mlpLoop);
    M.Cycles = static_cast<uint64_t>(std::round(Res.EstimatedCycles));
    M.MLP = TI.Analyzer->compute_mlp(region_instrs, TI.WindowWidthVal, opts::DepKind, opts::AssignKind, *TI.STI, *TI.MCII, *TI.MRI, M.MLP_R, mlpLoop);
    M.BaseCPI = Res.EstimatedCPI;
    M.FacileReason = Res.FacileReason;
    M.Valid = true;
    return M;
}

// --update-mlp: the cycle count of a previous run, with the MLP metrics
// recomputed.
static McaMetrics reusePreviousResult(const TargetInfo &TI, ArrayRef<Instr> region_instrs,
                                      const McaMetrics &Previous, bool mlpLoop) {
    McaMetrics M;
    M.RetiredInstructions = Previous.RetiredInstructions;
    M.Cycles = Previous.Cycles;
    M.LoadInstructions = TI.Analyzer->countPotentialMissLoads(region_instrs, *TI.STI, *TI.MCII, *TI.MRI, opts::DepKind, mlpLoop);
    float mlp_r = 0.0f;
    M.MLP = TI.Analyzer->compute_mlp(region_instrs, TI.WindowWidthVal, opts::DepKind, opts::AssignKind, *TI.STI, *TI.MCII, *TI.MRI, mlp_r, mlpLoop);
    M.MLP_R = mlp_r;
    if (M.RetiredInstructions > 0) {
        M.BaseCPI = static_cast<double>(M.Cycles) / static_cast<double>(M.RetiredInstructions);
    }
    M.Valid = true;
    return M;
}

int main(int argc, char **argv) {
    InitLLVM X(argc, argv);
    std::unique_ptr<ObjectFile> Obj;
    TargetInfo TI;
    if (!initializeFrontend(argc, argv, "automatic-llvm-mca optimized C++ tool\n", Obj, TI)) {
        return 1;
    }

    std::map<uint64_t, McaMetrics> csv_cache;
    if (!opts::UpdateMlp.empty() && !loadPreviousResults(opts::UpdateMlp, csv_cache)) {
        return 1;
    }

    printCsvHeader();
    FunctionBoundaries FunctionRanges = collectFunctionBoundaries(*Obj);

    for (const SectionRef &Section : Obj->sections()) {
        if (!Section.isText() || Section.getSize() == 0) continue;

        if (auto NameOrErr = Section.getName()) {
            if (isSkippedTextSection(*NameOrErr)) continue;
        }

        auto SectionInstrs = disassembleTextSection(Section, *TI.DisAsm, *TI.MCII, TI.MCIA.get());

        if (SectionInstrs.empty()) continue;

        std::vector<McaRegion> regions;
        walkRegions(SectionInstrs, FunctionRanges,
                    [&](const RegionSpan &Span) { regions.push_back(makeRegion(Span, /*IsLoop=*/true, SectionInstrs)); },
                    [&](const RegionSpan &Span) { regions.push_back(makeRegion(Span, /*IsLoop=*/false, SectionInstrs)); });

        auto runMca = [&](McaRegion &r) {
            uint64_t regionAddr = r.StartAddr;
            if (TI.TargetAddress != 0 && regionAddr != TI.TargetAddress) return;

            auto region_instrs = ArrayRef<Instr>(SectionInstrs).slice(r.AnalysisStart, r.AnalysisSize);
            if (isAllNopRegion(region_instrs, *TI.MCII)) return;

            const bool ignore = shouldIgnoreLoopCarried(r.IsLoop);
            const bool mlpLoop = shouldWrapMlpWindow(r.IsLoop);

            if (opts::Facile) {
                r.Metrics = analyzeFacileRegion(TI, region_instrs, ignore, mlpLoop);
                r.Valid = true;
                return;
            }

            bool reused = false;
            if (!opts::UpdateMlp.empty()) {
                auto it = csv_cache.find(regionAddr);
                if (it != csv_cache.end()) {
                    r.Metrics = reusePreviousResult(TI, region_instrs, it->second, mlpLoop);
                    reused = true;
                }
            }

            if (!reused) {
                r.Metrics = analyzeMcaRegion(region_instrs, *TI.STI, *TI.MCII,
                                             *TI.MRI, TI.MCIA.get(), TI.PO, opts::Iterations, TI.WindowWidthVal, opts::DepKind, opts::AssignKind,
                                             *TI.Analyzer, ignore, opts::OverrideLoadLatency, mlpLoop);
            }
            r.Valid = r.Metrics.Valid;
        };

        if (opts::CountOnly) {
            std::printf("TOTAL_REGIONS:%lu\n", static_cast<unsigned long>(regions.size()));
            continue;
        }

        ScopedSilence silence;

        for (McaRegion &r : regions) {
            runMca(r);
        }

        // Output the valid regions ordered by end address, then start address.
        std::vector<McaRegion> valid_regions;
        for (const auto &r : regions) {
            if (r.Valid) {
                valid_regions.push_back(r);
            }
        }
        std::sort(valid_regions.begin(), valid_regions.end(), [](const McaRegion &a, const McaRegion &b) {
            return a.EndAddr != b.EndAddr ? a.EndAddr < b.EndAddr : a.StartAddr < b.StartAddr;
        });

        for (const auto &r : valid_regions) {
            printResultCsv(r.StartAddr, r.EndAddr, r.AnalysisStartAddr, r.AnalysisEndAddr, r.SimulatedSize, r.IsLoop, r.Metrics);
        }
    }

    // stderr is silenced while regions are analysed (ScopedSilence above), so
    // the instructions InstrBuilder rejected are reported once, here.
    reportDroppedInstructions(llvm::errs());

    return 0;
}
