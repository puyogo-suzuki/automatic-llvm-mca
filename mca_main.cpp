#include "mca_common.h"
#include "custom_a55_sched.h"
#include <cstdio>
#include <algorithm>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <unistd.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace llvm::object;

static cl::opt<std::string> InputBinary(cl::Positional, cl::desc("<input binary>"), cl::Required);
static cl::opt<std::string> MTriple("mtriple", cl::desc("Target triple"));
static cl::opt<std::string> MCPU("mcpu", cl::desc("Target CPU"));
static cl::opt<int> WindowWidth("window-width", cl::desc("MLP window width"), cl::init(4));
static cl::opt<DependencyKind> DepKind("dependency", cl::desc("Dependency mode"),
    cl::values(
        clEnumValN(DependencyKind::None, "none", "No dependency tracking"),
        clEnumValN(DependencyKind::IO, "io", "In-order dependency"),
        clEnumValN(DependencyKind::OOO, "ooo", "Out-of-order dependency"),
        clEnumValN(DependencyKind::Dependency, "dependency", "Load-use dependency distance")
    ), cl::init(DependencyKind::None));
static cl::opt<MLPWindowAssignmentKind> AssignKind("mlp-window-assignment", cl::desc("Per-load MLP assignment mode"),
    cl::values(
        clEnumValN(MLPWindowAssignmentKind::Forward, "forward", "Forward window"),
        clEnumValN(MLPWindowAssignmentKind::MaxContaining, "max-containing", "Max MLP of containing windows")
    ), cl::init(MLPWindowAssignmentKind::MaxContaining));
static cl::opt<int> Iterations("iterations", cl::desc("Steady-state iteration multiplier"), cl::init(2));
static cl::opt<int> LoopMaxInstrs("loop-max-instrs", cl::desc("Maximum instructions in a loop to analyze"), cl::init(100));
static cl::opt<int> BBMaxInstrs("bb-max-instrs", cl::desc("Maximum instructions in a basic block to analyze"), cl::init(100));
static cl::opt<IgnoreLoopCarriedMode> IgnoreLoopCarried("ignore-loop-carried",
    cl::desc("Ignore loop-carried register dependencies mode"),
    cl::values(
        clEnumValN(IgnoreLoopCarriedMode::Default, "default", "Ignore in basic blocks, but not in loops"),
        clEnumValN(IgnoreLoopCarriedMode::Force, "force", "Ignore in both loops and basic blocks"),
        clEnumValN(IgnoreLoopCarriedMode::Disable, "disable", "Do not ignore loop-carried dependencies anywhere")
    ), cl::init(IgnoreLoopCarriedMode::Default));
static cl::opt<int> OverrideLoadLatency("override-load-latency",
    cl::desc("Override load instruction latency in cycles"),
    cl::init(-1));
static cl::opt<MlpWindowLoopMode> MlpWindowLoop("mlp-window-loop",
    cl::desc("Loop back to the start of the basic block mode"),
    cl::values(
        clEnumValN(MlpWindowLoopMode::Default, "default", "Loop back to the start only for loops"),
        clEnumValN(MlpWindowLoopMode::Force, "force", "Always loop back to the start (even for non-loops)"),
        clEnumValN(MlpWindowLoopMode::Disable, "disable", "Never loop back to the start")
    ), cl::init(MlpWindowLoopMode::Default));
static cl::opt<std::string> TargetAddressStr("target-address",
    cl::desc("Only analyze region starting at this hex address"),
    cl::init(""));

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
    initializeTargets();

    cl::ParseCommandLineOptions(argc, argv, "automatic-llvm-mca optimized C++ tool\n");

    auto BinaryOrErr = ObjectFile::createObjectFile(InputBinary);
    if (!BinaryOrErr) {
        WithColor::error() << "Failed to open binary: " << toString(BinaryOrErr.takeError()) << "\n";
        return 1;
    }
    ObjectFile &Obj = *BinaryOrErr.get().getBinary();

    Triple TT = Obj.makeTriple();
    if (!MTriple.empty()) TT = Triple(MTriple);
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
    if (!TheTarget) {
        WithColor::error() << "No target for " << TT.str() << ": " << Error << "\n";
        return 1;
    }

    std::string CPU = MCPU.empty() ? "generic" : std::string(MCPU);
    std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TT));
    MCTargetOptions MCOPT;
    std::unique_ptr<MCAsmInfo> MAI(TheTarget->createMCAsmInfo(*MRI, TT, MCOPT));
    std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
    std::unique_ptr<MCSubtargetInfo> STI(TheTarget->createMCSubtargetInfo(TT, CPU, ""));
    if (STI) {
        llvm::overrideCortexA55SchedModel(*STI);
    }
    MCContext Ctx(TT, MAI.get(), MRI.get(), STI.get());
    std::unique_ptr<MCDisassembler> DisAsm(TheTarget->createMCDisassembler(*STI, Ctx));
    std::unique_ptr<MCInstrAnalysis> MCIA(TheTarget->createMCInstrAnalysis(MCII.get()));

    if (!MRI || !MAI || !MCII || !STI || !DisAsm || !MCIA) {
        WithColor::error() << "Failed to initialize LLVM components\n";
        return 1;
    }

    mca::PipelineOptions PO(0, 0, 0, 0, 0, 0, true);
    const MCSchedModel &SM = STI->getSchedModel();
    int windowWidth = WindowWidth;
    if (WindowWidth.getNumOccurrences() == 0 && !MCPU.empty()) {
        if (SM.MicroOpBufferSize > 0) {
            windowWidth = SM.MicroOpBufferSize;
        } else {
            windowWidth = SM.IssueWidth * SM.MispredictPenalty;
        }
    }
    PO.MicroOpQueueSize = SM.MicroOpBufferSize;
    PO.DispatchWidth = SM.IssueWidth;
    if (STI->getCPU() == "cortex-a76" || STI->getCPU() == "cortex-a76ae" || STI->getCPU() == "neoverse-n1") {
        PO.DispatchWidth = 8;
    } else if (STI->getCPU() == "cortex-a78" || STI->getCPU() == "cortex-a78ae" || STI->getCPU() == "cortex-a78c") {
        PO.DispatchWidth = 12;
    } else if (STI->getCPU() == "cortex-a710" || STI->getCPU() == "cortex-a715" || STI->getCPU() == "cortex-a720" || STI->getCPU() == "cortex-a720ae" || STI->getCPU() == "neoverse-n2") {
        PO.DispatchWidth = 10;
    } else if (STI->getCPU() == "cortex-x1" || STI->getCPU() == "cortex-x1c" || STI->getCPU() == "neoverse-v1") {
        PO.DispatchWidth = 16;
    }
    PO.AssumeNoAlias = true;

    uint64_t TargetAddress = 0;
    if (!TargetAddressStr.empty()) {
        TargetAddress = std::stoull(TargetAddressStr, nullptr, 16);
    }

    std::unique_ptr<MLPAnalyzer> Analyzer = MLPAnalyzer::create(*STI);

    std::printf("start_address,end_address,length,loop,retired_instructions,load_instructions,cycles,mlp,mlp_r\n");
    FunctionBoundaries FunctionRanges = collectFunctionBoundaries(Obj);

    for (const SectionRef &Section : Obj.sections()) {
        if (!Section.isText() || Section.getSize() == 0) continue;

        auto SectionInstrs = disassembleTextSection(Section, *DisAsm, *MCII, MCIA.get());
        if (SectionInstrs.empty()) continue;

        std::vector<McaRegion> regions;

        walkRegions(SectionInstrs, FunctionRanges, LoopMaxInstrs, BBMaxInstrs,
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
            if (TargetAddress != 0 && regionAddr != TargetAddress) return;
            bool ignore = false;
            if (IgnoreLoopCarried == IgnoreLoopCarriedMode::Force) {
                ignore = true;
            } else if (IgnoreLoopCarried == IgnoreLoopCarriedMode::Default) {
                ignore = !r.IsLoop;
            } else if (IgnoreLoopCarried == IgnoreLoopCarriedMode::Disable) {
                ignore = false;
            }
            bool mlpLoop = false;
            if (MlpWindowLoop == MlpWindowLoopMode::Force) {
                mlpLoop = true;
            } else if (MlpWindowLoop == MlpWindowLoopMode::Default) {
                mlpLoop = r.IsLoop;
            } else if (MlpWindowLoop == MlpWindowLoopMode::Disable) {
                mlpLoop = false;
            }
            r.Metrics = analyzeMcaRegion(ArrayRef<Instr>(SectionInstrs).slice(r.Start, r.Size), *STI, *MCII,
                                         *MRI, MCIA.get(), PO, Iterations, windowWidth, DepKind, AssignKind,
                                         *Analyzer, ignore, OverrideLoadLatency, mlpLoop);
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
            if (a.StartAddr != b.StartAddr)
                return a.StartAddr < b.StartAddr;
            return a.EndAddr < b.EndAddr;
        });

        // Output all valid results
        for (const auto &r : valid_regions) {
            printResultCsv(SectionInstrs[r.Start], SectionInstrs[r.Start + r.Size - 1], r.SimulatedSize, r.IsLoop, r.Metrics);
        }
    }

    return 0;
}
