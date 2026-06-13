#include "mca_common.h"
#include "custom_a55_sched.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/MCA/InstrBuilder.h"
#include "llvm/MCA/Pipeline.h"
#include "llvm/MCA/SourceMgr.h"
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

namespace {

using SpanKey = std::pair<size_t, size_t>;

struct RegionMetrics {
    double mlp = 0.0;
    double base_cpi = 0.0;
    bool valid = false;
};

void printSectionHeader(StringRef Name) {
    std::cout << "Disassembly of section " << Name.str() << ":\n";
}

std::string formatAddress(uint64_t Addr) {
    std::ostringstream OS;
    OS << "0x" << std::hex << std::setw(16) << std::setfill('0') << Addr;
    return OS.str();
}

std::string formatMetrics(const RegionMetrics &M) {
    if (!M.valid) return "  --.-  --.--";
    std::ostringstream OS;
    OS << std::fixed << std::setw(5) << std::setprecision(1) << M.mlp << "  " << std::setw(6)
       << std::setprecision(2) << M.base_cpi;
    return OS.str();
}

void mergeMetrics(RegionMetrics &Dst, const McaMetrics &Src) {
    if (!Src.Valid) return;
    if (!Dst.valid) {
        Dst.mlp = Src.MLP;
        Dst.base_cpi = Src.BaseCPI;
        Dst.valid = true;
        return;
    }
    Dst.mlp = std::max(Dst.mlp, static_cast<double>(Src.MLP));
    Dst.base_cpi = std::min(Dst.base_cpi, Src.BaseCPI);
}

} // namespace

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
static cl::opt<int> Iterations("iterations", cl::desc("Steady-state iteration multiplier"), cl::init(100));
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

int main(int argc, char **argv) {
    InitLLVM X(argc, argv);
    initializeTargets();

    cl::ParseCommandLineOptions(argc, argv, "mlp-objdump\n");

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
    MCContext Ctx(TT, *MAI, *MRI, *STI);
    std::unique_ptr<MCDisassembler> DisAsm(TheTarget->createMCDisassembler(*STI, Ctx));
    std::unique_ptr<MCInstrAnalysis> MCIA(TheTarget->createMCInstrAnalysis(MCII.get()));
    std::unique_ptr<MCInstPrinter> IP(TheTarget->createMCInstPrinter(TT, 0, *MAI, *MCII, *MRI));

    if (!MRI || !MAI || !MCII || !STI || !DisAsm || !MCIA || !IP) {
        WithColor::error() << "Failed to initialize LLVM components\n";
        return 1;
    }

    IP->setPrintImmHex(true);
    IP->setPrintBranchImmAsAddress(true);

    mca::PipelineOptions PO(0, 0, 0, 0, 0, 0, true);
    const MCSchedModel &SM = STI->getSchedModel();
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

    FunctionBoundaries Boundaries = collectFunctionBoundaries(Obj);
    bool FirstSection = true;

    for (const SectionRef &Section : Obj.sections()) {
        if (!Section.isText() || Section.getSize() == 0) continue;

        auto SectionInstrs = disassembleTextSection(Section, *DisAsm, *MCII, MCIA.get());
        if (SectionInstrs.empty()) continue;

        if (!FirstSection) std::cout << "\n";
        FirstSection = false;

        std::map<SpanKey, RegionMetrics> MetricsBySpan;
        std::vector<SpanKey> SpanForInstr(SectionInstrs.size(), SpanKey{0, 0});

        auto accumulateSpan = [&](const RegionSpan &Span, bool isLoop) {
            if (Span.Size == 0) return;
            const SpanKey Key{Span.Start, Span.Size};
            bool ignore = false;
            if (IgnoreLoopCarried == IgnoreLoopCarriedMode::Force) {
                ignore = true;
            } else if (IgnoreLoopCarried == IgnoreLoopCarriedMode::Default) {
                ignore = !isLoop;
            } else if (IgnoreLoopCarried == IgnoreLoopCarriedMode::Disable) {
                ignore = false;
            }
            auto Result = analyzeMcaRegion(ArrayRef<Instr>(SectionInstrs).slice(Span.Start, Span.Size), *STI, *MCII,
                                           *MRI, MCIA.get(), PO, Iterations, WindowWidth, DepKind, AssignKind,
                                           ignore, OverrideLoadLatency);
            mergeMetrics(MetricsBySpan[Key], Result);
            for (size_t i = Span.Start; i < Span.Start + Span.Size; ++i) {
                if (SpanForInstr[i].second == 0) {
                    SpanForInstr[i] = Key;
                }
            }
        };

        auto rememberBasicBlock = [&](const RegionSpan &Span) {
            if (Span.Size == 0) return;
            const SpanKey Key{Span.Start, Span.Size};
            for (size_t i = Span.Start; i < Span.Start + Span.Size; ++i) SpanForInstr[i] = Key;
            accumulateSpan(Span, false);
        };

        walkRegions(SectionInstrs, Boundaries, LoopMaxInstrs, BBMaxInstrs,
                    [&](const RegionSpan &Span) { accumulateSpan(Span, true); },
                    rememberBasicBlock);

        if (auto NameOrErr = Section.getName()) printSectionHeader(*NameOrErr);

        for (size_t i = 0; i < SectionInstrs.size(); ++i) {
            const auto &I = SectionInstrs[i];
            auto It = MetricsBySpan.find(SpanForInstr[i]);
            const RegionMetrics *M = (It != MetricsBySpan.end()) ? &It->second : nullptr;

            std::string InstText;
            raw_string_ostream OS(InstText);
            IP->printInst(&I.Inst, I.Addr, "", *STI, OS);
            OS.flush();

            std::cout << formatMetrics(M ? *M : RegionMetrics{}) << " " << formatAddress(I.Addr) << ": "
                      << InstText << "\n";
        }
    }

    return 0;
}
