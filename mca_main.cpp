#include "mca_common.h"
#include <cstdio>
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
        clEnumValN(DependencyKind::OOO, "ooo", "Out-of-order dependency")
    ), cl::init(DependencyKind::None));
static cl::opt<MLPWindowAssignmentKind> AssignKind("mlp-window-assignment", cl::desc("Per-load MLP assignment mode"),
    cl::values(
        clEnumValN(MLPWindowAssignmentKind::Forward, "forward", "Forward window"),
        clEnumValN(MLPWindowAssignmentKind::MaxContaining, "max-containing", "Max MLP of containing windows")
    ), cl::init(MLPWindowAssignmentKind::MaxContaining));
static cl::opt<int> Iterations("iterations", cl::desc("Steady-state iteration multiplier"), cl::init(100));
static cl::opt<int> LoopMaxInstrs("loop-max-instrs", cl::desc("Maximum instructions in a loop to analyze"), cl::init(100));
static cl::opt<int> BBMaxInstrs("bb-max-instrs", cl::desc("Maximum instructions in a basic block to analyze"), cl::init(100));
static cl::opt<bool> IgnoreLoopCarried("ignore-loop-carried",
    cl::desc("Ignore loop-carried register dependencies during cycle estimation"),
    cl::init(false));

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

static void printResultCsv(const Instr &First, const Instr &Last, const McaMetrics &M) {
    std::printf("0x%lx,0x%lx,%lu,%lu,%u,%.2f\n", First.Addr, Last.Addr,
                static_cast<unsigned long>(M.RetiredInstructions),
                static_cast<unsigned long>(M.LoadInstructions),
                static_cast<unsigned>(M.Cycles), M.MLP);
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
    MCContext Ctx(TT, *MAI, *MRI, *STI);
    std::unique_ptr<MCDisassembler> DisAsm(TheTarget->createMCDisassembler(*STI, Ctx));
    std::unique_ptr<MCInstrAnalysis> MCIA(TheTarget->createMCInstrAnalysis(MCII.get()));

    if (!MRI || !MAI || !MCII || !STI || !DisAsm || !MCIA) {
        WithColor::error() << "Failed to initialize LLVM components\n";
        return 1;
    }

    mca::PipelineOptions PO(0, 0, 0, 0, 0, 0, true);
    const MCSchedModel &SM = STI->getSchedModel();
    PO.MicroOpQueueSize = SM.MicroOpBufferSize;
    PO.DispatchWidth = SM.IssueWidth;
    PO.AssumeNoAlias = true;

    std::printf("start_address,end_address,retired_instructions,load_instructions,cycles,mlp\n");
    FunctionBoundaries FunctionRanges = collectFunctionBoundaries(Obj);

    for (const SectionRef &Section : Obj.sections()) {
        if (!Section.isText() || Section.getSize() == 0) continue;

        auto SectionInstrs = disassembleTextSection(Section, *DisAsm, *MCII, MCIA.get());
        if (SectionInstrs.empty()) continue;

        ScopedSilence silence;
        auto emitRegion = [&](const RegionSpan &Span) {
            auto Result = analyzeMcaRegion(ArrayRef<Instr>(SectionInstrs).slice(Span.Start, Span.Size), *STI, *MCII,
                                           *MRI, MCIA.get(), PO, Iterations, WindowWidth, DepKind, AssignKind,
                                           IgnoreLoopCarried);
            if (Result.Valid) printResultCsv(SectionInstrs[Span.Start], SectionInstrs[Span.Start + Span.Size - 1], Result);
        };

        walkRegions(SectionInstrs, FunctionRanges, LoopMaxInstrs, BBMaxInstrs, emitRegion, emitRegion);
    }

    return 0;
}
