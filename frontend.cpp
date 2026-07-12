#include "frontend.h"
#include "custom_a55_sched.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;

namespace opts {
    cl::opt<std::string> InputBinary(cl::Positional, cl::desc("<input binary>"), cl::Required);
    cl::opt<std::string> MTriple("mtriple", cl::desc("Target triple"));
    cl::opt<std::string> MCPU("mcpu", cl::desc("Target CPU"));
    cl::opt<int> WindowWidth("window-width", cl::desc("MLP window width"), cl::init(4));
    cl::opt<DependencyKind> DepKind("dependency", cl::desc("Dependency mode"),
        cl::values(
            clEnumValN(DependencyKind::None, "none", "No dependency tracking"),
            clEnumValN(DependencyKind::IO, "io", "In-order dependency"),
            clEnumValN(DependencyKind::OOO, "ooo", "Out-of-order dependency"),
            clEnumValN(DependencyKind::Dependency, "dependency", "Load-use dependency distance")
        ), cl::init(DependencyKind::None));
    cl::opt<MLPWindowAssignmentKind> AssignKind("mlp-window-assignment", cl::desc("Per-load MLP assignment mode"),
        cl::values(
            clEnumValN(MLPWindowAssignmentKind::Forward, "forward", "Forward window"),
            clEnumValN(MLPWindowAssignmentKind::MaxContaining, "max-containing", "Max MLP of containing windows")
        ), cl::init(MLPWindowAssignmentKind::MaxContaining));
    cl::opt<int> Iterations("iterations", cl::desc("Steady-state iteration multiplier"), cl::init(3));
    cl::opt<int> LoopMaxInstrs("loop-max-instrs", cl::desc("Maximum instructions in a loop to analyze"), cl::init(3000));
    cl::opt<int> BBMaxInstrs("bb-max-instrs", cl::desc("Maximum instructions in a basic block to analyze"), cl::init(3000));
    cl::opt<int> NestLimitOuter("nest-limit-outer", cl::desc("Maximum nesting depth of loops to analyze from outer to inner"), cl::init(2));
    cl::opt<int> NestLimitInner("nest-limit-inner", cl::desc("Maximum nesting depth of loops to analyze from inner to outer"), cl::init(2));
    cl::opt<IgnoreLoopCarriedMode> IgnoreLoopCarried("ignore-loop-carried",
        cl::desc("Ignore loop-carried register dependencies mode"),
        cl::values(
            clEnumValN(IgnoreLoopCarriedMode::Default, "default", "Ignore in basic blocks, but not in loops"),
            clEnumValN(IgnoreLoopCarriedMode::Force, "force", "Ignore in both loops and basic blocks"),
            clEnumValN(IgnoreLoopCarriedMode::Disable, "disable", "Do not ignore loop-carried dependencies anywhere")
        ), cl::init(IgnoreLoopCarriedMode::Default));
    cl::opt<int> OverrideLoadLatency("override-load-latency",
        cl::desc("Override load instruction latency in cycles"),
        cl::init(-1));
    cl::opt<MlpWindowLoopMode> MlpWindowLoop("mlp-window-loop",
        cl::desc("Loop back to the start of the basic block mode"),
        cl::values(
            clEnumValN(MlpWindowLoopMode::Default, "default", "Loop back to the start only for loops"),
            clEnumValN(MlpWindowLoopMode::Force, "force", "Always loop back to the start (even for non-loops)"),
            clEnumValN(MlpWindowLoopMode::Disable, "disable", "Never loop back to the start")
        ), cl::init(MlpWindowLoopMode::Default));
    cl::opt<std::string> TargetAddressStr("target-address",
        cl::desc("Target address to analyze (hex)"), cl::init(""));

    cl::opt<std::string> UpdateMlp("update-mlp", cl::desc("CSV file containing previous MCA results to reuse"), cl::init(""));
}

bool initializeFrontend(int argc, char **argv, const char *Overview,
                        std::unique_ptr<llvm::object::ObjectFile> &Obj,
                        TargetInfo &TI) {
    initializeTargets();
    cl::ParseCommandLineOptions(argc, argv, Overview);

    auto BinaryOrErr = object::ObjectFile::createObjectFile(opts::InputBinary);
    if (!BinaryOrErr) {
        WithColor::error() << "Failed to open binary: " << toString(BinaryOrErr.takeError()) << "\n";
        return false;
    }
    auto Pair = BinaryOrErr.get().takeBinary();
    Obj = std::move(Pair.first);
    TI.BinaryBuffer = std::move(Pair.second);

    Triple TT = Obj->makeTriple();
    if (!opts::MTriple.empty()) TT = Triple(opts::MTriple);
    TI.TripleName = TT.str();

    std::string Error;
    TI.TheTarget = TargetRegistry::lookupTarget(TT, Error);
    if (!TI.TheTarget) {
        WithColor::error() << "No target for " << TT.str() << ": " << Error << "\n";
        return false;
    }

    TI.CPU = opts::MCPU.empty() ? "generic" : std::string(opts::MCPU);
    TI.MRI.reset(TI.TheTarget->createMCRegInfo(TT));
    MCTargetOptions MCOPT;
    TI.MAI.reset(TI.TheTarget->createMCAsmInfo(*TI.MRI, TT, MCOPT));
    TI.MCII.reset(TI.TheTarget->createMCInstrInfo());
    TI.STI.reset(TI.TheTarget->createMCSubtargetInfo(TT, TI.CPU, ""));
    if (TI.STI) {
        llvm::overrideCortexA55SchedModel(*TI.STI);
    }
    TI.Ctx = std::make_unique<MCContext>(TT, TI.MAI.get(), TI.MRI.get(), TI.STI.get());
    TI.DisAsm.reset(TI.TheTarget->createMCDisassembler(*TI.STI, *TI.Ctx));
    TI.MCIA.reset(TI.TheTarget->createMCInstrAnalysis(TI.MCII.get()));

    if (!TI.MRI || !TI.MAI || !TI.MCII || !TI.STI || !TI.DisAsm || !TI.MCIA) {
        WithColor::error() << "Failed to initialize LLVM components\n";
        return false;
    }

    TI.MCAContext = std::make_unique<mca::Context>(*TI.MRI, *TI.STI);

    const MCSchedModel &SM = TI.STI->getSchedModel();
    TI.WindowWidthVal = opts::WindowWidth;
    if (opts::WindowWidth.getNumOccurrences() == 0 && !opts::MCPU.empty()) {
        if (SM.MicroOpBufferSize > 0) {
            TI.WindowWidthVal = SM.MicroOpBufferSize;
        } else {
            TI.WindowWidthVal = SM.IssueWidth * SM.MispredictPenalty;
        }
    }
    TI.PO.MicroOpQueueSize = SM.MicroOpBufferSize;
    TI.PO.DispatchWidth = SM.IssueWidth;
    if (TI.STI->getCPU() == "cortex-a76" || TI.STI->getCPU() == "cortex-a76ae" || TI.STI->getCPU() == "neoverse-n1") {
        TI.PO.DispatchWidth = 8;
    } else if (TI.STI->getCPU() == "cortex-a78" || TI.STI->getCPU() == "cortex-a78ae" || TI.STI->getCPU() == "cortex-a78c") {
        TI.PO.DispatchWidth = 12;
    } else if (TI.STI->getCPU() == "cortex-a710" || TI.STI->getCPU() == "cortex-a715" || TI.STI->getCPU() == "cortex-a720" || TI.STI->getCPU() == "cortex-a720ae" || TI.STI->getCPU() == "neoverse-n2") {
        TI.PO.DispatchWidth = 10;
    } else if (TI.STI->getCPU() == "cortex-x1" || TI.STI->getCPU() == "cortex-x1c" || TI.STI->getCPU() == "neoverse-v1") {
        TI.PO.DispatchWidth = 16;
    }
    TI.PO.AssumeNoAlias = true;

    TI.TargetAddress = 0;
    if (!opts::TargetAddressStr.empty()) {
        TI.TargetAddress = std::stoull(opts::TargetAddressStr, nullptr, 16);
    }

    TI.Analyzer = MLPAnalyzer::create(*TI.STI);

    return true;
}

#include <fcntl.h>
#include <unistd.h>
#include "llvm/Support/raw_ostream.h"

ScopedSilence::ScopedSilence() {
    devNull = ::open("/dev/null", O_WRONLY);
    if (devNull != -1) {
        oldStderr = dup(STDERR_FILENO);
        if (dup2(devNull, STDERR_FILENO) != -1) active = true;
    }
}

ScopedSilence::~ScopedSilence() {
    if (active) {
        llvm::errs().flush();
        dup2(oldStderr, STDERR_FILENO);
    }
    if (oldStderr != -1) ::close(oldStderr);
    if (devNull != -1) ::close(devNull);
}
