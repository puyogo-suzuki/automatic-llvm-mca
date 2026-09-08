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
    cl::opt<bool> Facile("facile", cl::desc("Enable Facile static analytical throughput prediction (AArch64)"), cl::init(false));
    cl::opt<bool> FacileReason("facile-reason", cl::desc("Output dominant bottleneck reason (inst, exec, prec) for Facile analysis"), cl::init(false));
    cl::opt<int> ChainThreshold("chain-threshold", cl::desc("Max chain threshold for merged loop region analysis"), cl::init(5));
    cl::opt<bool> CountOnly("count-only", cl::desc("Only count total generated regions without running MCA simulation"), cl::init(false));
    cl::opt<bool> DisableAlwaysHitLoadsHeuristic("disable-always-hit-loads-heuristic",
        cl::desc("Disable the same-cache-line 'line reuse' always-hit heuristic: by default "
                 "a load that repeats a same-base-register+cache-line access already seen in "
                 "the analyzed window is treated as an unconditional cache hit and excluded "
                 "from both the load_instructions/BB_LD_RETIRED count (countPotentialMissLoads) "
                 "and the BB_MLP potential-miss averaging (compute_mlp). Passing this flag "
                 "turns that exclusion off, so such loads are counted normally; it is a "
                 "diagnostic escape hatch for generating 'no exclusion at all' reference data. "
                 "Stack/frame-pointer always-hit exclusion is unrelated and unconditional "
                 "regardless of this flag; see -no-stack-exclusion to turn that off instead."),
        cl::init(false));
    cl::opt<bool> ForwardingAwareHitHeuristic("forwarding-aware-hit-heuristic",
        cl::desc("Replace the blanket stack/frame-pointer always-hit assumption with an "
                 "exact-address store-to-load forwarding check (applies to any base register, "
                 "not just sp/fp)"),
        cl::init(false));
    cl::opt<bool> StackOnlyMissLoadCount("stack-only-miss-load-count",
        cl::desc("In countPotentialMissLoads only (the load_instructions column), exclude "
                 "stack/frame-pointer accesses but keep same-cache-line repeat accesses. "
                 "Diagnostic knob that decomposes the two always-hit heuristics; does not "
                 "affect the MLP computation itself."),
        cl::init(false));
    cl::opt<bool> StackConstOffsetOnly("stack-const-offset-only",
        cl::desc("With -stack-only-miss-load-count, additionally require a constant "
                 "(immediate) offset for the stack always-hit rule, so register-indexed "
                 "stack accesses (large local arrays) are still counted as possible misses."),
        cl::init(false));
    cl::opt<bool> StackSpillOnly("stack-spill-only",
        cl::desc("With -stack-only-miss-load-count, only treat a stack/frame-pointer load "
                 "as an always-hit when a store to the exact same (base_reg, offset) is "
                 "seen in the analyzed region (a provable spill/reload pair)."),
        cl::init(false));
    cl::opt<bool> StackLoopResident("stack-loop-resident",
        cl::desc("With -stack-only-miss-load-count -stack-spill-only, additionally treat a "
                 "constant-offset stack load inside a steady-state loop region as an "
                 "always-hit: the same slot is touched every iteration, so it is L1 "
                 "resident from the second iteration on."),
        cl::init(false));
    cl::opt<bool> NoStackExclusion("no-stack-exclusion",
        cl::desc("Do not give stack/frame-pointer (sp/wsp/x29/w29/fp) loads any special "
                 "always-hit treatment: treat them exactly like ordinary loads in both the "
                 "MLP computation (compute_mlp, the 'mlp' column) and the potential-miss "
                 "load count (countPotentialMissLoads, the 'load_instructions' column), "
                 "subjecting them to the same OOO/dependency same-cache-line heuristics as "
                 "any other load. Takes effect only where the blanket stack always-hit "
                 "check would otherwise fire; -forwarding-aware-hit-heuristic already does "
                 "not special-case stack accesses, so it takes precedence when combined."),
        cl::init(false));
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
    std::string llvm_cpu = TI.CPU;
    if (TI.CPU == "icestorm" || TI.CPU == "firestorm") {
        llvm_cpu = "apple-m1";
    }
    TI.MRI.reset(TI.TheTarget->createMCRegInfo(TT));
    MCTargetOptions MCOPT;
    TI.MAI.reset(TI.TheTarget->createMCAsmInfo(*TI.MRI, TT, MCOPT));
    TI.MCII.reset(TI.TheTarget->createMCInstrInfo());
    TI.STI.reset(TI.TheTarget->createMCSubtargetInfo(TT, llvm_cpu, ""));
    if (TI.STI) {
        llvm::overrideCortexA55SchedModel(*TI.STI, TI.CPU);
        TI.STI = llvm::wrapCustomSubtargetInfo(std::move(TI.STI), TI.CPU);
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
