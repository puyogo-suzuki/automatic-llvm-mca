#include "frontend/frontend.h"
#include "core/object_loader.h"
#include "cpu/cpu_traits.h"
#include "cpu/sched_model.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <fcntl.h>
#include <unistd.h>

using namespace llvm;

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

    // The one place a --mcpu spelling is resolved (cpu_traits.h): every
    // per-CPU decision below and in the analyses is keyed by TI.CPU.
    TI.CPU = canonicalCpuName(opts::MCPU.empty() ? StringRef("generic") : StringRef(opts::MCPU)).str();
    TI.MRI.reset(TI.TheTarget->createMCRegInfo(TT));
    MCTargetOptions MCOPT;
    TI.MAI.reset(TI.TheTarget->createMCAsmInfo(*TI.MRI, TT, MCOPT));
    TI.MCII.reset(TI.TheTarget->createMCInstrInfo());
    // Remaps MCII's SchedClass indices onto the local tables (a large share of
    // the 9135 opcodes disagree with libLLVM's numbering, docs/docs.md section
    // 5.A) and installs the local sched model.
    if (TI.MCII)
        TI.STI = llvm::createLocalSubtargetInfo(*TI.TheTarget, TT, TI.CPU, *TI.MCII);
    TI.Ctx = std::make_unique<MCContext>(TT, TI.MAI.get(), TI.MRI.get(), TI.STI.get());
    TI.DisAsm.reset(TI.TheTarget->createMCDisassembler(*TI.STI, *TI.Ctx));
    TI.MCIA.reset(TI.TheTarget->createMCInstrAnalysis(TI.MCII.get()));

    if (!TI.MRI || !TI.MAI || !TI.MCII || !TI.STI || !TI.DisAsm || !TI.MCIA) {
        WithColor::error() << "Failed to initialize LLVM components\n";
        return false;
    }

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
    // Real uop dispatch width and the SOG's Mop cap come from cpu_traits.cpp
    // (which also documents where each number comes from).  The Mop cap is a
    // SECOND front-end constraint independent of PO.DispatchWidth (the uop
    // cap): every one of these SOGs states "the dispatch stage can process up
    // to N Mops per cycle and dispatch up to M uops per cycle", with M = 2N, so
    // the Mop cap is the tighter of the two whenever a region averages fewer
    // than two uops per instruction (docs/docs.md section 4.E).  See
    // frontend.h's MopDispatchWidth comment for why it can't come from
    // llvm::MCSchedModel itself.  0 = not applied, see calculateIssueBound().
    const CpuTraits &Traits = getCpuTraits(TI.CPU);
    if (Traits.UopDispatchWidth)
        TI.PO.DispatchWidth = Traits.UopDispatchWidth;
    TI.PO.AssumeNoAlias = true;
    TI.MopDispatchWidth = Traits.MopDispatchWidth;

    TI.TargetAddress = 0;
    if (!opts::TargetAddressStr.empty()) {
        TI.TargetAddress = std::stoull(opts::TargetAddressStr, nullptr, 16);
    }

    TI.Analyzer = MLPAnalyzer::create(*TI.STI);
    if (TI.Analyzer)
        TI.Analyzer->setWindowResources(Traits.MlpWindow);

    return true;
}

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
