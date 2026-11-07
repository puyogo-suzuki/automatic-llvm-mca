#pragma once

// Shared helpers of the per-core SOG tests (tests/<core>_sog_test.cpp) and of facile_test.cpp:
//  * LocalModelContext: an AArch64 context with the scheduling model mca_tool installs;
//  * sogDesc(): the per-instruction data InstrBuilder takes from the model;
//  * sogDump(): prints sogDesc() of a list of instructions (the X1SogTest / A720SogTest Dump tests);
//  * runFacile() / sogFacile(): the Facile prediction of a loop body;
//  * inOrderRun(): steady-state cycles per iteration simulated by the MCA pipeline
//    (the in-order A55 / A520 issue stages).

#include "test_support.h"
#include "mca/zero_aware_instr_builder.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// EXPECT_NEAR tolerances: Facile estimates (out-of-order cores) and in-order simulations.
constexpr double kSogFacileTol = 0.01;
constexpr double kSogInOrderTol = 0.1;

// AArch64TestContext with the scheduling model mca_tool installs for `Cpu` (what frontend.cpp's
// initializeFrontend() does: remap, install, wrap), also for the CPUs AArch64TestContext leaves on the
// stock model (cortex-a76, cortex-a78, cortex-a78c, ...).
struct LocalModelContext : AArch64TestContext {
    explicit LocalModelContext(const std::string &Cpu) : AArch64TestContext(Cpu) {
        if (testContextInstallsLocalModel(Cpu))
            return;
        llvm::remapSchedClassIndices(*MCII, Cpu);
        llvm::installLocalSchedModel(*STI, Cpu);
        STI = llvm::wrapCustomSubtargetInfo(std::move(STI), Cpu);
    }
};

struct SogInstInfo {
    std::string Opcode;
    unsigned MaxLatency = 0;
    unsigned NumMicroOps = 0;
    std::vector<unsigned> WriteLatencies;  // per def, in operand order (implicit defs last)
    // Total resource cycles per resource name, as the scheduling tables list them (a use of a
    // pipeline also appears under every group that contains it, e.g. M0 under M and I).
    std::map<std::string, unsigned> ResourceCycles;
    // ReadAdvance entries of the sched class: "use index:cycles(wWriteResourceID)".
    std::vector<std::string> ReadAdvances;
    bool BeginGroup = false;
    bool EndGroup = false;
    bool Valid = false;  // exactly one instruction, accepted by InstrBuilder
};

// How ResourceCycles counts a write-resource entry: ReleaseAtCycle - AcquireAtCycle (the
// out-of-order cores' tests) or ReleaseAtCycle alone (the A55 / A520 tests).
enum class ResourceCycleCount { ReleaseMinusAcquire, Release };

// The data of the single instruction in `Asm` (directives allowed) for `Cpu`, from a plain
// mca::InstrBuilder (one instruction, so ZeroAwareInstrBuilder would change nothing).
inline SogInstInfo sogDesc(const std::string &Cpu, const std::string &Asm,
                           ResourceCycleCount Count = ResourceCycleCount::ReleaseMinusAcquire) {
    initLLVMAArch64();
    LocalModelContext TC(Cpu);
    auto Instrs = parseAsm(TC, Asm);
    SogInstInfo Info;
    if (Instrs.size() != 1)
        return Info;
    Info.Opcode = TC.MCII->getName(Instrs[0].Inst.getOpcode()).str();
    mca::InstrumentManager IM(*TC.STI, *TC.MCII);
    mca::InstrBuilder IB(*TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), IM, 0);
    auto ExpectedInst = IB.createInstruction(Instrs[0].Inst, {});
    if (!ExpectedInst) {
        consumeError(ExpectedInst.takeError());
        return Info;
    }
    const mca::Instruction &IS = **ExpectedInst;
    Info.MaxLatency = IS.getLatency();
    Info.NumMicroOps = IS.getNumMicroOps();
    Info.BeginGroup = IS.getBeginGroup();
    Info.EndGroup = IS.getEndGroup();
    for (const mca::WriteState &WS : IS.getDefs())
        Info.WriteLatencies.push_back(WS.getLatency());
    const MCSchedModel &SM = TC.STI->getSchedModel();
    if (const MCSchedClassDesc *SCD = SM.getSchedClassDesc(IS.getDesc().SchedClassID)) {
        for (const MCWriteProcResEntry *E = TC.STI->getWriteProcResBegin(SCD),
                                       *End = TC.STI->getWriteProcResEnd(SCD);
             E != End; ++E)
            Info.ResourceCycles[SM.getProcResource(E->ProcResourceIdx)->Name] +=
                Count == ResourceCycleCount::Release ? E->ReleaseAtCycle
                                                     : E->ReleaseAtCycle - E->AcquireAtCycle;
        for (const MCReadAdvanceEntry &RA : TC.STI->getReadAdvanceEntries(*SCD))
            Info.ReadAdvances.push_back(std::to_string(RA.UseIdx) + ":" + std::to_string(RA.Cycles) +
                                        "(w" + std::to_string(RA.WriteResourceID) + ")");
    }
    Info.Valid = true;
    return Info;
}

// Prints sogDesc() of every instruction in the environment variable `EnvVar` (separated by ';'),
// one line each; nothing when the variable is unset.
inline void sogDump(const std::string &Cpu, const char *EnvVar) {
    const char *Env = std::getenv(EnvVar);
    if (!Env)
        return;
    std::stringstream SS(Env);
    std::string Line;
    while (std::getline(SS, Line, ';')) {
        if (Line.empty())
            continue;
        SogInstInfo I = sogDesc(Cpu, Line);
        std::cout << Line << " | " << I.Opcode << " | lat " << I.MaxLatency << " [";
        for (unsigned L : I.WriteLatencies)
            std::cout << " " << L;
        std::cout << " ] uops " << I.NumMicroOps << " |";
        for (const auto &[N, C] : I.ResourceCycles)
            std::cout << " " << N << ":" << C;
        std::cout << " | RA";
        for (const auto &R : I.ReadAdvances)
            std::cout << " " << R;
        std::cout << (I.Valid ? "" : " INVALID") << "\n";
    }
}

struct FacileOptions {
    unsigned UopWidth = 0;      // 0: the model's IssueWidth
    unsigned MopWidth = 0;      // 0: no Mop cap
    bool MemoryEdges = false;   // pass the MemAccessInfo of every instruction (store -> load edges,
                                // mca_tool --facile-memory-deps); off by default, as in mca_tool
    bool LoopCarried = true;
};

// Facile prediction for `Asm` (one instruction per line) in the context `TC`, the instructions
// built with ZeroAwareInstrBuilder as mca_tool does; instructions InstrBuilder rejects are left out.
inline facile::FacileResult runFacile(const AArch64TestContext &TC, const std::string &Asm,
                                      const FacileOptions &O = {}) {
    initLLVMAArch64();
    auto Instrs = parseAsm(TC, Asm);
    std::unique_ptr<MLPAnalyzer> Analyzer;
    if (O.MemoryEdges)
        Analyzer = MLPAnalyzer::create(*TC.STI);
    mca::InstrumentManager IM(*TC.STI, *TC.MCII);
    ZeroAwareInstrBuilder IB(*TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), IM, 0);
    std::vector<std::unique_ptr<mca::Instruction>> SimInstrs;
    std::vector<const MCInst *> MCInsts;
    std::vector<MemAccessInfo> MemInfos;
    for (const auto &I : Instrs) {
        auto ExpectedInst = IB.createInstruction(I.Inst, {});
        if (!ExpectedInst) {
            consumeError(ExpectedInst.takeError());
            continue;
        }
        SimInstrs.push_back(std::move(*ExpectedInst));
        MCInsts.push_back(&I.Inst);
        if (O.MemoryEdges)
            MemInfos.push_back(Analyzer->getMemAccessInfo(I.Inst, TC.MCII->get(I.Inst.getOpcode()),
                                                          *TC.MRI, *TC.MCII));
    }
    return facile::computeFacilePrediction(*TC.STI, *TC.MCII, *TC.MRI, SimInstrs, MCInsts, O.UopWidth,
                                           MemInfos, O.MopWidth, O.LoopCarried);
}

// Facile prediction for the loop body `Body` (one instruction per line, no loop branch), with the
// widths (cpu_traits.cpp) and fusion of `mca_tool --mcpu <Cpu> --dependency ooo --facile`.  Like
// mca_tool, it adds no store -> load memory edge unless `MemoryEdges` is true (`mca_tool
// --facile-memory-deps`).
inline facile::FacileResult sogFacile(const std::string &Cpu, const std::string &Body,
                                      bool MemoryEdges = false) {
    initLLVMAArch64();
    LocalModelContext TC(Cpu);
    const CpuTraits &T = getCpuTraits(Cpu);
    FacileOptions O;
    O.UopWidth = T.UopDispatchWidth;
    O.MopWidth = T.MopDispatchWidth;
    O.MemoryEdges = MemoryEdges;
    return runFacile(TC, Body, O);
}

struct InOrderRun {
    double CyclesPerIter = -1.0;  // -1 when the simulation gives no valid result
    size_t NumInstrs = 0;
    uint64_t Retired = 0;
};

// Steady-state cycles per iteration of `Body` (one instruction per line) for `Cpu`, simulated as a
// loop (loop-carried dependencies kept), --dependency dependency.  With `IgnoreLoopCarried`, as a
// basic block under --ignore-loop-carried=default (dependencies on the previous iteration ignored).
inline InOrderRun inOrderRun(const std::string &Cpu, const std::string &Body, bool IgnoreLoopCarried) {
    initLLVMAArch64();
    LocalModelContext TC(Cpu);
    auto Instrs = parseAsm(TC, Body);
    InOrderRun R;
    R.NumInstrs = Instrs.size();
    AArch64MLPAnalyzer Analyzer;
    mca::PipelineOptions PO(0, 0, 0, 0, 0, 0, true);
    McaMetrics M = analyzeMcaRegion(Instrs, *TC.STI, *TC.MCII, *TC.MRI, TC.MCIA.get(), PO,
                                    200, 16, DependencyKind::Dependency,
                                    MLPWindowAssignmentKind::Forward, Analyzer,
                                    IgnoreLoopCarried, -1, /*mlpWindowLoop=*/false);
    R.Retired = M.RetiredInstructions;
    if (!M.Valid || M.RetiredInstructions == 0 || Instrs.empty())
        return R;
    R.CyclesPerIter = static_cast<double>(M.Cycles) /
                      (static_cast<double>(M.RetiredInstructions) / static_cast<double>(Instrs.size()));
    return R;
}
