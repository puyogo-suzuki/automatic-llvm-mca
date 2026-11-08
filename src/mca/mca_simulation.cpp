#include "mca/mca_simulation.h"
#include "a55/a55_adaptations.h"
#include "a55/a55_issue_stage.h"
#include "a520/a520_issue_stage.h"
#include "cpu/cpu_traits.h"
#include "cpu/sched_model.h"
#include "mca/steady_state_tracker.h"
#include "mca/zero_aware_instr_builder.h"
#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MCA/Context.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/MCA/HWEventListener.h"
#include "llvm/MCA/InstrBuilder.h"
#include "llvm/MCA/Pipeline.h"
#include "llvm/MCA/SourceMgr.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

template<typename Tag, typename Tag::type M>
struct RobStore {
  friend typename Tag::type get(Tag) { return M; }
};

struct Context_Hardware_Tag {
  typedef SmallVector<std::unique_ptr<mca::HardwareUnit>, 4> mca::Context::*type;
  friend type get(Context_Hardware_Tag);
};
template struct RobStore<Context_Hardware_Tag, &mca::Context::Hardware>;

unsigned getWarmupWindowSize(const MCSubtargetInfo &STI) {
    const auto &SM = STI.getSchedModel();
    if (SM.isOutOfOrder()) return std::max(SM.MicroOpBufferSize, SM.LoopMicroOpBufferSize);
    return std::max(1u, SM.IssueWidth);
}

// The register file among the hardware units of `Ctx`, or null.
mca::RegisterFile *findRegisterFile(mca::Context &Ctx) {
    auto &Hardware = Ctx.*get(Context_Hardware_Tag{});
    for (auto &HU : Hardware) {
        if (auto *RF = dynamic_cast<mca::RegisterFile *>(HU.get()))
            return RF;
    }
    return nullptr;
}

// --override-load-latency: every write of a load with a latency above one
// cycle gets `Latency` (writes of at most one cycle, e.g. a base-register
// update, keep theirs), and MaxLatency is recomputed.  Writes the InstrDesc, which
// InstrBuilder shares between instructions.
void applyLoadLatencyOverride(mca::Instruction &Inst, int Latency) {
    mca::InstrDesc &MutableDesc = const_cast<mca::InstrDesc &>(Inst.getDesc());
    for (auto &W : MutableDesc.Writes) {
        if (W.Latency > 1) {
            W.Latency = Latency;
        }
    }
    unsigned MaxLatency = 0;
    for (const auto &W : MutableDesc.Writes) {
        MaxLatency = std::max(MaxLatency, W.Latency);
    }
    MutableDesc.MaxLatency = MaxLatency;
}

// Lets `Tracker` drop the previous iteration's producers before the issue
// stage's register hazard check (see the call site in analyzeMcaRegion()).
template <typename IssueStageT>
void installLoopCarriedHook(IssueStageT *Issue, LoopCarriedIgnoreSteadyStateTracker *Tracker) {
    if (!Issue)
        return;
    Issue->setBeforeRegisterHazardCheckHook(
        [Tracker](const mca::InstRef &IR) { Tracker->beforeRegisterHazardCheck(IR); });
}

struct DroppedInstructionTally {
    std::mutex Lock;
    std::map<std::string, uint64_t> ByOpcode;
};
DroppedInstructionTally &droppedInstructionTally() {
    static DroppedInstructionTally T;
    return T;
}
} // namespace

void noteDroppedInstruction(StringRef OpcodeName) {
    DroppedInstructionTally &T = droppedInstructionTally();
    std::lock_guard<std::mutex> G(T.Lock);
    ++T.ByOpcode[OpcodeName.str()];
}

void reportDroppedInstructions(raw_ostream &OS) {
    DroppedInstructionTally &T = droppedInstructionTally();
    std::lock_guard<std::mutex> G(T.Lock);
    if (T.ByOpcode.empty())
        return;
    uint64_t Total = 0;
    for (const auto &KV : T.ByOpcode)
        Total += KV.second;
    OS << "[WARN] " << Total << " instruction(s) were left out of the MCA simulation because "
       << "llvm::mca::InstrBuilder rejected them (scheduling model has no usable data); "
       << "regions containing them are simulated without them:\n";
    for (const auto &KV : T.ByOpcode)
        OS << "[WARN]   " << KV.first << ": " << KV.second << "\n";
    OS.flush();
    T.ByOpcode.clear();
}

void buildRegionInstructions(
    ZeroAwareInstrBuilder &IB, const MCSubtargetInfo &STI, const MCInstrInfo &MCII,
    ArrayRef<Instr> instrs,
    function_ref<void(const Instr &, std::unique_ptr<mca::Instruction>)> Accept) {
    const bool CoalescedROB = isCoalescedROBSubtarget(STI);
    for (const auto &I : instrs) {
        auto ExpectedInst = IB.createInstruction(I.Inst, {});
        if (!ExpectedInst) {
            // Not silently: see noteDroppedInstruction() in mca_simulation.h.
            consumeError(ExpectedInst.takeError());
            noteDroppedInstruction(MCII.getName(I.Inst.getOpcode()));
            continue;
        }
        std::unique_ptr<mca::Instruction> Inst = std::move(*ExpectedInst);
        if (CoalescedROB) {
            mca::InstrDesc &MutableDesc = const_cast<mca::InstrDesc &>(Inst->getDesc());
            MutableDesc.NumMicroOps = 1;
        }
        Accept(I, std::move(Inst));
    }
}

unsigned computeSteadyIterations(const MCSubtargetInfo &STI, size_t regionInstrCount, unsigned iterations) {
    if (regionInstrCount == 0) return 1;
    const unsigned WindowSize = std::max(1u, getWarmupWindowSize(STI));
    return std::max(1u, (WindowSize * iterations + static_cast<unsigned>(regionInstrCount) - 1) /
                            static_cast<unsigned>(regionInstrCount));
}

unsigned computeWarmupIterations(const MCSubtargetInfo &STI, size_t regionInstrCount) {
    return computeSteadyIterations(STI, regionInstrCount, 1);
}

McaMetrics analyzeMcaRegion(ArrayRef<Instr> instrs, const MCSubtargetInfo &STI, const MCInstrInfo &MCII,
                            const MCRegisterInfo &MRI, const MCInstrAnalysis *MCIA, const mca::PipelineOptions &PO,
                            int iterations, int windowWidth, DependencyKind depKind,
                            MLPWindowAssignmentKind assignKind, const MLPAnalyzer &analyzer,
                            bool ignoreLoopCarriedDep,
                            int overrideLoadLatency, bool mlpWindowLoop) {
    if (instrs.empty()) return {};

    mca::Context MCAContext(MRI, STI);
    mca::InstrumentManager IM(STI, MCII);
    ZeroAwareInstrBuilder IB(STI, MCII, MRI, MCIA, IM, 0);
    const CpuModel Model = getCpuTraits(STI).Model;
    const bool IsA55 = (Model == CpuModel::A55);
    const bool IsA520 = (Model == CpuModel::A520);
    std::vector<mca::SourceMgr::UniqueInst> Sequence;
    buildRegionInstructions(IB, STI, MCII, instrs,
                            [&](const Instr &I, std::unique_ptr<mca::Instruction> Inst) {
        if (IsA55)
            applyA55InstructionAdaptations(*Inst, I.Inst);
        if (Inst->getMayLoad() && overrideLoadLatency > 0)
            applyLoadLatencyOverride(*Inst, overrideLoadLatency);
        Sequence.push_back(std::move(Inst));
    });

    const unsigned WarmupIterations = computeWarmupIterations(STI, instrs.size());
    const unsigned SteadyIterations = computeSteadyIterations(STI, instrs.size(), iterations);
    mca::CircularSourceMgr CSM(Sequence, WarmupIterations + SteadyIterations);
    mca::CustomBehaviour CB(STI, CSM, MCII);
    std::unique_ptr<mca::Pipeline> P;
    mca::A55DecoupledIssueStage *A55Issue = nullptr;
    mca::A520InOrderIssueStage *A520Issue = nullptr;
    if (STI.getSchedModel().isOutOfOrder()) {
        P = MCAContext.createDefaultPipeline(PO, CSM, CB);
    } else if (IsA520) {
        // Cortex-A520: the stock in-order pipeline with the project's copy of
        // InOrderIssueStage (src/a520/a520_issue_stage.h): no general write-back
        // ordering [a520_no_wb_order] (docs/a520-sog.md D-9), a hook before the
        // register hazard check [a520_lc_hook] (docs/a520-sog.md section 2.2), and the flag bypass
        // borrowed from the Cortex-A55 [a520_from_a55_flags] (docs/a520-sog.md B-4).
        auto PRF_unique = std::make_unique<mca::RegisterFile>(STI.getSchedModel(), MRI, PO.RegisterFileSize);
        auto LSU_unique = std::make_unique<mca::LSUnit>(STI.getSchedModel(), PO.LoadQueueSize,
                                                        PO.StoreQueueSize, PO.AssumeNoAlias);
        mca::RegisterFile *PRF_ptr = PRF_unique.get();
        mca::LSUnit *LSU_ptr = LSU_unique.get();
        MCAContext.addHardwareUnit(std::move(PRF_unique));
        MCAContext.addHardwareUnit(std::move(LSU_unique));
        P = mca::createA520InOrderPipeline(CSM, CB, STI, *PRF_ptr, *LSU_ptr, &A520Issue,
                                           /*EnforceWriteBackOrder=*/false, &MCII,
                                           CSM.getInstructions());
    } else if (IsA55) {
        auto PRF_unique = std::make_unique<mca::RegisterFile>(STI.getSchedModel(), MRI);
        mca::RegisterFile *PRF_ptr = PRF_unique.get();
        MCAContext.addHardwareUnit(std::move(PRF_unique));
        P = mca::createA55DecoupledPipeline(PO, CSM, CB, STI, MRI, *PRF_ptr, MCII, CSM.getInstructions(),
                                            &A55Issue);
    } else {
        P = MCAContext.createInOrderPipeline(PO, CSM, CB);
    }

    mca::RegisterFile *PRF = findRegisterFile(MCAContext);

    std::unique_ptr<BaseSteadyStateTracker> Tracker;
    unsigned WarmupLimit = instrs.size() * WarmupIterations;

    if (ignoreLoopCarriedDep) {
        auto LCI = std::make_unique<LoopCarriedIgnoreSteadyStateTracker>(
            WarmupLimit, instrs.size(), PRF);
        // #12: the tracker drops the previous iteration's producers on the
        // Dispatched event, which the issue stages raise only after their
        // register hazard check, so the first instruction of every iteration
        // still waited for the previous iteration.  The A55 issue stage calls the
        // tracker before that check, and so does the Cortex-A520 issue stage
        // (src/a520/a520_issue_stage.h).  The stock out-of-order pipeline has no
        // such hook and does not need one (docs.md 2.C).
        installLoopCarriedHook(A55Issue, LCI.get());
        installLoopCarriedHook(A520Issue, LCI.get());  // [a520_lc_hook] docs/a520-sog.md section 2.2
        Tracker = std::move(LCI);
    } else {
        Tracker = std::make_unique<BaseSteadyStateTracker>(
            WarmupLimit);
    }

    P->addEventListener(Tracker.get());

    auto ExpectedCycles = P->run();

    if (!ExpectedCycles) {
        auto err = ExpectedCycles.takeError();
        llvm::errs() << "[TARGET_ERR] Region start 0x" << llvm::format_hex(instrs[0].Addr, 8) << ": " << llvm::toString(std::move(err)) << "\n";
        return {};
    }

    McaMetrics M;
    M.RetiredInstructions = Tracker->SteadyRetired > 0 ? Tracker->SteadyRetired : Tracker->TotalRetired;
    M.LoadInstructions = analyzer.countPotentialMissLoads(instrs, STI, MCII, MRI, depKind, mlpWindowLoop);
    M.Cycles = Tracker->SteadyCycles > 0 ? Tracker->SteadyCycles : Tracker->CurrentCycle;
    if (IsA55)
        M.Cycles = applyA55BranchOverlapCorrection(M.Cycles, Tracker->SteadyRetired, instrs.size());

    M.MLP = analyzer.compute_mlp(instrs, windowWidth, depKind, assignKind, STI, MCII, MRI, M.MLP_R, mlpWindowLoop);

    if (M.RetiredInstructions > 0) {
        M.BaseCPI = static_cast<double>(M.Cycles) / static_cast<double>(M.RetiredInstructions);
    }
    M.Valid = true;
    return M;
}
