#include "mca/mca_simulation.h"
#include "a55/a55_adaptations.h"
#include "a55/a55_issue_stage.h"
#include "cpu/cpu_traits.h"
#include "cpu/sched_model.h"
#include "mca/steady_state_tracker.h"
#include <algorithm>
#include <memory>

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

} // namespace

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
    mca::InstrBuilder IB(STI, MCII, MRI, MCIA, IM, 0);
    std::vector<mca::SourceMgr::UniqueInst> Sequence;
    SmallVector<mca::Instrument *> IVec;
    for (const auto &I : instrs) {
        auto ExpectedInst = IB.createInstruction(I.Inst, IVec);
        if (!ExpectedInst) {
            consumeError(ExpectedInst.takeError());
            continue;
        }
        
        std::unique_ptr<mca::Instruction> Inst = std::move(*ExpectedInst);
        if (getCpuTraits(STI.getCPU()).Model == CpuModel::A55)
            applyA55InstructionAdaptations(*Inst, I.Inst);
        if (Inst->getMayLoad() && overrideLoadLatency > 0) {
            mca::InstrDesc &MutableDesc = const_cast<mca::InstrDesc &>(Inst->getDesc());
            for (auto &W : MutableDesc.Writes) {
                if (W.Latency > 1) {
                    W.Latency = overrideLoadLatency;
                }
            }
            unsigned MaxLatency = 0;
            for (const auto &W : MutableDesc.Writes) {
                MaxLatency = std::max(MaxLatency, W.Latency);
            }
            MutableDesc.MaxLatency = MaxLatency;
        }
        if (isCoalescedROBSubtarget(STI)) {
            mca::InstrDesc &MutableDesc = const_cast<mca::InstrDesc &>(Inst->getDesc());
            MutableDesc.NumMicroOps = 1;
        }
        
        Sequence.push_back(std::move(Inst));
    }

    const unsigned WarmupIterations = computeWarmupIterations(STI, instrs.size());
    const unsigned SteadyIterations = computeSteadyIterations(STI, instrs.size(), iterations);
    mca::CircularSourceMgr CSM(Sequence, WarmupIterations + SteadyIterations);
    mca::CustomBehaviour CB(STI, CSM, MCII);
    bool IsA55 = (getCpuTraits(STI.getCPU()).Model == CpuModel::A55);
    std::unique_ptr<mca::Pipeline> P;
    if (STI.getSchedModel().isOutOfOrder()) {
        P = MCAContext.createDefaultPipeline(PO, CSM, CB);
    } else if (IsA55) {
        auto PRF_unique = std::make_unique<mca::RegisterFile>(STI.getSchedModel(), MRI);
        mca::RegisterFile *PRF_ptr = PRF_unique.get();
        MCAContext.addHardwareUnit(std::move(PRF_unique));
        P = mca::createA55DecoupledPipeline(PO, CSM, CB, STI, MRI, *PRF_ptr, MCII, CSM.getInstructions());
    } else {
        P = MCAContext.createInOrderPipeline(PO, CSM, CB);
    }

    mca::RegisterFile *PRF = nullptr;
    auto &hardware = MCAContext.*get(Context_Hardware_Tag{});
    for (auto &hu : hardware) {
        if (auto *r = dynamic_cast<mca::RegisterFile *>(hu.get())) {
            PRF = r;
            break;
        }
    }

    std::unique_ptr<BaseSteadyStateTracker> Tracker;
    unsigned WarmupLimit = instrs.size() * WarmupIterations;

    if (ignoreLoopCarriedDep) {
        Tracker = std::make_unique<LoopCarriedIgnoreSteadyStateTracker>(
            WarmupLimit, instrs.size(), PRF);
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
