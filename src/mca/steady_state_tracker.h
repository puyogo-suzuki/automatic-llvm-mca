#ifndef MCA_STEADY_STATE_TRACKER_H
#define MCA_STEADY_STATE_TRACKER_H

// Event listeners that turn an MCA run into a steady-state cycle / retired
// instruction count: cycles before the warm-up limit are not counted.

#include "llvm/MCA/HWEventListener.h"
#include "llvm/MCA/HardwareUnits/RegisterFile.h"

struct BaseSteadyStateTracker : llvm::mca::HWEventListener {
    unsigned WarmupRetiredLimit;
    unsigned TotalRetired = 0;
    unsigned SteadyRetired = 0;
    unsigned CurrentCycle = 0;
    unsigned SteadyStartCycle = 0;
    unsigned SteadyCycles = 0;
    bool WarmupComplete = false;

    explicit BaseSteadyStateTracker(unsigned WarmupRetiredLimit)
        : WarmupRetiredLimit(WarmupRetiredLimit) {}

    virtual ~BaseSteadyStateTracker() = default;

    void onCycleBegin() override {
        ++CurrentCycle;
    }

    void onCycleEnd() override {
        if (WarmupComplete && CurrentCycle > SteadyStartCycle) ++SteadyCycles;
    }

    void onEvent(const llvm::mca::HWInstructionEvent &Event) override {
        if (Event.Type == llvm::mca::HWInstructionEvent::Retired) {
            ++TotalRetired;
            if (WarmupComplete) {
                ++SteadyRetired;
                return;
            }
            if (TotalRetired >= WarmupRetiredLimit) {
                WarmupComplete = true;
                SteadyStartCycle = CurrentCycle;
            }
        } else if (Event.Type == llvm::mca::HWInstructionEvent::Dispatched) {
            onInstructionDispatched(Event);
        }
    }

    virtual void onInstructionDispatched(const llvm::mca::HWInstructionEvent &Event) {}
};

// Additionally removes the dependence of a loop iteration on the previous one
// (marks reads whose critical producer belongs to an earlier iteration as
// ready and drops those producers from the register mappings).
struct LoopCarriedIgnoreSteadyStateTracker : public BaseSteadyStateTracker {
    unsigned LoopSize;
    llvm::mca::RegisterFile *PRF;
    unsigned CurrentIteration = 0;

    LoopCarriedIgnoreSteadyStateTracker(unsigned WarmupRetiredLimit, unsigned LoopSize, llvm::mca::RegisterFile *PRF)
        : BaseSteadyStateTracker(WarmupRetiredLimit), LoopSize(LoopSize), PRF(PRF) {}

    void onInstructionDispatched(const llvm::mca::HWInstructionEvent &Event) override;
};

#endif // MCA_STEADY_STATE_TRACKER_H
