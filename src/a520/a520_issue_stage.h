#ifndef A520_ISSUE_STAGE_H
#define A520_ISSUE_STAGE_H

// In-order issue stage for Cortex-A520 (docs/a520-sog.md, D-9 and section 2.2).
//
// This is llvm::mca::InOrderIssueStage of LLVM 22.1.8
// (llvm/lib/MCA/Stages/InOrderIssueStage.cpp, Apache-2.0 WITH LLVM-exception)
// with exactly three changes, all confined to the cortex-a520 path:
//  1. [a520_no_wb_order] (docs/a520-sog.md D-9; docs/a55-sog.md X-1): the stock rule
//     "an instruction may not issue if it would write back before the last
//     issued one" (LastWriteBackCycle) is not applied.  The Cortex-A520 SOG
//     states no such rule; its co-issue limits (sec. 4.1) are the only issue
//     constraints besides the hazards.  `EnforceWriteBackOrder = true` restores
//     the stock behaviour (used to check that the copy is faithful).
//  2. [a520_lc_hook] (docs/a520-sog.md section 2.2; docs/a55-sog.md X-2): an optional
//     hook runs before the register hazard check, so that
//     --ignore-loop-carried can drop the previous iteration's producers before
//     the first instruction of an iteration is checked (issue #12).
//  3. [a520_from_a55_flags] (docs/a520-sog.md B-4): the Cortex-A55's same-cycle
//     flag bypass (src/a55/a55_bypass.h, a55IsFlagBypassEligible; docs/a55-sog.md
//     C-4, D-8) is applied in the register hazard check: an NZCV read does not
//     stall unless the consumer is ADC/ADCS/SBC/SBCS or the producer is an FP
//     compare (FCMP/FCCMP).  Borrowed from the A55; the A520 SOG has no such
//     statement.  Active only when the stage is given the MCInstrInfo and the
//     static instruction sequence (`MCII` not null); without them the hazard
//     check is the stock one.
// Everything else (issue width counted in NumMicroOps, BeginGroup/EndGroup,
// carry-over of instructions wider than the issue width, the stall handling)
// is the stock code.

#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/MCA/HardwareUnits/LSUnit.h"
#include "llvm/MCA/HardwareUnits/RegisterFile.h"
#include "llvm/MCA/HardwareUnits/ResourceManager.h"
#include "llvm/MCA/Pipeline.h"
#include "llvm/MCA/SourceMgr.h"
#include "llvm/MCA/Stages/InOrderIssueStage.h"  // StallInfo
#include "llvm/MCA/Stages/Stage.h"
#include <functional>
#include <memory>

namespace llvm {
namespace mca {

class A520InOrderIssueStage final : public Stage {
  const MCSubtargetInfo &STI;
  RegisterFile &PRF;
  ResourceManager RM;
  CustomBehaviour &CB;
  LSUnitBase &LSU;
  const bool EnforceWriteBackOrder;
  // [a520_from_a55_flags] docs/a520-sog.md B-4: the flag bypass needs the opcode of
  // the NZCV producer; null disables it.
  const MCInstrInfo *MCII;
  ArrayRef<SourceMgr::UniqueInst> SimInstrs;

  SmallVector<InstRef, 4> IssuedInst;
  unsigned NumIssued = 0;
  StallInfo SI;
  InstRef CarriedOver;
  unsigned CarryOver = 0;
  unsigned Bandwidth = 0;
  // Only used when EnforceWriteBackOrder (the stock rule).
  unsigned LastWriteBackCycle = 0;

  std::function<void(const InstRef &)> BeforeRegisterHazardCheck;

  A520InOrderIssueStage(const A520InOrderIssueStage &) = delete;
  A520InOrderIssueStage &operator=(const A520InOrderIssueStage &) = delete;

  bool canExecute(const InstRef &IR);
  unsigned checkRegisterHazard(const InstRef &IR) const;
  Error tryIssue(InstRef &IR);
  void updateIssuedInst();
  void updateCarriedOver();
  void notifyStallEvent();
  void retireInstruction(InstRef &IR);

public:
  A520InOrderIssueStage(const MCSubtargetInfo &STI, RegisterFile &PRF,
                        CustomBehaviour &CB, LSUnitBase &LSU,
                        bool EnforceWriteBackOrder = false,
                        const MCInstrInfo *MCII = nullptr,
                        ArrayRef<SourceMgr::UniqueInst> SimInstrs = {});

  unsigned getIssueWidth() const;
  bool isAvailable(const InstRef &) const override;
  bool hasWorkToComplete() const override;
  Error execute(InstRef &IR) override;
  Error cycleStart() override;
  Error cycleEnd() override;

  // `Hook` runs on every issue attempt, before the register hazard check
  // (--ignore-loop-carried: LoopCarriedIgnoreSteadyStateTracker::
  // beforeRegisterHazardCheck).  [a520_lc_hook]
  void setBeforeRegisterHazardCheckHook(std::function<void(const InstRef &)> Hook) {
    BeforeRegisterHazardCheck = std::move(Hook);
  }
};

// Builds Entry -> A520InOrderIssueStage, like Context::createInOrderPipeline.
// `PRF` and `LSU` must outlive the pipeline (the caller hands them to the
// mca::Context).  `IssueStageOut`, if not null, receives the issue stage.
// `MCII` and `SimInstrs` (the static instruction sequence of the source manager)
// enable the flag bypass [a520_from_a55_flags]; with `MCII` null it is off.
std::unique_ptr<Pipeline> createA520InOrderPipeline(SourceMgr &SrcMgr, CustomBehaviour &CB,
                                                    const MCSubtargetInfo &STI,
                                                    RegisterFile &PRF, LSUnitBase &LSU,
                                                    A520InOrderIssueStage **IssueStageOut = nullptr,
                                                    bool EnforceWriteBackOrder = false,
                                                    const MCInstrInfo *MCII = nullptr,
                                                    ArrayRef<SourceMgr::UniqueInst> SimInstrs = {});

} // namespace mca
} // namespace llvm

#endif // A520_ISSUE_STAGE_H
