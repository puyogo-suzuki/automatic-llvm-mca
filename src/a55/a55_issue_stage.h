#ifndef A55_ISSUE_STAGE_H
#define A55_ISSUE_STAGE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/MCA/Context.h"
#include "llvm/MCA/HardwareUnits/RegisterFile.h"
#include "llvm/MCA/HardwareUnits/ResourceManager.h"
#include "llvm/MCA/HardwareUnits/LSUnit.h"
#include "llvm/MCA/HardwareUnits/RetireControlUnit.h"
#include "llvm/MCA/Pipeline.h"
#include "llvm/MCA/SourceMgr.h"
#include "llvm/MCA/Stages/Stage.h"
#include "llvm/MCA/Stages/InOrderIssueStage.h"
#include <functional>
#include <memory>

namespace llvm {
namespace mca {

/// 2-wide in-order issue stage for Cortex-A55 (SOG v3 sec. 3.2, 3.4, 5.1).
/// - Issues at most an (instruction-0, instruction-1) pair per cycle, strictly
///   in program order: the younger instruction never issues ahead of a stalled
///   older one (SOG sec. 5.1 "does not support any instruction reordering at
///   issue").
/// - Honours the SOG dual-issue column through the scheduling class's
///   BeginGroup (01: slot 0 only) / EndGroup (10: slot 1 only) / SingleIssue
///   (00) bits.
/// - Applies the cmp -> b.cc / csel flag bypass (not for ADC/SBC consumers nor
///   FCMP/FCCMP producers, SOG 3.2 Table 2 / 5.11) and the adrp -> ldr pointer
///   forwarding before the RAW hazard check (a55_bypass.h).
/// - Applies the dedicated ALU0 ex1 -> ALU1 ex1 forwarding path (SOG sec.
///   3.1.1): a 1-cycle ALU instruction-0 and an ALU instruction-1 that reads its
///   result in the ALU (not in the shifter) issue in the same cycle.
/// - The only write-back ordering rule is SOG sec. 3.4's: nothing younger than
///   an in-flight integer divide may complete before it.
class A55DecoupledIssueStage final : public Stage {
  const MCSubtargetInfo &STI;
  const MCRegisterInfo &MRI;
  RegisterFile &PRF;
  ResourceManager RM;
  CustomBehaviour &CB;
  const MCInstrInfo &MCII;
  ArrayRef<SourceMgr::UniqueInst> SimInstrs;
  std::unique_ptr<LSUnitBase> LSU_Owner;
  LSUnitBase &LSU;

  // 2-instruction decode/issue window
  SmallVector<InstRef, 2> DecodeWindow;
  SmallVector<InstRef, 16> IssuedInst;

  StallInfo Stalls[2]; // Stalls for instructions currently in DecodeWindow

  unsigned NumIssued = 0;
  unsigned NumRetired = 0;
  // Cycles until the youngest in-flight integer divide writes back (SOG 3.4).
  unsigned DivWriteBackCycle = 0;
  // True once an EndGroup (dual-issue 10 / 00) instruction issued this cycle.
  bool LastIssuedEndsGroup = false;

  // Same-cycle ALU0 -> ALU1 forwarding (SOG 3.1.1), see isSameCycleALUForward().
  // Source index of this cycle's instruction-0, valid while NumIssued == 1, and
  // whether it is a 1-cycle ALU-pipe instruction (a producer for that path).
  unsigned Slot0SourceIndex = 0;
  bool Slot0IsALUForwardProducer = false;
  // CortexA55UnitALU's index in the scheduling model (0 = not found).
  unsigned ALUProcResIdx = 0;
  // CortexA55UnitVarShift's index (0 = not found): the variable-shift
  // throughput cap inside the ALU pipes [a55_varshift], see isALUPipeOnly().
  unsigned VarShiftProcResIdx = 0;
  // WriteResourceID of WriteLD (the LDRXui write); a consumer read whose
  // ReadAdvance from it is >= 1 is an operand read by the ALU (the load-use
  // path of SOG 3.3), as opposed to one read by the shifter.
  unsigned LoadWriteResID = 0;

  // Called with each instruction before its register hazard check (see
  // setBeforeRegisterHazardCheckHook()); empty = nothing to do.
  std::function<void(const InstRef &)> BeforeRegisterHazardCheck;

  A55DecoupledIssueStage(const A55DecoupledIssueStage &) = delete;
  A55DecoupledIssueStage &operator=(const A55DecoupledIssueStage &) = delete;

  unsigned checkRegisterHazard(const InstRef &IR) const;
  bool isALUPipeOnly(unsigned SchedClassID) const;
  bool isALUForwardProducer(const Instruction &IS) const;
  bool isSameCycleALUForward(const InstRef &IR, const ReadState &RS) const;
  bool canExecute(const InstRef &IR, StallInfo &SI);
  Error issue(InstRef &IR);
  void drainDecodeWindow();
  void updateIssuedInst();
  void retireInstruction(InstRef &IR);

public:
  A55DecoupledIssueStage(const MCSubtargetInfo &STI, const MCRegisterInfo &MRI,
                         RegisterFile &PRF, CustomBehaviour &CB,
                         const MCInstrInfo &MCII, ArrayRef<SourceMgr::UniqueInst> SimInstrs,
                         std::unique_ptr<LSUnitBase> LSUB);

  bool isAvailable(const InstRef &IR) const override;
  bool hasWorkToComplete() const override;
  Error execute(InstRef &IR) override;
  Error cycleStart() override;
  Error cycleEnd() override;
  unsigned getNumRetired() const { return NumRetired; }

  // `Hook` runs on every issue attempt, before the register hazard check.  Used
  // by --ignore-loop-carried (LoopCarriedIgnoreSteadyStateTracker::
  // beforeRegisterHazardCheck) to drop the previous iteration's producers before
  // the first instruction of an iteration is checked, not only at its
  // Dispatched event, which comes after the check.
  void setBeforeRegisterHazardCheckHook(std::function<void(const InstRef &)> Hook) {
    BeforeRegisterHazardCheck = std::move(Hook);
  }
};

// `IssueStageOut`, if not null, receives the issue stage (owned by the pipeline).
std::unique_ptr<Pipeline> createA55DecoupledPipeline(const PipelineOptions &Opts,
                                                     SourceMgr &SrcMgr,
                                                     CustomBehaviour &CB,
                                                     const MCSubtargetInfo &STI,
                                                     const MCRegisterInfo &MRI,
                                                     RegisterFile &PRF,
                                                     const MCInstrInfo &MCII,
                                                     ArrayRef<SourceMgr::UniqueInst> SimInstrs,
                                                     A55DecoupledIssueStage **IssueStageOut = nullptr);

} // namespace mca
} // namespace llvm

#endif // A55_ISSUE_STAGE_H
