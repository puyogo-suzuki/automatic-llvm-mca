#ifndef A55_ISSUE_STAGE_H
#define A55_ISSUE_STAGE_H

#include "llvm/ADT/SmallVector.h"
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
#include <memory>

namespace llvm {
namespace mca {

/// True 2-wide Decoupled Issue Stage for Cortex-A55.
/// - Models the 2-instruction dual-issue decode window.
/// - If Slot 0 holds a stalled FP instruction (e.g. fmul waiting for RAW data),
///   an independent Integer instruction in Slot 1 (e.g. add/subs/b.ne) can bypass
///   Slot 0 and issue concurrently to the integer pipeline.
/// - Integer-to-Integer and FP-to-FP remain strictly In-Order.
/// - Integer instructions strictly commit in program order.
class A55DecoupledIssueStage final : public Stage {
  const MCSubtargetInfo &STI;
  const MCRegisterInfo &MRI;
  RegisterFile &PRF;
  ResourceManager RM;
  CustomBehaviour &CB;
  std::unique_ptr<LSUnitBase> LSU_Owner;
  LSUnitBase &LSU;

  // 2-instruction decode/issue window
  SmallVector<InstRef, 2> DecodeWindow;
  SmallVector<InstRef, 16> IssuedInst;

  StallInfo Stalls[2]; // Stalls for instructions currently in DecodeWindow

  unsigned NumIssued = 0;
  unsigned NumRetired = 0;
  unsigned LastWriteBackCycle = 0;

  A55DecoupledIssueStage(const A55DecoupledIssueStage &) = delete;
  A55DecoupledIssueStage &operator=(const A55DecoupledIssueStage &) = delete;

  bool isFPInstruction(const InstRef &IR) const;
  bool canExecute(const InstRef &IR, StallInfo &SI);
  bool checkInterSlotDependency(const InstRef &Producer, const InstRef &Consumer) const;
  Error issue(InstRef &IR);
  void drainDecodeWindow();
  void updateIssuedInst();
  void retireInstruction(InstRef &IR);

public:
  A55DecoupledIssueStage(const MCSubtargetInfo &STI, const MCRegisterInfo &MRI,
                         RegisterFile &PRF, CustomBehaviour &CB,
                         std::unique_ptr<LSUnitBase> LSUB);

  bool isAvailable(const InstRef &IR) const override;
  bool hasWorkToComplete() const override;
  Error execute(InstRef &IR) override;
  Error cycleStart() override;
  Error cycleEnd() override;
  unsigned getNumRetired() const { return NumRetired; }
};

std::unique_ptr<Pipeline> createA55DecoupledPipeline(const PipelineOptions &Opts,
                                                     SourceMgr &SrcMgr,
                                                     CustomBehaviour &CB,
                                                     const MCSubtargetInfo &STI,
                                                     const MCRegisterInfo &MRI,
                                                     RegisterFile &PRF);

} // namespace mca
} // namespace llvm

#endif // A55_ISSUE_STAGE_H
