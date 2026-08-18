#include "a55_issue_stage.h"
#include "llvm/MCA/Stages/EntryStage.h"
#include "llvm/MCA/HWEventListener.h"

namespace llvm {
namespace mca {

static bool hasResourceHazard(const ResourceManager &RM, const InstRef &IR) {
  return RM.checkAvailability(IR.getInstruction()->getDesc()) != 0;
}

static unsigned findFirstWriteBackCycle(const InstRef &IR) {
  unsigned FirstWBCycle = IR.getInstruction()->getLatency();
  for (const WriteState &WS : IR.getInstruction()->getDefs()) {
    int CyclesLeft = WS.getCyclesLeft();
    if (CyclesLeft == UNKNOWN_CYCLES)
      CyclesLeft = WS.getLatency();
    if (CyclesLeft < 0)
      CyclesLeft = 0;
    FirstWBCycle = std::min(FirstWBCycle, (unsigned)CyclesLeft);
  }
  return FirstWBCycle;
}

static unsigned checkRegisterHazard(const RegisterFile &PRF,
                                    const MCSubtargetInfo &STI,
                                    const InstRef &IR) {
  for (const ReadState &RS : IR.getInstruction()->getUses()) {
    RegisterFile::RAWHazard Hazard = PRF.checkRAWHazards(STI, RS);
    if (Hazard.isValid())
      return Hazard.hasUnknownCycles() ? 1U : Hazard.CyclesLeft;
  }
  return 0;
}

static void addRegisterReadWrite(RegisterFile &PRF, Instruction &IS,
                                 unsigned SourceIndex,
                                 const MCSubtargetInfo &STI,
                                 SmallVectorImpl<unsigned> &UsedRegs) {
  for (ReadState &RS : IS.getUses())
    PRF.addRegisterRead(RS, STI);

  for (WriteState &WS : IS.getDefs())
    PRF.addRegisterWrite(WriteRef(SourceIndex, &WS), UsedRegs);
}

A55DecoupledIssueStage::A55DecoupledIssueStage(const MCSubtargetInfo &STI,
                                               const MCRegisterInfo &MRI,
                                               RegisterFile &PRF,
                                               CustomBehaviour &CB,
                                               std::unique_ptr<LSUnitBase> LSUB)
    : STI(STI), MRI(MRI), PRF(PRF), RM(STI.getSchedModel()), CB(CB),
      LSU_Owner(std::move(LSUB)), LSU(*LSU_Owner), LastWriteBackCycle(0) {}

bool A55DecoupledIssueStage::isFPInstruction(const InstRef &IR) const {
  const InstrDesc &Desc = IR.getInstruction()->getDesc();
  // In CortexA55ModelProcResources:
  // #4: CortexA55UnitFPALU (1ULL << 4 = 0x10)
  // #5: CortexA55UnitFPDIV (1ULL << 5 = 0x20)
  // #6: CortexA55UnitFPMAC (1ULL << 6 = 0x40)
  constexpr uint64_t FPMask = (1ULL << 4) | (1ULL << 5) | (1ULL << 6);
  for (const std::pair<uint64_t, ResourceUsage> &R : Desc.Resources) {
    if (R.first & FPMask) {
      return true;
    }
  }
  return false;
}

bool A55DecoupledIssueStage::checkInterSlotDependency(const InstRef &Producer,
                                                      const InstRef &Consumer) const {
  const Instruction &ProdIS = *Producer.getInstruction();
  const Instruction &ConsIS = *Consumer.getInstruction();

  for (const WriteState &WS : ProdIS.getDefs()) {
    MCPhysReg DefReg = WS.getRegisterID();
    if (!DefReg)
      continue;
    for (const ReadState &RS : ConsIS.getUses()) {
      if (RS.getRegisterID() == DefReg)
        return true;
    }
  }
  return false;
}

bool A55DecoupledIssueStage::canExecute(const InstRef &IR, StallInfo &SI) {
  if (unsigned Cycles = checkRegisterHazard(PRF, STI, IR)) {
    SI.update(IR, Cycles, StallInfo::StallKind::REGISTER_DEPS);
    return false;
  }

  if (hasResourceHazard(RM, IR)) {
    SI.update(IR, 1, StallInfo::StallKind::DISPATCH);
    return false;
  }

  if (IR.getInstruction()->isMemOp() && !LSU.isReady(IR)) {
    SI.update(IR, 1, StallInfo::StallKind::LOAD_STORE);
    return false;
  }

  if (unsigned CustomStallCycles = CB.checkCustomHazard(IssuedInst, IR)) {
    SI.update(IR, CustomStallCycles, StallInfo::StallKind::CUSTOM_STALL);
    return false;
  }

  // Integer-to-Integer In-Order Commit enforcement
  if (LastWriteBackCycle && !IR.getInstruction()->getRetireOOO()) {
    unsigned NextWriteBackCycle = findFirstWriteBackCycle(IR);
    if (NextWriteBackCycle < LastWriteBackCycle) {
      SI.update(IR, LastWriteBackCycle - NextWriteBackCycle,
                StallInfo::StallKind::DELAY);
      return false;
    }
  }

  return true;
}

Error A55DecoupledIssueStage::issue(InstRef &IR) {
  Instruction &IS = *IR.getInstruction();
  unsigned SourceIndex = IR.getSourceIndex();
  const InstrDesc &Desc = IS.getDesc();

  unsigned RCUTokenID = RetireControlUnit::UnhandledTokenID;
  IS.dispatch(RCUTokenID);

  SmallVector<unsigned, 4> UsedRegs(PRF.getNumRegisterFiles());
  addRegisterReadWrite(PRF, IS, SourceIndex, STI, UsedRegs);

  unsigned NumMicroOps = IS.getNumMicroOps();
  notifyEvent<HWInstructionEvent>(
      HWInstructionDispatchedEvent(IR, UsedRegs, NumMicroOps));

  SmallVector<ResourceUse, 4> UsedResources;
  RM.issueInstruction(Desc, UsedResources);
  IS.execute(SourceIndex);

  if (IS.isMemOp())
    LSU.onInstructionIssued(IR);

  for (ResourceUse &Use : UsedResources) {
    uint64_t Mask = Use.first.first;
    Use.first.first = RM.resolveResourceMask(Mask);
  }
  notifyEvent<HWInstructionEvent>(
      HWInstructionEvent(HWInstructionEvent::Ready, IR));
  notifyEvent<HWInstructionEvent>(
      HWInstructionIssuedEvent(IR, UsedResources));

  NumIssued += NumMicroOps;

  if (IS.isExecuted()) {
    PRF.onInstructionExecuted(&IS);
    LSU.onInstructionExecuted(IR);
    notifyEvent<HWInstructionEvent>(
        HWInstructionEvent(HWInstructionEvent::Executed, IR));
    retireInstruction(IR);
    return ErrorSuccess();
  }

  IssuedInst.push_back(IR);

  if (!IR.getInstruction()->getRetireOOO())
    LastWriteBackCycle = IS.getCyclesLeft();

  return ErrorSuccess();
}

void A55DecoupledIssueStage::drainDecodeWindow() {
  if (DecodeWindow.empty())
    return;

  bool IssuedSlot0 = false;
  bool IssuedSlot1 = false;

  // 1. Evaluate Slot 0
  InstRef &IR0 = DecodeWindow[0];
  if (canExecute(IR0, Stalls[0])) {
    (void)issue(IR0);
    IssuedSlot0 = true;
  }

  // 2. Evaluate Slot 1 (if present and haven't exceeded issue width of 2)
  if (DecodeWindow.size() >= 2 && NumIssued < 2) {
    InstRef &IR1 = DecodeWindow[1];
    if (IssuedSlot0) {
      // Slot 0 issued successfully, Slot 1 can issue normally if ready
      if (canExecute(IR1, Stalls[1])) {
        (void)issue(IR1);
        IssuedSlot1 = true;
      }
    } else {
      // Slot 0 is STALLED:
      // In Cortex-A55, Decoupled Issue is ONLY allowed across different pipeline domains:
      // - Slot 0 is FP (stalled) and Slot 1 is Integer (independent) -> Issue Slot 1 to Integer Pipe
      // - Slot 0 is Integer (stalled) and Slot 1 is FP (independent) -> Issue Slot 1 to FP Pipe
      // Intra-pipe (Int->Int, FP->FP) remains strictly In-Order!
      bool Slot0IsFP = isFPInstruction(IR0);
      bool Slot1IsFP = isFPInstruction(IR1);
      bool IsCrossPipe = (Slot0IsFP != Slot1IsFP);

      if (IsCrossPipe && !checkInterSlotDependency(IR0, IR1) && canExecute(IR1, Stalls[1])) {
        (void)issue(IR1);
        IssuedSlot1 = true;
      }
    }
  }

  // 3. Compact DecodeWindow
  SmallVector<InstRef, 2> NextWindow;
  if (!IssuedSlot0)
    NextWindow.push_back(DecodeWindow[0]);
  if (DecodeWindow.size() >= 2 && !IssuedSlot1)
    NextWindow.push_back(DecodeWindow[1]);

  DecodeWindow = std::move(NextWindow);
}

void A55DecoupledIssueStage::updateIssuedInst() {
  SmallVector<InstRef, 16> StillExecuting;
  for (InstRef &IR : IssuedInst) {
    Instruction &IS = *IR.getInstruction();
    IS.cycleEvent();

    if (!IS.isExecuted()) {
      StillExecuting.push_back(IR);
      continue;
    }

    PRF.onInstructionExecuted(&IS);
    LSU.onInstructionExecuted(IR);
    notifyEvent<HWInstructionEvent>(
        HWInstructionEvent(HWInstructionEvent::Executed, IR));

    retireInstruction(IR);
  }
  IssuedInst = std::move(StillExecuting);
}

void A55DecoupledIssueStage::retireInstruction(InstRef &IR) {
  Instruction &IS = *IR.getInstruction();
  IS.retire();
  NumRetired++;

  SmallVector<unsigned, 4> FreedRegs(PRF.getNumRegisterFiles());
  for (const WriteState &WS : IS.getDefs()) {
    PRF.removeRegisterWrite(WS, FreedRegs);
  }

  if (IS.isMemOp())
    LSU.onInstructionRetired(IR);

  notifyEvent<HWInstructionEvent>(
      HWInstructionRetiredEvent(IR, FreedRegs));
}

bool A55DecoupledIssueStage::isAvailable(const InstRef &IR) const {
  return DecodeWindow.size() < 2;
}

bool A55DecoupledIssueStage::hasWorkToComplete() const {
  return !IssuedInst.empty() || !DecodeWindow.empty();
}

Error A55DecoupledIssueStage::execute(InstRef &IR) {
  Instruction &IS = *IR.getInstruction();
  if (IS.isMemOp())
    IS.setLSUTokenID(LSU.dispatch(IR));

  DecodeWindow.push_back(IR);
  drainDecodeWindow();
  return ErrorSuccess();
}

Error A55DecoupledIssueStage::cycleStart() {
  NumIssued = 0;

  PRF.cycleStart();
  LSU.cycleEvent();

  SmallVector<ResourceRef, 4> Freed;
  RM.cycleEvent(Freed);

  updateIssuedInst();
  drainDecodeWindow();

  return ErrorSuccess();
}

Error A55DecoupledIssueStage::cycleEnd() {
  PRF.cycleEnd();
  Stalls[0].cycleEnd();
  Stalls[1].cycleEnd();

  if (LastWriteBackCycle > 0)
    --LastWriteBackCycle;

  return ErrorSuccess();
}

std::unique_ptr<Pipeline> createA55DecoupledPipeline(const PipelineOptions &Opts,
                                                     SourceMgr &SrcMgr,
                                                     CustomBehaviour &CB,
                                                     const MCSubtargetInfo &STI,
                                                     const MCRegisterInfo &MRI,
                                                     RegisterFile &PRF) {
  auto P = std::make_unique<Pipeline>();
  auto Entry = std::make_unique<EntryStage>(SrcMgr);
  auto LSU = std::make_unique<LSUnit>(
      STI.getSchedModel(), Opts.LoadQueueSize, Opts.StoreQueueSize, Opts.AssumeNoAlias);
  auto Issue = std::make_unique<A55DecoupledIssueStage>(
      STI, MRI, PRF, CB, std::move(LSU));

  Entry->setNextInSequence(Issue.get());

  P->appendStage(std::move(Entry));
  P->appendStage(std::move(Issue));
  return P;
}

} // namespace mca
} // namespace llvm
