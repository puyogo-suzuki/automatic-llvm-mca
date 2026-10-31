// Cortex-A520 in-order issue stage: llvm::mca::InOrderIssueStage of LLVM 22.1.8
// (llvm/lib/MCA/Stages/InOrderIssueStage.cpp; part of the LLVM Project, under
// the Apache License v2.0 with LLVM Exceptions, see https://llvm.org/LICENSE.txt)
// with the three changes described in a520_issue_stage.h.  The code below follows
// the stock file function by function; the comments mark the changes.

#include "a520/a520_issue_stage.h"
#include "a55/a55_bypass.h"  // [a520_from_a55_flags] a55IsFlagBypassEligible
#include "llvm/MCA/HWEventListener.h"
#include "llvm/MCA/HardwareUnits/RetireControlUnit.h"
#include "llvm/MCA/Instruction.h"
#include "llvm/MCA/Stages/EntryStage.h"
#include <algorithm>

namespace llvm {
namespace mca {

A520InOrderIssueStage::A520InOrderIssueStage(const MCSubtargetInfo &STI, RegisterFile &PRF,
                                             CustomBehaviour &CB, LSUnitBase &LSU,
                                             bool EnforceWriteBackOrder,
                                             const MCInstrInfo *MCII,
                                             ArrayRef<SourceMgr::UniqueInst> SimInstrs)
    : STI(STI), PRF(PRF), RM(STI.getSchedModel()), CB(CB), LSU(LSU),
      EnforceWriteBackOrder(EnforceWriteBackOrder), MCII(MCII), SimInstrs(SimInstrs) {}

unsigned A520InOrderIssueStage::getIssueWidth() const {
  return STI.getSchedModel().IssueWidth;
}

bool A520InOrderIssueStage::hasWorkToComplete() const {
  return !IssuedInst.empty() || SI.isValid() || CarriedOver;
}

bool A520InOrderIssueStage::isAvailable(const InstRef &IR) const {
  if (SI.isValid() || CarriedOver)
    return false;

  const Instruction &Inst = *IR.getInstruction();
  unsigned NumMicroOps = Inst.getNumMicroOps();

  bool ShouldCarryOver = NumMicroOps > getIssueWidth();
  if (Bandwidth < NumMicroOps && !ShouldCarryOver)
    return false;

  // Instruction with BeginGroup must be the first instruction to be issued in a
  // cycle.
  if (Inst.getBeginGroup() && NumIssued != 0)
    return false;

  return true;
}

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

unsigned A520InOrderIssueStage::checkRegisterHazard(const InstRef &IR) const {
  for (const ReadState &RS : IR.getInstruction()->getUses()) {
    // [a520_from_a55_flags] docs/a520-sog.md B-4: the Cortex-A55's flag bypass
    // (A55DecoupledIssueStage::checkRegisterHazard, docs/a55-sog.md C-4, D-8), with
    // the same eligibility: a flag-setting producer and a conditional consumer
    // (B.cond, CSEL, CCMP, ...) issue in the same cycle, except when the consumer is
    // ADC/ADCS/SBC/SBCS (arithmetic with carry) or the producer is FCMP/FCCMP (FP
    // flags).  The A520 SOG states no flag forwarding; not SOG-based.
    if (MCII && RS.getRegisterID() == AArch64::NZCV &&
        a55IsFlagBypassEligible(PRF, *MCII, SimInstrs, IR.getInstruction()->getOpcode()))
      continue;
    RegisterFile::RAWHazard Hazard = PRF.checkRAWHazards(STI, RS);
    if (Hazard.isValid())
      return Hazard.hasUnknownCycles() ? 1U : Hazard.CyclesLeft;
  }
  return 0;
}

bool A520InOrderIssueStage::canExecute(const InstRef &IR) {
  // [a520_lc_hook] --ignore-loop-carried: forget the previous iteration's
  // producers before the first instruction of an iteration is checked (#12).
  if (BeforeRegisterHazardCheck)
    BeforeRegisterHazardCheck(IR);

  if (unsigned Cycles = checkRegisterHazard(IR)) {
    SI.update(IR, Cycles, StallInfo::StallKind::REGISTER_DEPS);
    return false;
  }

  if (hasResourceHazard(RM, IR)) {
    SI.update(IR, /* delay */ 1, StallInfo::StallKind::DISPATCH);
    return false;
  }

  if (IR.getInstruction()->isMemOp() && !LSU.isReady(IR)) {
    SI.update(IR, /* delay */ 1, StallInfo::StallKind::LOAD_STORE);
    return false;
  }

  if (unsigned CustomStallCycles = CB.checkCustomHazard(IssuedInst, IR)) {
    SI.update(IR, CustomStallCycles, StallInfo::StallKind::CUSTOM_STALL);
    return false;
  }

  // [a520_no_wb_order] docs/a520-sog.md D-9: the stock "writes happen in
  // program order" delay is applied only when EnforceWriteBackOrder is set.
  if (EnforceWriteBackOrder && LastWriteBackCycle) {
    if (!IR.getInstruction()->getRetireOOO()) {
      unsigned NextWriteBackCycle = findFirstWriteBackCycle(IR);
      if (NextWriteBackCycle < LastWriteBackCycle) {
        SI.update(IR, LastWriteBackCycle - NextWriteBackCycle,
                  StallInfo::StallKind::DELAY);
        return false;
      }
    }
  }

  return true;
}

static void addRegisterReadWrite(RegisterFile &PRF, Instruction &IS, unsigned SourceIndex,
                                 const MCSubtargetInfo &STI,
                                 SmallVectorImpl<unsigned> &UsedRegs) {
  for (ReadState &RS : IS.getUses())
    PRF.addRegisterRead(RS, STI);

  for (WriteState &WS : IS.getDefs())
    PRF.addRegisterWrite(WriteRef(SourceIndex, &WS), UsedRegs);
}

Error A520InOrderIssueStage::execute(InstRef &IR) {
  Instruction &IS = *IR.getInstruction();
  if (IS.isMemOp())
    IS.setLSUTokenID(LSU.dispatch(IR));

  if (Error E = tryIssue(IR))
    return E;

  if (SI.isValid())
    notifyStallEvent();

  return ErrorSuccess();
}

Error A520InOrderIssueStage::tryIssue(InstRef &IR) {
  Instruction &IS = *IR.getInstruction();
  unsigned SourceIndex = IR.getSourceIndex();
  const InstrDesc &Desc = IS.getDesc();

  if (!canExecute(IR)) {
    Bandwidth = 0;
    return ErrorSuccess();
  }

  unsigned RCUTokenID = RetireControlUnit::UnhandledTokenID;
  IS.dispatch(RCUTokenID);

  SmallVector<unsigned, 4> UsedRegs(PRF.getNumRegisterFiles());
  addRegisterReadWrite(PRF, IS, SourceIndex, STI, UsedRegs);

  unsigned NumMicroOps = IS.getNumMicroOps();
  notifyEvent<HWInstructionEvent>(HWInstructionDispatchedEvent(IR, UsedRegs, NumMicroOps));

  SmallVector<ResourceUse, 4> UsedResources;
  RM.issueInstruction(Desc, UsedResources);
  IS.execute(SourceIndex);

  if (IS.isMemOp())
    LSU.onInstructionIssued(IR);

  for (ResourceUse &Use : UsedResources) {
    uint64_t Mask = Use.first.first;
    Use.first.first = RM.resolveResourceMask(Mask);
  }
  notifyEvent<HWInstructionEvent>(HWInstructionEvent(HWInstructionEvent::Ready, IR));
  notifyEvent<HWInstructionEvent>(HWInstructionIssuedEvent(IR, UsedResources));

  bool ShouldCarryOver = NumMicroOps > Bandwidth;
  if (ShouldCarryOver) {
    CarryOver = NumMicroOps - Bandwidth;
    CarriedOver = IR;
    Bandwidth = 0;
    NumIssued += Bandwidth;
  } else {
    NumIssued += NumMicroOps;
    Bandwidth = IS.getEndGroup() ? 0 : Bandwidth - NumMicroOps;
  }

  if (IS.isExecuted() && !ShouldCarryOver) {
    PRF.onInstructionExecuted(&IS);
    LSU.onInstructionExecuted(IR);
    notifyEvent<HWInstructionEvent>(HWInstructionEvent(HWInstructionEvent::Executed, IR));
    retireInstruction(IR);
    return ErrorSuccess();
  }

  IssuedInst.push_back(IR);

  if (!IR.getInstruction()->getRetireOOO())
    LastWriteBackCycle = IS.getCyclesLeft();

  return ErrorSuccess();
}

void A520InOrderIssueStage::updateIssuedInst() {
  unsigned NumExecuted = 0;
  for (auto I = IssuedInst.begin(), E = IssuedInst.end(); I != (E - NumExecuted);) {
    InstRef &IR = *I;
    Instruction &IS = *IR.getInstruction();

    IS.cycleEvent();
    if (!IS.isExecuted()) {
      ++I;
      continue;
    }

    if (!CarriedOver) {
      PRF.onInstructionExecuted(&IS);
      LSU.onInstructionExecuted(IR);
      notifyEvent<HWInstructionEvent>(HWInstructionEvent(HWInstructionEvent::Executed, IR));
      retireInstruction(*I);
    }

    ++NumExecuted;
    std::iter_swap(I, E - NumExecuted);
  }

  if (NumExecuted)
    IssuedInst.resize(IssuedInst.size() - NumExecuted);
}

void A520InOrderIssueStage::updateCarriedOver() {
  if (!CarriedOver)
    return;

  if (CarryOver > Bandwidth) {
    CarryOver -= Bandwidth;
    Bandwidth = 0;
    return;
  }

  if (CarriedOver.getInstruction()->getEndGroup())
    Bandwidth = 0;
  else
    Bandwidth -= CarryOver;

  if (CarriedOver.getInstruction()->isExecuted()) {
    PRF.onInstructionExecuted(CarriedOver.getInstruction());
    LSU.onInstructionExecuted(CarriedOver);
    notifyEvent<HWInstructionEvent>(HWInstructionEvent(HWInstructionEvent::Executed, CarriedOver));
    retireInstruction(CarriedOver);
  }

  CarriedOver = InstRef();
  CarryOver = 0;
}

void A520InOrderIssueStage::retireInstruction(InstRef &IR) {
  Instruction &IS = *IR.getInstruction();
  IS.retire();

  SmallVector<unsigned, 4> FreedRegs(PRF.getNumRegisterFiles());
  for (const WriteState &WS : IS.getDefs())
    PRF.removeRegisterWrite(WS, FreedRegs);

  if (IS.isMemOp())
    LSU.onInstructionRetired(IR);

  notifyEvent<HWInstructionEvent>(HWInstructionRetiredEvent(IR, FreedRegs));
}

void A520InOrderIssueStage::notifyStallEvent() {
  const InstRef &IR = SI.getInstruction();

  switch (SI.getStallKind()) {
  default:
    break;
  case StallInfo::StallKind::REGISTER_DEPS:
    notifyEvent<HWStallEvent>(HWStallEvent(HWStallEvent::RegisterFileStall, IR));
    notifyEvent<HWPressureEvent>(HWPressureEvent(HWPressureEvent::REGISTER_DEPS, IR));
    break;
  case StallInfo::StallKind::DISPATCH:
    notifyEvent<HWStallEvent>(HWStallEvent(HWStallEvent::DispatchGroupStall, IR));
    notifyEvent<HWPressureEvent>(HWPressureEvent(HWPressureEvent::RESOURCES, IR));
    break;
  case StallInfo::StallKind::CUSTOM_STALL:
    notifyEvent<HWStallEvent>(HWStallEvent(HWStallEvent::CustomBehaviourStall, IR));
    break;
  }
}

Error A520InOrderIssueStage::cycleStart() {
  NumIssued = 0;
  Bandwidth = getIssueWidth();

  PRF.cycleStart();
  LSU.cycleEvent();

  SmallVector<ResourceRef, 4> Freed;
  RM.cycleEvent(Freed);

  updateIssuedInst();

  updateCarriedOver();

  if (SI.isValid()) {
    if (!SI.getCyclesLeft()) {
      InstRef IR = SI.getInstruction();
      SI.clear();

      if (Error E = tryIssue(IR))
        return E;
    }

    if (SI.getCyclesLeft()) {
      notifyStallEvent();
      Bandwidth = 0;
      return ErrorSuccess();
    }
  }

  return ErrorSuccess();
}

Error A520InOrderIssueStage::cycleEnd() {
  PRF.cycleEnd();
  SI.cycleEnd();

  if (LastWriteBackCycle > 0)
    --LastWriteBackCycle;

  return ErrorSuccess();
}

std::unique_ptr<Pipeline> createA520InOrderPipeline(SourceMgr &SrcMgr, CustomBehaviour &CB,
                                                    const MCSubtargetInfo &STI,
                                                    RegisterFile &PRF, LSUnitBase &LSU,
                                                    A520InOrderIssueStage **IssueStageOut,
                                                    bool EnforceWriteBackOrder,
                                                    const MCInstrInfo *MCII,
                                                    ArrayRef<SourceMgr::UniqueInst> SimInstrs) {
  auto Entry = std::make_unique<EntryStage>(SrcMgr);
  auto Issue = std::make_unique<A520InOrderIssueStage>(STI, PRF, CB, LSU, EnforceWriteBackOrder,
                                                       MCII, SimInstrs);
  if (IssueStageOut)
    *IssueStageOut = Issue.get();
  auto P = std::make_unique<Pipeline>();
  P->appendStage(std::move(Entry));
  P->appendStage(std::move(Issue));
  return P;
}

} // namespace mca
} // namespace llvm
