#include "a55/a55_issue_stage.h"
#include "a55/a55_bypass.h"
#include "mca/in_order_issue_helpers.h"
#include "llvm/MCA/Stages/EntryStage.h"
#include "llvm/MCA/HWEventListener.h"
#include <algorithm>

namespace llvm {
namespace mca {

using in_order_issue::addRegisterReadWrite;
using in_order_issue::findFirstWriteBackCycle;
using in_order_issue::hasResourceHazard;

// Applies Cortex-A55's same-cycle flag bypass (cmp -> csel/b.cc) and
// pointer-forwarding (adrp -> ldr base) bypasses *before* consulting the
// register file's RAW hazard, so that they suppress the stall they are meant
// to suppress (a Dispatched-event listener would run only after canExecute()
// had already gated the instruction on this hazard).
unsigned A55DecoupledIssueStage::checkRegisterHazard(const InstRef &IR) const {
  StringRef CurrName = MCII.getName(IR.getInstruction()->getOpcode());
  for (const ReadState &RS : IR.getInstruction()->getUses()) {
    MCPhysReg RegID = RS.getRegisterID();
    if (RegID == AArch64::NZCV &&
        a55IsFlagBypassEligible(PRF, MCII, SimInstrs, IR.getInstruction()->getOpcode()))
      continue;
    // The base register Rn of LDR*ui and STR*ui is operand 1 (Rt, Rn, imm).
    const ReadDescriptor &RD = RS.getDescriptor();
    if (!RD.isImplicitRead() &&
        a55IsAdrpForwardConsumer(CurrName, /*BaseOperandIndex=*/1, RD.OpIndex) &&
        a55IsPointerForwardEligible(PRF, RegID, MCII, SimInstrs))
      continue;
    RegisterFile::RAWHazard Hazard = PRF.checkRAWHazards(STI, RS);
    if (Hazard.isValid()) {
      if (isSameCycleALUForward(IR, RS))
        continue;
      return Hazard.hasUnknownCycles() ? 1U : Hazard.CyclesLeft;
    }
  }
  return 0;
}

// True if every write of the scheduling class goes to the integer ALU pipes
// (CortexA55UnitALU, i.e. ALU0/ALU1 of SOG fig. 1) and nothing else.
// CortexA55UnitVarShift is not a separate pipe: it only caps the variable
// shifts (ASRV/LSLV/LSRV/RORV, SOG 4.4 throughput 1) at one per cycle inside
// the ALU pipes [a55_varshift], so it counts as part of them here.
bool A55DecoupledIssueStage::isALUPipeOnly(unsigned SchedClassID) const {
  if (!ALUProcResIdx)
    return false;
  const MCSchedModel &SM = STI.getSchedModel();
  const MCSchedClassDesc *SC = SM.getSchedClassDesc(SchedClassID);
  if (!SC || !SC->isValid() || SC->isVariant())
    return false;
  bool UsesALU = false;
  for (const MCWriteProcResEntry *E = STI.getWriteProcResBegin(SC),
                                 *End = STI.getWriteProcResEnd(SC);
       E != End; ++E) {
    if (E->ReleaseAtCycle == 0)
      continue;
    if (VarShiftProcResIdx && E->ProcResourceIdx == VarShiftProcResIdx)
      continue;
    if (E->ProcResourceIdx != ALUProcResIdx)
      return false;
    UsesALU = true;
  }
  return UsesALU;
}

// A producer for the ALU0 -> ALU1 path: an instruction executing in the ALU
// pipes.  Its result must also be available at the end of ex1, which is
// checked per write in isSameCycleALUForward() (write latency 1: SOG 4.3 "ALU,
// basic" / "Conditional compare" / "Conditional select", 4.4 "Address
// generation" / "Move immed" / "Variable shift" [a55_varshift], 4.7 "Sign/zero
// extend" / "Count leading" / "Reverse bytes", the LSL/LSR/ASR-immediate
// aliases of UBFM/SBFM [a55_shift_imm] and the ROR-immediate alias of EXTR
// [a55_ror_imm]).  CRC32W/CW are 1-cycle in the .td but SOG 4.21
// does not say they run in the ALU pipes, so they are kept out.
bool A55DecoupledIssueStage::isALUForwardProducer(const Instruction &IS) const {
  unsigned Opc = IS.getOpcode();
  if (Opc == AArch64::CRC32Wrr || Opc == AArch64::CRC32CWrr)
    return false;
  return isALUPipeOnly(IS.getDesc().SchedClassID);
}

// SOG v3 sec. 3.1.1: "There is also a dedicated forwarding path from the ex1
// stage of the ALU0 to the ex1 stage of the ALU1".  Both stages being ex1, the
// path can only connect the two instructions of one dual-issue pair: the older
// instruction-0 (in ALU0) and the younger instruction-1 (in ALU1) issued in the
// same cycle.  (The SOG does not state the slot -> pipe mapping explicitly;
// instruction-0 -> ALU0 is the only one under which an ALU0 -> ALU1 path
// within one stage can serve an in-order core.)  Across cycles an ALU result
// is forwarded normally ("from the end of ex1 and ex2 to earlier stages of
// both ALU pipelines") and seen at its exec latency, 1 for ALU basic: that is
// the RegisterFile hazard itself, as the .td has no ReadAdvance from
// the ALU writes [a55_alu_fwd], so nothing is subtracted twice.
//
// Returns true if the hazard of `RS` (a read of instruction-1 candidate `IR`)
// comes only from this cycle's instruction-0, that instruction is a 1-cycle
// ALU producer, `IR` itself executes in the ALU pipes and `RS` is an operand
// the ALU (not the shifter) consumes.  The latter is the operand class the
// .td gives the load-use advance (ReadI / CortexA55ReadNotShifted vs
// CortexA55ReadShifted for shifted/extended Rm, UBFM/SBFM/BFM Rn, EXTR,
// variable shifts), queried through WriteLD's ReadAdvance.
bool A55DecoupledIssueStage::isSameCycleALUForward(const InstRef &IR,
                                                   const ReadState &RS) const {
  if (NumIssued != 1 || !Slot0IsALUForwardProducer || !LoadWriteResID)
    return false;
  const Instruction &IS = *IR.getInstruction();
  if (!isALUPipeOnly(IS.getDesc().SchedClassID))
    return false;

  const ReadDescriptor &RD = RS.getDescriptor();
  const MCSchedClassDesc *SC = STI.getSchedModel().getSchedClassDesc(RD.SchedClassID);
  if (!SC || STI.getReadAdvanceCycles(SC, RD.UseIndex, LoadWriteResID) < 1)
    return false;

  SmallVector<WriteRef, 4> Writes;
  SmallVector<WriteRef, 4> CommittedWrites;
  PRF.collectWrites(STI, RS, Writes, CommittedWrites);
  bool FromSlot0 = false;
  for (const WriteRef &WR : Writes) {
    const WriteState *WS = WR.getWriteState();
    if (!WS)
      continue;
    int CyclesLeft = WS->getCyclesLeft();
    int Advance = STI.getReadAdvanceCycles(SC, RD.UseIndex, WS->getWriteResourceID());
    if (CyclesLeft != UNKNOWN_CYCLES && CyclesLeft - Advance <= 0)
      continue;  // this write is not the hazard
    if (WR.getSourceIndex() != Slot0SourceIndex)
      return false;
    if (CyclesLeft == UNKNOWN_CYCLES || WS->getLatency() != 1 || CyclesLeft > 1)
      return false;
    FromSlot0 = true;
  }
  return FromSlot0;
}

A55DecoupledIssueStage::A55DecoupledIssueStage(const MCSubtargetInfo &STI,
                                               const MCRegisterInfo &MRI,
                                               RegisterFile &PRF,
                                               CustomBehaviour &CB,
                                               const MCInstrInfo &MCII,
                                               ArrayRef<SourceMgr::UniqueInst> SimInstrs,
                                               std::unique_ptr<LSUnitBase> LSUB)
    : STI(STI), MRI(MRI), PRF(PRF), RM(STI.getSchedModel()), CB(CB), MCII(MCII),
      SimInstrs(SimInstrs), LSU_Owner(std::move(LSUB)), LSU(*LSU_Owner) {
  // Model data used by isSameCycleALUForward() (SOG 3.1.1).
  const MCSchedModel &SM = STI.getSchedModel();
  for (unsigned I = 1, E = SM.getNumProcResourceKinds(); I < E; ++I) {
    StringRef Name(SM.getProcResource(I)->Name);
    if (Name == "CortexA55UnitALU")
      ALUProcResIdx = I;
    else if (Name == "CortexA55UnitVarShift")
      VarShiftProcResIdx = I;
  }
  const MCSchedClassDesc *LdSC = SM.getSchedClassDesc(MCII.get(AArch64::LDRXui).getSchedClass());
  if (LdSC && LdSC->isValid() && !LdSC->isVariant() && LdSC->NumWriteLatencyEntries)
    LoadWriteResID = STI.getWriteLatencyEntry(LdSC, 0)->WriteResourceID;
}

// Integer divides (SOG v3 sec. 3.4).
static bool isIntegerDivide(unsigned Opcode) {
  return Opcode == AArch64::SDIVWr || Opcode == AArch64::SDIVXr ||
         Opcode == AArch64::UDIVWr || Opcode == AArch64::UDIVXr;
}

bool A55DecoupledIssueStage::canExecute(const InstRef &IR, StallInfo &SI) {
  const Instruction &IS = *IR.getInstruction();

  // Dual-issue column of the SOG instruction tables (sec. 4.1: "00 not
  // dual-issuable, 01 dual-issuable from slot 0, 10 dual-issuable from slot 1,
  // 11 dual-issuable from both slots"; sec. 3.2 Tables 1/2).  The .td encodes
  // 01 as BeginGroup (the instruction may only be instruction-0 of a pair),
  // 10 as EndGroup (it may only be instruction-1, i.e. nothing younger issues
  // with it) and 00 as SingleIssue (= BeginGroup + EndGroup).  InstrBuilder
  // copies both bits from the MCSchedClassDesc onto the mca::Instruction.
  if (NumIssued != 0 && (IS.getBeginGroup() || LastIssuedEndsGroup)) {
    SI.update(IR, 1, StallInfo::StallKind::DISPATCH);
    return false;
  }

  // --ignore-loop-carried: forget the previous iteration's producers before
  // the first instruction of a new iteration is checked (#12).
  if (BeforeRegisterHazardCheck)
    BeforeRegisterHazardCheck(IR);

  if (unsigned Cycles = checkRegisterHazard(IR)) {
    SI.update(IR, Cycles, StallInfo::StallKind::REGISTER_DEPS);
    return false;
  }

  if (hasResourceHazard(RM, IR)) {
    SI.update(IR, 1, StallInfo::StallKind::DISPATCH);
    return false;
  }

  if (IS.isMemOp() && !LSU.isReady(IR)) {
    SI.update(IR, 1, StallInfo::StallKind::LOAD_STORE);
    return false;
  }

  if (unsigned CustomStallCycles = CB.checkCustomHazard(IssuedInst, IR)) {
    SI.update(IR, CustomStallCycles, StallInfo::StallKind::CUSTOM_STALL);
    return false;
  }

  // SOG v3 sec. 3.4: "Integer divide instructions are serializing and do not
  // allow younger instructions to retire underneath to ensure that the integer
  // divide results is retired in-order."  Modelled as: an instruction younger
  // than an in-flight SDIV/UDIV may not issue if it would complete (write back)
  // before that divide does.  This is the only write-back ordering rule on
  // A55: llvm::mca::InOrderIssueStage's generic "no younger instruction may
  // write back before an older one" check is not applied, because the SOG
  // pipelines have fixed completion stages (sec. 3.1) and its throughput /
  // dual-issue tables allow e.g. FADD+ADD or LDR+ADD pairs every cycle, which
  // that check would turn into up to three stall cycles per pair.
  if (DivWriteBackCycle) {
    unsigned NextWriteBackCycle = findFirstWriteBackCycle(IR);
    if (NextWriteBackCycle < DivWriteBackCycle) {
      SI.update(IR, DivWriteBackCycle - NextWriteBackCycle,
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

  // The SOG dual-issue limit is in instructions (an instruction-0 / instruction-1
  // pair, sec. 3.2), not micro-ops.  TableGen sums NumMicroOps over the
  // SchedWrites of a class, so e.g. every writeback-form load/store ([WriteAdr,
  // WriteLD]) and every LDP (two data writes) has NumMicroOps = 2 and, counted
  // in micro-ops, would fill the whole issue width on its own: an LDP X
  // ("01") or an STR post-index ("11") could never dual-issue.
  if (NumIssued == 0) {
    Slot0SourceIndex = SourceIndex;
    Slot0IsALUForwardProducer = isALUForwardProducer(IS);
  }
  ++NumIssued;
  if (IS.getEndGroup())
    LastIssuedEndsGroup = true;

  if (IS.isExecuted()) {
    PRF.onInstructionExecuted(&IS);
    LSU.onInstructionExecuted(IR);
    notifyEvent<HWInstructionEvent>(
        HWInstructionEvent(HWInstructionEvent::Executed, IR));
    retireInstruction(IR);
    return ErrorSuccess();
  }

  IssuedInst.push_back(IR);

  if (isIntegerDivide(IS.getOpcode()))
    DivWriteBackCycle = std::max<unsigned>(DivWriteBackCycle, IS.getCyclesLeft());

  return ErrorSuccess();
}

void A55DecoupledIssueStage::drainDecodeWindow() {
  if (DecodeWindow.empty())
    return;

  // Cortex-A55 SOG v3 sec. 3.2 "Dual-issue": the core issues AT MOST an
  // (instruction-0, instruction-1) PAIR per cycle -- the whole section is
  // framed as the conditions under which a single older/younger pair may go
  // together (Table 1 "Instruction-0 dual issue conditions", Table 2
  // "Instruction-1 dual issue conditions"); there is no instruction-2.
  // AArch64SchedA55.td encodes the same fact as `let IssueWidth = 2`.
  //
  // drainDecodeWindow() is called more than once per cycle: once from
  // cycleStart(), and then AGAIN from execute() for every instruction
  // EntryStage hands over, because isAvailable() re-opens
  // (DecodeWindow.size() < 2) as soon as a drain empties the window.
  // NumIssued is only reset in cycleStart(), so it accumulates across those
  // calls and caps the per-cycle issue width at 2 here (and the group rules in
  // canExecute() see whether an instruction would be instruction-0 or -1).
  if (NumIssued >= 2)
    return;

  // Strictly in-order issue (SOG v3 sec. 5.1: "the Cortex-A55 core does not
  // support any instruction reordering at issue"; sec. 3.5.1: "the Cortex-A55
  // core only issues instructions in-order").  The younger instruction of the
  // window may only issue in the same cycle as, and after, the older one: if
  // the oldest instruction stalls, nothing behind it issues.
  unsigned NumIssuedFromWindow = 0;
  if (canExecute(DecodeWindow[0], Stalls[0])) {
    (void)issue(DecodeWindow[0]);
    NumIssuedFromWindow = 1;
    if (DecodeWindow.size() >= 2 && NumIssued < 2 &&
        canExecute(DecodeWindow[1], Stalls[1])) {
      (void)issue(DecodeWindow[1]);
      NumIssuedFromWindow = 2;
    }
  }

  DecodeWindow.erase(DecodeWindow.begin(),
                     DecodeWindow.begin() + NumIssuedFromWindow);
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

bool A55DecoupledIssueStage::isAvailable(const InstRef & /*IR*/) const {
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
  LastIssuedEndsGroup = false;
  Slot0IsALUForwardProducer = false;

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

  if (DivWriteBackCycle > 0)
    --DivWriteBackCycle;

  return ErrorSuccess();
}

std::unique_ptr<Pipeline> createA55DecoupledPipeline(const PipelineOptions &Opts,
                                                     SourceMgr &SrcMgr,
                                                     CustomBehaviour &CB,
                                                     const MCSubtargetInfo &STI,
                                                     const MCRegisterInfo &MRI,
                                                     RegisterFile &PRF,
                                                     const MCInstrInfo &MCII,
                                                     ArrayRef<SourceMgr::UniqueInst> SimInstrs,
                                                     A55DecoupledIssueStage **IssueStageOut) {
  auto P = std::make_unique<Pipeline>();
  auto Entry = std::make_unique<EntryStage>(SrcMgr);
  auto LSU = std::make_unique<LSUnit>(
      STI.getSchedModel(), Opts.LoadQueueSize, Opts.StoreQueueSize, Opts.AssumeNoAlias);
  auto Issue = std::make_unique<A55DecoupledIssueStage>(
      STI, MRI, PRF, CB, MCII, SimInstrs, std::move(LSU));
  if (IssueStageOut)
    *IssueStageOut = Issue.get();

  Entry->setNextInSequence(Issue.get());

  P->appendStage(std::move(Entry));
  P->appendStage(std::move(Issue));
  return P;
}

} // namespace mca
} // namespace llvm
