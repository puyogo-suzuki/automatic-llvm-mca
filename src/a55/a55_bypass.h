#ifndef A55_A55_BYPASS_H
#define A55_A55_BYPASS_H

// Cortex-A55 register bypass networks, evaluated against the register file's
// current producer mapping.  NOTE: the flag bypass (cmp -> csel/b.cc) and the
// pointer forwarding (adrp -> ldr base) are applied inside
// A55DecoupledIssueStage::checkRegisterHazard() (a55_issue_stage.cpp), before
// the hazard is computed.  Applying them on the Dispatched event, as an earlier
// version did, was too late: that event fires only after the instruction has
// already cleared canExecute()/checkRegisterHazard() for the attempt.

#include "mca/private_access.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MCA/SourceMgr.h"
#include "llvm-source/llvm/lib/Target/AArch64/MCTargetDesc/AArch64MCTargetDesc.h"

// Looks up the instruction currently mapped as the producer of `RegID` in
// `PRF`, and returns its opcode name, or an empty StringRef if there is no
// valid mapping. `SimInstrs` is the static (per-iteration) instruction
// sequence, indexed by `WriteRef::getSourceIndex() % SimInstrs.size()`.
inline llvm::StringRef a55GetProducerOpcodeName(const llvm::mca::RegisterFile &PRF,
                                                llvm::MCPhysReg RegID,
                                                const llvm::MCInstrInfo &MCII,
                                                llvm::ArrayRef<llvm::mca::SourceMgr::UniqueInst> SimInstrs) {
  auto member_ptr = get_mappings(RegisterFile_RegisterMappings_Tag{});
  const auto &mappings = PRF.*member_ptr;
  if (RegID >= mappings.size())
    return {};
  const llvm::mca::WriteRef &WR = mappings[RegID].first;
  if (!WR.isValid() || SimInstrs.empty())
    return {};
  unsigned staticIID = WR.getSourceIndex() % SimInstrs.size();
  const llvm::mca::Instruction *DepInst = SimInstrs[staticIID].get();
  return MCII.getName(DepInst->getOpcode());
}

// Cortex-A55's zero-latency condition-flag bypass network: cmp -> csel/b.cc
// same-cycle forwarding. Does not cover FP flag producers (FCMP/VMRS/VMSR).
inline bool a55IsFlagBypassEligible(const llvm::mca::RegisterFile &PRF,
                                    const llvm::MCInstrInfo &MCII,
                                    llvm::ArrayRef<llvm::mca::SourceMgr::UniqueInst> SimInstrs) {
  llvm::StringRef DepName = a55GetProducerOpcodeName(PRF, llvm::AArch64::NZCV, MCII, SimInstrs);
  if (DepName.empty())
    return false;
  return !(DepName.starts_with_insensitive("FCMP") ||
           DepName.starts_with_insensitive("VMRS") ||
           DepName.starts_with_insensitive("VMSR"));
}

// A64 low-latency pointer forwarding: an LDR/LDUR base register produced by
// ADRP is available without waiting for ADRP's full write-back latency.
inline bool a55IsPointerForwardEligible(const llvm::mca::RegisterFile &PRF,
                                        llvm::MCPhysReg BaseRegID,
                                        const llvm::MCInstrInfo &MCII,
                                        llvm::ArrayRef<llvm::mca::SourceMgr::UniqueInst> SimInstrs) {
  llvm::StringRef DepName = a55GetProducerOpcodeName(PRF, BaseRegID, MCII, SimInstrs);
  return DepName.equals_insensitive("ADRP");
}

#endif // A55_A55_BYPASS_H
