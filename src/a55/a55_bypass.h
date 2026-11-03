#ifndef A55_A55_BYPASS_H
#define A55_A55_BYPASS_H

// Cortex-A55 register bypass networks, evaluated against the register file's
// current producer mapping.  NOTE: the flag bypass (cmp -> csel/b.cc) and the
// pointer forwarding (adrp -> ldr base) are applied inside
// A55DecoupledIssueStage::checkRegisterHazard() (a55_issue_stage.cpp), before
// the hazard is computed.  Applying them on the Dispatched event would be too
// late: that event fires only after the instruction has already cleared
// canExecute()/checkRegisterHazard() for the attempt.

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

// Arithmetic with carry: ADC/ADCS/SBC/SBCS, W and X (NGC/NGCS are SBC/SBCS
// with Rn = ZR and share these opcodes).
inline bool a55IsArithmeticWithCarry(unsigned Opcode) {
  switch (Opcode) {
  case llvm::AArch64::ADCWr:  case llvm::AArch64::ADCXr:
  case llvm::AArch64::ADCSWr: case llvm::AArch64::ADCSXr:
  case llvm::AArch64::SBCWr:  case llvm::AArch64::SBCXr:
  case llvm::AArch64::SBCSWr: case llvm::AArch64::SBCSXr:
    return true;
  default:
    return false;
  }
}

// [a55_flag_bypass] docs/a55-sog.md C-4, D-8, U-2.
// Cortex-A55's zero-latency condition-flag bypass network: cmp -> csel/b.cc
// same-cycle forwarding.  SOG v3 sec. 3.2 Table 2 ("Conditional"): a
// conditional (flag-dependent) instruction-1 can be dual issued with a flag
// setting instruction-0 except
//   - instructions that execute an RRX operation      (AArch32 only),
//   - arithmetic with carry instructions               -> consumer ADC/ADCS/SBC/SBCS (NGC/NGCS),
//   - instruction-0 is MULS/MLAS                       (AArch32 only),
//   - instruction-0 is a NEON instruction             (no A64 NEON instruction sets NZCV).
// FP flag producers are not on the network either: SOG sec. 5.11, the flag
// transfer from the floating-point flags to the integer flags takes a cycle
// (VMRS/VMSR in AArch32; FCMP{E} and FCCMP{E}, the A64 FP compares that write
// NZCV, SOG 4.11).  When this returns false the consumer sees the producer's
// NZCV write at its latency through the ordinary RAW hazard.
// `ConsumerOpcode` is the opcode of the instruction reading NZCV.
inline bool a55IsFlagBypassEligible(const llvm::mca::RegisterFile &PRF,
                                    const llvm::MCInstrInfo &MCII,
                                    llvm::ArrayRef<llvm::mca::SourceMgr::UniqueInst> SimInstrs,
                                    unsigned ConsumerOpcode) {
  if (a55IsArithmeticWithCarry(ConsumerOpcode))
    return false;
  llvm::StringRef DepName = a55GetProducerOpcodeName(PRF, llvm::AArch64::NZCV, MCII, SimInstrs);
  if (DepName.empty())
    return false;
  return !(DepName.starts_with_insensitive("FCMP") ||
           DepName.starts_with_insensitive("FCCMP") ||
           DepName.starts_with_insensitive("VMRS") ||
           DepName.starts_with_insensitive("VMSR"));
}

// A64 low-latency pointer forwarding (SOG sec. 5.10, p.44): "adrp x0, <const>"
// followed by a load from "[x0, #lo12 <const>]" executes without a dependency
// stall; the SOG calls the sequence common "to generate load-store addresses".
// Applied to a load or a store with an unsigned immediate offset (LDR*ui and
// STR*ui: the GPR and FP/SIMD forms of every size), and only to its base
// register.  Not applied to LDUR/STUR, pre/post-index and register-offset forms,
// nor to the index register or to the data of a store (docs/a55-sog.md C-5).
// Whether the offset is the low 12 bits of the same symbol as the ADRP cannot be
// known from the instructions and is not checked.
inline bool a55IsAdrpForwardConsumer(llvm::StringRef ConsumerName, int BaseOperandIndex,
                                     int ReadOperandIndex) {
  return (ConsumerName.starts_with("LDR") || ConsumerName.starts_with("STR")) &&
         ConsumerName.ends_with("ui") && ReadOperandIndex == BaseOperandIndex;
}

inline bool a55IsPointerForwardEligible(const llvm::mca::RegisterFile &PRF,
                                        llvm::MCPhysReg BaseRegID,
                                        const llvm::MCInstrInfo &MCII,
                                        llvm::ArrayRef<llvm::mca::SourceMgr::UniqueInst> SimInstrs) {
  llvm::StringRef DepName = a55GetProducerOpcodeName(PRF, BaseRegID, MCII, SimInstrs);
  return DepName.equals_insensitive("ADRP");
}

#endif // A55_A55_BYPASS_H
