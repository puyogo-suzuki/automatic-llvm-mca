#include "custom_a55_sched.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm-source/llvm/lib/Target/AArch64/MCTargetDesc/AArch64AddressingModes.h"
#include "llvm-source/llvm/lib/Target/AArch64/MCTargetDesc/AArch64MCTargetDesc.h"

// Define the private member access rob structures
template <typename Tag, typename Tag::type M>
struct Rob {
  friend typename Tag::type get(Tag) { return M; }
};

struct MCSubtargetInfo_CPUSchedModel {
  typedef const llvm::MCSchedModel *llvm::MCSubtargetInfo::*type;
  friend type get(MCSubtargetInfo_CPUSchedModel);
};
template struct Rob<MCSubtargetInfo_CPUSchedModel, &llvm::MCSubtargetInfo::CPUSchedModel>;

struct MCSubtargetInfo_WriteProcResTable {
  typedef const llvm::MCWriteProcResEntry *llvm::MCSubtargetInfo::*type;
  friend type get(MCSubtargetInfo_WriteProcResTable);
};
template struct Rob<MCSubtargetInfo_WriteProcResTable, &llvm::MCSubtargetInfo::WriteProcResTable>;

struct MCSubtargetInfo_WriteLatencyTable {
  typedef const llvm::MCWriteLatencyEntry *llvm::MCSubtargetInfo::*type;
  friend type get(MCSubtargetInfo_WriteLatencyTable);
};
template struct Rob<MCSubtargetInfo_WriteLatencyTable, &llvm::MCSubtargetInfo::WriteLatencyTable>;

struct MCSubtargetInfo_ReadAdvanceTable {
  typedef const llvm::MCReadAdvanceEntry *llvm::MCSubtargetInfo::*type;
  friend type get(MCSubtargetInfo_ReadAdvanceTable);
};
template struct Rob<MCSubtargetInfo_ReadAdvanceTable, &llvm::MCSubtargetInfo::ReadAdvanceTable>;

// Use a macro hack to force internal linkage for generated tables to avoid linker collisions
#define extern static
#define resolveVariantSchedClassImpl resolveVariantSchedClassImpl_custom
#define GET_SUBTARGETINFO_MC_DESC
#include "AArch64GenSubtargetInfo.inc"
#undef GET_SUBTARGETINFO_MC_DESC
#undef resolveVariantSchedClassImpl
#undef extern

namespace llvm {
void overrideCortexA55SchedModel(llvm::MCSubtargetInfo &STI) {
    if (STI.getCPU() == "cortex-a55") {
        STI.*get(MCSubtargetInfo_CPUSchedModel()) = &CortexA55Model;
        STI.*get(MCSubtargetInfo_WriteProcResTable()) = AArch64WriteProcResTable;
        STI.*get(MCSubtargetInfo_WriteLatencyTable()) = AArch64WriteLatencyTable;
        STI.*get(MCSubtargetInfo_ReadAdvanceTable()) = AArch64ReadAdvanceTable;
    } else if (STI.getCPU() == "cortex-a76" || STI.getCPU() == "cortex-a76ae") {
        STI.*get(MCSubtargetInfo_CPUSchedModel()) = &NeoverseN1Model;
        STI.*get(MCSubtargetInfo_WriteProcResTable()) = AArch64WriteProcResTable;
        STI.*get(MCSubtargetInfo_WriteLatencyTable()) = AArch64WriteLatencyTable;
        STI.*get(MCSubtargetInfo_ReadAdvanceTable()) = AArch64ReadAdvanceTable;
    }
}
}
