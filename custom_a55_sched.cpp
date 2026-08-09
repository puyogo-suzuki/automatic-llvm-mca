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

class CustomSubtargetInfo : public MCSubtargetInfo {
    std::unique_ptr<MCSubtargetInfo> BaseSTI;
public:
    CustomSubtargetInfo(std::unique_ptr<MCSubtargetInfo> Base)
        : MCSubtargetInfo(*Base), BaseSTI(std::move(Base)) {}

    unsigned resolveVariantSchedClass(unsigned SchedClass, const MCInst *MCI, const MCInstrInfo *MCII, unsigned CPUID) const override {
        unsigned res = AArch64_MC::resolveVariantSchedClassImpl_custom(SchedClass, MCI, MCII, *this, CPUID);
        if (res != 0) return res;
        // Fallback 1: try generic CPUID (0)
        res = AArch64_MC::resolveVariantSchedClassImpl_custom(SchedClass, MCI, MCII, *this, 0);
        if (res != 0) return res;
        // Fallback 2: base SchedClass (never return 0 for write variants)
        return SchedClass;
    }
};

std::unique_ptr<MCSubtargetInfo> wrapCustomSubtargetInfo(std::unique_ptr<MCSubtargetInfo> STI, StringRef CPUName) {
    if (!STI) return STI;
    return std::make_unique<CustomSubtargetInfo>(std::move(STI));
}

void overrideCortexA55SchedModel(llvm::MCSubtargetInfo &STI, llvm::StringRef CPUName) {
    if (CPUName == "cortex-a55") {
        STI.*get(MCSubtargetInfo_CPUSchedModel()) = &CortexA55Model;
        STI.*get(MCSubtargetInfo_WriteProcResTable()) = AArch64WriteProcResTable;
        STI.*get(MCSubtargetInfo_WriteLatencyTable()) = AArch64WriteLatencyTable;
        STI.*get(MCSubtargetInfo_ReadAdvanceTable()) = AArch64ReadAdvanceTable;
    } else if (CPUName == "cortex-a520" || CPUName == "cortex-a520ae") {
        STI.*get(MCSubtargetInfo_CPUSchedModel()) = &CortexA520Model;
        STI.*get(MCSubtargetInfo_WriteProcResTable()) = AArch64WriteProcResTable;
        STI.*get(MCSubtargetInfo_WriteLatencyTable()) = AArch64WriteLatencyTable;
        STI.*get(MCSubtargetInfo_ReadAdvanceTable()) = AArch64ReadAdvanceTable;
    } else if (CPUName == "cortex-a76" || CPUName == "cortex-a76ae") {
        STI.*get(MCSubtargetInfo_CPUSchedModel()) = &NeoverseN1Model;
        STI.*get(MCSubtargetInfo_WriteProcResTable()) = AArch64WriteProcResTable;
        STI.*get(MCSubtargetInfo_WriteLatencyTable()) = AArch64WriteLatencyTable;
        STI.*get(MCSubtargetInfo_ReadAdvanceTable()) = AArch64ReadAdvanceTable;
    } else if (CPUName == "cortex-a78" || CPUName == "cortex-a78ae" || CPUName == "cortex-a78c") {
        STI.*get(MCSubtargetInfo_CPUSchedModel()) = &NeoverseN2Model;
        STI.*get(MCSubtargetInfo_WriteProcResTable()) = AArch64WriteProcResTable;
        STI.*get(MCSubtargetInfo_WriteLatencyTable()) = AArch64WriteLatencyTable;
        STI.*get(MCSubtargetInfo_ReadAdvanceTable()) = AArch64ReadAdvanceTable;
    } else if (CPUName == "cortex-a720" || CPUName == "cortex-a720ae") {
        STI.*get(MCSubtargetInfo_CPUSchedModel()) = &CortexA720Model;
        STI.*get(MCSubtargetInfo_WriteProcResTable()) = AArch64WriteProcResTable;
        STI.*get(MCSubtargetInfo_WriteLatencyTable()) = AArch64WriteLatencyTable;
        STI.*get(MCSubtargetInfo_ReadAdvanceTable()) = AArch64ReadAdvanceTable;
    } else if (CPUName == "cortex-x1" || CPUName == "cortex-x1c") {
        STI.*get(MCSubtargetInfo_CPUSchedModel()) = &NeoverseV1Model;
        STI.*get(MCSubtargetInfo_WriteProcResTable()) = AArch64WriteProcResTable;
        STI.*get(MCSubtargetInfo_WriteLatencyTable()) = AArch64WriteLatencyTable;
        STI.*get(MCSubtargetInfo_ReadAdvanceTable()) = AArch64ReadAdvanceTable;
    } else if (CPUName == "icestorm") {
        STI.*get(MCSubtargetInfo_CPUSchedModel()) = &IcestormModel;
        STI.*get(MCSubtargetInfo_WriteProcResTable()) = AArch64WriteProcResTable;
        STI.*get(MCSubtargetInfo_WriteLatencyTable()) = AArch64WriteLatencyTable;
        STI.*get(MCSubtargetInfo_ReadAdvanceTable()) = AArch64ReadAdvanceTable;
    } else if (CPUName == "firestorm") {
        STI.*get(MCSubtargetInfo_CPUSchedModel()) = &FirestormModel;
        STI.*get(MCSubtargetInfo_WriteProcResTable()) = AArch64WriteProcResTable;
        STI.*get(MCSubtargetInfo_WriteLatencyTable()) = AArch64WriteLatencyTable;
        STI.*get(MCSubtargetInfo_ReadAdvanceTable()) = AArch64ReadAdvanceTable;
    }
}
}
