#include "custom_a55_sched.h"
#include "cpu_traits.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include <iterator>
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

// Locally re-generated instruction descriptors: provides AArch64Descs (the
// MCInstrDesc / implicit-operand / operand-info table) plus the matching
// InitAArch64MCInstrInfo().  See remapSchedClassIndices().
#define extern static
#define GET_INSTRINFO_MC_DESC
#include "AArch64GenInstrInfo.inc"
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

// True for exactly the CPU names for which overrideCortexA55SchedModel()
// installs a LOCALLY re-generated MCSchedModel below.  Only those CPUs need
// the SchedClass index remap; leaving any other CPU (e.g. "generic") alone is
// mandatory, because such a CPU keeps libLLVM's own stock MCSchedModel, whose
// SchedClassTable is indexed with the stock numbering.
bool usesLocalSchedTables(llvm::StringRef CPUName) {
    return getCpuTraits(CPUName).Model != CpuModel::None;
}

// Re-point MCInstrInfo at the LOCALLY re-generated AArch64 instruction descriptor
// table, so that MCInstrDesc::SchedClass carries OUR TableGen numbering instead of
// stock libLLVM's (which differs for ~11% of opcodes once ModifiedTarget/*.td add
// InstRW rules).  This is a correctness fix, not an accuracy claim.  The full
// rationale, the measured extent of the mismatch and the safety checks:
// docs.md section 5.A.
void remapSchedClassIndices(llvm::MCInstrInfo &MCII, llvm::StringRef CPUName) {
    if (!usesLocalSchedTables(CPUName))
        return;
    // If the two tables ever stop describing the same instruction set, leaving
    // the stock descriptors in place is far safer than swapping in ours.
    if (MCII.getNumOpcodes() != std::size(AArch64Descs.Insts))
        return;
    InitAArch64MCInstrInfo(&MCII);
}

void overrideCortexA55SchedModel(llvm::MCSubtargetInfo &STI, llvm::StringRef CPUName) {
    const llvm::MCSchedModel *Model = nullptr;
    switch (getCpuTraits(CPUName).Model) {
    case CpuModel::A55:       Model = &CortexA55Model;   break;
    case CpuModel::A520:      Model = &CortexA520Model;  break;
    case CpuModel::A76:       Model = &NeoverseN1Model;  break;
    case CpuModel::A78:       Model = &NeoverseN2Model;  break;
    case CpuModel::A720:      Model = &CortexA720Model;  break;
    case CpuModel::X1:        Model = &NeoverseV1Model;  break;
    case CpuModel::Icestorm:  Model = &IcestormModel;    break;
    case CpuModel::Firestorm: Model = &FirestormModel;   break;
    case CpuModel::None:      return;
    }
    STI.*get(MCSubtargetInfo_CPUSchedModel()) = Model;
    STI.*get(MCSubtargetInfo_WriteProcResTable()) = AArch64WriteProcResTable;
    STI.*get(MCSubtargetInfo_WriteLatencyTable()) = AArch64WriteLatencyTable;
    STI.*get(MCSubtargetInfo_ReadAdvanceTable()) = AArch64ReadAdvanceTable;
}
}
