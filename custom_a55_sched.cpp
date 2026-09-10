#include "custom_a55_sched.h"
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
    return CPUName == "cortex-a55" ||
           CPUName == "cortex-a520" || CPUName == "cortex-a520ae" ||
           CPUName == "cortex-a76" || CPUName == "cortex-a76ae" ||
           CPUName == "cortex-a78" || CPUName == "cortex-a78ae" || CPUName == "cortex-a78c" ||
           CPUName == "cortex-a720" || CPUName == "cortex-a720ae" ||
           CPUName == "cortex-x1" || CPUName == "cortex-x1c" ||
           CPUName == "icestorm" || CPUName == "firestorm";
}

// Re-point MCInstrInfo at the LOCALLY re-generated AArch64 instruction
// descriptor table, so that MCInstrDesc::SchedClass carries OUR TableGen
// numbering instead of stock libLLVM's.
//
// WHY THIS IS NEEDED (2026-09-10).  The tool mixes two independently generated
// halves of the same AArch64 target description:
//
//   (a) frontend.cpp calls TheTarget->createMCInstrInfo(), i.e. the MCInstrDesc
//       array COMPILED INTO the installed libLLVM-22 (22.1.8).  Its
//       MCInstrDesc::SchedClass values use the numbering that stock
//       AArch64.td produces.
//   (b) overrideCortexA55SchedModel() below installs FirestormModelSchedClasses
//       et al. from build/AArch64GenSubtargetInfo.inc, which this project
//       re-generates from AArch64.td plus ModifiedTarget/AArch64/*.td.  Those
//       extra files (AArch64SchedIcestorm.td, AArch64SchedFirestorm.td, and the
//       modified A55/N1/N2/V1/A520/A720 models) declare additional InstRW rules,
//       and every InstRW that partitions an existing SchedClass makes TableGen
//       CREATE new SchedClass records.  Our class list is 2324 entries; stock's
//       is 2050.  The two numberings therefore DIVERGE, and MCA was indexing
//       table (b) with indices from table (a).
//
// Measured extent of the mismatch (all 9135 AArch64 opcodes compared, stock
// libLLVM vs build/AArch64GenInstrInfo.inc):
//   - opcode numbering:      0 / 9131 mismatched (the two .td revisions agree)
//   - SchedClass numbering:  1051 / 9131 mismatched (11.5%)
//   - first divergence at class 498 (stock) / 499 (ours); delta is +1 for 953
//     opcodes, and -776..+738 for the rest.
// Stock's largest SchedClass index (2049) stays below our NumSchedClasses
// (2324), so the mis-indexing never read out of bounds - it silently returned
// ANOTHER instruction's latency and port assignment, which is why it survived
// unnoticed.
//
// The symptom that exposed it: `fmadd d31, d30, d29, d31` simulated at 3c
// instead of 4c.  FMADDDrrr is Sched<[WriteFMul]> (AArch64InstrFormats.td:5953)
// and WriteFMul is Latency=4 in both Apple models, so the .td was already
// correct - but stock libLLVM hands MCA SchedClass 649, and OUR class 649 is
// FADDDrr_FADDSrr_FSUBDrr_FSUBSrr (WriteF, Latency=3).  Our FMADD class is 650.
// That is why the previous attempts failed to move the number: editing
// WriteFMul's latency, or adding an InstRW for the FMADD family, both correctly
// changed class 650, while the simulation kept reading class 649.  Plain `fmul`
// appeared right only because FMULDrr's stock and local indices happen to agree.
//
// WHY THE WHOLE TABLE, and not just the SchedClass field.  Two reasons, both
// found the hard way:
//
//  1. A per-class translation (rewriting table (b) into stock numbering) is
//     impossible in principle.  The two numberings are not a shift, they are
//     DIFFERENT PARTITIONS of the instruction set - stock puts FMADDDrrr in
//     class 649 and FADDDrr in 1343, we put FADDDrr in 649 and FMADDDrrr in
//     650.  Our extra InstRW rules exist precisely to split stock classes, so
//     one stock class index can correspond to several of ours and cannot carry
//     their differing latencies.  The remap has to be per-OPCODE.
//  2. A per-opcode copy of the descriptors (std::vector<MCInstrDesc> with the
//     SchedClass field overwritten) is also impossible: MCInstrDesc is
//     explicitly non-copyable-in-practice, because it locates its own operand
//     and implicit-operand arrays from its OWN ADDRESS -
//     MCInstrDesc::operands() is `reinterpret_cast<const MCOperandInfo *>(this
//     + Opcode + 1) + OpInfoOffset` (MCInstrDesc.h:240), which only resolves
//     inside the generated <Target>InstrTable struct, where Insts[] is stored
//     in DESCENDING opcode order and is immediately followed by ImplicitOps[]
//     and OperandInfo[].  Copying the descs elsewhere sends operands() into
//     unrelated heap memory; the first attempt at this fix aborted on every
//     input, including a two-instruction `nop; ret`.
//
// So install our AArch64Descs wholesale, via the InitAArch64MCInstrInfo() that
// TableGen emits alongside it - exactly the call libLLVM's own
// createMCInstrInfo() makes, just with our table.  That keeps Insts /
// ImplicitOps / OperandInfo mutually consistent, and makes SchedClass agree
// with the sched-class tables installed by overrideCortexA55SchedModel().
//
// SAFETY of swapping in a table generated from llvm-source (LLVM main) while
// linking libLLVM-22.1.8.  Verified field by field over all 9135 opcodes, our
// table against libLLVM's: SchedClass is the ONLY field that differs anywhere.
//   SchedClass            1051 differ
//   Flags 0    TSFlags 0    NumOperands 0    NumDefs 0    Size 0
//   NumImplicitUses 0    NumImplicitDefs 0    OpInfoOffset 0
//   opcode slot ordering  exact match (our Insts[N-1-i].Opcode == i for all i)
// So this is a pure SchedClass correction: it cannot change mayLoad / mayStore
// / isBranch / operand shape, and any change in simulation output is
// attributable to the sched-class mapping alone.  The only 4 instruction NAMES
// that differ (LOAD_STACK_GUARD, PATCHABLE_EVENT_CALL,
// PATCHABLE_TYPED_EVENT_CALL, PREALLOCATED_ARG, emitted as anonymous_* by our
// run) are generic TargetOpcode pseudos that never occur in a disassembled
// AArch64 binary.  The opcode-count guard below refuses the swap outright if a
// future libLLVM bump breaks that agreement.
//
// EFFECT ON ACCURACY (2026-09-10, eval_estimation.py, CPI-Stack R2, all four
// datasets regenerated through analyze.bash + concat.py).  The fix is NOT an
// accuracy win; it is a correctness fix whose net metric effect is mixed:
//            CPI_S          CPI_B          SF_S            SF_B
//   data      0.183>0.182    0.709>0.707   -0.135>+0.179   -1.074>-0.590
//   0818_a78  0.311>0.268    0.645>0.624   -0.041>-0.174   -0.046>-0.119
//   0907_a78  0.331>0.306    0.660>0.618   +0.029>+0.126   +0.181>-0.267
//   macbook   0.139>0.128    0.638>0.608   +0.309>+0.254   +0.492>+0.419
// MAPE mostly improves (e.g. data SF 0.299>0.270, macbook CPI_S 0.161>0.155).
// Working hypothesis for the regressions: the InstRW rules in
// ModifiedTarget/AArch64/*.td were tuned empirically WHILE the indices were
// scrambled, so part of that tuning was compensating for the mis-indexing -
// and worse, each added InstRW itself renumbers the classes, so it perturbed a
// different arbitrary set of instructions each time.  Those models likely need
// re-tuning on top of this fix rather than the fix being backed out.
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
