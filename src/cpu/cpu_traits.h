#ifndef CPU_TRAITS_H
#define CPU_TRAITS_H

#include "llvm/ADT/StringRef.h"

// Per-CPU knowledge that LLVM's scheduling model cannot express, used by the
// frontend, the Facile predictor, the sched-model installer and the MCA driver
// instead of comparing CPU names.  The table lives in cpu_traits.cpp; adding a
// CPU or a CPU variant (e.g. cortex-a78c) is one line there.

// Which locally generated sched model (ModifiedTarget/AArch64/*.td) is
// installed for a CPU.  None = libLLVM's own stock model is left in place.
enum class CpuModel {
    None,
    A55,
    A520,
    A76,       // NeoverseN1Model
    A78,       // NeoverseN2Model
    A720,
    X1,        // NeoverseV1Model
    Icestorm,
    Firestorm,
};

// Adjacent-pair macro-op fusion table (facile/fusion.cpp, computeFusedMask()).  Each
// core has its OWN SOG table; they are deliberately not shared.
enum class FusionTable {
    None,
    A78,          // A78 SOG sec. 4.14
    X1,           // X1 SOG "Instruction fusion": the A78 table minus CMP + CSEL/CSET
    A720,         // A720 SOG sec. 4.11
};

// Out-of-order window limits for the MLP window, for cores whose published
// documentation gives several independent in-flight limits instead of one
// uop-counted ROB.  The MLP window (mlp/mlp_window.cpp) ends at the first
// instruction that would exceed any of them.  All zero = not used (the window is
// then `MicroOpBufferSize` uops, or --window-width).
struct WindowResources {
    // 0 = this resource is not limited (the source gives no figure for it).
    unsigned RenameEntries = 0;   // architectural-register writes in flight (zero registers excluded)
    unsigned LoadSlots = 0;       // loads in flight (an LDP counts as 2)
    unsigned StoreSlots = 0;      // stores in flight (an STP counts as 2)
    unsigned BranchSlots = 0;     // branches (incl. calls/returns) in flight
    unsigned MaxInsts = 0;        // absolute cap on instructions in the window; 0 = struct unused
    bool enabled() const { return MaxInsts != 0; }
};

struct CpuTraits {
    CpuModel Model = CpuModel::None;

    // See WindowResources.  Only Firestorm and Icestorm set it.
    WindowResources MlpWindow;

    // Real uOP dispatch/issue width from the SOG, in uops per cycle.  Used as
    // PipelineOptions::DispatchWidth instead of the borrowed model's
    // MCSchedModel::IssueWidth.  0 = keep the model's IssueWidth.
    unsigned UopDispatchWidth = 0;

    // SOG "dispatch stage can process up to N Mops per cycle": a second,
    // independent front-end cap (frontend/frontend.h, TargetInfo::MopDispatchWidth).
    // 0 = not known for this CPU / not applicable.
    unsigned MopDispatchWidth = 0;

    FusionTable Fusion = FusionTable::None;

    // The model's Latency==0 writes are exactly the SOG's rename-time
    // zero-latency moves, so facile may drop its 1-cycle latency floor for
    // them (see getEdgeLatency() in facile/facile.cpp).  NOT set for models that use
    // Latency==0 as an accounting marker.
    bool ZeroLatencyMov = false;
};

// CPU names.  Every --mcpu spelling is resolved ONCE, by the frontend, to a
// canonical name (canonicalCpuName()), and everything per-CPU is keyed by that
// name: getCpuTraits(), the local sched model (installLocalSchedModel(),
// remapSchedClassIndices()), the MLP window limits, and - through
// getCpuTraits(const MCSubtargetInfo &) in sched_model.h - Facile's fusion table
// and zero-latency moves and the MCA driver's in-order pipelines.
//
// The canonical name is the --mcpu string itself, except for the short Apple
// aliases "ice" -> "icestorm" and "fire" -> "firestorm".  CPU variants that LLVM
// knows by their own name (cortex-a78c, cortex-x1c, the -ae parts, ...) are
// canonical names of their own and share their core's row in the table.
llvm::StringRef canonicalCpuName(llvm::StringRef MCPU);

// The CPU name handed to Target::createMCSubtargetInfo() for a canonical name.
// LLVM has no per-core Apple CPU, so "icestorm" and "firestorm" both become
// "apple-m1"; LLVM resolves that alias to "apple-a14", which is what
// MCSubtargetInfo::getCPU() then reports for both cores.  Every other name is
// passed unchanged.
llvm::StringRef llvmCpuName(llvm::StringRef CanonicalCPU);

// Unknown CPU names (including "generic") get a default-constructed CpuTraits.
// Keyed by the canonical name (canonicalCpuName()); an alias such as "ice" is
// not a key.
const CpuTraits &getCpuTraits(llvm::StringRef CanonicalCPU);

#endif // CPU_TRAITS_H
