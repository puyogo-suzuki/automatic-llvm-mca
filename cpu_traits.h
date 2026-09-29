#ifndef CPU_TRAITS_H
#define CPU_TRAITS_H

#include "llvm/ADT/StringRef.h"

// Per-CPU knowledge that LLVM's scheduling model cannot express and that used
// to be spread over `CPU == "cortex-a78" || CPU == "cortex-a78ae" || ...`
// ladders in frontend.cpp, facile.cpp, custom_a55_sched.cpp, mca.cpp and
// convergence_test.cpp.  The table lives in cpu_traits.cpp; adding a CPU or a
// CPU variant (e.g. cortex-a78c) is one line there.

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

// Adjacent-pair macro-op fusion table (facile.cpp, computeFusedMask()).  Each
// core has its OWN SOG table; they are deliberately not shared.
enum class FusionTable {
    None,
    A78,          // A78 SOG sec. 4.14
    X1,           // X1 SOG "Instruction fusion": the A78 table minus CMP + CSEL/CSET
    A720,         // A720 SOG sec. 4.11
};

struct CpuTraits {
    CpuModel Model = CpuModel::None;

    // Real uOP dispatch/issue width from the SOG, in uops per cycle.  Used as
    // PipelineOptions::DispatchWidth instead of the borrowed model's
    // MCSchedModel::IssueWidth.  0 = keep the model's IssueWidth.
    unsigned UopDispatchWidth = 0;

    // SOG "dispatch stage can process up to N Mops per cycle": a second,
    // independent front-end cap (frontend.h, TargetInfo::MopDispatchWidth).
    // 0 = not known for this CPU / not applicable.
    unsigned MopDispatchWidth = 0;

    FusionTable Fusion = FusionTable::None;

    // The model's Latency==0 writes are exactly the SOG's rename-time
    // zero-latency moves, so facile may drop its 1-cycle latency floor for
    // them (see getEdgeLatency() in facile.cpp).  NOT set for models that use
    // Latency==0 as an accounting marker.
    bool ZeroLatencyMov = false;

    // The core is modelled by NeoverseN2Model, which disagrees with the
    // Cortex-A78 SOG on register-offset load/store uops, M0 occupancy, LDPSW
    // and integer divide; facile.cpp corrects those on top of the model.
    // (The X1's NeoverseV1Model does not have these defects.)
    bool N2ModelCorrections = false;
};

// Unknown CPU names (including "generic") get a default-constructed CpuTraits.
// Keyed by the CPU name as MCSubtargetInfo::getCPU() reports it.
const CpuTraits &getCpuTraits(llvm::StringRef CPU);

#endif // CPU_TRAITS_H
