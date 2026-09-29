#include "cpu_traits.h"
#include "llvm/ADT/StringMap.h"

namespace {

// Sources for the widths (SOG = Arm Software Optimization Guide of that core):
//   uop dispatch: Cortex-A76 8, A78 12, A720 10, X1 16.
//   Mops per cycle: A76 4 (SOG PJDOC-466751330-7215 sec 4.1), A78 6 (sec 4.1),
//   A720 5 (sec 4.1), X1 8 (PJDOC-466751330-12804 sec 4.1).
// Decode width (e.g. A78 4, X1 5) is deliberately NOT modelled: the tool
// bounds hot-loop front-end throughput by the rename/dispatch width.
//
// Cores that only share a uop DispatchWidth with a modelled core and whose own
// SOG has not been consulted keep MopDispatchWidth = 0 (bound not applied):
// cortex-a710/a715/neoverse-n2 (10) and neoverse-v1 (16) - do not assume their
// Mop width equals the A720's or X1's.
//
// The Apple icestorm/firestorm entries have no Mop width either:
// computeFacilePrediction() already forces NumMicroOps = 1 for the
// coalesced-ROB CPUs (isCoalescedROBCPU()), so their dispatch width already
// IS a Mop-level width and must not get a second, redundant cap.
llvm::StringMap<CpuTraits> buildTable() {
    llvm::StringMap<CpuTraits> T;

    CpuTraits A55;  A55.Model = CpuModel::A55;
    T["cortex-a55"] = A55;

    CpuTraits A520; A520.Model = CpuModel::A520;
    T["cortex-a520"] = A520;
    T["cortex-a520ae"] = A520;

    CpuTraits A76;  A76.Model = CpuModel::A76; A76.UopDispatchWidth = 8; A76.MopDispatchWidth = 4;
    T["cortex-a76"] = A76;
    T["cortex-a76ae"] = A76;
    CpuTraits N1;   N1.UopDispatchWidth = 8;   N1.MopDispatchWidth = 4;   // stock model
    T["neoverse-n1"] = N1;

    CpuTraits A78;  A78.Model = CpuModel::A78; A78.UopDispatchWidth = 12; A78.MopDispatchWidth = 6;
    A78.Fusion = FusionTable::A78; A78.ZeroLatencyMov = true; A78.N2ModelCorrections = true;
    T["cortex-a78"] = A78;
    T["cortex-a78ae"] = A78;
    T["cortex-a78c"] = A78;

    CpuTraits A720; A720.Model = CpuModel::A720; A720.UopDispatchWidth = 10; A720.MopDispatchWidth = 5;
    A720.Fusion = FusionTable::A720; A720.ZeroLatencyMov = true;
    T["cortex-a720"] = A720;
    T["cortex-a720ae"] = A720;

    CpuTraits Shared10; Shared10.UopDispatchWidth = 10;   // stock models, Mop width not consulted
    T["cortex-a710"] = Shared10;
    T["cortex-a715"] = Shared10;
    T["neoverse-n2"] = Shared10;

    CpuTraits X1;   X1.Model = CpuModel::X1; X1.UopDispatchWidth = 16; X1.MopDispatchWidth = 8;
    X1.Fusion = FusionTable::X1; X1.ZeroLatencyMov = true;
    T["cortex-x1"] = X1;
    T["cortex-x1c"] = X1;
    CpuTraits V1;   V1.UopDispatchWidth = 16;                            // stock model
    T["neoverse-v1"] = V1;

    CpuTraits Ice;  Ice.Model = CpuModel::Icestorm;
    T["icestorm"] = Ice;
    CpuTraits Fire; Fire.Model = CpuModel::Firestorm;
    T["firestorm"] = Fire;

    return T;
}

} // namespace

const CpuTraits &getCpuTraits(llvm::StringRef CPU) {
    static const llvm::StringMap<CpuTraits> Table = buildTable();
    static const CpuTraits Default;
    auto It = Table.find(CPU);
    return It == Table.end() ? Default : It->second;
}
