#include "cpu/cpu_traits.h"
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
// Mop width equals the A720's or X1's.  neoverse-n1 is the exception: it has
// the Cortex-A76 widths (8 uops, 4 Mops, from the A76 SOG) although the
// Neoverse N1's own SOG has not been consulted (docs/docs.md section 4.E).
//
// The Apple icestorm/firestorm entries have no Mop width either:
// computeFacilePrediction() already forces NumMicroOps = 1 for the
// coalesced-ROB CPUs (isCoalescedROBSubtarget()), so their dispatch width already
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
    A78.Fusion = FusionTable::A78; A78.ZeroLatencyMov = true;
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
    // MLP window limits, from Dougall Johnson's Icestorm page
    // (https://dougallj.github.io/applecpu/icestorm.html): ~111 renames in flight,
    // 30 load buffers, 18 store buffers.  The page gives NO branch limit (0 =
    // unlimited here, so the window can come out too wide), no retire-group size
    // (MaxInsts reuses Firestorm's "at most 7 uops per group": ~60 groups * 7), and
    // itself warns the M1 does not use "an entirely conventional reorder buffer" so
    // the figures "may or may not be accurate".
    Ice.MlpWindow = {111, 30, 18, 0, 60 * 7};
    T["icestorm"] = Ice;
    CpuTraits Fire; Fire.Model = CpuModel::Firestorm;
    // MLP window limits, from Dougall Johnson's Firestorm page
    // (https://dougallj.github.io/applecpu/firestorm.html).  There is no Arm SOG
    // for Apple cores, so these are not SOG values; the page's numbers are
    // approximate ("~").
    //   rename retire queue ~623 entries: every write to an architectural register
    //     takes one (add/mov 1, adds/ldp 2, cbz/str 0);
    //   ~130 loads, ~60 stores, ~144 branches in flight;
    //   MaxInsts: the coalesced retire queue holds ~330 retire groups of at most 7
    //     uops, so no window holds more than 330 * 7 instructions (the page quotes
    //     "over 2200 nop").  The group-forming rules themselves are not modelled.
    // The physical register files are deliberately not used (the page marks their
    // sizes uncertain).
    Fire.MlpWindow = {623, 130, 60, 144, 330 * 7};
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
