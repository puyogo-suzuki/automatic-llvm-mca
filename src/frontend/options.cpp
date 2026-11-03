#include "frontend/options.h"

using namespace llvm;

namespace opts {
    cl::opt<std::string> InputBinary(cl::Positional, cl::desc("<input binary>"), cl::Required);
    cl::opt<std::string> MTriple("mtriple", cl::desc("Target triple"));
    cl::opt<std::string> MCPU("mcpu", cl::desc("Target CPU"));
    cl::opt<int> WindowWidth("window-width", cl::desc("MLP window width"), cl::init(4));
    cl::opt<DependencyKind> DepKind("dependency", cl::desc("Dependency mode"),
        cl::values(
            clEnumValN(DependencyKind::None, "none", "No dependency tracking"),
            clEnumValN(DependencyKind::IO, "io", "In-order dependency"),
            clEnumValN(DependencyKind::OOO, "ooo", "Out-of-order dependency"),
            clEnumValN(DependencyKind::Dependency, "dependency", "Load-use dependency distance")
        ), cl::init(DependencyKind::None));
    cl::opt<MLPWindowAssignmentKind> AssignKind("mlp-window-assignment", cl::desc("Per-load MLP assignment mode"),
        cl::values(
            clEnumValN(MLPWindowAssignmentKind::Forward, "forward", "Forward window"),
            clEnumValN(MLPWindowAssignmentKind::MaxContaining, "max-containing", "Max MLP of containing windows")
        ), cl::init(MLPWindowAssignmentKind::MaxContaining));
    cl::opt<int> Iterations("iterations", cl::desc("Steady-state iteration multiplier"), cl::init(3));
    cl::opt<IgnoreLoopCarriedMode> IgnoreLoopCarried("ignore-loop-carried",
        cl::desc("Ignore loop-carried register dependencies mode"),
        cl::values(
            clEnumValN(IgnoreLoopCarriedMode::Default, "default", "Ignore in basic blocks, but not in loops"),
            clEnumValN(IgnoreLoopCarriedMode::Force, "force", "Ignore in both loops and basic blocks"),
            clEnumValN(IgnoreLoopCarriedMode::Disable, "disable", "Do not ignore loop-carried dependencies anywhere")
        ), cl::init(IgnoreLoopCarriedMode::Default));
    cl::opt<int> OverrideLoadLatency("override-load-latency",
        cl::desc("Override load instruction latency in cycles"),
        cl::init(-1));
    cl::opt<MlpWindowLoopMode> MlpWindowLoop("mlp-window-loop",
        cl::desc("Loop back to the start of the basic block mode"),
        cl::values(
            clEnumValN(MlpWindowLoopMode::Default, "default", "Loop back to the start only for loops"),
            clEnumValN(MlpWindowLoopMode::Force, "force", "Always loop back to the start (even for non-loops)"),
            clEnumValN(MlpWindowLoopMode::Disable, "disable", "Never loop back to the start")
        ), cl::init(MlpWindowLoopMode::Default));
    cl::opt<std::string> TargetAddressStr("target-address",
        cl::desc("Target address to analyze (hex)"), cl::init(""));

    cl::opt<std::string> UpdateMlp("update-mlp", cl::desc("CSV file containing previous MCA results to reuse"), cl::init(""));
    cl::opt<bool> Facile("facile", cl::desc("Enable Facile static analytical throughput prediction (AArch64)"), cl::init(false));
    cl::opt<bool> FacileReason("facile-reason", cl::desc("Output dominant bottleneck reason (inst, exec, prec) for Facile analysis"), cl::init(false));
    cl::opt<bool> NoFacileMemoryDeps("no-facile-memory-deps",
        cl::desc("Disable store->load memory RAW edges in the --facile precedence-constraint "
                 "(Maximum Cycle Ratio) graph, restoring the register-dependences-only "
                 "behaviour. The edges are on by default: the graph already carries "
                 "loop-carried REGISTER recurrences, so omitting loop-carried MEMORY "
                 "recurrences was an inconsistency that made the model blind to any value "
                 "that round-trips through memory across a backedge (e.g. a scalar "
                 "carried via a stack slot instead of a register). A missing precedence bound is masked by "
                 "IssueWidth on narrow cores and only becomes visible on the widest core of "
                 "a pair, which is why it presented as a FireStorm-specific "
                 "under-prediction. This flag exists to A/B the effect."),
        cl::init(false));
    cl::opt<int> ChainThreshold("chain-threshold", cl::desc("Max chain threshold for merged loop region analysis"), cl::init(5));
    cl::opt<bool> MergeSameHeader("merge-same-header", cl::desc("Keep only the widest backward-branch span per loop header (natural-loop definition: all back-edges to one header are one loop); drops same-header partial-path spans such as an early-continue back-edge. Applied after abab merging, never changes a kept span. See region_splitter.cpp mergeSameHeaderLoops."), cl::init(false));
    cl::opt<bool> CountOnly("count-only", cl::desc("Only count total generated regions without running MCA simulation"), cl::init(false));
    cl::opt<bool> DisableAlwaysHitLoadsHeuristic("disable-always-hit-loads-heuristic",
        cl::desc("Disable the same-cache-line 'line reuse' always-hit heuristic: by default "
                 "a load that repeats a same-base-register+cache-line access already seen in "
                 "the analyzed window is treated as an unconditional cache hit and excluded "
                 "from both the load_instructions/BB_LD_RETIRED count (countPotentialMissLoads) "
                 "and the BB_MLP potential-miss averaging (compute_mlp). Passing this flag "
                 "turns that exclusion off, so such loads are counted normally; it is a "
                 "diagnostic escape hatch for generating 'no exclusion at all' reference data. "
                 "Stack/frame-pointer always-hit exclusion is unrelated and unconditional "
                 "regardless of this flag; see -no-stack-exclusion to turn that off instead."),
        cl::init(false));
    cl::opt<bool> LineReuseInOrder("line-reuse-in-order",
        cl::desc("Also apply the same-cache-line 'line reuse' always-hit heuristic in the "
                 "non-OOO code paths (--dependency dependency|io|none, i.e. the in-order "
                 "small-core configuration such as cortex-a55), which otherwise only apply "
                 "the stack/frame-pointer always-hit rule. The heuristic's physical "
                 "justification is a stall-on-use IN-ORDER pipeline: once a real miss stalls "
                 "the machine, a later load to the same cache line is guaranteed resident, so "
                 "it is a guaranteed hit. That argument applies literally to an in-order core, "
                 "so this flag exists to A/B-test whether extending the exclusion to the "
                 "in-order path improves prediction accuracy. Ignored under "
                 "-disable-always-hit-loads-heuristic, and a no-op for --dependency ooo "
                 "(which already enables the heuristic unconditionally)."),
        cl::init(false));
    cl::opt<bool> StackOnlyMissLoadCount("stack-only-miss-load-count",
        cl::desc("In countPotentialMissLoads only (the load_instructions column), exclude "
                 "stack/frame-pointer accesses but keep same-cache-line repeat accesses. "
                 "Diagnostic knob that decomposes the two always-hit heuristics; does not "
                 "affect the MLP computation itself."),
        cl::init(false));
    cl::opt<bool> StackConstOffsetOnly("stack-const-offset-only",
        cl::desc("With -stack-only-miss-load-count, additionally require a constant "
                 "(immediate) offset for the stack always-hit rule, so register-indexed "
                 "stack accesses (large local arrays) are still counted as possible misses."),
        cl::init(false));
    cl::opt<bool> StackSpillOnly("stack-spill-only",
        cl::desc("With -stack-only-miss-load-count, only treat a stack/frame-pointer load "
                 "as an always-hit when a store to the exact same (base_reg, offset) is "
                 "seen in the analyzed region (a provable spill/reload pair)."),
        cl::init(false));
    cl::opt<bool> StackLoopResident("stack-loop-resident",
        cl::desc("With -stack-only-miss-load-count -stack-spill-only, additionally treat a "
                 "constant-offset stack load inside a steady-state loop region as an "
                 "always-hit: the same slot is touched every iteration, so it is L1 "
                 "resident from the second iteration on."),
        cl::init(false));
    cl::opt<bool> NoStackExclusion("no-stack-exclusion",
        cl::desc("Do not give stack/frame-pointer (sp/wsp/x29/w29/fp) loads any special "
                 "always-hit treatment: treat them exactly like ordinary loads in both the "
                 "MLP computation (compute_mlp, the 'mlp' column) and the potential-miss "
                 "load count (countPotentialMissLoads, the 'load_instructions' column), "
                 "subjecting them to the same OOO/dependency same-cache-line heuristics as "
                 "any other load. Takes effect only where the blanket stack always-hit "
                 "check would otherwise fire."),
        cl::init(false));
}
