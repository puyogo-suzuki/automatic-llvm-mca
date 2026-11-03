# automatic-llvm-mca Technical Documentation

This document provides an exhaustive technical overview of the processor scheduling model custom adaptations, the C++ simulation pipeline modifications, and the static Memory Level Parallelism (MLP) calculation methodology implemented here. Each modelled Arm core and its conformance to its Arm Software Optimization Guide is described separately, item by item: Cortex-A55 in [a55-sog.md](a55-sog.md), Cortex-A520 in [a520-sog.md](a520-sog.md), Cortex-A76 in [a76-sog.md](a76-sog.md), Cortex-A78 in [a78-sog.md](a78-sog.md), Cortex-A720 in [a720-sog.md](a720-sog.md) and Cortex-X1 in [x1-sog.md](x1-sog.md). Those documents are the reference for each core; where this document mentions a core-specific value, the per-core document is authoritative. [README.md](README.md) lists the documents.

---

## 1. Processor Scheduling Model Customizations & Forwarding Paths

The scheduling models are customized at the **TableGen** description level (`ModifiedTarget/AArch64/*.td`, compiled into the tool's own tables, see section 5) and, for the in-order Cortex-A55, in the **C++ runtime simulation layer** (`src/mca/mca_simulation.cpp`, `src/a55/`).

The Cortex-A55 model (the TableGen model `AArch64SchedA55.td`, the A55 issue stage, the flag and pointer bypasses and the other C++ corrections) is described in [a55-sog.md](a55-sog.md), the Cortex-A520 model and its issue stage in [a520-sog.md](a520-sog.md), the Cortex-A76, Cortex-A78 and Cortex-A720 models in [a76-sog.md](a76-sog.md), [a78-sog.md](a78-sog.md) and [a720-sog.md](a720-sog.md). This chapter keeps only the overview of the Neoverse-derived models (1.A) and the Cortex-X1 machine parameters (1.B).

---

### A. Neoverse Architecture Validations (N1, N2, V1)

The out-of-order Cortex-A76, Cortex-A78 and Cortex-X1 use local copies of LLVM's Neoverse models (`AArch64SchedNeoverseN1.td`, `AArch64SchedNeoverseN2.td`, `AArch64SchedNeoverseV1.td`), installed only for those cores (`neoverse-n1`, `neoverse-n2` and `neoverse-v1` keep libLLVM's stock models). The cores' own differences from the Neoverse rows are corrected in those files and listed in [a76-sog.md](a76-sog.md), [a78-sog.md](a78-sog.md) and [x1-sog.md](x1-sog.md).
*   The shifted-register integer ALU forms resolve through a `SchedWriteVariant` on the stock predicate `IsCheapLSL` (shift type LSL and amount 0 to 4): 1 cycle on an I pipe (`N1Write_1c_1I`, `N2Write_1c_1I`, `V1Write_1c_1I`) when it holds, otherwise 2 cycles on M (`N1Write_2c_1M`, `N2Write_2c_1M`, `V1Write_2c_1M`). The extended-register forms and the per-core rules are in the "arithmetic and logical" items of the per-core documents (a76-sog.md C-8, a78-sog.md C-9, x1-sog.md C-9).
*   These models are out of order (`MicroOpBufferSize > 0`); no branch latency override is applied to them.

---

### B. Cortex-X1 Model (`NeoverseV1Model`)

The Cortex-X1 runs on `AArch64SchedNeoverseV1.td`. That file is compiled into the tool's own tables and installed **only** for `cortex-x1` / `cortex-x1c` (`neoverse-v1` keeps libLLVM's stock model), so its machine parameters are the X1's:

| Parameter | Value | Where |
|---|---|---|
| Decode width | 5 | not modelled |
| Alloc/rename width | 8 Mops/cycle | `MopDispatchWidth` in `cpu_traits.cpp` (X1 SOG sec. 4.1: "up to 8 MOPs per cycle") |
| Issue (dispatch) width | 16 uops/cycle | `UopDispatchWidth` in `cpu_traits.cpp`, which overrides the `.td`'s `IssueWidth` (also 16) |
| ROB | 224 | `MicroOpBufferSize`; on the X1 path it is used as the MLP window ([x1-sog.md](x1-sog.md) U-1) |
| Scheduler entries | 150 | not modelled |

The ROB size, the decode width and the scheduler size are not in the SOG; they were supplied by the project owner. The decode width and the scheduler size are not modelled, for the same reasons as for the A78 (whose 4-wide decode and scheduler size are likewise absent from `AArch64SchedNeoverseN2.td`): neither `MCSchedModel` nor MCA has a decode stage, and the tool bounds hot-loop front-end throughput by the rename/dispatch width. MCA's only handle for scheduler entries is a per-`ProcResource` `BufferSize`, i.e. one queue per pipe or group, and nothing says how the 150 entries are split over the fifteen issue pipelines; inventing a split would add stalls the hardware need not have, so the queues stay unbounded and the ROB is the only window limit.

The model's rows were compared with the Cortex-X1 Software Optimization Guide item by item in [x1-sog.md](x1-sog.md), which lists every difference from the stock V1 model, the corrections made on top of it (tagged `[x1_...]` in the `.td`), the deviations and the open questions. The dispatch widths are C-1 there, the ROB size U-1, the decode width and scheduler size D-12. Facile-side X1 behaviour: the fusion table is the A78 table without the CMP + CSEL/CSET rows, and the zero-latency MOV list is the A78's (sections 4.B and 4.C; x1-sog.md C-4, C-5). The Cortex-A78 changes of section 4.D are made in the A78's own model file and do not reach the X1.

## 2. LLVM MCA Simulation & Steady-State Estimation

Calculating representative cycle metrics requires isolating the stable, recurring phase of execution (the steady-state) from transient startup effects.

```
[ Cold Pipeline ] ----> [ Warm-up Phase ] ----> [ Transition ] ----> [ Steady-State Phase ]
                        (WarmupIterations)                           (SteadyIterations)
                                                                     * Only this phase is measured
```

### A. Warm-up Iterations Calculation
To completely fill the reservation stations, reorder buffers, and pipeline stages:
1.  We query the subtarget's maximum inflight instruction capability (`getWarmupWindowSize` in `src/mca/mca_simulation.cpp`):
    *   For out-of-order cores, the larger of `MicroOpBufferSize` and `LoopMicroOpBufferSize`.
    *   For in-order cores, `IssueWidth` (at least 1).
2.  The number of iterations needed to fill that window once (`computeWarmupIterations`), with $n$ the number of instructions of the region:
    $$\text{WarmupIterations} = \max\left(1, \left\lceil \frac{\text{WarmupWindowSize}}{n} \right\rceil\right)$$
3.  The measured iterations (`computeSteadyIterations`) fill the same window `--iterations` times (default 3); `--iterations` is a multiplier, not an iteration count:
    $$\text{SteadyIterations} = \max\left(1, \left\lceil \frac{\text{WarmupWindowSize} \times \text{iterations}}{n} \right\rceil\right)$$
    The region is simulated for $\text{WarmupIterations} + \text{SteadyIterations}$ iterations.

### B. Event-Driven Metric Tracking (`SteadyStateTracker`)
We attach a custom `HWEventListener` to the MCA pipeline:
*   **`onCycleBegin` / `onCycleEnd`**: Tracks the progression of execution cycles.
*   **`onEvent` (Retired)**: Increments the total retired instructions.
    *   When the retired instruction count is less than `WarmupRetiredLimit` ($\text{WarmupIterations} \times \text{RegionSize}$), no metrics are stored.
    *   As soon as it crosses the limit, `WarmupComplete` becomes `true` and the tracker marks the current cycle as `SteadyStartCycle`.
    *   Subsequent cycles and retirements are accumulated into `SteadyCycles` and `SteadyRetired`, respectively.

### C. Loop-Carried Dependency Elimination
For basic block throughput estimation, loop-carried register dependencies (e.g., induction variables `add x4, x4, x3` spanning across iterations) must be ignored.
*   **Mechanism**: We use a template-based member pointer extraction trick to bypass C++ access controls and retrieve `RegisterFile::RegisterMappings` from the active register file:
    ```cpp
    auto member_ptr = get_mappings(RegisterFile_RegisterMappings_Tag{});
    auto &mappings = (*PRF).*member_ptr;
    ```
*   When a new iteration starts (tracked by `readerIID / LoopSize`), we clear the history of register writes from prior iterations by resetting their `WriteRef`.
*   We intercept the instruction dispatch phase and manually mark any pending input operands (`mca::ReadState`) as ready if their critical dependency originates from a previous loop iteration.
*   **Timing of the reset** (the point the code and the tests tag `#12`). The reset must reach the first instruction of an iteration before that instruction is checked for its register operands. In the out-of-order pipeline the `Dispatched` event is raised at dispatch, before the instruction can issue, so the tracker's `onInstructionDispatched()` is in time for every instruction. The in-order issue stages raise `Dispatched` only after their register hazard check, which is too late for the first instruction of an iteration; the Cortex-A55 and Cortex-A520 issue stages therefore call `LoopCarriedIgnoreSteadyStateTracker::beforeRegisterHazardCheck()` before each hazard check (`setBeforeRegisterHazardCheckHook`, wired in `analyzeMcaRegion`; see [a55-sog.md](a55-sog.md#22-issue-stage-srca55a55_issue_stagecpp) and [a520-sog.md](a520-sog.md#22-issue-stage-srca520a520_issue_stagecpp)). Check: the basic block `fadd d0, d0, d1; add x2, x2, #1; add x3, x3, #1; ret` (`--iterations=200`) gives the same cycles/iteration with the `fadd` first or second on all six Arm cores (A78/X1/A720 0.5, A76 1.0, A55 2.01, A520 1.35 under `--ignore-loop-carried=default`; 2.0 for the out-of-order cores and 4.0 to 4.01 for the in-order ones under `disable`).

### D. Basic Block Partitioning under Size Limits
There are no region size limits: no region is skipped or cut for its length. A region is either a loop span or a non-loop span of a function (section 2.E), and a function without loops is one region however long it is. Regions whose instructions are all NOPs (`isAllNopRegion`) are not analysed. The sections `.plt`, `.plt.*`, `.init` and `.fini` are skipped by `mca_tool`.

### E. CFG and Post-Dominator Based Region Splitting
No control flow graph, dominator or post-dominator analysis is built. The splitter (`walkRegions()` in `src/core/region_splitter.cpp`) works on the linear disassembly of each executable section, function by function, with backward branches as the loop criterion (the top-level [README](../README.md#region-splitting--merging-algorithm) has the same steps):
1.  **Functions**: the instructions are cut at the start addresses of the ELF function symbols (`collectFunctionBoundaries`); the start of the section also starts a function.
2.  **Loops** (`detectBackwardBranchLoops`): every branch that is not a call and whose target lies at or before the branch and not before the start of the function gives a loop span `[target, branch]`. A span is dropped if it contains a return that is reached from the header without passing a forward branch (an unconditional exit, not a loop body).
3.  **Interlocking (abab) chains** (`detectAndMergeAbabChains`): two spans $a$, $b$ with `a.h < b.h < a.l < b.l` overlap without nesting. The length of the longest chain of such pairs is computed by dynamic programming over the spans sorted by (header ascending, latch descending), looking back at most 50 spans. If it reaches `--chain-threshold` (default 5), every span that takes part in an overlap is removed and the overlapping groups are replaced by their merged bounding spans (marked `IsAbabMerged`); spans that overlap nothing, including spans nested inside others, are kept. Below the threshold all spans are kept as they are.
4.  **Same header** (`mergeSameHeaderLoops`, only with `--merge-same-header`, off by default): of several spans with the same header only the widest is kept.
5.  **Non-loop spans**: in a function with loops, each maximal run of instructions outside every kept loop span is reported as a region of its own whose metrics are those of a loop of the same function: the first loop span of step 2 (before steps 3 and 4) that starts at or after the end of the run, or else the last one. The CSV carries that loop as `analysis_start`/`analysis_end` and `loop` = 1. In a function without loops the whole function is one non-loop region (`loop` = 0), analysed by itself.

Nested loops are separate regions; the outer span includes the inner loop's instructions. `mca_tool` prints the regions sorted by end address, then by start address.

### F. Instructions rejected by InstrBuilder
An instruction whose scheduling class cannot be modelled (e.g. a `SchedWrite` marked `Unsupported`) is left out of the simulated sequence. `buildRegionInstructions` (`src/mca/mca_simulation.cpp`), which builds the instructions of a region for `analyzeMcaRegion` and for the `--facile` path of `mca_tool`, counts each drop per opcode (`noteDroppedInstruction`), and `mca_tool` / `mlp-objdump` print a `[WARN]` summary to stderr once the (stderr-silenced) analysis is over. The CSV output does not mark the affected regions.

---

## 3. Memory Level Parallelism (MLP) Calculation Method

Memory Level Parallelism (MLP) represents the average number of memory loads that can be processed concurrently by the memory subsystem.

### A. Window-Based Outstanding Load Evaluation
The static MLP analyzer (`MLPAnalyzer`) runs a sliding-window sweep over the instruction vector:

1.  **Instruction Properties Extraction**:
    For each instruction, we extract an `MLPInstInfo` struct. This contains the register input/output sets, the micro-op count, and memory properties (such as whether it is a load, a store, or a function call).
2.  **Sliding Window Sweep**:
    For each load instruction index $i$, we open a speculative window:
    *   Instructions $j \ge i$ are added to the window.
    *   The window size is constrained by the sum of micro-ops ($\sum \text{uops} \le W$, where $W$ is the `--window-width` or reorder buffer size).
    *   **Apple Firestorm and Icestorm (in-flight-limit window)**: the ROB of both Apple cores counts coalesced retire *groups* (up to 7 uops each; Firestorm ~330, Icestorm ~60), not uops, so for `--mcpu firestorm|icestorm` the window instead ends at the first instruction that would exceed any one of the limits on Dougall Johnson's pages (https://dougallj.github.io/applecpu/firestorm.html, https://dougallj.github.io/applecpu/icestorm.html; approximate "~" values). Cost per instruction: rename retire queue = one entry per architectural-register write (zero registers excluded, NZCV included: `add` 1, `adds`/`ldp` 2, `cbz`/`str` 0); loads (`ldp` counts 2; every load counts, including always-hit and stack loads); stores (`stp` counts 2); branches (all branch/call/return instructions); plus at most groups × 7 instructions. Firestorm: rename 623, loads 130, stores 60, branches 144, 330 × 7 instructions. Icestorm: rename ~111, loads 30, stores 18, **no branch limit** (the page gives none; 0 = unlimited in `WindowResources`, so the window can come out too wide), 60 × 7 instructions (the 7 is Firestorm's figure; Icestorm's page gives no group size). The Icestorm page itself warns that the M1 does not use "an entirely conventional reorder buffer" and its figures "may or may not be accurate". The group-forming rules and the physical register files are not modelled. An explicit `--window-width` overrides this and uses the plain uop window. The limits live in `cpu_traits.cpp` (`WindowResources`) and are handed to the analyzer by the frontend from the `--mcpu` name (`MCSubtargetInfo::getCPU()` reports `apple-a14` for both Apple cores, so it cannot be used). Other cores are unchanged.
3.  **Dependency Stall Check**:
    For each instruction $j$ added to the window, we check if it consumes any register defined by an active load already inside the window:
    ```cpp
    bool has_dep = has_intersection(inst_infos[j].io_regs.inputs, load_dep_regs);
    ```
    If a register dependency is detected, the window expansion halts at that index (representing a data hazard stall).
4.  **Loop Wrap-Around (`--mlp-window-loop`)**:
    If loop-mode is enabled, when the window reaches the end of the basic block, it wraps around to the beginning, carrying register dependency masks and seen-base registers into the next virtual iteration.

### B. Cache Hit and Spatial Locality Filtering
Not all memory loads translate to concurrent outstanding memory requests. The analyzer filters out loads that are highly likely to result in L1 cache hits:

1.  **Stack and PC-Relative Load Filtering**:
    Loads referencing the stack pointer (e.g., `ldr w0, [sp, #16]`) or PC-relative loads (e.g., literal pool loads) are assumed to be L1 cache hits and do not contribute to MLP.
2.  **Base Register Spatial Locality Filtering (`SeenBaseRegs`)**:
    If multiple loads within the same window access different offsets using the same base register (e.g., `ldrb w1, [x0, #0]` and `ldrb w2, [x0, #1]`), they are likely targeting the same cache line.
    *   We track base registers and their cache line boundaries ($64$-byte granularity):
        $$\text{CacheLine} = \frac{\text{Offset}}{64}$$
    *   **Stall-on-Use Cache Line Hit Requirement**: Even if a base register and cache line pair `(base_reg, cache_line)` has been seen, a subsequent load targeting the same cache line is only classified as a guaranteed cache hit if the processor has already encountered a user instruction consuming the output of the first load (causing an in-order "stall-on-use" that ensures the cache line has finished loading). If no user of the first load appears between the first and second loads, the second load is assumed to still have a chance of cache missing (not treated as a hit).
    *   **Call-Instruction Invalidation**: at a function call instruction (`bl`, `blr`; on RISC-V a `jal`/`jalr` that links into `ra`/`t0`) only the return-value registers are invalidated (`getReturnRegisters`; on AArch64 the registers numbered 0 to 7 of the X/W and the FP/SIMD register files): they are removed from the load-dependence set and from the seen base registers (and the seen store addresses). Every other register dependence and every other seen (base register, cache line) pair stays in force across the call.
3.  **No-Load Default Specification**:
    If a basic block or loop region contains no eligible memory load instructions (or is an empty instruction sequence), both $MLP$ and $MLP_R$ values default to $1.0$ (representing the sequential memory baseline).

## 4. Facile: SOG-Derived Modelling Notes

Rationale for the code in `src/facile/`. Each subsection names the function it belongs to. "SOG" is the Arm Software Optimization Guide of the core in question; the guides themselves are not part of this repository. Per-CPU switches (which core gets which behaviour) are in `cpu_traits.cpp`.

### A. Dependence-edge latency (`getEdgeLatency`)

Latency to put on a RAW dependency edge  (Writer defines Reg) -> (Reader reads Reg through use slot Use).

The edge latency is read per write and per operand from the scheduling model rather than taken as the instruction's single latency (getInstLatency(Writer)); three points:

(a) PER-DEFINITION latency.  mca::Instruction::getLatency() is InstrDesc::MaxLatency, i.e. the MAXIMUM over all of an instruction's writes, which would charge every out-edge of a multi-def instruction the slowest one.  AArch64 pre/post-index memory ops are exactly that shape: `ldr x2, [x1], #8` is Sched<[WriteAdr, WriteLD]> - 1c for the x1 base update, 4c for the x2 load result - and the base-pointer recurrence that every post-increment / stride loop carries (x1 -> x1 across the backedge, distance 1) closes in 1 cycle, not in the load's 4.  The edge therefore uses WriteState::getLatency(), that individual write's own latency.

(b) REGISTER UNITS.  Registers are tracked by their register units (`regKeys` in `src/facile/facile.cpp`), not by register id.  Aliasing sub-/super-registers (W0/X0, B0/H0/S0/D0/Q0) have the same unit, so `ldr w0, [x0]` chains through x0.  A register tuple (the destination of LD1-LD4, the source of ST1-ST4, e.g. `Q0_Q1`) has one unit per register: the tuple is a producer and a consumer of each of its registers, so `ld1 {v0.16b, v1.16b}, [x0]; add v1.16b, v1.16b, v1.16b` has an edge through v1 as it has through v0.  The edge latency is the latency of the tuple's write.

(c) READADVANCE (operand bypass / late forwarding).  A scheduling model states, per (consumer sched class, consumer operand slot, producer write-resource), how many cycles earlier than write-back that operand is actually needed.  MCA's own RegisterFile::addRegisterRead applies it via WriteState::addUser(IID, Use, ReadAdvance), giving an effective dependence latency of WS.getLatency() - ReadAdvance.  Facile applies it the same way; without it, all operands of a multi-input instruction would need the producer's result at the same (worst) cycle, which over-predicts a multiply-accumulate recurrence.

The case that actually matters here is the ACCUMULATOR of a multiply-accumulate.  Arm Cortex-A78 Core Software Optimization Guide (PJDOC-466751330-9691, Issue 4.0) Table 3-16 "AArch64 FP data processing instructions" lists

```text
       "FP multiply accumulate | FMADD, FMSUB, FNMADD, FNMSUB |
        Execution Latency 4 (2) | Throughput 2 | Pipeline V"
```

with note 3: "FP multiply-accumulate pipelines support late-forwarding of accumulate operands from similar uOPs, allowing a typical sequence of multiply-accumulate uOPs to issue one every N cycles (accumulate latency N shown in parentheses)."  Table 3-7 likewise gives "Multiply accumulate, W-form / X-form | MADD, MSUB | 2(1)". AArch64SchedNeoverseN2.td (the model installed for cortex-a78) encodes precisely that:

```text
       def N2Wr_FMA : SchedWriteRes<[N2UnitV]> { let Latency = 4; }
       def N2Rd_FMA : SchedReadAdvance<2, [WriteFMul, N2Wr_FMA]>;
       def : InstRW<[N2Wr_FMA, ReadDefault, ReadDefault, N2Rd_FMA],
                    (instregex "^FN?M(ADD|SUB)[HSD]rrr$")>;
```

AArch64SchedNeoverseN1.td (cortex-a76) has the same rows, and its `N1Rd_FMA` also advances from `N1Wr_FMul` (`SchedReadAdvance<2, [WriteFMul, N1Wr_FMul, N1Wr_FMA]>`).  So an FMADD -> FMADD accumulator recurrence - the shape of FP reductions, dot products and stencil accumulations - closes in 2 cycles, not 4.  The per-core forwarding rules are in the per-core documents.

Apple's two models use (c) for the forwarding paths Dougall Johnson documents for them ("MADD's output can be passed to its third operand (the addend) with 1c latency, but if it's chained with other instructions it has 3c latency", "Loads may be passed to the base address of other loads with 3c latency ... but chaining with ALU operations gives a latency of 4c"), which are per-operand facts that no WriteRes latency can express: `AArch64SchedFirestorm.td` and `AArch64SchedIcestorm.td` both declare `ReadAdvance<ReadIMA, 2, [WriteIM32, WriteIM64]>` and `ReadAdvance<ReadAdrBase, 1, [WriteLD, WriteLDIdx, WriteLDHi]>`.

The latency is floored at 1 cycle, as in getInstLatency(): a dependent operation never issues in the same cycle as its producer in this model.  The only exception is the zero-latency MOVs of 4.B.

### B. Zero-latency MOVs (`isZeroLatencyMovTarget`, `getEdgeLatency`)

Gate: `CpuTraits::ZeroLatencyMov` (A78 family, A720 family, X1 family).

TRUE ZERO-LATENCY WRITES (rename-time move elimination / zero idioms).  Arm Cortex-A78 SOG sec. 4.15 "Zero Latency MOVs": "A subset of register-to-register move operations and move immediate operations are executed with zero latency.  These instructions do not utilize the scheduling and execution resources of the machine. These are as follows: MOV Xd,#0 / MOV Xd,XZR / MOV Wd,#0 / MOV Wd,WZR / MOV Rd,#0 (AArch32) / MOV Wd,Wn / MOV Xd,Xn / MOV Rd,Rn (AArch32)".  Such a MOV is resolved by pointing the destination's rename entry at the source's physical register, so a consumer of the MOV's result reads the SAME physical register the producer wrote and waits exactly zero additional cycles.

The scheduling model carries this.  AArch64SchedNeoverseN2.td (the model installed for cortex-a78) has

```text
       def N2Write_0c : SchedWriteRes<[]> { let Latency = 0; }
       def N2Write_0or1c_1I : SchedWriteVariant<[
             SchedVar<NeoverseZeroMove, [N2Write_0c]>,
             SchedVar<NoSchedPred,      [N2Write_1c_1I]>]>;
       def : InstRW<[N2Write_0or1c_1I], (instregex "^MOVZ[WX]i$")>;
       def : InstRW<[N2Write_0or1c_1I], (instregex "^ORR[WX]rs$")>;
```

and AArch64SchedPredNeoverse.td's NeoverseZeroMove enumerates precisely sec. 4.15's list (MOVZ[WX]i with both immediates zero, i.e. MOV Wd/Xd,#0; ORR[WX]rs with Rn == ZR and shift 0, which is how MOV Wd,WZR / MOV Xd,XZR / MOV Wd,Wn / MOV Xd,Xn all encode). SchedWriteRes<[]> is also exactly the SOG's "do not utilize the scheduling and execution resources" - no ProcResource is listed - and calculatePortUsageBound() sees no resource for it, because resolveSchedClass() resolves the variant.  For the dependence latency, getEdgeLatency() returns 0 for a Latency==0 write on the gated targets instead of applying the 1-cycle floor of 4.A.

The floor is right for every other write (a dependent operation cannot issue in the same cycle as a producer that really has to compute something), but a zero-latency MOV is not computed at all: there is no producer to wait for.  Charging it a cycle would inflate every dependence chain that passes through a register copy. WHY THIS IS GATED ON THE TARGET AND NOT JUST ON "Latency == 0". A zero-latency write means "renamed away" only in some models; in the stock N1 and V1 models and in the Apple models a blanket rule would be wrong:

```text
      * N2 (cortex-a78): N2Write_0c is reached only through the
        NeoverseZeroMove arm of the MOVZ[WX]i and ORR[WX]rs variants
        (the FMOV-from-ZR and MOVI #0 rows do not use the variant on the
        A78, whose sec. 4.15 does not list them: a78-sog.md C-4).  Here
        Latency==0 <=> SOG sec. 4.15 move.
      * N1 (cortex-a76): the stock model uses N1Write_0c_0Z (Latency 0,
        NumMicroOps 0) as the SECOND write of LDPWi / LDNPWi / LDPSWi,
        i.e. the second destination register of a load pair: a micro-op
        accounting marker, NOT a statement that the second loaded
        register is available immediately - that register comes out of
        the same L1 access as the first.  The local model uses
        N1Write_4c_0Z / N1Write_5c_0Z there (the row's latency, zero
        uops; a76-sog.md), so it has no Latency==0 write.  The gate is
        off for the A76: its SOG documents no zero-latency-MOV section
        (a76-sog.md C-4).
      * Firestorm / Icestorm: WriteZeroFire/WriteZeroIce covers MOVZ,
        MOVN, *MOVK*, FMOVDr, FMOVSr and NOP.  MOVK is a read-modify-
        write of its own destination (it merges a 16-bit field into the
        existing value) and so genuinely carries a dependence; these
        models are left alone here.
      * A720 (cortex-a720/-a720ae): see the A720 paragraph below.  Its
        only Latency==0 write is A720Write_0c, which - like N2's - exists
        solely as the zero-move arm of a SchedWriteVariant.
      * X1 (cortex-x1/-x1c): its SOG sec. 4.15 "Zero Latency MOVs" is
        word for word A78's list (MOV Xd/Wd,#0; MOV Xd/Wd,XZR/WZR;
        MOV Wd,Wn; MOV Xd,Xn), and AArch64SchedNeoverseV1.td, the model
        installed for it, has V1Write_0c as the zero-move arm of one
        SchedWriteVariant, V1Write_0or1c_1I, which two InstRW rows bind
        to MOVZ[WX]i and ORR[WX]rs, as N2 does (x1-sog.md C-4).
        The stock V1 model also has the N1-style marker above -
        V1Write_0c_0Z as the second write of the load pairs - so the X1
        model uses V1Write_{4,5,6}c_0Z there instead (same latency as the
        load, still zero uops); otherwise this gate would make the high
        half of every load pair free.  No instruction of the X1 model
        uses a Latency==0 write other than V1Write_0c.
      * A55 (in-order) is not covered by this gate, and A520 defines
        no Latency==0 write.
```

So the gate is the honest scope of what has actually been verified, not a convenience.

CORTEX-A720.  (The item-by-item comparison of the A720 model with its SOG, including this list, is [a720-sog.md](a720-sog.md); the zero-latency forms are C-4 and D-9 there.)  Arm Cortex-A720 Core Software Optimization Guide (109720, Issue 7.0) sec. 4.12 "Zero Latency Instructions" - note the title is "Instructions", not A78 sec. 4.15's "MOVs" - reads: "A subset of register-to-register move operations, move immediate operations, predicates operations are executed with zero latency.  These instructions do not utilize the scheduling and execution resources of the machine.  These are as follows: MOV Xd,#{12{1'b0},imm[3:0]} / MOV Xd,XZR / MOV Wd,#{12{1'b0},imm[3:0]} / MOV Wd,WZR / MOV Hd,WZR / MOV Hd,XZR / MOV Sd,WZR / MOV Dd,XZR / MOVI Dd,#0 / MOVI Vd.2D,#0 / MOV Wd,Wn / MOV Xd,Xn / FMOV Sd,Sn / FMOV Dd,Dn / MOV Vd,Vn (vector) / MOV Zd.D,Zn.D / PTRUE / PFALSE / SETFFR", with the caveat "The MOV Wd,Wn, MOV Xd,Xn and FMOV Sd,Sn, FMOV Dd,Dn, MOV Vd,Vn (vector), MOV Zd.D,Zn.D instructions may not be executed with zero latency under certain conditions."  That is a strict superset of A78 sec. 4.15's list: A720 adds the GPR<->FP/ASIMD zero moves (MOV Hd/Sd/Dd from WZR/XZR), the vector zero idioms (MOVI Dd,#0 and MOVI Vd.2D,#0), FP/vector register-to-register copies (FMOV Sd,Sn / FMOV Dd,Dn / MOV Vd,Vn), and the SVE predicate forms (MOV Zd.D,Zn.D / PTRUE / PFALSE / SETFFR).  It also narrows the immediate form: A78 says "MOV Xd,#0", A720 says "MOV Xd,#{12{1'b0},imm[3:0]}", i.e. any 16-bit immediate whose top 12 bits are zero (0..15), not just zero.

WHY THE GATE NEEDS THE MODEL DATA.  The gate only lets Facile read a Latency==0 write as "renamed away"; it adds no latency 0 of its own.  cortex-a720/cortex-a720ae resolve to CortexA720Model in AArch64SchedA720.td (the local AArch64Processors.td; stock LLVM maps them to NeoverseN2Model), so the zero-latency forms must be in that model for the gate to have an effect.

THE MODEL DATA. AArch64SchedA720.td carries sec. 4.12 directly (see the commentary there for the opcode-by-opcode derivation):

```text
       def A720Write_0c : SchedWriteRes<[]> { let Latency = 0; }
       def A720Write_0or1c_1I / _0or2c_1V / _0or3c_1M0 : SchedWriteVariant
           selecting A720Write_0c under the A720ZeroMove predicate and
           otherwise the SOG's own non-eliminated write (1c/I for
           MOVZ[WX]i and ORR[WX]rs, 3c/M0 for the GPR-sourced
           FMOV[WX][HSD]r, 2c/V for FMOV[SD]r, MOVID/MOVIv2d_ns and
           ORRv16i8/ORRv8i8);
       InstRW rows binding those to MOVZWi, MOVZXi, ORRWrs, ORRXrs,
           FMOVWHr, FMOVXHr, FMOVWSr, FMOVXDr, MOVID, MOVIv2d_ns,
           ORRv16i8, ORRv8i8, FMOVSr, FMOVDr.
```

A720ZeroMove is deliberately NOT stock LLVM's NeoverseZeroMove: that predicate is Neoverse N2/V1's and is too narrow for A720 in two ways (it requires MOVZ imm == 0, where sec. 4.12 documents imm[3:0], i.e. 0..15; and it has no FMOV Sd,Sn / FMOV Dd,Dn arm at all) and too wide in one (its SVE ORR_ZZZ arm is unreachable under A720's UnsupportedFeatures = SVEUnsupported.F).  AArch64SchedPredNeoverse.td is left untouched so the eight Neoverse models sharing it do not move.

The condition this gate depends on therefore holds for A720 the same way it holds for N2: A720Write_0c is referenced ONLY as the A720ZeroMove arm of those three variants and nowhere else in the file, so within CortexA720Model "Latency == 0" is equivalent to "this is a sec. 4.12 zero-latency form".  None of the newly bound writes is an accounting marker of the N1Write_0c_0Z kind (that one is the second destination of a load pair, where the value genuinely is not available early), and none is a read-modify-write of the Firestorm/Icestorm MOVK kind: every opcode listed above fully overwrites its destination from a source that the rename stage can point at (or from a constant it can fabricate). MOVK is specifically NOT bound, and neither is FMOVHr (FMOV Hd,Hn), which sec. 4.12 does not list.

NOT VALIDATED AGAINST HARDWARE.  This change rests entirely on SOG sec. 4.12 being transcribed correctly, which is what the unit tests in tests/facile_test.cpp assert: that the documented forms carry 0 and the deliberately excluded neighbours (non-ZR ORR, shifted ORR, MOVZ with imm > 15) still carry their normal latency.  Accuracy is NOT claimed.

### C. Macro-op fusion tables (`computeFusedMask`)

Each core has its **own** table (`CpuTraits::Fusion`); there is deliberately no shared "Arm fusion" path. The second instruction of a fused pair consumes no issue slot and no execution-port resource (for "NOP + Any" it is the first).

#### Cortex-A78 (SOG sec. 4.14)

The row-by-row comparison with the SOG, and the open questions about the register forms and the AES pairs, are in [a78-sog.md](a78-sog.md) C-5, D-11, U-6 and U-11.

Cortex-A78 Software Optimization Guide sec. 4.14 "Instruction fusion": specific adjacent AArch64 instruction pairs execute as a single fused operation on Cortex-A78 ("These instruction pairs must be adjacent to each other in program code"). The second instruction of a fused pair consumes no additional issue-width or execution-port resource. Confirmed this is A78-specific: the Cortex-A76 SOG documents no equivalent CMP/CSEL or CMP/B.cond fusion (only unrelated FP fused-multiply-accumulate), so this must not be applied there.

THE SOG's LIST IS NOT A CROSS PRODUCT.  Quoting sec. 4.14 verbatim:

```text
  1. CMP/CMN (immediate) + B.cond      6. CMP (register) + CSET
  2. CMP/CMN (register) + B.cond       7. TST (immediate) + B.cond
  3. CMP (immediate) + CSEL            8. TST (register) + B.cond
  4. CMP (register) + CSEL             9. BICS (register) + B.cond
  5. CMP (immediate) + CSET            10. NOP + Any instruction
```

Note which producer each consumer is paired with.  B.cond accepts all four producers (CMP, CMN, TST, BICS), but CSEL and CSET are listed with CMP ONLY - items 3-6 say "CMP", not "CMP/CMN" the way items 1-2 do, and TST and BICS appear only with B.cond.  Treating the producer set and the consumer set as independent would also fuse CMN+CSEL, TST+CSEL and BICS+CSEL, none of which the SOG lists.  The table (`isA78FusiblePair()` in `src/facile/fusion.cpp`) encodes the pairing exactly as printed.

CMP/CMN/TST are flag-setting SUBS/ADDS/ANDS with a discarded (WZR/XZR) destination; only those forms qualify, not general flag-setting ops with a real destination register.

NOT MODELLED HERE, DELIBERATELY: sec. 4.14's SECOND list ("The following instruction pairs are fused in both Aarch32 and Aarch64 modes: 1. AESE + AESMC, 2. AESD + AESIMC (see Section 4.6 on AES Encryption/Decryption)"). That cross-reference resolves to sec. 4.7 "AES encryption/decryption", which states "Cortex-A78 can issue TWO AESE/AESMC/AESD/AESIMC instruction every cycle (fully pipelined) with an execution latency of two cycles" and "Pairs of dependent AESE/AESMC and AESD/AESIMC instructions exhibit higher performance when they are adjacent in the program code and both instructions use the same destination register".  So the AES "fusion" is a LATENCY / forwarding effect on a dependent pair, not the issue-slot and port elimination that FusedMask expresses: both halves still occupy V-pipe throughput (two AES instructions per cycle, per the quote), so masking the second one out of calculatePortUsageBound() would claim four AES instructions per cycle and contradict the SOG.  Wiring it through this mask would therefore be a modelling error, not a fix.

#### Cortex-X1 (SOG sec. 4.14)

The row-by-row comparison with the SOG and the open questions are in [x1-sog.md](x1-sog.md) C-5, D-11, U-6 and U-11.

Cortex-X1 SOG (PJDOC-466751330-12804 Issue 4.0) "Instruction fusion" prints a STRICT SUBSET of A78 sec. 4.14:

```text
  1. CMP/CMN (immediate) + B.cond     4. TST (register) + B.cond
  2. CMP/CMN (register) + B.cond      5. BICS (register) + B.cond
  3. TST (immediate) + B.cond         6. NOP + Any instruction
```

i.e. A78 items 1,2,7,8,9,10 - the CMP + CSEL and CMP + CSET rows (A78 items 3-6) do NOT exist on the X1.  (The AESE+AESMC / AESD+AESIMC list is the same and is left out for the reason given for the A78.)  The A78 machinery is reused with the select/set consumers switched off; see isA78FusiblePair() in `src/facile/fusion.cpp` and FusionTable::X1 in `src/cpu/cpu_traits.h`.

#### Cortex-A720 (SOG sec. 4.11)

The row-by-row comparison with the SOG and the open questions (immediate restrictions, BTI pairs, AES pairs) are in [a720-sog.md](a720-sog.md) C-5, D-11, U-6, U-10 and U-11.

Arm Cortex-A720 Core Software Optimization Guide (109720, Issue 7.0) sec. 4.11 "Instruction fusion": "Cortex-A720 core can accelerate certain instruction pairs in an operation called fusion.  Specific instruction pairs that can be fused are as follows: [...] These instruction pairs must be adjacent to each other in program code."

WHY THIS IS A SEPARATE TABLE AND NOT `FusionTable::A78` FOR cortex-a720 TOO.  A720 is A78's architectural successor, so widening the A78 gate is the obvious-looking move - and it is wrong.  The two SOGs print DIFFERENT tables.  Side by side, A78 sec. 4.14 vs. A720 sec. 4.11:

```text
  A78 sec. 4.14                      A720 sec. 4.11
  --------------------------------   -------------------------------------
  CMP/CMN (immediate) + B.cond       CMP/CMN (immediate) + B.cond
  CMP/CMN (register)  + B.cond       CMP/CMN (register Rn != ZR) + B.cond
  CMP (immediate) + CSEL             CMP (immediate) + CSEL
  CMP (register)  + CSEL             CMP (register)  + CSEL
  CMP (immediate) + CSET             CMP (immediate) + CSET
  CMP (register)  + CSET             CMP (register)  + CSET
  TST (immediate) + B.cond           TST (immediate) + B.cond
  TST (register)  + B.cond           TST (register)  + B.cond
  BICS (register) + B.cond           BICS ZR (register) + B.cond
  NOP + Any instruction              -- ABSENT --
  -- absent --                       BTI + Integer DP/BR/BLR/RET/B uncond/
                                         CBZ/TBZ
  -- absent --                       SHL + SRI (both scalar or both vector)
  -- absent --                       FCMP + AXFLAG
  AESE + AESMC / AESD + AESIMC       AESE + AESMC / AESD + AESIMC
  -- absent --                       MOVPRFX + supported SVE instruction
```

THREE load-bearing differences, each of which a widened A78 gate gets wrong, all three in the direction of OVER-fusing (i.e. under-predicting cycles):

(1) A720 DOES NOT DOCUMENT "NOP + Any instruction".  A78 sec. 4.14 item 10 has no counterpart anywhere in A720 sec. 4.11.  NOP is common in real code, and reusing A78's table for A720 would silently delete every such NOP's issue slot and port cycles, which A720's SOG never says are free.

(2) A720 restricts the register form of CMP/CMN + B.cond to "Rn != ZR"; A78 prints the same row with no such qualifier.  A CMP/CMN (register) with Rn == ZR followed by B.cond is fusible on A78, NOT fusible on A720.  Note the qualifier is printed on the B.cond row ONLY; the "CMP (register) + CSEL" and "CMP (register) + CSET" rows carry no Rn restriction, so - transcribing exactly as printed, which is the same discipline the A78 table needed - the restriction is applied to the B.cond pairing only.

(3) A720 says "BICS ZR (register)", A78 says "BICS (register)".  The A78 implementation above deliberately accepts a real (non-ZR) destination because A78's SOG names BICS by its own mnemonic with no destination qualifier.  A720 names the destination explicitly, so on A720 the BICS must discard its result into WZR/XZR.  A non-ZR-destination BICS + B.cond is fusible on A78, not on A720.

NOT MODELLED HERE, DELIBERATELY - and why, row by row:

* AESE+AESMC / AESD+AESIMC.  Identical reasoning to the A78 case above, against A720's own text.  Sec. 4.11's cross-reference resolves to A720 sec. 4.6 "AES encryption/decryption": "Cortex-A720 core can issue two AESE/AESMC/AESD/AESIMC instruction every cycle (fully pipelined) with an execution latency of two cycles.  Plus note, pairs of dependent AESE/AESMC and AESD/AESIMC instructions are higher performance when they are adjacent in the program code and both instructions use the same destination register since they are fused".  As on A78, that is a latency/forwarding effect on a dependent pair, not the issue-slot and port elimination FusedMask expresses - both halves still occupy V-pipe throughput, so masking the second out would claim four AES instructions per cycle and contradict sec. 4.6.

* SHL + SRI, and FCMP + AXFLAG.  Both are genuine issue-slot fusions and are modelled (`isA720ShlSriPair`, `isA720FcmpAxflagPair` in `fusion.cpp`; SHL and SRI both scalar or both vector; the second instruction is absorbed): [a720-sog.md](a720-sog.md) C-5.

* BTI + Integer DP/BR/BLR/RET/B uncond/CBZ/TBZ ([a720-sog.md](a720-sog.md) U-11).  BTI is a function-entry landing pad, i.e. normally not inside a loop body, which is the only thing facile is ever asked to analyse.  Modelling it would therefore change almost no prediction this tool can make, while requiring a definition of "Integer DP" that the SOG does not give precisely enough to encode without guessing.  Recorded, not written.

* MOVPRFX + supported SVE instruction.  CortexA720Model in AArch64SchedA720.td declares `UnsupportedFeatures = SVEUnsupported.F`, so SVE instructions carry no scheduling data in this model at all, and MOVPRFX only has a use together with SVE.  Out of scope twice over.

### D. Corrections on top of `NeoverseN2Model` (Cortex-A78)

The A78 borrows LLVM's `NeoverseN2Model`. The local copy `ModifiedTarget/AArch64/AArch64SchedNeoverseN2.td` is installed only for cortex-a78 / -a78ae / -a78c (neoverse-n2, cortex-a710 and cortex-a715 keep libLLVM's stock model), so the places where the N2 rows disagree with the Cortex-A78 SOG are corrected in that file; Facile has no A78-specific port or uop correction. This covers, among others, the register-offset loads and stores (no I µOP except the halfword forms with a scale, and LDRH/LDRSH scaled latency 5), the M0 occupancy of SCVTF/UCVTF, FMOV and DUP from a GPR (one cycle), LDPSW (I and two L-cycles, both registers 5) and the integer divide (best case: latency 5, the divider held 5 cycles). The comparison with the SOG, item by item, and every change are in [a78-sog.md](a78-sog.md) (register-offset accesses and LDPSW: C-12; M0 occupancy: C-15 and C-20; divide: C-10 and D-3).

### E. Front-end bounds: uops and Mops (`calculateIssueBound`)

Two independent front-end bounds. The Software Optimization Guides of the Cortex-A76, A78, A720 and X1 each state a pair of numbers of the form "the dispatch stage can process up to N Mops per cycle and dispatch up to M uops per cycle" (e.g. A76 SOG sec. 4.1 p.41: N = 4, M = 8). Facile counts one Mop (macro-operation) per non-fused instruction (the SOGs: instructions are decoded into Mops, and a Mop can be split into two uops further down the pipeline, e.g. A76 SOG sec. 2.1, A78 SOG sec. 2.1). The issue bound is therefore the larger of (uops of the region) / M and (non-fused instructions of the region) / N. M is the uop dispatch width (`PO.DispatchWidth`, set from `UopDispatchWidth`); N is `MopDispatchWidth`. With M = 2N on all four cores, the Mop bound is the tighter one whenever a region averages fewer than two uops per instruction. `MopDispatchWidth` = 0 skips the Mop bound.

The numbers per core are in `cpu_traits.cpp`: uop dispatch width A76 8, A78 12, A720 10, X1 16; Mops per cycle A76 4, A78 6, A720 5, X1 8. The decode width (A78 4, X1 5) is deliberately not modelled: the tool bounds hot-loop front-end throughput by the rename/dispatch width. Cores that only share a uop width with a modelled core (`cortex-a710`, `cortex-a715`, `neoverse-n2`; `neoverse-v1`) keep no Mop cap because their own SOGs have not been consulted. `neoverse-n1` is the exception: it has the Cortex-A76 entry's widths (8 uops, 4 Mops, from the A76 SOG), although the Neoverse N1's own SOG has not been consulted either. The Apple cores have no Mop cap because `computeFacilePrediction()` already forces `NumMicroOps = 1` for coalesced-ROB CPUs, so their dispatch width is already a Mop-level width.

### F. Stores with a writeback base update (`buildDependencyGraph`)

A pre- or post-index store (`str w1,[x0],#4`, `str w1,[x0,#4]!`, `stp ...,[x0],#8`, `st1 ...,[x0],#16`) writes the base register it reads. The SOGs split a store into an address uOP and a data uOP at dispatch; the base update belongs to the address side, so it is available about one cycle (`WriteAdr`) after the base register, **regardless of when the store data is ready**.

With one node per instruction, the base update would inherit the data dependence: in the in-place update loop

```text
ldr w1, [x0]
sub w1, w1, #10
str w1, [x0], #4      // base update
cmp x0, x21
```

the cycle ldr (4) + sub (1) + str (1) = 6 cycles per iteration would be the precedence bound, although the only real recurrence is the one-cycle base update (the same loop with the update written as a separate `add x0, x0, #4` has no such cycle). Vectorised in-place updates (`ldr q`, arithmetic, `str q,[x],#16`) have the same shape.

Such a store therefore gets a second node. That node reads only the address registers (the base and, for ASIMD post-index forms, the register increment: every register use whose explicit operand index is at or after the tied base operand) and is the producer of the base register; the store's own node keeps all its inputs, so the store data and the address still feed the store itself. The extra nodes are numbered after the instructions, so instruction indices, the port and issue bounds and the memory-dependence edges are unchanged. The node also reads the base register's value from the previous iteration, which keeps the one-cycle base update a recurrence: the loop above has a precedence bound of 1 cycle per iteration on the Cortex-A78 (`mca_tool --facile`). The writeback def is recognised as "a def of a store that the same instruction also reads"; the latency is the one the scheduling model gives that def (`WriteAdr`), not a new constant.

Loads with writeback need no second node (their only use is the base), and neither do stores whose base is updated by a separate `add`.

### G. Memory dependence when the index register changes (`addMemoryDependencies`)

For a store followed by a load with the same base register and a register index (`strh w3,[x0,x4]` ... `ldrsh w5,[x0,x4]`), a memory edge is added when the base register is unchanged in between. If the index register is advanced in between (`add x4,x4,#2`), the load reads the next element, so the SOG's store-to-load forwarding (which needs the load to start at the start or the middle of the older store) cannot apply, and an edge would create a recurrence the hardware does not have (test `FacileTest.MemoryNoEdgeWhenIndexAdvancesPastTheStore`).

The edge is skipped only when the two accesses are **provably disjoint**: both use the same register-offset form (same index register, extend type and scaled/unscaled flag), the only instructions between them that write the index register are `add/sub idx, idx, #imm` (no shift), and the resulting byte distance (immediate sum, times the access size for a scaled index) is at least the store size forward or the load size backward. Different forms, any other kind of index update, an unknown access size or an overlap keep the edge, so nothing that is not statically known is assumed. The loop-carried rule is unchanged. The SOGs give no latency for store-to-load forwarding, only the conditions above; the edge latency (store write + load) is not touched.

### H. FP multiply-accumulate on the A720

The A720 SOG lists FMADD/FMSUB/FNMADD/FNMSUB and ASIMD FMLA/FMLS/FMLAL/FMLSL as "4 (2)" and adds that an FP multiply result is forwarded late to the accumulate operand of a multiply-accumulate (1 cycle after the multiply issues). Without FMA rows, FMADD would behave like an FMUL (latency 3, no early operand read), a 3-cycle accumulate chain against 2 on the A76/A78/X1, whose models carry the same "4 (2)". The A720 model has `A720Wr_FMA` (latency 4) and `A720Rd_FMA` (`SchedReadAdvance<2, [WriteFMul, A720Wr_FMA, A720Wr_VFMUL]>`, the last being the ASIMD FMUL/FMULX write): 2 cycles from a previous multiply-accumulate and 3 - 2 = 1 from an FMUL. See [a720-sog.md](a720-sog.md) C-15 and C-20.

### I. FP divide and square root: latency and occupancy

The SOGs give FP divide and square root a latency and a throughput that depend on the operands for the Neoverse-derived cores (A76/A78/X1: ranges such as "7 to 15" and "1/7 to 2/7" per form) and a fixed value per form for the A720. The model has one latency and one occupancy per `SchedWriteRes`, so a range needs one chosen value. Every form has its own write (`N1WriteFDS_*`, `N2WriteFDS_*`, `V1WriteFDS_*`, `A720Write_*`), listed next to the `InstRW` rows with the SOG values they come from.

* **A76 (N1), A78 (N2), X1 (V1)**: the BEST of each range. Latency: the shortest (7 for the scalar and 2S/4S/2D ASIMD forms; Q-form F16 FDIV 10 and FSQRT 11, where the SOG ranges "10 to 13" and "11 to 13" start). Occupancy: the unit is busy for ceil(1 / best throughput) cycles. For the X1, whose SOG throughput is for the V0/V2 pair, the occupancy of one unit is ceil(2 / best throughput), so the instruction time is the same as on the A76 up to the pair. Examples (A76/A78): scalar H and S 4/7 -> 2, scalar D 2/7 -> 4; ASIMD D-form F32 2/7 -> 4, Q-form F32 and F64 1/7 -> 7, Q-form F16 1/10 (FDIV) and 1/11 (FSQRT) -> 10 and 11. The uOP counts of the A78 writes are those of the N2 writes they replace. On the A76, the A78 and the X1 the occupancy is held on an FP divide unit of its own (`N1UnitFDiv`, `A78UnitFDiv`; `X1UnitFDiv` with two units, one in V0 and one in V2), the pipeline only one cycle ([a76-sog.md](a76-sog.md) D-3, [a78-sog.md](a78-sog.md) D-3, [x1-sog.md](x1-sog.md) D-3).
* **A720**: one latency per form, as in the SOG: scalar H 5, S 7, D 12 and ASIMD D-form F16 8, D-form F32 8, Q-form F16 12, Q-form F32 10, Q-form F64 13, for FDIV and FSQRT alike. The scalar throughput is 1 ("fully pipelined"), so the unit is busy one cycle. The ASIMD table's throughput column (1/4, 1/2, 1/8 -> occupancy 4, 2, 8) contradicts the note under it ("fully pipelined"); the table is used as printed ([a720-sog.md](a720-sog.md) C-15, C-20, I-2).
* **Apple (firestorm, icestorm)**: one latency per form from Dougall Johnson's tables (scalar H/S/D and vector 4H/8H/2S/4S/2D). Port and occupancy are those of the Apple models (there is no SOG for Apple cores).

### J. Loop-carried edges only for loops (`computeFacilePrediction`, `LoopCarried`)

The dependence graph has, besides the edges within one iteration, an edge from the last write of a register to its first read in the NEXT iteration (and the same for a store and a load through memory). That is a recurrence only if the region repeats. For a region that is not a loop (a function body or other straight-line code, `loop` = 0 in the CSV) the edge is false: the value a function returns in `w0` is not the argument `x0` of its next execution, yet the edge, together with the shared register unit of W and X names (4.A (b)), would chain the whole critical path of the region into a recurrence.

`mca_tool` therefore passes `LoopCarried = !ignore` to `computeFacilePrediction`, where `ignore` is the decision the MCA path already takes from `--ignore-loop-carried`: `default` ignores loop-carried dependences in regions that are not loops, `force` ignores them everywhere, `disable` nowhere. Without loop-carried edges the precedence bound is 0 and the estimate of such a region is its issue / port bound, the same quantity the MCA path measures when the iterations overlap. Loops keep their loop-carried edges unless `force` is given.

### K. Constant-register operands and the `InstrBuilder` descriptor cache (`ZeroAwareInstrBuilder`)

`llvm::mca::InstrBuilder` caches the `InstrDesc` of a scheduling class that is not a variant by (opcode, scheduling class id), and a descriptor made from an instruction whose source operand is a constant register (WZR / XZR) has no Read for that operand. With a single `InstrBuilder`, every later instruction of the same key would inherit that descriptor: `mul w1, w1, w2` is `madd w1, w1, w2, wzr`, so a following `madd w1, w3, w4, w1` would lose its accumulator read, and `str wzr, [x0]` followed by `str w9, [x0]` would lose the data read of the second store. The dependences would then depend on which instruction comes first in the region, and the lost reads would remove dependence edges (a variant class such as the MADD of the Cortex-A78, X1 and A720 is cached per instruction and is not affected; on the Cortex-A76 and the Cortex-A55 it is not variant).

`src/mca/zero_aware_instr_builder.h` (`ZeroAwareInstrBuilder`) keeps one `InstrBuilder` per pattern of constant-register operands (bit I set when operand I is WZR / XZR), so an instruction is built by a builder whose cache holds only descriptors made from instructions with the same pattern. It is the builder of `analyzeMcaRegion` (`src/mca/mca_simulation.cpp`), of the Facile path of `mca_tool` (`analyzeFacileRegion` in `apps/mca_main.cpp`; both build through `buildRegionInstructions`), and of the test helpers that build a sequence of instructions (`tests/sog_test_support.h`). The builders live as long as the wrapper, because an `mca::Instruction` refers to the descriptor its builder owns. Tests: `FacileTest.ZeroRegisterDoesNotHideLaterReads`, `FacileTest.MulBeforeMaddKeepsTheAccumulatorDependence`, `A76SogTest.MultiplyThenMultiplyAccumulateKeepsTheChain`.

## 5. Scheduling-Model Plumbing

### A. SchedClass index remapping (`remapSchedClassIndices`)

Re-point MCInstrInfo at the LOCALLY re-generated AArch64 instruction descriptor table, so that MCInstrDesc::SchedClass carries OUR TableGen numbering instead of stock libLLVM's. It is done only for the CPUs that get a local scheduling model (`usesLocalSchedTables()`, i.e. a `CpuModel` other than `None` in `cpu_traits.cpp`); every other CPU keeps libLLVM's stock model and its stock numbering.

WHY THIS IS NEEDED.  The tool mixes two independently generated halves of the same AArch64 target description:

(a) src/frontend/frontend.cpp calls TheTarget->createMCInstrInfo(), i.e. the MCInstrDesc array COMPILED INTO the installed libLLVM-22 (22.1.8).  Its MCInstrDesc::SchedClass values use the numbering that stock AArch64.td produces. (b) installLocalSchedModel() installs the sched-class tables of the local models from the build directory's AArch64GenSubtargetInfo.inc, which this project re-generates from llvm-source's AArch64.td plus ModifiedTarget/AArch64/*.td.  Those files (AArch64SchedIcestorm.td, AArch64SchedFirestorm.td, and the modified A55/N1/N2/V1/A520/A720 models) declare additional InstRW rules, and every InstRW that partitions an existing SchedClass makes TableGen CREATE new SchedClass records.  The two numberings therefore DIVERGE: indexing table (b) with the indices of table (a) silently returns ANOTHER instruction's latency and port assignment (stock's largest per-opcode SchedClass index stays below the local NumSchedClasses, so nothing reads out of bounds).

Extent: both tables have the same 9135 AArch64 opcodes in the same order; the SchedClass differs for a large share of them, and the share changes with every InstRW added to or removed from a local model.  The local class list is the longer one (stock libLLVM 22.1.8: per-opcode indices 0-2049, `NumSchedClasses` 2304 including the variant classes; the local `NumSchedClasses` is larger and also changes with every InstRW).

WHY THE WHOLE TABLE, and not just the SchedClass field.  Two reasons:

1. A per-class translation (rewriting table (b) into stock numbering) is impossible in principle.  The two numberings are not a shift, they are DIFFERENT PARTITIONS of the instruction set: the local InstRW rules exist precisely to split stock classes, so one stock class index can correspond to several local ones and cannot carry their differing latencies.  The remap has to be per-OPCODE.
2. A per-opcode copy of the descriptors (std::vector<MCInstrDesc> with the SchedClass field overwritten) is also impossible: MCInstrDesc locates its own operand and implicit-operand arrays from its OWN ADDRESS - MCInstrDesc::operands() is `reinterpret_cast<const MCOperandInfo *>(this + Opcode + 1)` plus `OpInfoOffset` (MCInstrDesc.h of LLVM 22), which only resolves inside the generated <Target>InstrTable struct, where Insts[] is stored in DESCENDING opcode order and is immediately followed by ImplicitOps[] and OperandInfo[].  Copied descriptors send operands() into unrelated memory.

So the local AArch64Descs are installed wholesale, via the InitAArch64MCInstrInfo() that TableGen emits alongside them - exactly the call libLLVM's own createMCInstrInfo() makes, just with the local table.  That keeps Insts / ImplicitOps / OperandInfo mutually consistent, and makes SchedClass agree with the sched-class tables installed by installLocalSchedModel().

SAFETY of swapping in the table generated from llvm-source while linking libLLVM-22.1.8.  Compared field by field over all 9135 opcodes, the local table against libLLVM's: SchedClass is the ONLY field that differs.

```text
  SchedClass            differs for a large share of the opcodes (see above)
  Flags 0    TSFlags 0    NumOperands 0    NumDefs 0    Size 0
  NumImplicitUses 0    NumImplicitDefs 0    instruction names 0
```

So this is a pure SchedClass correction: it cannot change mayLoad / mayStore / isBranch / operand shape, and any change in simulation output is attributable to the sched-class mapping alone.  The opcode-count guard in remapSchedClassIndices() refuses the swap outright if a future libLLVM no longer has the same number of opcodes.

The remap makes the instruction table agree with the installed scheduling tables; it is a matter of correctness, not of accuracy.

### B. Per-CPU parameters (`cpu_traits.cpp`)

Facts about a core that LLVM's scheduling model cannot express live in one table, `getCpuTraits()`: which local `.td` model is installed, the SOG's uop and Mop dispatch widths, which fusion table applies, and whether the model's zero-latency writes are rename-time moves. Adding a core or a variant of one is a single line there.
