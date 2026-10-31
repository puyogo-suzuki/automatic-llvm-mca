# automatic-llvm-mca Technical Documentation

This document provides an exhaustive technical overview of the processor scheduling model custom adaptations, the C++ simulation pipeline modifications, and the static Memory Level Parallelism (MLP) calculation methodology implemented here. The Cortex-A55 model and its conformance to the Arm Cortex-A55 Software Optimization Guide are described separately in [a55-sog.md](a55-sog.md), the Cortex-A520 and Cortex-A76 ones in [a520-sog.md](a520-sog.md) and [a76-sog.md](a76-sog.md); [README.md](README.md) lists the documents.

---

## 1. Processor Scheduling Model Customizations & Forwarding Paths

The scheduling models are customized at the **TableGen** description level (`ModifiedTarget/AArch64/*.td`, compiled into the tool's own tables, see section 5) and, for the in-order Cortex-A55, in the **C++ runtime simulation layer** (`src/mca/mca_simulation.cpp`, `src/a55/`).

The Cortex-A55 material that used to be here is now in [a55-sog.md](a55-sog.md), which also compares each part of the A55 model with the Cortex-A55 Software Optimization Guide. (It was the former sections 1.A, the TableGen model `AArch64SchedA55.td`, and 1.B, the A55 issue stage, the flag and pointer bypasses and the other C++ corrections.) The remaining sections of this chapter were renumbered: the former 1.C is now 1.A and the former 1.D is now 1.B.

---

### A. Neoverse Architecture Validations (N1, N2, V1)

We verified the scheduling models for high-performance out-of-order Neoverse cores (`NeoverseN1.td`, `NeoverseN2.td`, `NeoverseV1.td`).
*   Unlike the Cortex-A55, these architectures utilize the TableGen predicate `IsCheapLSL` (which checks if the shift type is LSL and the shift amount is $\le 4$).
*   If the shift predicate is satisfied (or there is no shift), the scheduler variant maps the instruction to a 1-cycle latency pipeline (e.g. `N1Write_1c_1I`, `N2Write_1c_1I`, `V1Write_1c_1I`).
*   Otherwise, it falls back to a 2-cycle latency pipeline (e.g. `N1Write_2c_1M`, `N2Write_2c_1M`, `V1Write_2c_1M`).
*   Since out-of-order cores have large instruction window buffers (`MicroOpBufferSize > 0`), speculative branch bubbles are naturally absorbed, meaning no manual branch latency overrides are necessary for Neoverse targets.

---

### B. Cortex-X1 Model (`NeoverseV1Model`)

The Cortex-X1 runs on `AArch64SchedNeoverseV1.td`. That file is compiled into the tool's own tables and installed **only** for `cortex-x1` / `cortex-x1c` (`neoverse-v1` keeps libLLVM's stock model), so its machine parameters are the X1's:

| Parameter | Value | Where |
|---|---|---|
| Decode width | 5 | not modelled |
| Alloc/rename width | 8 Mops/cycle | `MopDispatchWidth` in `cpu_traits.cpp` (X1 SOG sec. 4.1: "up to 8 MOPs per cycle") |
| Issue (dispatch) width | 16 uops/cycle | `IssueWidth` in the `.td`, and `UopDispatchWidth` in `cpu_traits.cpp` |
| ROB | 224 | `MicroOpBufferSize` (also the MLP window size) |
| Scheduler entries | 150 | not modelled |

The decode width and the scheduler size are not in the SOG; they were supplied by the project owner. They are not modelled for the same reasons as for the A78 (whose 4-wide decode and scheduler size are likewise absent from `AArch64SchedNeoverseN2.td`): neither `MCSchedModel` nor MCA has a decode stage, and the tool bounds hot-loop front-end throughput by the rename/dispatch width. MCA's only handle for scheduler entries is a per-`ProcResource` `BufferSize`, i.e. one queue per pipe or group, and nothing says how the 150 entries are split over the fifteen issue pipelines; inventing a split would add stalls the hardware need not have, so the queues stay unbounded and the ROB is the only window limit.

**Differences from the stock V1 model that follow the X1 SOG** (Software Optimization Guide table numbers in parentheses):

* AES (`AESD/AESE/AESMC/AESIMC`): V01, 2 per cycle (Table 3-34). Stock V1 uses the 4-wide V group.
* `LDPSW`: one I uop and two L slots, 1.5 per cycle (Table 3-12). Expressed as occupancy so `NumMicroOps` stays 2.
* `FCSEL`: V02 (Table 3-16). `FCVT[AMNPZ][SU]` FP to GPR: V02 but one per cycle (Table 3-18), expressed as a two-cycle occupancy of the V02 group.
* `SDOT/UDOT`: latency 2 with accumulator forwarding 1 (Table 3-24).
* `SQSHRUN`: the stock regular expression `^SQSHU?RNv` cannot match `SQSHRUN`; it now falls in the "shift by immed, complex" row (V13, latency 4).
* `TBL` (1-2 table registers) and `TBX` (1 table register): a single V01 uop, 2 per cycle (Table 3-28).
* `PMULL` (64x64): V01, latency 2, 2 per cycle. `SHA1SU0` / `SHA256SU1` three-register forms: V0 (the stock regular expressions only matched the two-register forms).
* High half of `LDPW`/`LDPSW`/`LDP[SD]`: latency of the load itself, zero uops (`V1Write_{4,5,6}c_0Z`). Stock V1 uses `V1Write_0c_0Z` (latency 0), which cannot be told apart from a rename-time zero-latency MOV (see section 4.B) and would make the high half free.

Facile-side X1 behaviour: the fusion table is the A78 table without the CMP + CSEL/CSET rows, and the zero-latency MOV list is the A78's (sections 4.B and 4.C). The A78-specific corrections of section 4.D are not needed because V1 does not have those defects.

The model was compared against the X1 SOG tables with `tools/sog_compare.py` (see the README). Differences that were left as they are: `SMADDL`-family throughput (the SOG prints throughput 2 for the single M0 pipe), `LD4` Q-form throughput (SOG 1/2, model about 3/4 per cycle), `EXTR` with two distinct source registers (SOG: 3 cycles, I and M; model: 1 cycle, I), and the data-side pipes of ASIMD stores (the SOG table says V, its Table 2-1 says only V0/V1 take store data; the model follows Table 2-1).

## 2. LLVM MCA Simulation & Steady-State Estimation

Calculating representative cycle metrics requires isolating the stable, recurring phase of execution (the steady-state) from transient startup effects.

```
[ Cold Pipeline ] ----> [ Warm-up Phase ] ----> [ Transition ] ----> [ Steady-State Phase ]
                        (First N Iterations)                         (Remaining Iterations)
                                                                     * Only this phase is measured
```

### A. Warm-up Iterations Calculation
To completely fill the reservation stations, reorder buffers, and pipeline stages:
1.  We query the subtarget's maximum inflight instruction capability (`getWarmupWindowSize`):
    *   For out-of-order cores, this corresponds to the Reorder Buffer size (`MicroOpBufferSize` or `LoopMicroOpBufferSize`).
    *   For in-order cores, it defaults to the processor `IssueWidth`.
2.  We calculate the minimum number of loop iterations required to reach this occupancy threshold (`computeWarmupIterations`):
    $$\text{WarmupIterations} = \max\left(1, \frac{\text{WarmupWindowSize}}{\text{RegionInstructionCount}}\right)$$
3.  The remaining iterations specified by the `--iterations` parameter are designated as the `SteadyIterations`.

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
*   **Timing of the reset (#12).** The Dispatched event is raised by the issue stage only after the instruction has passed its register hazard check. Resetting the mappings there is in time for every instruction of an iteration except the first one, whose own check still saw the previous iteration's producers; a basic block whose first instruction depended on the previous iteration was therefore simulated as if the dependency were kept (e.g. `fadd d0, d0, d1; add; add`: 4.0 cycles/iteration under both `--ignore-loop-carried=default` and `disable`, against 2.01 with the `fadd` in second position; measured with `--iterations=200`). For the Cortex-A55, whose issue stage is the project's own, the reset is now performed before each hazard check (`LoopCarriedIgnoreSteadyStateTracker::beforeRegisterHazardCheck()`; see [a55-sog.md](a55-sog.md#22-issue-stage-srca55a55_issue_stagecpp)). The Cortex-A520, which used the stock `llvm::mca::InOrderIssueStage` (no such hook), now has the same call in its own copy of that stage (since 2026-10-31; `fadd d0, d0, d1; fadd d2, d3, d1` as a basic block: 4.0 -> 1.0 cycles/iteration; see [a520-sog.md](a520-sog.md#22-issue-stage-srca520a520_issue_stagecpp)). **Not fixed:** the out-of-order pipeline.

### D. Basic Block Partitioning under Size Limits
To prevent long instruction sequences from being completely skipped when size limits are applied:
*   If a loop exceeds the maximum loop size limit (`loop-max-instrs`), it is excluded from the loop analysis but its instructions are fallback-evaluated as basic blocks.
*   If a basic block exceeds the maximum basic block size limit (`bb-max-instrs`), instead of ignoring the block, the analyzer dynamically splits it into multiple contiguous sub-blocks (chunks) of size `bb-max-instrs` (with the final chunk containing the remainder). This ensures that every instruction in the binary is covered by the performance evaluation.

### E. CFG and Post-Dominator Based Region Splitting
Instead of simple linear scanning, we build a Control Flow Graph (CFG) of each function to perform precise analysis:
1.  **CFG Construction**: We divide the instruction vector into disjoint Basic Blocks (BBs) based on branch targets and terminators (e.g. conditional branches, returns, size limits). We then add directed edges between BBs to construct the CFG.
2.  **Natural Loop Detection**: We run DFS to find back-edges. For each back-edge `U -> V` where `V` is an active ancestor, we collect all nodes reachable from `U` to `V` without going through `V` to identify the Natural Loop nodes. We build a Loop Nest Tree to filter loops by nesting depth (`depth < nestLimitOuter`) and height (`height < nestLimitInner`).
3.  **Post-Dominator Analysis**: We compute the Immediate Post-dominator (`ipdom`) relation for all CFG nodes using the Lengauer-Tarjan algorithm on the reverse CFG (starting from a virtual exit node).
4.  **Loop Merging**: If a basic block node `U` is not part of a valid loop but is post-dominated by a loop's header, it is classified as a pre-header or post-exit block that will always execute in tandem with the loop. Rather than processing it independently, `U` is merged into the loop, and is not emitted as a standalone basic block. This eliminates redundant basic block simulations.

### F. Instructions rejected by InstrBuilder
An instruction whose scheduling class cannot be modelled (e.g. a `SchedWrite` marked `Unsupported`) is left out of the simulated sequence. This is no longer silent: `analyzeMcaRegion` (and the `--facile` path of `mca_tool`) count each drop per opcode, and `mca_tool` / `mlp-objdump` print a `[WARN]` summary to stderr once the (stderr-silenced) analysis is over. The CSV output is unchanged.

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
    *   **Call-Instruction Invalidation**: If a function call instruction (`bl`) is encountered in the window, we assume all register dependencies and active cache line tracking states are invalidated (since the callee may modify return registers and memory states).
3.  **No-Load Default Specification**:
    If a basic block or loop region contains no eligible memory load instructions (or is an empty instruction sequence), both $MLP$ and $MLP_R$ values default to $1.0$ (representing the sequential memory baseline).

## 4. Facile: SOG-Derived Modelling Notes

Rationale that used to sit in comments in `src/facile/`. Each subsection names the function it belongs to. "SOG" is the Arm Software Optimization Guide of the core in question; the guides themselves are not part of this repository. Per-CPU switches (which core gets which behaviour) are in `cpu_traits.cpp`.

### A. Dependence-edge latency (`getEdgeLatency`)

Latency to put on a RAW dependency edge  (Writer defines Reg) -> (Reader reads Reg through use slot Use).

Refines two coarse approximations that the plain getInstLatency(Writer) form made.  Both are cases where the scheduling model already carries the right number and facile was simply not reading it.

(a) PER-DEFINITION latency.  mca::Instruction::getLatency() is InstrDesc::MaxLatency, i.e. the MAXIMUM over all of an instruction's writes, so every out-edge of a multi-def instruction was charged the slowest one.  AArch64 pre/post-index memory ops are exactly that shape: `ldr x2, [x1], #8` is Sched<[WriteAdr, WriteLD]> - 1c for the x1 base update, 4c for the x2 load result - so the base-pointer recurrence that every post-increment / stride loop carries (x1 -> x1 across the backedge, distance 1) was being given the LOAD's 4c and produced a precedence bound of 4 cycles/iteration for a loop whose address recurrence really closes in 1.  WriteState::getLatency() is that individual write's own latency.

(b) READADVANCE (operand bypass / late forwarding).  A scheduling model states, per (consumer sched class, consumer operand slot, producer write-resource), how many cycles earlier than write-back that operand is actually needed.  MCA's own RegisterFile::addRegisterRead applies it via WriteState::addUser(IID, Use, ReadAdvance), giving an effective dependence latency of WS.getLatency() - ReadAdvance.  facile never consulted it, so all operands of a multi-input instruction were assumed to need the producer's result at the same (worst) cycle - the textbook way to over-predict a multiply-accumulate recurrence.

The case that actually matters here is the ACCUMULATOR of a multiply-accumulate.  Arm Cortex-A78 Software Optimization Guide (r1p2, PJDOC-466751330-9691) Table 3-24 lists

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

(AArch64SchedNeoverseN1.td, used for cortex-a76, is identical.)  So an FMADD -> FMADD accumulator recurrence - the shape of every FP reduction, dot product and stencil accumulation in real FP workloads - really closes in 2 cycles, and facile was calling it 4: a clean 2x over-prediction of the precedence bound on the exact loops where the precedence bound is the binding constraint.

Apple's two models declare ReadAdvance 0 everywhere, so (b) is a no-op for icestorm/firestorm today; it is what makes it POSSIBLE to encode the forwarding paths Dougall Johnson documents for them ("MADD's output can be passed to its third operand (the addend) with 1c latency, but if it's chained with other instructions it has 3c latency", "Loads may be passed to the base address of other loads with 3c latency ... but chaining with ALU operations gives a latency of 4c"), which are per-operand facts that no WriteRes latency can express.

The 1-cycle floor is kept from getInstLatency(): a dependent operation can never issue in the same cycle as its producer in this model, and MCA's own zero-latency writes (register-renamed MOVs) already relied on it.

### B. Zero-latency MOVs (`isZeroLatencyMovTarget`, `getEdgeLatency`)

Gate: `CpuTraits::ZeroLatencyMov` (A78 family, A720 family, X1 family).

TRUE ZERO-LATENCY WRITES (rename-time move elimination / zero idioms).  Arm Cortex-A78 SOG sec. 4.15 "Zero Latency MOVs": "A subset of register-to-register move operations and move immediate operations are executed with zero latency.  These instructions do not utilize the scheduling and execution resources of the machine. These are as follows: MOV Xd,#0 / MOV Xd,XZR / MOV Wd,#0 / MOV Wd,WZR / MOV Rd,#0 (AArch32) / MOV Wd,Wn / MOV Xd,Xn / MOV Rd,Rn (AArch32)".  Such a MOV is resolved by pointing the destination's rename entry at the source's physical register, so a consumer of the MOV's result reads the SAME physical register the producer wrote and waits exactly zero additional cycles.

The scheduling model ALREADY carries this and facile was, again, simply not reading it.  AArch64SchedNeoverseN2.td (the model installed for cortex-a78, see AArch64Processors.td) has

```text
       def N2Write_0c : SchedWriteRes<[]> { let Latency = 0; }
       def N2Write_0or1c_1I : SchedWriteVariant<[
             SchedVar<NeoverseZeroMove, [N2Write_0c]>,
             SchedVar<NoSchedPred,      [N2Write_1c_1I]>]>;
       def : InstRW<[N2Write_0or1c_1I], (instregex "^MOVZ[WX]i$")>;
       def : InstRW<[N2Write_0or1c_1I], (instregex "^ORR[WX]rs$")>;
```

and AArch64SchedPredNeoverse.td's NeoverseZeroMove enumerates precisely sec. 4.15's list (MOVZ[WX]i with both immediates zero, i.e. MOV Wd/Xd,#0; ORR[WX]rs with Rn == ZR and shift 0, which is how MOV Wd,WZR / MOV Xd,XZR / MOV Wd,Wn / MOV Xd,Xn all encode). SchedWriteRes<[]> is also exactly the SOG's "do not utilize the scheduling and execution resources" - no ProcResource is listed - and calculatePortUsageBound() already gets that part right, because resolveSchedClass() resolves the variant.  Only the dependence latency was being lost, to the max(Lat, 1.0) floor below.

The floor is right for every other write (a dependent operation cannot issue in the same cycle as a producer that really has to compute something), but a zero-latency MOV is not computed at all: there is no producer to wait for.  Charging it a cycle inflates every dependence chain that passes through a register copy. WHY THIS IS GATED ON THE TARGET AND NOT JUST ON "Latency == 0". A zero-latency write means "renamed away" only in the N2 model. Checked every other model this tool uses, and a blanket rule would have been wrong in two of them:

```text
      * N2 (cortex-a78): N2Write_0c appears at exactly three places,
        all three of them the NeoverseZeroMove arm of a
        SchedWriteVariant.  Here Latency==0 <=> SOG sec. 4.15 move.
      * N1 (cortex-a76): N1Write_0c_0Z (Latency 0, NumMicroOps 0) was
        used as the SECOND write of LDPWi / LDNPWi / LDPSWi, i.e. the
        second destination register of a load pair.  It was a micro-op
        accounting marker, NOT a statement that the second loaded
        register is available immediately - that register comes out of
        the same 4-cycle L1 access as the first.  Since 2026-10-31 these
        writes are N1Write_4c_0Z / N1Write_5c_0Z (the row's latency, zero
        uops; docs/a76-sog.md C-11), so the N1 model has no Latency==0
        write left.  The gate stays off for the A76 anyway: its SOG
        documents no zero-latency-MOV section (a76-sog.md C-4).
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
        installed for it, has V1Write_0c as the zero-move arm of two
        SchedWriteVariants (MOVZ[WX]i, ORR[WX]rs) exactly as N2 does.
        V1 ALSO has the N1-style trap above - V1Write_0c_0Z, the
        Latency-0 second write of LDPW/LDPSW/LDP[SD] - so the X1 model
        replaces it with V1Write_{4,5,6}c_0Z (same latency as the load,
        still zero uops); without that edit this gate would have made the
        high half of every load pair free.  V1Write_0c is now the only
        Latency==0 write in the model.
      * A55 (in-order) is not covered by this gate, and A520 defines
        no Latency==0 write.
```

So the gate is the honest scope of what has actually been verified, not a convenience.

CORTEX-A720.  This was a known, unfixed gap until the sec. 4.12 data was added to AArch64SchedA720.td; the paragraphs below record what was missing, what was added, and why the gate can now honestly include it. The gap was never "the core has no zero-latency moves" - that was a fact about this project's SCHEDULING MODEL, not about the core.  A720 does have zero-latency moves, and its SOG documents MORE of them than A78's does.  Arm Cortex-A720 Core Software Optimization Guide (109720, Issue 7.0) sec. 4.12 "Zero Latency Instructions" - note the title is "Instructions", not A78 sec. 4.15's "MOVs" - reads: "A subset of register-to-register move operations, move immediate operations, predicates operations are executed with zero latency.  These instructions do not utilize the scheduling and execution resources of the machine.  These are as follows: MOV Xd,#{12{1'b0},imm[3:0]} / MOV Xd,XZR / MOV Wd,#{12{1'b0},imm[3:0]} / MOV Wd,WZR / MOV Hd,WZR / MOV Hd,XZR / MOV Sd,WZR / MOV Dd,XZR / MOVI Dd,#0 / MOVI Vd.2D,#0 / MOV Wd,Wn / MOV Xd,Xn / FMOV Sd,Sn / FMOV Dd,Dn / MOV Vd,Vn (vector) / MOV Zd.D,Zn.D / PTRUE / PFALSE / SETFFR", with the caveat "The MOV Wd,Wn, MOV Xd,Xn and FMOV Sd,Sn, FMOV Dd,Dn, MOV Vd,Vn (vector), MOV Zd.D,Zn.D instructions may not be executed with zero latency under certain conditions."  That is a strict superset of A78 sec. 4.15's list: A720 adds the GPR<->FP/ASIMD zero moves (MOV Hd/Sd/Dd from WZR/XZR), the vector zero idioms (MOVI Dd,#0 and MOVI Vd.2D,#0), FP/vector register-to-register copies (FMOV Sd,Sn / FMOV Dd,Dn / MOV Vd,Vn), and the SVE predicate forms (MOV Zd.D,Zn.D / PTRUE / PFALSE / SETFFR).  It also narrows the immediate form: A78 says "MOV Xd,#0", A720 says "MOV Xd,#{12{1'b0},imm[3:0]}", i.e. any 16-bit immediate whose top 12 bits are zero (0..15), not just zero.

WHY THE GATE COULD NOT SIMPLY BE WIDENED.  The A78 fix worked because the data was already in the model and facile was merely not reading it: AArch64SchedNeoverseN2.td defines N2Write_0c and gates it on the NeoverseZeroMove predicate, so `Lat == 0.0` is a reliable signal and the only change needed was to stop applying the max(Lat,1.0) floor. A720 was the opposite case.  cortex-a720/cortex-a720ae resolve to CortexA720Model in AArch64SchedA720.td (AArch64Processors.td:1448- 1450), which was a hand-written WriteRes-only model: ZERO InstRW entries, no SchedWriteRes<[]>, no `Latency = 0` anywhere, and no zero-move predicate.  So `Lat == 0.0` could never be true for A720, and adding "cortex-a720" to isZeroLatencyMovTarget() on its own would have been pure dead code - it would have READ as a fix while changing nothing.

WHAT WAS ADDED, AND WHY THE GATE IS NOW CORRECT. AArch64SchedA720.td now carries sec. 4.12 directly (see the long commentary there for the opcode-by-opcode derivation):

```text
       def A720Write_0c : SchedWriteRes<[]> { let Latency = 0; }
       def A720Write_0or1c_1I / _0or2c_1V / _0or3c_1M0 : SchedWriteVariant
           selecting A720Write_0c under a new A720ZeroMove predicate and
           otherwise the SOG's own non-eliminated write (1c/I for
           MOVZ[WX]i and ORR[WX]rs, 3c/M0 for the GPR-sourced
           FMOV[WX][HSD]r, 2c/V for FMOV[SD]r, MOVID/MOVIv2d_ns and
           ORRv16i8/ORRv8i8);
       InstRW rows binding those to MOVZWi, MOVZXi, ORRWrs, ORRXrs,
           FMOVWHr, FMOVXHr, FMOVWSr, FMOVXDr, MOVID, MOVIv2d_ns,
           ORRv16i8, ORRv8i8, FMOVSr, FMOVDr.
```

A720ZeroMove is deliberately NOT stock LLVM's NeoverseZeroMove: that predicate is Neoverse N2/V1's and is too narrow for A720 in two ways (it requires MOVZ imm == 0, where sec. 4.12 documents imm[3:0], i.e. 0..15; and it has no FMOV Sd,Sn / FMOV Dd,Dn arm at all) and too wide in one (its SVE ORR_ZZZ arm is unreachable under A720's UnsupportedFeatures = SVEUnsupported.F).  AArch64SchedPredNeoverse.td is left untouched so the eight Neoverse models sharing it do not move.

The condition this gate depends on therefore holds for A720 the same way it holds for N2, and was re-checked rather than assumed: A720Write_0c is referenced ONLY as the A720ZeroMove arm of those three variants and nowhere else in the file, so within CortexA720Model "Latency == 0" is equivalent to "this is a sec. 4.12 zero-latency form".  None of the newly bound writes is an accounting marker of the N1Write_0c_0Z kind (that one is the second destination of a load pair, where the value genuinely is not available early), and none is a read-modify-write of the Firestorm/Icestorm MOVK kind: every opcode listed above fully overwrites its destination from a source that the rename stage can point at (or from a constant it can fabricate). MOVK is specifically NOT bound, and neither is FMOVHr (FMOV Hd,Hn), which sec. 4.12 does not list.

NOT VALIDATED AGAINST HARDWARE.  This change rests entirely on SOG sec. 4.12 being transcribed correctly, which is what the unit tests in tests/facile_test.cpp assert: that the documented forms carry 0 and the deliberately excluded neighbours (non-ZR ORR, shifted ORR, MOVZ with imm > 15) still carry their normal latency.  Accuracy is NOT claimed.

### C. Macro-op fusion tables (`computeFusedMask`)

Each core has its **own** table (`CpuTraits::Fusion`); there is deliberately no shared "Arm fusion" path. The second instruction of a fused pair consumes no issue slot and no execution-port resource (for "NOP + Any" it is the first).

#### Cortex-A78 (SOG sec. 4.14)

Cortex-A78 Software Optimization Guide sec. 4.14 "Instruction fusion": specific adjacent AArch64 instruction pairs execute as a single fused operation on Cortex-A78 ("These instruction pairs must be adjacent to each other in program code"). The second instruction of a fused pair consumes no additional issue-width or execution-port resource. Confirmed this is A78-specific: the Cortex-A76 SOG documents no equivalent CMP/CSEL or CMP/B.cond fusion (only unrelated FP fused-multiply-accumulate), so this must not be applied there.

THE SOG's LIST IS NOT A CROSS PRODUCT.  Quoting sec. 4.14 verbatim:

```text
  1. CMP/CMN (immediate) + B.cond      6. CMP (register) + CSET
  2. CMP/CMN (register) + B.cond       7. TST (immediate) + B.cond
  3. CMP (immediate) + CSEL            8. TST (register) + B.cond
  4. CMP (register) + CSEL             9. BICS (register) + B.cond
  5. CMP (immediate) + CSET            10. NOP + Any instruction
```

Note which producer each consumer is paired with.  B.cond accepts all four producers (CMP, CMN, TST, BICS), but CSEL and CSET are listed with CMP ONLY - items 3-6 say "CMP", not "CMP/CMN" the way items 1-2 do, and TST and BICS appear only with B.cond.  The previous implementation treated the producer set and the consumer set as independent and so also fused CMN+CSEL, TST+CSEL and BICS+CSEL, none of which the SOG lists.  The table below encodes the pairing exactly as printed.

CMP/CMN/TST are flag-setting SUBS/ADDS/ANDS with a discarded (WZR/XZR) destination; only those forms qualify, not general flag-setting ops with a real destination register.

NOT MODELLED HERE, DELIBERATELY: sec. 4.14's SECOND list ("The following instruction pairs are fused in both Aarch32 and Aarch64 modes: 1. AESE + AESMC, 2. AESD + AESIMC (see Section 4.6 on AES Encryption/Decryption)"). That cross-reference resolves to sec. 4.7 "AES encryption/decryption", which states "Cortex-A78 can issue TWO AESE/AESMC/AESD/AESIMC instruction every cycle (fully pipelined) with an execution latency of two cycles" and "Pairs of dependent AESE/AESMC and AESD/AESIMC instructions exhibit higher performance when they are adjacent in the program code and both instructions use the same destination register".  So the AES "fusion" is a LATENCY / forwarding effect on a dependent pair, not the issue-slot and port elimination that FusedMask expresses: both halves still occupy V-pipe throughput (two AES instructions per cycle, per the quote), so masking the second one out of calculatePortUsageBound() would claim four AES instructions per cycle and contradict the SOG.  Wiring it through this mask would therefore be a modelling error, not a fix.

#### Cortex-X1 (SOG "Instruction fusion")

Cortex-X1 SOG (PJDOC-466751330-12804 Issue 4.0) "Instruction fusion" prints a STRICT SUBSET of A78 sec. 4.14:

```text
  1. CMP/CMN (immediate) + B.cond     4. TST (register) + B.cond
  2. CMP/CMN (register) + B.cond      5. BICS (register) + B.cond
  3. TST (immediate) + B.cond         6. NOP + Any instruction
```

i.e. A78 items 1,2,7,8,9,10 - the CMP + CSEL and CMP + CSET rows (A78 items 3-6) do NOT exist on the X1.  (The AESE+AESMC / AESD+AESIMC list is the same and is left out for the reason given for the A78.)  The A78 machinery is reused with the select/set consumers switched off; see isA78FusiblePair() and FusionTable::X1 in cpu_traits.h.

#### Cortex-A720 (SOG sec. 4.11)

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

* SHL + SRI, and FCMP + AXFLAG.  Both are genuine issue-slot fusions and would be correct to model, but are not implemented because no test exercises them (AXFLAG is Armv8.5 FEAT_FlagM2).  They are recorded here rather than written.  To add them, add them as SHL->SRI (matched on both operands being scalar, or both vector) and FCMP->AXFLAG rows in isA720FusiblePair().

* BTI + Integer DP/BR/BLR/RET/B uncond/CBZ/TBZ.  BTI is a function-entry landing pad, i.e. normally not inside a loop body, which is the only thing facile is ever asked to analyse.  Modelling it would therefore change almost no prediction this tool can make, while requiring a definition of "Integer DP" that the SOG does not give precisely enough to encode without guessing.  Recorded, not written.

* MOVPRFX + supported SVE instruction.  CortexA720Model in AArch64SchedA720.td declares `UnsupportedFeatures = SVEUnsupported.F`, so SVE instructions carry no scheduling data in this model at all, and MOVPRFX only has a use together with SVE.  Out of scope twice over.

### D. Corrections on top of `NeoverseN2Model` (Cortex-A78)

The A78 borrows `NeoverseN2Model`. Where it disagrees with the A78 SOG, `src/facile/n2_corrections.cpp` and `calculatePortUsageBound()` correct the port usage after the fact (`CpuTraits::N2ModelCorrections`).

#### Register-offset loads and stores (`isA78RegOffsetNoAlu`)

A78 SOG correction: integer register-offset loads/stores on Cortex-A78 do NOT use an integer (I) pipe.

cortex-a78 borrows NeoverseN2Model (AArch64Processors.td), whose

```text
  SchedAlias<WriteLDIdx, N2Write_4c_1I_1L>        (every reg-offset load)
  SchedAlias<WriteSTIdx, N2Write_1c_1L01_1D_1I>   (every reg-offset store)
```

charge one extra N2UnitI uop to EVERY integer register-offset access. The Arm Cortex-A78 Core SOG (PJDOC-466751330-9691 Issue 4.0, Table 3-12/3-14) says otherwise:

```text
  Load register, register offset, basic            4  3  L
  Load register, register offset, scale by 4/8     4  3  L
  Load register, register offset, scale by 2       5  3  I, L   (LDRH/LDRSH)
  Load register, register offset, extend           4  3  L
```

Load register, register offset, extend, sc. 4/8  4  3  L

```text
  Load register, register offset, extend, sc. 2    5  3  I, L   (LDRH/LDRSH)
```

Store register, register offset, basic/scaled 4/8/extend  1 2 L01, D

```text
  Store register, register offset, scaled by 2 (STRH)       2 2 I, L01, D
```

i.e. only the halfword forms WITH a shift need the I uop. (Cortex-A76's NeoverseN1Model already has WriteLDIdx = 1L, WriteSTIdx = 1L_1D, matching the A76 SOG, which is why only the A78 is affected; since 2026-10-31 it also gives the halfword forms scaled by 2 their I uop, see [a76-sog.md](a76-sog.md) C-11 and C-12.) Consequence of the bug: integer-ALU-heavy loops with many indexed accesses (e.g. a 59-instruction loop with 17 LDRWroX + 10 STRWroX) become falsely N2UnitI-bound.

#### M0 occupancy, `LDPSW` and integer divide (`classifyA78ResOverride`)

A78 SOG correction: NeoverseN2Model gives several Cortex-A78 instructions a MULTI-cycle occupancy of the single M0 pipe although the A78 SOG lists them as fully pipelined (throughput 1):

```text
  SCVTF/UCVTF (gen->vec)  N2Write_3c_1M0  ReleaseAtCycles=3   SOG 3-18: lat 3, thr 1, M0
  FMOV W/X -> S/D/H       N2Write_0or3c_1M0 (3)               SOG: lat 3, thr 1, M0
  DUP (vector, from GPR)  N2Write_3c_1M0 (3)                  SOG 3-26: lat 3, thr 1, M0
```

and LDPSW, which the SOG lists as "5, 3/2, I, L" (as NeoverseN1Model models it for the A76: N1Write_5c_1I_1L), is modelled as N2Write_5c_1M0 (5 cycles of M0, no load pipe at all).

A76/A78 model-convention alignment: both the A76 and the A78 SOG give integer divide "latency 5 to 12(20), throughput 1/12(20) to 1/5". NeoverseN1Model (A76) occupied its divider for the BEST case (5, N1Write_12c5_1M / N1Write_20c5_1M); NeoverseN2Model (A78) for the WORST case (12 / 20). (Since 2026-10-31 the A76 has latency 5 as well and holds a divider resource of its own for the 5 cycles, M only one cycle: [a76-sog.md](a76-sog.md) D-1, D-3. The A78 correction below is unchanged.) This is a convention mismatch between the two borrowed upstream models, not a hardware difference; the A76 model's occupancy (5) is applied to the A78 so the two cores are modelled alike. This correction is always on.

### E. Front-end bounds: uops and Mops (`calculateIssueBound`)

```text
1. Calculate Dispatch / Issue Width Limit
```

TWO INDEPENDENT front-end bounds, both documented for every Arm core this tool models via a locally-modified InstRW-bearing .td (Cortex-A76/A78/A720): their Software Optimization Guides each state a pair of numbers of the form "the dispatch stage can process up to N Mops per cycle and dispatch up to M uops per cycle" (e.g. A76 SOG sec 4.1 p.41: N=4, M=8). A Mop (macro-operation) is what one non-fused instruction decodes into 1:1 (the SOGs describe splitting only going the OTHER way, Mop -> up to two uops, at the dispatch stage - see frontend.cpp's MopDispatchWidth comment); M (the uop cap) is what DispatchWidthOverride already carries. Since AArch64 averages ~1.1-1.3 uops/instruction, N/4 is normally the TIGHTER of the two bounds and was, before this change, never computed at all - only the uop bound M was. MopWidth==0 (a CPU this hasn't been verified for, or Apple's coalesced-ROB cores where DispatchWidthOverride is already a Mop-level width) skips this second bound entirely, leaving prior behavior unchanged.

The numbers per core are in `cpu_traits.cpp`: uop dispatch width A76 8, A78 12, A720 10, X1 16; Mops per cycle A76 4, A78 6, A720 5, X1 8. The decode width (A78 4, X1 5) is deliberately not modelled: the tool bounds hot-loop front-end throughput by the rename/dispatch width. Cores that only share a uop width with a modelled core (`cortex-a710`, `cortex-a715`, `neoverse-n2`; `neoverse-v1`) keep no Mop cap because their own SOGs have not been consulted; the Apple cores have none because `computeFacilePrediction()` already forces `NumMicroOps = 1` for coalesced-ROB CPUs.

### F. Stores with a writeback base update (`buildDependencyGraph`)

A pre- or post-index store (`str w1,[x0],#4`, `str w1,[x0,#4]!`, `stp ...,[x0],#8`, `st1 ...,[x0],#16`) writes the base register it reads. The SOGs split a store into an address uOP and a data uOP at dispatch; the base update belongs to the address side, so it is available about one cycle (`WriteAdr`) after the base register, **regardless of when the store data is ready**.

The dependence graph has one node per instruction, so the base update used to inherit the data dependence: in the in-place update loop

```text
ldr w1, [x0]
sub w1, w1, #10
str w1, [x0], #4      // base update
cmp x0, x21
```

the cycle ldr (4) + sub (1) + str (1) = 6 cycles per iteration was reported as the precedence bound, although the only real recurrence is the one-cycle base update. The same loop with the update written as a separate `add x0, x0, #4` was not affected. Vectorised in-place updates (`ldr q`, arithmetic, `str q,[x],#16`) were affected the same way: 9 and 13 cycles became 1 and 2.

The fix gives such a store a second node. That node reads only the address registers (the base and, for ASIMD post-index forms, the register increment: every register use whose explicit operand index is at or after the tied base operand) and is the producer of the base register; the store's own node keeps all its inputs, so the store data and the address still feed the store itself. The extra nodes are numbered after the instructions, so instruction indices, the port and issue bounds and the memory-dependence edges are unchanged. The node also reads the base register's value from the previous iteration, which keeps the one-cycle base update a recurrence. The writeback def is recognised as "a def of a store that the same instruction also reads"; the latency is the one the scheduling model gives that def (`WriteAdr`), not a new constant.

Loads with writeback were not affected (their only use is the base), and neither were stores whose base is updated by a separate `add`.

### G. Memory dependence when the index register changes (`addMemoryDependencies`)

For a store followed by a load with the same base register and a register index (`strh w3,[x0,x4]` ... `ldrsh w5,[x0,x4]`), the memory edge used to be added whenever the base register was unchanged in between. If the index register was advanced in between (`add x4,x4,#2`), the load reads the next element, so the SOG's store-to-load forwarding (which needs the load to start at the start or the middle of the older store) cannot apply, and the edge created a recurrence the hardware does not have (x264 `quant_4x4`: 9 cycles instead of 2).

The edge is now skipped only when the two accesses are **provably disjoint**: both use the same register-offset form (same index register, extend type and scaled/unscaled flag), the only instructions between them that write the index register are `add/sub idx, idx, #imm` (no shift), and the resulting byte distance (immediate sum, times the access size for a scaled index) is at least the store size forward or the load size backward. Different forms, any other kind of index update, an unknown access size or an overlap keep the edge, so nothing that is not statically known is assumed. The loop-carried rule is unchanged. The SOGs give no latency for store-to-load forwarding, only the conditions above; the edge latency (store write + load) is not touched.

### H. FP multiply-accumulate on the A720

The A720 SOG lists FMADD/FMSUB/FNMADD/FNMSUB and ASIMD FMLA/FMLS/FMLAL/FMLSL as "4 (2)" and adds that an FP multiply result is forwarded late to the accumulate operand of a multiply-accumulate (1 cycle after the multiply issues). The A720 model had no FMA rows, so FMADD behaved like an FMUL (latency 3, no early operand read), a 3-cycle accumulate chain against 2 on the A76/A78/X1, whose models already carry the same "4 (2)". The model now has `A720Wr_FMA` (latency 4) and `A720Rd_FMA` (`SchedReadAdvance<2, [WriteFMul, A720Wr_FMA]>`): 2 cycles from a previous multiply-accumulate and 3 - 2 = 1 from an FMUL.

### I. FP divide and square root: latency and occupancy

The SOGs give FP divide and square root a latency and a throughput that depend on the operands for the Neoverse-derived cores (A76/A78/X1: ranges such as "7 to 15" and "1/7 to 2/7" per form) and a fixed value per form for the A720. The model has one latency and one occupancy per `SchedWriteRes`, so a range needs one chosen value. Every form has its own write (`N1WriteFDS_*`, `N2WriteFDS_*`, `V1WriteFDS_*`, `A720Write_*`), listed next to the `InstRW` rows with the SOG values they come from.

* **A76 (N1), A78 (N2), X1 (V1)**: the BEST of each range. Latency: the shortest (7 for the scalar and 2S/4S/2D ASIMD forms; Q-form F16 FDIV 10 and FSQRT 11, where the SOG ranges "10 to 13" and "11 to 13" start). Occupancy: the unit is busy for ceil(1 / best throughput) cycles. For the X1, whose SOG throughput is for the V0/V2 pair, the occupancy of one unit is ceil(2 / best throughput), so the instruction time is the same as on the A76 up to the pair. Examples (A76/A78): scalar H and S 4/7 -> 2, scalar D 2/7 -> 4; ASIMD D-form F32 2/7 -> 4, Q-form F32 and F64 1/7 -> 7, Q-form F16 1/10 (FDIV) and 1/11 (FSQRT) -> 10 and 11. The uOP counts of the A78 writes are those of the N2 writes they replace. The X1 SVE classes `V1Write_*_1V0` are not touched. On the A76 the occupancy is held on an FP divide unit of its own (`N1UnitFDiv`), V0 only one cycle, since 2026-10-31 ([a76-sog.md](a76-sog.md) D-3).
* **A720**: one latency per form, as in the SOG: scalar H 5, S 7, D 12 and ASIMD D-form F16 8, D-form F32 8, Q-form F16 12, Q-form F32 10, Q-form F64 13, for FDIV and FSQRT alike. The scalar throughput is 1 ("fully pipelined"), so the unit is busy one cycle. The ASIMD table's throughput column (1/4, 1/2, 1/8 -> occupancy 4, 2, 8) contradicts the note under it ("fully pipelined"); the table is used as printed.
* **Apple (firestorm, icestorm)**: one latency per form from Dougall Johnson's tables (scalar H/S/D and vector 4H/8H/2S/4S/2D), replacing the single worst-case value. Port and occupancy are as before (Apple is not in a SOG).

## 5. Scheduling-Model Plumbing

### A. SchedClass index remapping (`remapSchedClassIndices`)

Re-point MCInstrInfo at the LOCALLY re-generated AArch64 instruction descriptor table, so that MCInstrDesc::SchedClass carries OUR TableGen numbering instead of stock libLLVM's.

WHY THIS IS NEEDED (2026-09-10).  The tool mixes two independently generated halves of the same AArch64 target description:

(a) src/frontend/frontend.cpp calls TheTarget->createMCInstrInfo(), i.e. the MCInstrDesc array COMPILED INTO the installed libLLVM-22 (22.1.8).  Its MCInstrDesc::SchedClass values use the numbering that stock AArch64.td produces. (b) installLocalSchedModel() below installs FirestormModelSchedClasses et al. from build/AArch64GenSubtargetInfo.inc, which this project re-generates from AArch64.td plus ModifiedTarget/AArch64/*.td.  Those extra files (AArch64SchedIcestorm.td, AArch64SchedFirestorm.td, and the modified A55/N1/N2/V1/A520/A720 models) declare additional InstRW rules, and every InstRW that partitions an existing SchedClass makes TableGen CREATE new SchedClass records.  Our class list is 2324 entries; stock's is 2050.  The two numberings therefore DIVERGE, and MCA was indexing table (b) with indices from table (a).

Measured extent of the mismatch (all 9135 AArch64 opcodes compared, stock libLLVM vs build/AArch64GenInstrInfo.inc):

```text
  - opcode numbering:      0 / 9131 mismatched (the two .td revisions agree)
  - SchedClass numbering:  1051 / 9131 mismatched (11.5%)
  - first divergence at class 498 (stock) / 499 (ours); delta is +1 for 953
    opcodes, and -776..+738 for the rest.
```

Stock's largest SchedClass index (2049) stays below our NumSchedClasses (2324), so the mis-indexing never read out of bounds - it silently returned ANOTHER instruction's latency and port assignment, which is why it survived unnoticed.

The symptom that exposed it: `fmadd d31, d30, d29, d31` simulated at 3c instead of 4c.  FMADDDrrr is Sched<[WriteFMul]> (AArch64InstrFormats.td:5953) and WriteFMul is Latency=4 in both Apple models, so the .td was already correct - but stock libLLVM hands MCA SchedClass 649, and OUR class 649 is FADDDrr_FADDSrr_FSUBDrr_FSUBSrr (WriteF, Latency=3).  Our FMADD class is 650. That is why the previous attempts failed to move the number: editing WriteFMul's latency, or adding an InstRW for the FMADD family, both correctly changed class 650, while the simulation kept reading class 649.  Plain `fmul` appeared right only because FMULDrr's stock and local indices happen to agree.

WHY THE WHOLE TABLE, and not just the SchedClass field.  Two reasons, both found the hard way:

```text
 1. A per-class translation (rewriting table (b) into stock numbering) is
```

impossible in principle.  The two numberings are not a shift, they are DIFFERENT PARTITIONS of the instruction set - stock puts FMADDDrrr in class 649 and FADDDrr in 1343, we put FADDDrr in 649 and FMADDDrrr in

```text
    650.  Our extra InstRW rules exist precisely to split stock classes, so
```

one stock class index can correspond to several of ours and cannot carry their differing latencies.  The remap has to be per-OPCODE.

```text
 2. A per-opcode copy of the descriptors (std::vector<MCInstrDesc> with the
```

SchedClass field overwritten) is also impossible: MCInstrDesc is explicitly non-copyable-in-practice, because it locates its own operand and implicit-operand arrays from its OWN ADDRESS - MCInstrDesc::operands() is `reinterpret_cast<const MCOperandInfo *>(this + Opcode + 1) + OpInfoOffset` (MCInstrDesc.h:240), which only resolves inside the generated <Target>InstrTable struct, where Insts[] is stored in DESCENDING opcode order and is immediately followed by ImplicitOps[] and OperandInfo[].  Copying the descs elsewhere sends operands() into unrelated heap memory; the first attempt at this fix aborted on every input, including a two-instruction `nop; ret`.

So install our AArch64Descs wholesale, via the InitAArch64MCInstrInfo() that TableGen emits alongside it - exactly the call libLLVM's own createMCInstrInfo() makes, just with our table.  That keeps Insts / ImplicitOps / OperandInfo mutually consistent, and makes SchedClass agree with the sched-class tables installed by installLocalSchedModel().

SAFETY of swapping in a table generated from llvm-source (LLVM main) while linking libLLVM-22.1.8.  Verified field by field over all 9135 opcodes, our table against libLLVM's: SchedClass is the ONLY field that differs anywhere.

```text
  SchedClass            1051 differ
  Flags 0    TSFlags 0    NumOperands 0    NumDefs 0    Size 0
  NumImplicitUses 0    NumImplicitDefs 0    OpInfoOffset 0
  opcode slot ordering  exact match (our Insts[N-1-i].Opcode == i for all i)
```

So this is a pure SchedClass correction: it cannot change mayLoad / mayStore / isBranch / operand shape, and any change in simulation output is attributable to the sched-class mapping alone.  The only 4 instruction NAMES that differ (LOAD_STACK_GUARD, PATCHABLE_EVENT_CALL, PATCHABLE_TYPED_EVENT_CALL, PREALLOCATED_ARG, emitted as anonymous_* by our run) are generic TargetOpcode pseudos that never occur in a disassembled AArch64 binary.  The opcode-count guard below refuses the swap outright if a future libLLVM bump breaks that agreement.

This is a correctness fix, not an accuracy claim: InstRW rules in ModifiedTarget/AArch64/*.td written while the indices were scrambled may have been compensating for the mis-indexing, and each added InstRW itself renumbers the classes.

### B. Per-CPU parameters (`cpu_traits.cpp`)

Facts about a core that LLVM's scheduling model cannot express live in one table, `getCpuTraits()`: which local `.td` model is installed, the SOG's uop and Mop dispatch widths, which fusion table applies, whether the model's zero-latency writes are rename-time moves, and whether the `NeoverseN2Model` corrections of section 4.D apply. Adding a core or a variant of one is a single line there.
