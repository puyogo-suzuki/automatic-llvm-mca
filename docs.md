# automatic-llvm-mca Technical Documentation

This document provides an exhaustive technical overview of the processor scheduling model custom adaptations, the C++ simulation pipeline modifications, and the static Memory Level Parallelism (MLP) calculation methodology implemented in this project.

---

## 1. Processor Scheduling Model Customizations & Forwarding Paths

To accurately simulate the execution timeline of target processors (specifically the dual-issue in-order **ARM Cortex-A55**), the scheduling model has been customized at both the **TableGen** description level (`ModifiedTarget/AArch64/AArch64SchedA55.td`) and the **C++ runtime simulation layer** (`mca.cpp`).

### A. TableGen Scheduling Model Customizations (`AArch64SchedA55.td`)

The scheduling model defines how machine instructions are dispatched, executed, and retired on specific hardware ports. Key modifications and optimizations include:

#### 1. Hardware Pipeline & Resource Unit Definitions
The Cortex-A55 is modeled as a fully in-order processor (`let MicroOpBufferSize = 0;`). The execution units are represented by five core pipeline resources with zero queuing buffer size, preventing out-of-order execution inside the simulation:
*   `CortexA55UnitALU` (2 ports): Dual integer arithmetic units.
*   `CortexA55UnitMAC` (1 port): 64-bit wide multiply-accumulate pipe.
*   `CortexA55UnitDiv` (1 port): Non-pipelined integer division unit.
*   `CortexA55UnitLd` (1 port): Memory load pipeline.
*   `CortexA55UnitSt` (1 port): Memory store pipeline.
*   `CortexA55UnitB` (1 port): Branch unit.

#### 2. Forwarding Paths (Bypassing & Read-Advance Paths)
On physical hardware, to avoid stalling subsequent instructions when data is produced by a preceding instruction, the CPU implements **bypassing network/forwarding paths**. In the TableGen model, these are represented as `ReadAdvance` statements:

*   **EX1-stage ALU-to-ALU Forwarding (`ReadI`)**:
    Under standard operations, ALU input operands are consumed in the **EX1** stage. If an operand is produced in **EX2** by a preceding ALU operation, a forwarding path allows the data to be routed directly to the EX1 stage of the next instruction without waiting for register write-back.
    ```tablegen
    def : ReadAdvance<ReadI, 1>; // Decrements latency by 1 for back-to-back ALU execution
    ```
*   **Shifted vs. Shift-less Register Operands (`CortexA55ReadISReg`)**:
    If an ALU instruction requires a shift (e.g., `ADD Xd, Xn, Xm, LSL #2`), the shift operation must occur early in the **ISS (Issue)** stage. Therefore, it cannot benefit from EX2-to-EX1 forwarding. However, if the shift amount is 0, the instruction behaves like a standard ALU instruction and can receive forwarded operands in EX1.
    We implement a `SchedReadVariant` using the subtarget predicate `RegShiftedPred` to resolve this:
    ```tablegen
    def CortexA55ReadShifted    : SchedReadAdvance<0>; // Stalls: must be ready at ISS
    def CortexA55ReadNotShifted : SchedReadAdvance<1>; // Forwarded: ready at EX1
    def CortexA55ReadISReg : SchedReadVariant<[
            SchedVar<RegShiftedPred, [CortexA55ReadShifted]>,
            SchedVar<NoSchedPred,    [CortexA55ReadNotShifted]>]>;
    def : SchedAlias<ReadISReg, CortexA55ReadISReg>;
    ```

#### 3. Multiply-Accumulate Forwarding (`ReadIM` / `ReadIMA`)
For integer multiplications, the operands are forwarded with varying latencies:
*   `ReadIM` (Multiplicand/Multiplier): Forwarded with 1-cycle advance from producing ALU operations.
*   `ReadIMA` (Accumulator operand): Consumed later in the multiply pipe, permitting a 2-cycle advance. This allows back-to-back multiply-accumulate chains (like `madd`) to execute without bubbles.

---

### B. C++ Runtime Simulation Adaptations (`mca.cpp`)

While TableGen provides the static structural definition, the LLVM MCA runtime library does not always fully resolve variant scheduling classes at runtime because of limitations in the static analyzer's register-state operand representation. Therefore, several dynamic modifications are applied during the construction of the simulation pipeline:

#### 1. Conditional Flags (NZCV) Dependency Breaking
Conditional instructions (like `csel`, `cset`, `csneg`) and conditional branches (`b.ne`, `cbz`) depend on the status register `NZCV` written by flags-setting instructions (like `cmp`, `subs`).
*   **LLVM MCA Behavior**: By default, LLVM MCA introduces a mandatory 1-cycle data dependency bubble between the flag-producer and flag-consumer.
*   **Physical Hardware**: Cortex-A55 features a zero-latency condition flag bypass network enabling same-cycle dual-issue of `cmp` $\to$ `b.ne` or `cmp` $\to$ `csel`.
*   **C++ Override**: During instruction construction in `analyzeMcaRegion` and dispatch in `SteadyStateTracker::onEvent`, we programmatically break the NZCV dependency:
    ```cpp
    // For Writes (Defs)
    if (WS.getRegisterID() == AArch64::NZCV) {
        WS.setRegisterID(0); // Erase register write-back dependency
    }
    // For Reads (Uses)
    if (RS.getRegisterID() == AArch64::NZCV) {
        RS.IsReady = true;               // Force ready state
        RS.setIndependentFromDef();      // Prevent dependency stall
    }
    ```

#### 2. Shift-less Register-Shifted ALU Latency Correction
For instruction opcodes that permit optional register shifts (`ADDXrs`, `ADDWrs`, `SUBXrs`, `SUBWrs`, `SUBSXrs`, `SUBSWrs`, `ADDSXrs`, `ADDSWrs`), LLVM statically assigns a latency of 2 cycles.
*   **C++ Override**: We check the immediate value of the shift operand (operand index 3). If the shift value resolves to `0` (or the operand is missing), we dynamically overwrite the instruction descriptor:
    ```cpp
    MutableDesc.MaxLatency = 1;
    for (auto &W : MutableDesc.Writes) {
        if (W.Latency > 1) W.Latency = 1;
    }
    ```
    This shortens the ALU path to a single cycle, allowing back-to-back execution when combined with the EX1 forwarding path.

#### 3. Speculative Branch Predictor Bubble Correction
Cortex-A55 utilizes a non-blocking speculative branch predictor. When branch prediction succeeds, instruction fetching from the target address continues without bubble cycles.
*   **LLVM MCA Behavior**: Because the in-order pipeline simulation model enforces strict retirement order, the branch unit (`CortexA55UnitB`) remains locked until the branch retires, causing a mandatory 1-cycle dispatch bubble in every loop iteration.
*   **C++ Override**:
    1.  At construction, conditional and unconditional branches (`Bcc`, `B`) are stripped of their pipeline resource consumption, converting them to true zero-latency instructions:
        ```cpp
        MutableDesc.MaxLatency = 0;
        MutableDesc.Resources.clear();
        MutableDesc.UsedProcResUnits = 0;
        MutableDesc.UsedProcResGroups = 0;
        MutableDesc.UsedBuffers = 0;
        ```
    2.  Since the LLVM MCA simulation engine still incurs a 1-cycle retirement overhead at loop boundaries, we compensate for the speculative fetch capability at the metric aggregation step. We subtract `0.5` cycles per loop iteration from the steady-state cycle count:
        ```cpp
        double correctedCycles = static_cast<double>(M.Cycles) - (static_cast<double>(NumSteadyIterations) * 0.5);
        M.Cycles = static_cast<unsigned>(correctedCycles + 0.5);
        ```
        This brings the calculated loop CPI of tight blocks down to **`0.56`** (matching the actual hardware test loop CPI of **`0.57`**).

---

### C. Neoverse Architecture Validations (N1, N2, V1)

We verified the scheduling models for high-performance out-of-order Neoverse cores (`NeoverseN1.td`, `NeoverseN2.td`, `NeoverseV1.td`).
*   Unlike the Cortex-A55, these architectures utilize the TableGen predicate `IsCheapLSL` (which checks if the shift type is LSL and the shift amount is $\le 4$).
*   If the shift predicate is satisfied (or there is no shift), the scheduler variant maps the instruction to a 1-cycle latency pipeline (e.g. `N1Write_1c_1I`, `N2Write_1c_1I`, `V1Write_1c_1I`).
*   Otherwise, it falls back to a 2-cycle latency pipeline (e.g. `N1Write_2c_1M`, `N2Write_2c_1M`, `V1Write_2c_1M`).
*   Since out-of-order cores have large instruction window buffers (`MicroOpBufferSize > 0`), speculative branch bubbles are naturally absorbed, meaning no manual branch latency overrides are necessary for Neoverse targets.

---

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
    *   If a base register and cache line pair `(base_reg, cache_line)` has already been seen in the current window, the load is classified as a spatial cache hit and excluded from the outstanding MLP load count.
    *   **Call-Instruction Invalidation**: If a function call instruction (`bl`) is encountered in the window, we assume all register dependencies and active cache line tracking states are invalidated (since the callee may modify return registers and memory states).
