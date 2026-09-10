# automatic-llvm-mca

Estimate throughput for ELF binaries using a direct C++ interface to LLVM MCA.

## Procedure

1.  Load the ELF binary and identify executable sections.
2.  Perform linear disassembly to discover basic blocks and loops.
3.  Run the LLVM Machine Code Analyzer (MCA) on each region to obtain retired instructions and elapsed cycles.
4.  Print a CSV with start address, end address, retired instructions, load
    instructions, elapsed cycles, and Memory Level Parallelism (MLP) for every region.

This tool is implemented in pure C++ using LLVM 23 APIs, providing extreme performance by avoiding all Python overhead and subprocess creation.

## Supported Architectures

*   **x86 / x86-64**
*   **AArch64** (64-bit ARM)
*   **32-bit ARM**
*   **RISC-V** (RV32IC, RV64IC)

## Build Instructions

### Prerequisites

*   LLVM 22 development headers (`llvm-22-dev`)
*   CMake 3.10+
*   C++17 compatible compiler

To compile the customized TableGen-based pipeline, you need a local copy of the LLVM AArch64 target sources. Clone `llvm-project` into a directory named `llvm-source` at the project root using a sparse checkout (to save disk space and download time):

```bash
git clone --depth 1 --sparse --branch llvmorg-22.1.8 https://github.com/llvm/llvm-project.git llvm-source
cd llvm-source
git sparse-checkout set llvm/lib/Target/AArch64 llvm/include
cd ..
```

### Build

```bash
mkdir -p build && cd build
cmake ..
make
```

This will produce the main tool `build/mca_tool`, the secondary tools `build/mlp-objdump`, `build/mlp-check`, `build/mca-insts-info`, and the unit tests `build/mca_unit_tests`.

## Usage

```bash
# Main tool to analyze all loops and basic blocks, outputting metrics to stdout as CSV
# Use --facile to enable Facile analytical throughput prediction (AArch64) instead of full cycle-by-cycle MCA simulation
# Use --update-mlp=<csv file> to reuse the simulation results (Cycles, RetiredInsts) from a previous CSV run
./build/mca_tool [--facile] [--update-mlp <csv file>] [--mcpu <cpu>] [--mtriple <triple>] [--window-width <W>] [--dependency <mode>] [--mlp-window-assignment <mode>] [--iterations <N>] [--ignore-loop-carried] [--override-load-latency <N>] <elf-binary>

# Disassembles text sections and prints per-basic-block MLP/baseCPI next to each instruction address
./build/mlp-objdump [--mcpu <cpu>] [--mtriple <triple>] [--window-width <W>] [--dependency <mode>] [--mlp-window-assignment <mode>] [--iterations <N>] [--ignore-loop-carried] [--override-load-latency <N>] <elf-binary>

# Disassembles and runs step-by-step debug logging for a specific target address to trace register dependencies and MLP window evaluation
./build/mlp-check --target-address <hex> [--window-width <W>] [--dependency <mode>] <elf-binary>

# Outputs a comprehensive table of all target machine instructions along with their TableGen internal names, assembly mnemonics, scheduling classes, execution latencies, reciprocal throughputs, and resource usages. It dynamically resolves variant scheduling classes using register-class operand matching.
./build/mca-insts-info --mtriple aarch64-linux-gnu --mcpu cortex-a55 [--format csv/tsv] > output.csv
```

*   `<elf-binary>` — Path to the ELF binary to analyze.
*   `mlp-objdump` — Disassembles text sections and prints per-basic-block MLP/baseCPI next to each instruction address.
*   `--facile` — Enable Facile static analytical throughput prediction (AArch64). Calculates analytical throughput bounds for Issue, Execution Ports, and Precedence Constraints without cycle-by-cycle simulation overhead.
*   `--no-facile-memory-deps` — Drop store→load memory RAW edges from the `--facile` precedence graph (register dependences only). See "Memory dependences in the precedence graph" below.
*   `--mcpu <cpu>` — (Optional) Specify a target CPU (e.g., `cortex-a55`, `cortex-a720`, `firestorm`, `icestorm`).
*   `--mtriple <triple>` — (Optional) Specify a target triple (e.g., `aarch64-linux-gnu`).
*   `--window-width <W>` — (Optional) Window width for MLP estimation (default: 4).
*   `--dependency <mode>` — (Optional) MLP dependency mode (`none`, `io`, `ooo`, `dependency`).
*   `--mlp-window-assignment <mode>` — (Optional) MLP assignment mode (`forward`, `max-containing`).
*   `--iterations <N>` — (Optional) Steady-state repetition multiplier (default: 100).
*   `--override-load-latency <N>` — (Optional) Override load instruction latency (default: -1, inactive).
*   `--ignore-loop-carried <mode>` - (Optional) Mode for ignoring loop-carried register dependencies during cycle estimation.
    *   `default`: Ignores loop-carried dependencies in basic blocks, but considers them in loops (default behavior if option is omitted).
    *   `force`: Ignores loop-carried dependencies in both loops and basic blocks.
    *   `disable`: Considers loop-carried dependencies in both loops and basic blocks.

## Region Splitting & Merging Algorithm

`automatic-llvm-mca` automatically partitions ELF binary text sections into basic blocks and loop regions using an adaptive region splitting and merging algorithm implemented in `splitter.cpp`:

1. **Loop Detection**:
   - Scans disassembled instructions per function.
   - Identifies backward branches (`BranchTarget <= Addr` and `BranchTarget >= FuncBase`) to construct basic loop regions `[header_idx, latch_idx]`.

2. **$abab$ Interlocking Chain Detection**:
   - Identifies partially overlapping loop pairs $a$ and $b$ ($abab$ pattern) where `a.h_idx < b.h_idx < a.l_idx < b.l_idx` (loops intersect but neither is fully nested within the other).
   - Computes maximum interlocking chain depth using dynamic programming.

3. **Adaptive Chain Merging & Pre-merged Region Discarding (`--chain-threshold=<K>`)**:
   - Compares the maximum chain depth against the threshold $K$ (configurable via `--chain-threshold=<K>`, default: `5`).
   - **When `max_chain >= K`**:
     - Interlocking $abab$ loops are merged into unified outer bounding spans `merged_abab`.
     - **Pre-merged inner loops participating in the $abab$ chain are discarded** to eliminate redundant region counts and significantly accelerate full-binary analysis.
     - **Nested loops (e.g. $c$ in $abccab$) and non-$abab$ loops are preserved** as distinct inner regions (since fully nested loops do not partially overlap).
   - **When `max_chain < K`**:
     - Loop regions remain unmerged as standard individual loops.

4. **Innermost-First Region Sorting**:
   - Output regions are sorted by `EndAddr` ascending (and `StartAddr` ascending) to ensure downstream tools match the innermost nested loop span first.

## Facile Analytical Throughput Predictor

When passing `--facile` to `mca_tool`, the tool computes steady-state basic-block throughput analytically based on the **Facile** model instead of full step-by-step cycle-by-cycle MCA simulation.

Facile calculates three independent throughput limits from LLVM's machine scheduling model and takes their maximum ($\max$):
1. **Issue Limit**: Dispatch width bottleneck ($\sum \text{uops} / \text{IssueWidth}$).
2. **Execution Ports Limit**: Contention on execution unit resource ports ($\max_p \text{ResourceCycles}(p) / \text{NumUnits}(p)$).
3. **Precedence Constraints Limit**: Maximum Cycle Ratio (MCR) of loop-carried Read-After-Write (RAW) dependency chains, over both **register** and **memory** (store→load) dependences.

$$\text{Facile Throughput (Cycles/Iter)} = \max\left( \text{Issue Limit},\; \text{Execution Ports Limit},\; \text{Precedence Limit} \right)$$

#### Memory dependences in the precedence graph

Facile as published analyses a basic block in isolation and assumes aliasing is
statically unknowable (sec. 3.3). This tool goes further — it adds
*inter-iteration* edges so the third limit is a genuine loop-recurrence bound —
and therefore must also model recurrences that round-trip through **memory**,
not just through registers. `addMemoryDependencies()` in `facile.cpp` adds a
store→load edge under a deliberately conservative may-alias rule (different
base register ⇒ no alias; same base with both displacements constant ⇒ alias
only on an exact offset match; same base with a register index ⇒ may alias; a
base redefined in between breaks the match), plus the classic
dependence-distance test that suppresses the loop-carried edge when both
accesses have identical subscript expressions over a loop-varying index (an
`a[i] = f(a[i])` read-modify-write has dependence distance 0). Edge latency is
the store's own `WriteST` latency, exactly as the register path uses the
producer's latency — no new constant is introduced.

Pass `--no-facile-memory-deps` to restore the register-only behaviour. Because a
missing precedence bound is masked whenever the issue or port limit is larger,
this matters most on the **widest** core modelled with the **coarsest** port
table — in practice Apple FireStorm, where 456.hmmer's P7Viterbi D-state
recurrence moves the hot block's prediction from 7 to 12 cycles against 11.7
measured on real M1 hardware (Cortex-A76/A78 and IceStorm are unchanged on that
block: their port/issue limits already exceeded the recurrence).

### Reference & Citation
* **Paper**: *Facile: Fast, Accurate, and Interpretable Basic-Block Throughput Prediction*
* **Authors**: Shrey Sharma, Jan Reineke, Andreas Abel (Saarland University)
* **Preprint**: [arXiv:2310.13212 [cs.PF]](https://arxiv.org/abs/2310.13212) (2023)

## Tests

Run the C++ unit tests (GoogleTest):

```bash
./build/mca_unit_tests
```


## Misc

### Cortex-A55 Scheduling Model Customization
 * Models the physical hardware of Cortex-A55 accurately based on the *Cortex-A55 Software Optimization Guide (SOG)*.
 * Includes a specialized **0-cycle same-cycle flag bypass** (`cmp` $\to$ `csel`/conditional branch) which prevents artificial 1-cycle flag latency stalls, bringing the simulated CPI of tight loop kernels down under 1.0 (matching actual hardware).
 * Customized TableGen files are located in `ModifiedTarget/AArch64/AArch64SchedA55.td` and integrated into the build pipeline using `llvm-tblgen-23`.
