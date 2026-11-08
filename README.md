# automatic-llvm-mca

Estimate throughput for ELF binaries using a direct C++ interface to LLVM MCA.

## Procedure

1.  Load the ELF binary and identify executable sections.
2.  Perform linear disassembly and split each function into loop and non-loop regions (see "Region Splitting & Merging Algorithm" below).
3.  Run the LLVM Machine Code Analyzer (MCA) on each region to obtain retired instructions and elapsed cycles (or, with `--facile`, the Facile analytical estimate).
4.  Print a CSV with one row per region: `start_address`, `end_address`, `analysis_start`,
    `analysis_end`, `length`, `loop`, `retired_instructions`, `load_instructions`, `cycles`,
    `mlp`, `mlp_r` (and `facile_reason` with `--facile-reason`).

This tool is implemented in C++ against the LLVM 22 libraries (LLVM's MCA, MC and disassembler APIs), without Python or subprocesses.

## Supported Architectures

*   **x86 / x86-64**
*   **AArch64** (64-bit ARM)
*   **32-bit ARM**
*   **RISC-V** (RV32IC, RV64IC)

The region splitter (`src/core/region_splitter.cpp`) and the end addresses in the CSV assume
fixed 4-byte instructions, as on AArch64; for x86 and compressed RISC-V code the loop detection
and the end addresses are only right where that assumption holds. The MLP analysis has its own
analyzer for x86 and RISC-V; every other architecture, 32-bit ARM included, uses the AArch64
analyzer (`MLPAnalyzer::create`). The scheduling models and SOG comparisons in `docs/` are for
AArch64 cores only.

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

`CMakeLists.txt` requires LLVM 22.1.8 (`find_package(LLVM 22.1.8 REQUIRED CONFIG)`; pass `-DLLVM_DIR=/usr/lib/llvm-22/cmake` if CMake does not find it) and runs `/usr/bin/llvm-tblgen-22` to generate the AArch64 tables from `llvm-source` and `ModifiedTarget/AArch64/*.td`. GoogleTest is fetched by CMake (`FetchContent`; `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=<dir>` uses a local copy).

This will produce the main tool `build/mca_tool`, the secondary tools `build/mlp-objdump`, `build/mlp-check`, `build/mca-insts-info`, `build/analyze_distribution`, and the unit tests `build/mca_unit_tests`.

## Usage

```bash
# Main tool to analyze all loops and basic blocks, outputting metrics to stdout as CSV
# Use --facile to enable Facile analytical throughput prediction (AArch64) instead of full cycle-by-cycle MCA simulation
# Use --update-mlp=<csv file> to reuse the simulation results (Cycles, RetiredInsts) from a previous CSV run
./build/mca_tool [--facile] [--facile-reason] [--update-mlp <csv file>] [--mcpu <cpu>] [--mtriple <triple>] [--window-width <W>] [--dependency <mode>] [--mlp-window-assignment <mode>] [--mlp-window-loop <mode>] [--iterations <N>] [--ignore-loop-carried <mode>] [--override-load-latency <N>] [--chain-threshold <K>] [--merge-same-header] [--count-only] <elf-binary>

# Disassembles text sections and prints per-basic-block MLP/baseCPI next to each instruction address
./build/mlp-objdump [--mcpu <cpu>] [--mtriple <triple>] [--window-width <W>] [--dependency <mode>] [--mlp-window-assignment <mode>] [--iterations <N>] [--ignore-loop-carried <mode>] [--override-load-latency <N>] <elf-binary>

# Disassembles and runs step-by-step debug logging for a specific target address to trace register dependencies and MLP window evaluation
./build/mlp-check --target-address <hex> [--window-width <W>] [--dependency <mode>] <elf-binary>

# Outputs a comprehensive table of all target machine instructions along with their TableGen internal names, assembly mnemonics, scheduling classes, execution latencies, reciprocal throughputs, and resource usages. It dynamically resolves variant scheduling classes using register-class operand matching.
./build/mca-insts-info --mtriple aarch64-linux-gnu --mcpu cortex-a55 [--format csv/tsv] > output.csv
```

*   `<elf-binary>` — Path to the ELF binary to analyze.
*   `mlp-objdump` — Disassembles text sections and prints per-basic-block MLP/baseCPI next to each instruction address.
*   `--facile` — Enable Facile static analytical throughput prediction (AArch64). Calculates analytical throughput bounds for Issue, Execution Ports, and Precedence Constraints without cycle-by-cycle simulation overhead.
*   `--facile-memory-deps` — (Optional, off by default) Add store→load memory RAW edges to the `--facile` precedence graph; without it the graph has register dependences only. See "Memory dependences in the precedence graph" below.
*   `--mcpu <cpu>` — (Optional) Specify a target CPU (e.g., `cortex-a55`, `cortex-a720`, `cortex-x1`, `firestorm`, `icestorm`; `fire` and `ice` are accepted as aliases of the two Apple cores).
*   `--mtriple <triple>` — (Optional) Specify a target triple (e.g., `aarch64-linux-gnu`).
*   `--window-width <W>` — (Optional) Window width for MLP estimation (default: 4 without `--mcpu`; with `--mcpu`, a value taken from the CPU's scheduling model, see `initializeFrontend()` in `src/frontend/frontend.cpp`).
*   `--dependency <mode>` — (Optional) MLP dependency mode (`none`, `io`, `ooo`, `dependency`; default `none`).
*   `--mlp-window-assignment <mode>` — (Optional) MLP assignment mode (`forward`, `max-containing`; default `max-containing`).
*   `--mlp-window-loop <mode>` — (Optional) Whether the MLP window wraps around to the start of the region (`default`: only for loops; `force`; `disable`).
*   `--iterations <N>` — (Optional) Steady-state repetition multiplier (default: 3): the simulation runs enough iterations to fill the CPU's instruction window once for warm-up and N times for the measurement ([docs/docs.md](docs/docs.md) section 2.A).
*   `--override-load-latency <N>` — (Optional) Override load instruction latency (default: -1, inactive).
*   `--chain-threshold <K>` — (Optional) abab chain depth at which interlocking loops are merged (default: 5; see below).
*   `--merge-same-header` — (Optional) Keep only the widest loop span per loop header (see below).
*   `--facile-reason` — (Optional) With `--facile`, add a column naming the dominant bound (`inst`, `exec`, `prec`).
*   `--count-only` — (Optional) Only print the number of regions per section, without analysing them.
*   `--ignore-loop-carried <mode>` - (Optional) Mode for ignoring loop-carried register dependencies during cycle estimation.
    *   `default`: Ignores loop-carried dependencies in basic blocks, but considers them in loops (default behavior if option is omitted).
    *   `force`: Ignores loop-carried dependencies in both loops and basic blocks.
    *   `disable`: Considers loop-carried dependencies in both loops and basic blocks.

## Region Splitting & Merging Algorithm

`automatic-llvm-mca` automatically partitions ELF binary text sections into basic blocks and loop regions using an adaptive region splitting and merging algorithm implemented in `src/core/region_splitter.cpp` (details: [docs/docs.md](docs/docs.md) sections 2.D and 2.E):

1. **Loop Detection**:
   - Scans disassembled instructions per function (functions are delimited by the ELF function symbols).
   - Identifies backward branches that are not calls (`BranchTarget <= Addr` and `BranchTarget >= FuncBase`) to construct basic loop regions `[header_idx, latch_idx]`.
   - Drops a candidate loop that contains a return reachable from its header without a forward branch.

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

4. **Same-header merging (`--merge-same-header`, off by default)**:
   - Of several loop spans with the same header only the widest is kept.

5. **Non-loop code**:
   - In a function with loops, each run of instructions outside the kept loops is output as its own region with the metrics of a nearby loop of the function (`analysis_start`/`analysis_end` name that loop, `loop` = 1).
   - A function without loops is one non-loop region (`loop` = 0).

6. **Output order**:
   - Output regions are sorted by `EndAddr` ascending, then `StartAddr` ascending.

## Facile Analytical Throughput Predictor

When passing `--facile` to `mca_tool`, the tool computes steady-state basic-block throughput analytically based on the **Facile** model instead of full step-by-step cycle-by-cycle MCA simulation.

Facile calculates three independent throughput limits from LLVM's machine scheduling model and takes their maximum ($\max$):
1. **Issue Limit**: Dispatch width bottleneck ($\sum \text{uops} / \text{IssueWidth}$).
2. **Execution Ports Limit**: Contention on execution unit resource ports ($\max_p \text{ResourceCycles}(p) / \text{NumUnits}(p)$).
3. **Precedence Constraints Limit**: Maximum Cycle Ratio (MCR) of loop-carried Read-After-Write (RAW) dependency chains over **register** dependences, and also over **memory** (store→load) dependences with `--facile-memory-deps`.

$$\text{Facile Throughput (Cycles/Iter)} = \max\left( \text{Issue Limit},\; \text{Execution Ports Limit},\; \text{Precedence Limit} \right)$$

#### Memory dependences in the precedence graph

By default the precedence graph has register dependences only: a value that goes
through memory (a store followed by a load of the same address) adds no edge.
With `--facile-memory-deps`, `addMemoryDependencies()` in `src/facile/facile.cpp`
adds store→load edges, within an iteration and, for loops, across the backedge,
under a conservative may-alias rule (different base register ⇒ no alias; same
base with both displacements constant ⇒ alias only on an exact offset match;
same base with a register index ⇒ may alias unless provably disjoint; a base
redefined in between breaks the match), plus a dependence-distance test that
suppresses the loop-carried edge when both accesses have identical subscript
expressions over a loop-varying index (an `a[i] = f(a[i])` read-modify-write has
dependence distance 0). The edge latency is the store's own `WriteST` latency, as
the register path uses the producer's latency. See [docs/docs.md](docs/docs.md)
section 4.G.

### Reference & Citation
* **Paper**: *Facile: Fast, Accurate, and Interpretable Basic-Block Throughput Prediction*
* **Authors**: Shrey Sharma, Jan Reineke, Andreas Abel (Saarland University)
* **Preprint**: [arXiv:2310.13212 [cs.PF]](https://arxiv.org/abs/2310.13212) (2023)

## Source layout

```
src/
  core/      types, ELF loading + disassembly (object_loader), loop/basic-block splitting (region_splitter)
  cpu/       cpu_traits (per-CPU table), sched_model (installs the local ModifiedTarget/*.td models)
  a55/       Cortex-A55 specifics: in-order issue stage, register bypasses, instruction/cycle corrections
  a520/      Cortex-A520 in-order issue stage
  mca/       LLVM MCA driver (mca_simulation) and the steady-state event listeners
  mlp/       memory-level-parallelism analysis: one analyzer per ISA, window/heuristic logic
  facile/    Facile analytical predictor and the macro-op fusion tables
  frontend/  command-line options and target/CPU initialisation shared by the tools
apps/        the executables (mca_tool, mlp-objdump, mlp-check, mca-insts-info, analyze_distribution)
tests/       GoogleTest suites (mlp, splitter, facile, cpu_traits, and one <core>_sog_test per
             modelled Arm core: a55, a520, a76, a78, a720, x1) and shared fixtures
tools/       helper scripts (sog_compare.py)
ModifiedTarget/AArch64/   TableGen scheduling models
```

Everything under `src/` is one static library (`mca_common_lib`); headers are included as
`"<module>/<file>.h"`.

## Tests

Run the C++ unit tests (GoogleTest):

```bash
./build/mca_unit_tests
```

## Checking a scheduling model against an Arm SOG

`tools/sog_compare.py` compares the AArch64 instruction tables of an Arm Software
Optimization Guide (latency, throughput, utilized pipelines) with the scheduling model
this tool installs for the same core, and lists the opcodes that disagree with every
SOG row naming their mnemonic.

```bash
pdftotext -layout Arm_Cortex-X1_Core_Software_Optimization_Guide.pdf x1sog.txt
./build/mca-insts-info --mtriple aarch64-linux-gnu --mcpu cortex-x1 --format csv > x1.csv
tools/sog_compare.py --sog x1sog.txt --csv x1.csv
```

The SOGs are Arm's documents and are not part of this repository. The output is a triage list: it has false
positives (rows the SOG itself prints inconsistently, forms that cannot be told apart by
mnemonic). The script knows the resource names of the Neoverse-derived models only (Cortex-A76,
A78 and X1). Each modelled Arm core (Cortex-A55, A520, A76, A78, A720, X1) is compared with its SOG
item by item in `docs/<core>-sog.md` (listed in [docs/README.md](docs/README.md)). See the
script's docstring for details.

## Per-CPU parameters

Facts about a core that LLVM's scheduling model cannot express - which local `.td` model
is installed, the SOG's uop dispatch width and Mop dispatch width, which macro-op fusion
table applies, whether the model's zero-latency writes are rename-time moves - live in one
table, `src/cpu/cpu_traits.cpp` (`getCpuTraits()`). Adding a core or a variant of one is a
single line there.


## Misc

### Cortex-A55 Scheduling Model Customization
 * The Cortex-A55 model follows the *Cortex-A55 Software Optimization Guide (SOG)*; its conformance, item by item, and its deviations are in [docs/a55-sog.md](docs/a55-sog.md).
 * Includes a **flag bypass**: a conditional instruction (`csel`, a conditional branch, ...) can dual-issue with the flag-setting instruction before it (SOG Table 2), except in the cases listed in `docs/a55-sog.md` C-4.
 * The customized TableGen file is `ModifiedTarget/AArch64/AArch64SchedA55.td`; it is compiled into the tool's own tables with `llvm-tblgen-22` (see `CMakeLists.txt`).
