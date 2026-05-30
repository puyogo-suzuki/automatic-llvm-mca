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

*   LLVM 23 development headers (`llvm-23-dev`)
*   CMake 3.10+
*   C++17 compatible compiler

To compile the customized TableGen-based pipeline, you need a local copy of the LLVM AArch64 target sources. Clone `llvm-project` into a directory named `llvm-source` at the project root using a sparse checkout (to save disk space and download time):

```bash
git clone --depth 1 --sparse https://github.com/llvm/llvm-project.git llvm-source
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

This will produce the main tool `build/mca_tool`, the secondary tools `build/mlp-objdump` and `build/mca-insts-info`, and the unit tests `build/mca_unit_tests`.

## Added/Updated Features and Tools

### 1. New Features
*   **`--ignore-loop-carried <mode>`**: Mode for ignoring loop-carried register dependencies during cycle estimation (options: `default`, `force`, `disable`).
    *   `default`: Ignores loop-carried dependencies in basic blocks, but considers them in loops (default behavior if option is omitted).
    *   `force`: Ignores loop-carried dependencies in both loops and basic blocks.
    *   `disable`: Considers loop-carried dependencies in both loops and basic blocks.
*   **`--override-load-latency <N>`**: Overrides load instruction latencies.
*   **Cortex-A55 Scheduling Model Customization**:
    *   Models the physical hardware of Cortex-A55 accurately based on the *Cortex-A55 Software Optimization Guide (SOG)*.
    *   Includes a specialized **0-cycle same-cycle flag bypass** (`cmp` $\to$ `csel`/conditional branch) which prevents artificial 1-cycle flag latency stalls, bringing the simulated CPI of tight loop kernels down under 1.0 (matching actual hardware).
    *   Customized TableGen files are located in `ModifiedTarget/AArch64/AArch64SchedA55.td` and integrated into the build pipeline using `llvm-tblgen-23`.

### 2. Standalone Cost Dumper (`mca-insts-info`)
Outputs a comprehensive table of all target machine instructions along with their TableGen internal names, assembly mnemonics, scheduling classes, execution latencies, reciprocal throughputs, and resource usages. It dynamically resolves variant scheduling classes using register-class operand matching.

```bash
./build/mca-insts-info --mtriple aarch64-linux-gnu --mcpu cortex-a55 [--format csv/tsv] > output.csv
```

## Usage

```bash
./build/mca_tool [--mcpu <cpu>] [--mtriple <triple>] [--window-width <W>] [--dependency <mode>] [--mlp-window-assignment <mode>] [--iterations <N>] [--ignore-loop-carried] [--override-load-latency <N>] <elf-binary>

./build/mlp-objdump [--mcpu <cpu>] [--mtriple <triple>] [--window-width <W>] [--dependency <mode>] [--mlp-window-assignment <mode>] [--iterations <N>] [--ignore-loop-carried] [--override-load-latency <N>] <elf-binary>
```

*   `<elf-binary>` — Path to the ELF binary to analyze.
*   `mlp-objdump` — Disassembles text sections and prints per-basic-block MLP/baseCPI next to each instruction address.
*   `--mcpu <cpu>` — (Optional) Specify a target CPU (e.g., `cortex-a55`, `cortex-a72`, `haswell`).
*   `--mtriple <triple>` — (Optional) Specify a target triple (e.g., `aarch64-linux-gnu`).
*   `--window-width <W>` — (Optional) Window width for MLP estimation (default: 4).
*   `--dependency <mode>` — (Optional) MLP dependency mode (`none`, `io`, `ooo`).
*   `--mlp-window-assignment <mode>` — (Optional) MLP assignment mode (`forward`, `max-containing`).
*   `--iterations <N>` — (Optional) Steady-state repetition multiplier (default: 100).
*   `--ignore-loop-carried <mode>` — (Optional) Ignore loop-carried register dependencies mode (`default`, `force`, or `disable`; default: `default`).
*   `--override-load-latency <N>` — (Optional) Override load instruction latency (default: -1, inactive).

## Tests

Run the C++ unit tests (GoogleTest):

```bash
./build/mca_unit_tests
```
