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

### Build

```bash
mkdir -p build && cd build
cmake ..
make
```

This will produce the standalone tool `build/mca_tool` and the unit tests `build/mca_unit_tests`.

## Usage

```bash
./build/mca_tool [--mcpu <cpu>] [--mtriple <triple>] [--window-width <W>] [--dependency <mode>] [--mlp-window-assignment <mode>] [--iterations <N>] <elf-binary>
./build/mlp-objdump [--mcpu <cpu>] [--mtriple <triple>] [--window-width <W>] [--dependency <mode>] [--mlp-window-assignment <mode>] [--iterations <N>] <elf-binary>
```

*   `<elf-binary>` — Path to the ELF binary to analyze.
*   `mlp-objdump` — Disassembles text sections and prints per-basic-block MLP/baseCPI next to each instruction address.
*   `--mcpu <cpu>` — (Optional) Specify a target CPU (e.g., `cortex-a72`, `haswell`, `sifive-u74`).
*   `--mtriple <triple>` — (Optional) Specify a target triple (e.g., `x86_64-linux-gnu`).
*   `--window-width <W>` — (Optional) Window width for MLP estimation (default: 4).
*   `--dependency <mode>` — (Optional) MLP dependency mode (`none`, `io`, `ooo`).
*   `--mlp-window-assignment <mode>` — (Optional) MLP assignment mode (`forward`, `max-containing`).
*   `--iterations <N>` — (Optional) Number of MCA iterations (default: 100).

Note: `call-latency` is fixed to 0.

## Tests

Run the C++ unit tests (GoogleTest):

```bash
./build/mca_unit_tests
```
