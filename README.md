# automatic-llvm-mca

Estimate throughput for ELF binaries using a direct C++ interface to LLVM MCA.

## Procedure

1.  Disassemble the ELF binary with `objdump`.
2.  Decompose each function into loops and (for non-loop code) basic blocks.
3.  Run the LLVM Machine Code Analyzer (MCA) on each region to obtain retired instructions and elapsed cycles.
4.  Print a CSV with start address, end address, retired instructions, load
    instructions, elapsed cycles, and Memory Level Parallelism (MLP) for every region.

This tool now uses a custom C++ shared library to interface directly with the LLVM MCA API, significantly improving performance by avoiding process creation overhead for every code region.

## Supported Architectures

*   **x86 / x86-64**
*   **AArch64** (64-bit ARM)
*   **32-bit ARM**
*   **RISC-V** (RV32IC, RV64IC)

## Build Instructions

The C++ library must be built before using the tool.

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

This will produce `build/libmca_runner.so`, which is required by `analyze.py`.

## Usage

```bash
python3 analyze.py [--mcpu <cpu>] [--march <arch>] [--window-width <W>] [--dependency <mode>] [--mlp-window-assignment <mode>] <elf-binary>
```

*   `<elf-binary>` — Path to the ELF binary to analyze.
*   `--mcpu <cpu>` — (Optional) Specify a target CPU for MCA (e.g., `cortex-a72`, `haswell`, `sifive-u74`).
*   `--march <arch>` — (Optional) Specify a target architecture (e.g., `x86-64`, `aarch64`).
*   `--window-width <W>` — (Optional) Specify the window width for MLP estimation (default is 4).
*   `--dependency <mode>` — (Optional) Specify the dependency tracking mode for MLP estimation (`none`, `io`, `ooo`).
*   `--mlp-window-assignment <mode>` — (Optional) Per-load MLP assignment mode (`forward`, `max-containing`).

Note: `call-latency` is fixed to 0 in this implementation.

## Tests

Tests can be run using `pytest` within the provided virtual environment:

```bash
./venv/bin/pytest tests/
```
