# automatic-llvm-mca

Estimate throughput for ELF binaries using `llvm-mca`.

## Procedure

1.  Disassemble the ELF binary with `objdump`.
2.  Decompose each function into loops and (for non-loop code) basic blocks.
3.  Run `llvm-mca` on each region to obtain retired instructions and elapsed
    cycles.
4.  Print a CSV with start address, end address, retired instructions, load
    instructions, and elapsed cycles for every region.

For nested loops the outer loop (including the inner loop body) and the inner
loop are reported separately.

## Supported Architectures

*   **x86 / x86-64**
*   **AArch64** (64-bit ARM)
*   **32-bit ARM**
*   **RISC-V** (RV32IC, RV64IC)

## Usage

```bash
python3 analyze.py [--mcpu <cpu>] <elf-binary>
```

*   `<elf-binary>` — Path to the ELF binary to analyze.
*   `--mcpu <cpu>` — (Optional) Specify a target CPU for `llvm-mca` (e.g.,
    `cortex-a72`, `skylake`, `sifive-u74`). If omitted, the tool attempts to
    auto-detect the host CPU (for x86) or uses a generic model.

### Output format

The output is a CSV with the following columns:

*   `start_address` — Hex address of the first instruction in the region.
*   `end_address` — Hex address of the last instruction in the region.
*   `retired_instructions` — Total number of instructions retired in the region
    (simulated over 100 iterations by default).
*   `load_instructions` — Total number of load instructions retired in the
    region.
*   `cycles` — Total simulated cycles for the region.

## Debugging / Dumping Assembly (`--dump`)

The `--dump` flag writes the formatted assembly for every analyzed region to a
`dump/` directory:

```bash
python3 analyze.py --dump <elf-binary>
```

Files are named `{start}_{end}.{arch}.txt`. These files contain the exact
assembly passed to `llvm-mca`, including labels and `LLVM-MCA-BEGIN` /
`LLVM-MCA-END` markers (simulated by our runner).

## Analysis from dump files (`analyze_str.py`)

You can run `llvm-mca` on a previously dumped assembly file:

```bash
python3 analyze_str.py [--mcpu <cpu>] <textfile>
```

This is useful for manually inspecting the assembly or for re-running the
simulation with a different CPU model without needing the original ELF binary.
The output format is the same as `analyze.py`.
