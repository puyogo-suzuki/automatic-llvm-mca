# automatic-llvm-mca
This automatically reports the estimated throughput for the given binary.

## Usage

```
python3 analyze.py [--mcpu <cpu>] <elf-binary>
```

The program disassembles the ELF binary with `objdump`, splits each function
into loops and basic blocks, runs `llvm-mca` on each region, and prints the
estimated IPC for every region:

```
0xSTART-0xEND IPC
```

For nested loops the outer loop (including the inner body) and the inner loop
are reported separately.

Supported architectures: x86/x86-64, AArch64, 32-bit ARM, RISC-V (RV32IC, RV64IC).

## How llvm-mca handles jump instructions

`llvm-mca` models throughput by simulating a **fixed instruction stream**
(the given code region) that repeats for a configurable number of iterations
(default: 100).  It does **not** simulate actual branch prediction or
speculative execution.

* **All instructions in the region are always executed** on every iteration,
  regardless of whether a conditional branch would be taken or not in real
  execution.
* A branch instruction (e.g. `jl`, `b.ne`, `bne`) contributes to throughput
  estimation only through its **resource consumption** — the execution unit(s)
  it occupies and the cycles it takes.  The branch outcome itself is ignored.
* Consequently, llvm-mca does **not** model branch misprediction penalties.
* For a loop body `analyze.py` includes the loop's back-edge branch in the
  region fed to llvm-mca, so its resource cost is counted once per iteration.
* For a basic block that ends with a conditional branch (but is not a loop),
  the branch is included in the region as well.

This means the reported IPC is a **best-case throughput** figure: it reflects
how fast the CPU can issue instructions assuming perfect branch prediction and
no misprediction bubbles.
