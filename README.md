# automatic-llvm-mca
This automatically reports the estimated throughput for the given binary.

## Usage

```
python3 analyze.py [--mcpu <cpu>] <elf-binary>
```

The program disassembles the ELF binary with `objdump`, splits each function
into loops and basic blocks, runs `llvm-mca` on each region, and prints a CSV
with the estimated throughput and proportion of load instructions for every
region:

```
start_address,end_address,throughput,load_proportion
0xSTART,0xEND,IPC,LOAD_PROPORTION
```

`load_proportion` is the fraction of instructions in the region that carry the
`MayLoad` attribute as reported by `llvm-mca -instruction-info` (e.g. `0.2500`
means 25 % of instructions may perform a memory load).

For nested loops the outer loop (including the inner body) and the inner loop
are reported separately.

Supported architectures: x86/x86-64, AArch64, 32-bit ARM, RISC-V (RV32IC, RV64IC).

### Cache-miss sensitivity (CPI vs instructions-per-cache-miss)

`ipc_relate.py` reports how the estimated CPI (the reciprocal of IPC) changes as
the instructions-per-cache-miss (IPCM) ratio varies.  A lower IPCM means more
frequent cache misses.

```
python3 ipc_relate.py [--mcpu <cpu>] [--cache-latency <cycles>] [--cache-miss-mode {stochastic,average}] [--ipcm-values IPCM [IPCM ...]] <elf-binary>
```

* `--cache-latency` (default: 100) sets the latency in cycles used for simulated
  cache misses via `# LLVM-MCA-LATENCY` directives.
* `--cache-miss-mode` (default: `stochastic`) controls how misses are distributed
  across load instructions.  `stochastic` gives a fraction of loads the full
  `--cache-latency` penalty; `average` gives all loads an averaged latency.
* `--ipcm-values` (default: `1 10 20 50 100 inf`) sets the list of
  instructions-per-cache-miss values to sweep over.  Provide one or more
  space-separated positive numbers and/or `inf` (no cache miss).
  Example: `--ipcm-values 1 10 100 inf`
* The script runs `llvm-mca` for each loop and basic block at the specified IPCM
  values and writes:

```
start_address,end_address,load_proportion,cpi_ipcm1,cpi_ipcm10,cpi_ipcm20,cpi_ipcm50,cpi_ipcm100,cpi_ipcm_inf
0xSTART,0xEND,LOAD_PROP,CPI_IPCM1,CPI_IPCM10,CPI_IPCM20,CPI_IPCM50,CPI_IPCM100,CPI_IPCM_INF
```

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
