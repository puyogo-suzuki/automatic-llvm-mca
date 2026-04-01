# automatic-llvm-mca
This automatically reports the estimated throughput for the given binary.

## Usage

```
python3 analyze.py [--mcpu <cpu>] <elf-binary>
```

The program disassembles the ELF binary with `objdump`, splits each function
into loops and basic blocks, runs `llvm-mca` on each region, and prints a CSV
with the retired instructions, elapsed cycles, and proportion of load
instructions for every region:

```
start_address,end_address,retired_instructions,elapsed_cycles,load_proportion
0xSTART,0xEND,RETIRED,CYCLES,LOAD_PROPORTION
```

`retired_instructions` is the total number of instructions retired across all
llvm-mca iterations for the region.  `elapsed_cycles` is the total number of
cycles elapsed.  Dividing `retired_instructions` by `elapsed_cycles` gives the
IPC estimate for the region.

`load_proportion` is the fraction of instructions in the region that carry the
`MayLoad` attribute as reported by `llvm-mca -instruction-info` (e.g. `0.2500`
means 25 % of instructions may perform a memory load).

For nested loops the outer loop (including the inner body) and the inner loop
are reported separately.

Supported architectures: x86/x86-64, AArch64, 32-bit ARM, RISC-V (RV32IC, RV64IC).

### Dump mode

`analyze.py --dump` writes the formatted assembly for each analysed region to a
text file in a `dump/` directory.  Each file is named
`{start_address}_{end_address}.{arch}.txt`:

```
python3 analyze.py --dump [--mcpu <cpu>] <elf-binary>
```

### Re-analysing a dump file (`analyze_str.py`)

`analyze_str.py` reads a single dump file produced by `analyze.py --dump` and
runs `llvm-mca` on it, printing the same CSV format as `analyze.py`:

```
python3 analyze_str.py [--mcpu <cpu>] [options] <textfile>
```

The textfile must follow the `{start}_{end}.{arch}.txt` naming convention (e.g.
`dump/1000_1080.x86.txt`).  The architecture and the start/end addresses are
inferred from the filename.

Optional cache-miss simulation can be applied to the dump on the fly:

```
python3 analyze_str.py --instructions-per-cache-miss 50 --cache-latency 200 dump/1000_1080.x86.txt
```

* `--mcpu` overrides the default CPU inferred from the filename.
* `--instructions-per-cache-miss` (default: `inf`, i.e. no simulation) sets
  the number of retired instructions per cache miss.
* `--cache-latency` (default: 0) sets the simulated cache-miss penalty in cycles.
* `--cache-miss-mode` (default: `stochastic`) selects the simulation mode
  (`stochastic`, `average`, or `early`); see `analyze.py` for a full description.

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
