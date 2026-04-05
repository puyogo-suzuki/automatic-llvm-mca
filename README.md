# automatic-llvm-mca
This automatically reports the estimated throughput for the given binary.

## Requirements

* **Python 3.8+**
* **llvm-mca 21 or later** — versions prior to 21 do not support the
  `--call-latency` flag, which is required for accurate cycle estimates on code
  regions that contain CALL instructions.
* **objdump** (binutils) — used to disassemble the target ELF binary.

## Usage

```
python3 analyze.py [--mcpu <cpu>] [--dump] <elf-binary> [<cache-latency> <cycles_0> <cycles_1> ...]
```

The program disassembles the ELF binary with `objdump`, splits each function
into loops and basic blocks, runs `llvm-mca` on each region, and prints a CSV
with the retired instructions, load instructions count, and cycles for every region:

```
start_address,end_address,retired_instructions,load_instructions,cycles[,cycles_0,cycles_1,...]
0xSTART,0xEND,RETIRED,LOAD_INSTRUCTIONS,CYCLES[,CYCLES_0,CYCLES_1,...]
```

`retired_instructions` is the total number of instructions retired across all
llvm-mca iterations for the region.  `load_instructions` is the number of load
instructions in the region (those that carry the `MayLoad` attribute).
`cycles` is the base cycle count without cache-miss simulation.  Dividing
`retired_instructions` by `cycles` gives the IPC estimate for the region.

When cache metrics are specified, additional `cycles_N` columns are added,
showing the cycle count with different cache-miss configurations.

For nested loops the outer loop (including the inner body) and the inner loop
are reported separately.

Supported architectures: x86/x86-64, AArch64, 32-bit ARM, RISC-V (RV32IC, RV64IC).

### Cache-miss simulation (`analyze.py`)

Cache-miss simulation can be enabled by specifying positional arguments after
the ELF binary path:

```
python3 analyze.py <elf-binary> <cache-latency> <cycles_0> [<cycles_1> ...]
```

The `<cache-latency>` argument (required when specifying any cycles metrics)
sets the simulated cache-miss penalty in cycles.

Each `<cycles_N>` argument specifies a cache-miss configuration and must be one of:

* `cm_<rate>` — Cache-miss **rate** per retired **load** instruction.  Values
  between 0 and 1 represent a miss probability (e.g. `cm_0.1` means 10 % of
  loads miss); values greater than 1 are also valid (e.g. `cm_1.5` means each
  load causes an average of 1.5 misses, as can happen in a multi-level cache
  hierarchy).

* `ipcm_<n>` — **Instructions** (all instructions) **per cache miss**.  Use
  `ipcm_inf` for no cache-miss simulation.  For example, `ipcm_50` means one
  cache miss occurs for every 50 retired instructions.

* `lipcm_<n>` — **Load instructions per cache miss**.  Use `lipcm_inf` for no
  cache-miss simulation.  For example, `lipcm_10` means one cache miss occurs
  for every 10 load instructions.  This is the reciprocal of `cm_<rate>`.

Example:

```
python3 analyze.py mybinary 200 cm_0.1 ipcm_50 lipcm_10
```

This will output:

* `cycles` — base cycles with no cache misses
* `cycles_0` — cycles with cache-miss rate 0.1 (10 % of loads miss) at 200 cycles per miss
* `cycles_1` — cycles with one cache miss per 50 instructions at 200 cycles per miss
* `cycles_2` — cycles with one cache miss per 10 load instructions at 200 cycles per miss

The `--cache-miss-mode` flag (default: `stochastic`) selects the simulation mode
(`stochastic`, `average`, or `early`):

* `stochastic`: the code block is repeated 100 times and load instructions
  receive the full `<cache-latency>` penalty according to the effective miss
  fraction; llvm-mca is run with `-iterations=1`.
* `average`: all load instructions receive a fixed latency derived from the
  miss fraction multiplied by `<cache-latency>` cycles.
* `early`: like `stochastic` but all cache misses are placed on the first
  loads in the repeated block rather than distributed uniformly.

### Dump mode

`analyze.py --dump` writes the formatted assembly for each analysed region to a
text file in a `dump/` directory.  Each file is named
`{start_address}_{end_address}.{arch}.txt`:

```
python3 analyze.py --dump [--mcpu <cpu>] <elf-binary> [<cache-latency> <cycles_spec>...]
```

### Re-analysing a dump file (`analyze_str.py`)

`analyze_str.py` reads a single dump file produced by `analyze.py --dump` and
runs `llvm-mca` on it, printing the same CSV format as `analyze.py`:

```
python3 analyze_str.py [--mcpu <cpu>] <textfile> [<cache-latency> <cycles_spec>...]
```

The textfile must follow the `{start}_{end}.{arch}.txt` naming convention (e.g.
`dump/1000_1080.x86.txt`).  The architecture and the start/end addresses are
inferred from the filename.

Optional cache-miss simulation can be applied to the dump on the fly:

```
python3 analyze_str.py dump/1000_1080.x86.txt 200 cm_0.1 ipcm_50 lipcm_10
```

* `--mcpu` overrides the default CPU inferred from the filename.
* Cache-miss specifications (`cm_`, `ipcm_`, `lipcm_`) work the same as in
  `analyze.py`.
* `--cache-miss-mode` (default: `stochastic`) selects the simulation mode
  (`stochastic`, `average`, or `early`); see the cache-miss simulation section
  above for a full description.

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
