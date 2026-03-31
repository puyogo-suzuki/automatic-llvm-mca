#!/usr/bin/env python3
"""ipc_relate.py: Estimate CPI at varying instructions-per-cache-miss rates for ELF binaries.

For each basic block / loop in the binary, estimates CPI (= 1/IPC) at
instructions-per-cache-miss values of 1, 5, 10, 100, 1000, and infinity
(no cache miss) using llvm-mca.

Output CSV columns:
  start_address, end_address, load_proportion,
  cpi_ipcm1, cpi_ipcm5, cpi_ipcm10, cpi_ipcm100, cpi_ipcm1000, cpi_ipcm_inf

Usage:
  python3 ipc_relate.py [--mcpu <cpu>] [--cache-latency <cycles>] <elf-binary>
"""

import argparse
import math
import os
import sys

import analyze

# Instructions-per-cache-miss values to sweep over.
# float('inf') represents no cache miss.
_INSTRUCTIONS_PER_CACHE_MISS = [1, 5, 10, 100, 1000, float("inf")]


def _region_cpis(region, mca_args, arch: analyze.ArchBase, cache_latency: int,
                 cache_miss_mode: str = "stochastic"):
    """Run llvm-mca on *region* at each instructions-per-cache-miss value.

    Returns ``(cpis, load_proportion)`` where *cpis* is a list of CPI values
    (one per entry in ``_INSTRUCTIONS_PER_CACHE_MISS``) and *load_proportion*
    is taken from the no-miss run.  Returns ``None`` if llvm-mca fails for any
    value.
    """
    cpis = []
    load_proportion = None
    for ipcm in _INSTRUCTIONS_PER_CACHE_MISS:
        if load_proportion == 0:
            cpis.append(cpis[-1])
            continue
        cache_mode = analyze._build_cache_mode(ipcm, cache_latency,
                                               cache_miss_mode)
        result = analyze._run_mca(region, mca_args, arch, cache_mode)
        if result is None:
            return None
        ipc, lp = result
        if load_proportion is None:
            load_proportion = lp
        # ipc == 0 should not occur for valid llvm-mca output, but guard anyway.
        cpis.append(1.0 / ipc if ipc != 0.0 else float("inf"))
    return cpis, load_proportion


def ipc_relate(binary: str, mcpu: str = "", cache_latency: int = 100,
               cache_miss_mode: str = "stochastic"):
    """Analyse *binary* and yield CPI-vs-instructions-per-cache-miss tuples.

    Yields ``(start, end, load_proportion, cpi_ipcm1, cpi_ipcm5, cpi_ipcm10,
    cpi_ipcm100, cpi_ipcm1000, cpi_ipcm_inf)`` for every loop and non-loop
    basic block in the binary.

    Parameters
    ----------
    binary:
        Path to the ELF binary to analyse.
    mcpu:
        If non-empty, overrides the default ``-mcpu`` value chosen by
        architecture auto-detection and is forwarded to llvm-mca.
    cache_latency:
        Cache-miss penalty in cycles used for the ``# LLVM-MCA-LATENCY``
        directive when simulating finite instructions-per-cache-miss values.
    cache_miss_mode:
        ``"stochastic"`` (default) — a fraction of loads receive the full
        *cache_latency* penalty, derived from the instructions-per-cache-miss
        ratio.  ``"average"`` — all loads receive an average latency computed
        from the ratio.
    """
    arch = analyze._detect_arch(binary)
    mca_args = arch.mca_args
    if mcpu:
        mca_args = [a for a in mca_args if not a.startswith("-mcpu=")]
        mca_args = mca_args + [f"-mcpu={mcpu}"]

    for _func_name, instrs in analyze.disassemble(binary, arch):
        loops = analyze._find_loops(instrs, arch)

        # --- Loops ---
        for ls, le in loops:
            region = [(a, m, o) for a, m, o in instrs if ls <= a <= le]
            if not region:
                continue
            result = _region_cpis(region, mca_args, arch, cache_latency,
                                   cache_miss_mode)
            if result is not None:
                cpis, load_proportion = result
                yield (region[0][0], region[-1][0], load_proportion) + tuple(cpis)

        # --- Basic blocks outside loops ---
        non_loop = [(a, m, o) for a, m, o in instrs
                    if not analyze._in_any_loop(a, loops)]
        bb = []
        for instr in non_loop:
            addr, mnemonic, operands = instr
            bb.append(instr)
            if arch.ends_basic_block(mnemonic, operands):
                if bb:
                    result = _region_cpis(bb, mca_args, arch, cache_latency,
                                          cache_miss_mode)
                    if result is not None:
                        cpis, load_proportion = result
                        yield (bb[0][0], bb[-1][0], load_proportion) + tuple(cpis)
                bb = []
        if bb:
            result = _region_cpis(bb, mca_args, arch, cache_latency,
                                   cache_miss_mode)
            if result is not None:
                cpis, load_proportion = result
                yield (bb[0][0], bb[-1][0], load_proportion) + tuple(cpis)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate CPI at varying instructions-per-cache-miss values for "
            "an ELF binary using llvm-mca."
        )
    )
    parser.add_argument("binary", help="Path to the ELF binary to analyse")
    parser.add_argument(
        "--mcpu",
        default="",
        metavar="CPU",
        help=(
            "Target CPU passed to llvm-mca via -mcpu (e.g. cortex-a72, "
            "neoverse-n1, sifive-u74). Overrides the default CPU chosen by "
            "architecture auto-detection."
        ),
    )
    parser.add_argument(
        "--cache-latency",
        type=int,
        default=100,
        metavar="CYCLES",
        help=(
            "Cache-miss latency in cycles (>=0, default 100). "
            "Used as the latency value in the # LLVM-MCA-LATENCY directive "
            "for finite instructions-per-cache-miss values."
        ),
    )
    parser.add_argument(
        "--cache-miss-mode",
        choices=["stochastic", "average"],
        default="stochastic",
        dest="cache_miss_mode",
        help=(
            "Cache-miss simulation mode (default: stochastic). "
            "'stochastic': load instructions receive the full --cache-latency "
            "penalty according to the effective miss fraction derived from the "
            "instructions-per-cache-miss ratio. "
            "'average': all loads receive an average latency derived from the "
            "instructions-per-cache-miss ratio and --cache-latency."
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        parser.error(f"{args.binary}: no such file")
    if args.cache_latency < 0:
        parser.error("--cache-latency must be >= 0")

    def _col_name(ipcm):
        return "cpi_ipcm_inf" if math.isinf(ipcm) else f"cpi_ipcm{int(ipcm)}"

    ipcm_cols = ",".join(_col_name(ipcm) for ipcm in _INSTRUCTIONS_PER_CACHE_MISS)
    print(f"start_address,end_address,load_proportion,{ipcm_cols}")

    results = sorted(
        ipc_relate(args.binary, args.mcpu, args.cache_latency,
                   args.cache_miss_mode),
        # Sort by end address ascending; for equal end, larger start first
        # (inner loop before outer).  Matches analyze.py's output ordering.
        key=lambda x: (x[1], -x[0]),
    )
    for row in results:
        start, end, load_proportion = row[0], row[1], row[2]
        cpis = row[3:]
        cpi_str = ",".join(f"{c:.4f}" for c in cpis)
        print(f"0x{start:x},0x{end:x},{load_proportion:.4f},{cpi_str}")


if __name__ == "__main__":
    main()
