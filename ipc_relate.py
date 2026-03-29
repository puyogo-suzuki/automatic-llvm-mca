#!/usr/bin/env python3
"""ipc_relate.py: Estimate CPI at varying cache-miss rates for ELF binaries.

For each basic block / loop in the binary, estimates CPI (= 1/IPC) at
cache-miss rates of 0%, 10%, 20%, 30%, 40%, and 50% using llvm-mca.

Output CSV columns:
  start_address, end_address, load_proportion,
  cpi0, cpi10, cpi20, cpi30, cpi40, cpi50

Usage:
  python3 ipc_relate.py [--mcpu <cpu>] [--cache-latency <cycles>] <elf-binary>
"""

import argparse
import os
import sys

import analyze

# Cache-miss rates (as fractions) to sweep over.
_CACHE_MISS_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def _region_cpis(region, mca_args, arch, cache_latency):
    """Run llvm-mca on *region* at each cache-miss rate.

    Returns ``(cpis, load_proportion)`` where *cpis* is a list of CPI values
    (one per entry in ``_CACHE_MISS_RATES``) and *load_proportion* is taken
    from the zero-miss-rate run.  Returns ``None`` if llvm-mca fails for any
    rate.
    """
    cpis = []
    load_proportion = None
    for miss_rate in _CACHE_MISS_RATES:
        result = analyze._run_mca(region, mca_args, arch,
                                   miss_rate, cache_latency)
        if result is None:
            return None
        ipc, lp = result
        if load_proportion is None:
            load_proportion = lp
        # ipc == 0 should not occur for valid llvm-mca output, but guard anyway.
        cpis.append(1.0 / ipc if ipc != 0.0 else float("inf"))
    return cpis, load_proportion


def ipc_relate(binary: str, mcpu: str = "", cache_latency: int = 100):
    """Analyse *binary* and yield CPI-vs-cache-miss tuples.

    Yields ``(start, end, load_proportion, cpi0, cpi10, cpi20, cpi30, cpi40,
    cpi50)`` for every loop and non-loop basic block in the binary.

    Parameters
    ----------
    binary:
        Path to the ELF binary to analyse.
    mcpu:
        If non-empty, overrides the default ``-mcpu`` value chosen by
        architecture auto-detection and is forwarded to llvm-mca.
    cache_latency:
        Cache-miss penalty in cycles used for the ``# LLVM-MCA-LATENCY``
        directive when simulating non-zero cache-miss rates.
    """
    arch_info = analyze._detect_arch(binary)
    mca_args = arch_info.mca_args
    if mcpu:
        mca_args = [a for a in mca_args if not a.startswith("-mcpu=")]
        mca_args = mca_args + [f"-mcpu={mcpu}"]

    for _func_name, instrs in analyze.disassemble(binary, arch_info.objdump,
                                                   arch_info.name):
        loops = analyze._find_loops(instrs, arch_info.name)

        # --- Loops ---
        for ls, le in loops:
            region = [(a, m, o) for a, m, o in instrs if ls <= a <= le]
            if not region:
                continue
            result = _region_cpis(region, mca_args, arch_info.name,
                                   cache_latency)
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
            if analyze._ends_basic_block(mnemonic, operands, arch_info.name):
                if bb:
                    result = _region_cpis(bb, mca_args, arch_info.name,
                                          cache_latency)
                    if result is not None:
                        cpis, load_proportion = result
                        yield (bb[0][0], bb[-1][0], load_proportion) + tuple(cpis)
                bb = []
        if bb:
            result = _region_cpis(bb, mca_args, arch_info.name, cache_latency)
            if result is not None:
                cpis, load_proportion = result
                yield (bb[0][0], bb[-1][0], load_proportion) + tuple(cpis)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate CPI at varying cache-miss rates for an ELF binary "
            "using llvm-mca."
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
            "for non-zero cache-miss rates."
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        parser.error(f"{args.binary}: no such file")
    if args.cache_latency < 0:
        parser.error("--cache-latency must be >= 0")

    miss_cols = ",".join(
        f"cpi{int(r * 100)}" for r in _CACHE_MISS_RATES
    )
    print(f"start_address,end_address,load_proportion,{miss_cols}")

    results = sorted(
        ipc_relate(args.binary, args.mcpu, args.cache_latency),
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
