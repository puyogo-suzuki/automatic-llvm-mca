#!/usr/bin/env python3
"""analyze_str.py: Estimate throughput for a dump-mode assembly file.

Accepts a text file in the format produced by ``analyze.py --dump`` (filename
``{start}_{end}.{arch}.txt``) and runs llvm-mca on its contents, reporting the
same CSV output as ``analyze.py``.

The assembly text is re-parsed into instruction tuples so that cache-miss
simulation (``--instructions-per-cache-miss``, ``--cache-latency``,
``--cache-miss-mode``) can be applied on demand, just as in ``analyze.py``.

Usage::

    python3 analyze_str.py [--mcpu <cpu>] [options] <textfile>
"""

import argparse
import math
import os
import re
import sys

from analyze import (
    _build_cache_mode,
    _build_cache_mode_from_rate,
    _run_mca,
)
from arch import X86Arch, AArch64Arch, ARMArch, RISCVArch, ArchBase


# Map arch.name → zero-argument arch constructor.
_ARCH_MAP: dict[str, type[ArchBase]] = {
    "x86": X86Arch,
    "aarch64": AArch64Arch,
    "arm": ARMArch,
    "riscv": RISCVArch,
}


def _arch_from_name(arch_name: str) -> ArchBase:
    """Return an :class:`~arch.ArchBase` instance for *arch_name*.

    Raises :class:`ValueError` when *arch_name* is not recognised.
    """
    cls = _ARCH_MAP.get(arch_name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown architecture {arch_name!r}. "
            f"Supported: {', '.join(sorted(_ARCH_MAP))}."
        )
    return cls()


def _parse_filename(path: str):
    """Parse ``{start}_{end}.{arch}.txt`` filename.

    Returns ``(start_int, end_int, arch_name)`` on success.
    Raises :class:`ValueError` when the filename does not match the expected
    format.
    """
    basename = os.path.basename(path)
    m = re.fullmatch(
        r"([0-9a-fA-F]+)_([0-9a-fA-F]+)\.([^.]+)\.txt", basename
    )
    if not m:
        raise ValueError(
            f"Cannot parse filename {basename!r}. "
            "Expected format: {start}_{end}.{arch}.txt "
            "(e.g. 1000_1080.x86.txt)."
        )
    start = int(m.group(1), 16)
    end = int(m.group(2), 16)
    arch_name = m.group(3)
    return start, end, arch_name


def _parse_instrs_from_dump(text: str) -> list:
    """Parse ``(addr, mnemonic, operands)`` tuples from a dump-file string.

    The dump format produced by ``analyze.py --dump`` looks like::

        .Lmca_1c:
            movq (%rdi,%rax,8), %rax
            addq %rax, %rcx
            jne .Lmca_1c
        .Lmca_end:

    This function:

    * Extracts instruction addresses from ``.Lmca_{hex}:`` label lines and
      assigns them to the following instruction.  Unlabeled instructions
      receive sequential fake addresses that do not overlap with labeled ones.
    * Replaces ``.Lmca_{hex}`` label references inside branch operands with
      their plain hex address strings so that
      :func:`~arch.ArchBase.get_branch_target` can parse them and the normal
      :func:`~analyze._compute_labeled_addrs` / :func:`~analyze._format_asm`
      machinery can re-format the block correctly.
    * Skips ``# LLVM-MCA-LATENCY`` directive lines and ``.Lmca_end:`` labels.
    """
    # Pass 1 — build a map from label names to addresses.
    label_map: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"^\.Lmca_([0-9a-fA-F]+):$", line.strip())
        if m:
            label_map[f".Lmca_{m.group(1)}"] = int(m.group(1), 16)

    # Pass 2 — parse instructions, assigning addresses.
    instrs: list = []
    pending_addr: int | None = None
    auto_addr = 0

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # .Lmca_{addr}: label — supplies the address for the next instruction.
        m = re.match(r"^\.Lmca_([0-9a-fA-F]+):$", stripped)
        if m:
            pending_addr = int(m.group(1), 16)
            continue

        # .Lmca_end: or any other assembler directive / label — skip.
        if stripped.startswith("."):
            continue

        # # LLVM-MCA-LATENCY directive — skip.
        if stripped.startswith("#"):
            continue

        # Instruction line (tab-indented in the dump format).
        if not line.startswith("\t"):
            continue

        parts = stripped.split(None, 1)
        mnemonic = parts[0]
        operands = parts[1] if len(parts) > 1 else ""

        # Replace .Lmca_{hex} references in operands with plain hex so that
        # get_branch_target() can recover the numeric target.
        for label, addr in label_map.items():
            if label in operands:
                operands = operands.replace(label, f"{addr:x}")

        if pending_addr is not None:
            addr = pending_addr
            pending_addr = None
            auto_addr = addr + 1
        else:
            addr = auto_addr
            auto_addr += 1

        instrs.append((addr, mnemonic, operands))

    return instrs


def load_str_file(path: str):
    """Load a dump file.

    Parameters
    ----------
    path:
        Path to a text file in the format produced by ``analyze.py --dump``
        (filename ``{start}_{end}.{arch}.txt``).
    """
    start, end, arch_name = _parse_filename(path)
    arch = _arch_from_name(arch_name)

    with open(path, encoding="utf-8") as f:
        text = f.read()

    instrs = _parse_instrs_from_dump(text)

    return instrs, arch, start, end

def analyze_str(instrs, arch: ArchBase, mcpu: str = "",
                cache_miss: float = 0.0,
                cache_latency: int = 0,
                cache_miss_mode: str = "stochastic",
                cache_miss_rate: float = float("inf")):
    """Run llvm-mca on pre-loaded instruction tuples.

    Parameters
    ----------
    instrs:
        List of ``(addr, mnemonic, operands)`` triples, as returned by
        :func:`load_str_file`.
    arch:
        Architecture object, as returned by :func:`load_str_file`.
    mcpu:
        If non-empty, overrides the default ``-mcpu`` value chosen by the
        architecture and is forwarded to llvm-mca.
    cache_miss:
        Number of retired instructions per cache miss.  Use ``0.0``
        (default) for no cache-miss simulation.
        Mutually exclusive with *cache_miss_rate*.
    cache_latency:
        Cache-miss penalty in cycles.  Only used when *cache_miss* or
        *cache_miss_rate* is finite.
    cache_miss_mode:
        ``"stochastic"`` (default), ``"average"``, or ``"early"``.  See
        :func:`analyze.analyze` for details.
    cache_miss_rate:
        Cache misses per retired load instruction.  Use ``float('inf')``
        (default) for no cache-miss simulation.  May be greater than 1.
        When finite, takes precedence over *cache_miss*.
        Mutually exclusive with *cache_miss*.

    Returns
    -------
    ``(retired_instructions, elapsed_cycles, load_instructions)`` on success,
    or ``None`` when llvm-mca produces no result.
    """
    if cache_miss_rate > 0.0:
        cache_mode = _build_cache_mode_from_rate(cache_miss_rate, cache_latency,
                                                 cache_miss_mode)
    else:
        cache_mode = _build_cache_mode(cache_miss, cache_latency, cache_miss_mode)

    # Build llvm-mca argument list, optionally overriding -mcpu.
    mca_args = arch.mca_args
    if mcpu:
        mca_args = [a for a in mca_args if not a.startswith("-mcpu=")]
        mca_args = mca_args + [f"-mcpu={mcpu}"]

    return _run_mca(instrs, mca_args, arch=arch, cache_mode=cache_mode)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate throughput for a dump-mode assembly file generated by "
            "analyze.py --dump."
        ),
        usage="%(prog)s [--mcpu CPU] <textfile> [<cache-latency> <cycles_spec>...]"
    )
    parser.add_argument(
        "textfile",
        help=(
            "Path to a text file produced by 'analyze.py --dump'. "
            "The filename must follow the {start}_{end}.{arch}.txt convention "
            "(e.g. 1000_1080.x86.txt)."
        ),
    )
    parser.add_argument(
        "metrics",
        nargs="*",
        metavar="ARGS",
        help=(
            "Optional: <cache-latency> <cycles_0> [<cycles_1> ...] where "
            "<cycles_n> is one of: cm_N (cache-miss rate N), ipcm_N "
            "(instructions per cache miss N), or lipcm_N (load instructions "
            "per cache miss N). Example: 200 cm_0.1 ipcm_50 lipcm_10"
        ),
    )
    parser.add_argument(
        "--mcpu",
        default="",
        metavar="CPU",
        help=(
            "Target CPU passed to llvm-mca via -mcpu (e.g. cortex-a72, "
            "neoverse-n1, sifive-u74). Overrides the default CPU inferred "
            "from the filename."
        ),
    )
    parser.add_argument(
        "--cache-miss-mode",
        choices=["stochastic", "average", "early"],
        default="stochastic",
        dest="cache_miss_mode",
        help=(
            "Cache-miss simulation mode (default: stochastic). "
            "'stochastic': the code block is repeated 100 times and load "
            "instructions receive the full --cache-latency penalty according "
            "to the effective miss fraction; llvm-mca is run with "
            "-iterations=1. "
            "'average': all load instructions receive a fixed latency derived "
            "from the miss fraction multiplied by --cache-latency cycles. "
            "'early': like 'stochastic' but all cache misses are placed on "
            "the first loads in the repeated block rather than distributed "
            "uniformly (e.g. miss miss miss hit hit instead of "
            "miss hit miss hit miss)."
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.textfile):
        parser.error(f"{args.textfile}: no such file")

    # Parse metrics arguments
    if len(args.metrics) == 0:
        # No cache metrics specified - run without cache miss simulation
        cache_specs = []
        cache_latency = 0
    elif len(args.metrics) == 1:
        parser.error("cache-latency requires at least one cycles_N specification")
    else:
        try:
            cache_latency = int(args.metrics[0])
            if cache_latency < 0:
                parser.error("cache-latency must be >= 0")
        except ValueError:
            parser.error(f"cache-latency must be an integer, got: {args.metrics[0]}")
        
        cache_specs = []
        for spec in args.metrics[1:]:
            cache_specs.append(_parse_cache_spec_str(spec, parser))

    instrs, arch, start, end = load_str_file(args.textfile)

    # Run analysis for each cache specification
    cycles_list = []
    
    if len(cache_specs) == 0:
        # No cache miss simulation
        result = analyze_str(instrs, arch, args.mcpu,
                           cache_miss=float("inf"),
                           cache_latency=0,
                           cache_miss_mode=args.cache_miss_mode,
                           cache_miss_rate=0.0)
        if result is None:
            return
        retired, cycles, load_instrs = result
        
        print(f"start_address,end_address,retired_instructions,load_instructions,cycles")
        print(f"0x{start:x},0x{end:x},{retired},{load_instrs},{cycles}")
    else:
        # Run analysis for each cache specification
        retired = None
        load_instrs = None
        base_cycles = None
        
        for i, (spec_type, spec_value) in enumerate(cache_specs):
            if spec_type == "cm":
                # Cache miss rate
                cache_miss_rate = spec_value
                cache_miss = float("inf")
            elif spec_type == "ipcm":
                # Instructions per cache miss
                cache_miss = spec_value
                cache_miss_rate = 0.0
            elif spec_type == "lipcm":
                # Load instructions per cache miss - convert to cache miss rate
                if spec_value == 0:
                    parser.error(f"lipcm value cannot be 0")
                if math.isinf(spec_value):
                    # No cache misses
                    cache_miss_rate = 0.0
                    cache_miss = float("inf")
                else:
                    cache_miss_rate = 1.0 / spec_value
                    cache_miss = float("inf")
            
            result = analyze_str(instrs, arch, args.mcpu,
                               cache_miss=cache_miss,
                               cache_latency=cache_latency,
                               cache_miss_mode=args.cache_miss_mode,
                               cache_miss_rate=cache_miss_rate)
            
            if result is None:
                return
            
            ret, cyc, load_i = result
            
            if i == 0:
                retired = ret
                load_instrs = load_i
                # Base cycles (no cache miss)
                base_result = analyze_str(instrs, arch, args.mcpu,
                                         cache_miss=float("inf"),
                                         cache_latency=0,
                                         cache_miss_mode=args.cache_miss_mode,
                                         cache_miss_rate=0.0)
                if base_result is None:
                    return
                _, base_cycles, _ = base_result
            
            cycles_list.append(cyc)
        
        # Print results
        cycles_headers = ",".join([f"cycles_{i}" for i in range(len(cache_specs))])
        print(f"start_address,end_address,retired_instructions,load_instructions,cycles,{cycles_headers}")
        
        cycles_str = ",".join([str(c) for c in cycles_list])
        print(f"0x{start:x},0x{end:x},{retired},{load_instrs},{base_cycles},{cycles_str}")


def _parse_cache_spec_str(spec: str, parser):
    """Parse a cache specification like cm_0.1, ipcm_50, or lipcm_10.
    
    Returns (spec_type, value) where spec_type is "cm", "ipcm", or "lipcm".
    """
    if spec.startswith("cm_"):
        try:
            value = float(spec[3:])
            if value < 0:
                parser.error(f"cache-miss rate must be >= 0, got: {spec}")
            return ("cm", value)
        except ValueError:
            parser.error(f"invalid cache-miss rate specification: {spec}")
    elif spec.startswith("ipcm_"):
        value_str = spec[5:]
        if value_str.lower() == "inf":
            return ("ipcm", float("inf"))
        try:
            value = float(value_str)
            if value <= 0:
                parser.error(f"instructions-per-cache-miss must be > 0, got: {spec}")
            return ("ipcm", value)
        except ValueError:
            parser.error(f"invalid instructions-per-cache-miss specification: {spec}")
    elif spec.startswith("lipcm_"):
        value_str = spec[6:]
        if value_str.lower() == "inf":
            return ("lipcm", float("inf"))
        try:
            value = float(value_str)
            if value <= 0:
                parser.error(f"load-instructions-per-cache-miss must be > 0, got: {spec}")
            return ("lipcm", value)
        except ValueError:
            parser.error(f"invalid load-instructions-per-cache-miss specification: {spec}")
    else:
        parser.error(f"invalid cycles specification: {spec} (expected cm_N, ipcm_N, or lipcm_N)")


if __name__ == "__main__":
    main()
