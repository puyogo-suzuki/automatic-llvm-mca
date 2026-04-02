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
                cache_miss: float = float("inf"),
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
        Number of retired instructions per cache miss.  Use ``float('inf')``
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
    ``(retired_instructions, elapsed_cycles, load_proportion)`` on success,
    or ``None`` when llvm-mca produces no result.
    """
    if not math.isinf(cache_miss_rate):
        cache_mode = _build_cache_mode_from_rate(cache_miss_rate, cache_latency,
                                                 cache_miss_mode)
    else:
        cache_mode = _build_cache_mode(cache_miss, cache_latency, cache_miss_mode)

    # Build llvm-mca argument list, optionally overriding -mcpu.
    mca_args = arch.mca_args
    if mcpu:
        mca_args = [a for a in mca_args if not a.startswith("-mcpu=")]
        mca_args = mca_args + [f"-mcpu={mcpu}"]

    return _run_mca(instrs, mca_args, arch, cache_mode)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate throughput for a dump-mode assembly file generated by "
            "analyze.py --dump."
        )
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
        "--instructions-per-cache-miss",
        type=float,
        default=float("inf"),
        metavar="N",
        dest="cache_miss",
        help=(
            "Number of retired instructions per cache miss (>0, default inf "
            "meaning no cache-miss simulation). "
            "Interpretation depends on --cache-miss-mode."
        ),
    )
    parser.add_argument(
        "--cache-latency",
        type=int,
        default=0,
        metavar="CYCLES",
        help=(
            "Cache-miss latency in cycles (≥0, default 0). "
            "Used as the latency value in the # LLVM-MCA-LATENCY directive "
            "when --instructions-per-cache-miss or --cache-miss-rate is finite."
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
    parser.add_argument(
        "--cache-miss-rate",
        type=float,
        default=float("inf"),
        metavar="R",
        dest="cache_miss_rate",
        help=(
            "Cache misses per retired load instruction (>0, default inf "
            "meaning no cache-miss simulation). "
            "May be greater than 1 (e.g. 1.5 means each load causes an "
            "average of 1.5 misses). "
            "Cannot be combined with --instructions-per-cache-miss. "
            "Interpretation depends on --cache-miss-mode."
        ),
    )

    parser.add_argument(
        "--ipcm-values",
        nargs="+",
        metavar="IPCM",
        default=None,
        dest="ipcm_values",
        help=(
            "Space-separated list of instructions-per-cache-miss values to "
            "sweep over. Each value must be a positive number; use 'inf' to "
            "represent no cache miss. "
            "Example: --ipcm-values 1 10 100 inf"
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.textfile):
        parser.error(f"{args.textfile}: no such file")

    if args.cache_miss <= 0:
        parser.error("--instructions-per-cache-miss must be > 0")
    if args.cache_latency < 0:
        parser.error("--cache-latency must be >= 0")
    if not math.isinf(args.cache_miss_rate) and args.cache_miss_rate <= 0:
        parser.error("--cache-miss-rate must be > 0")

    if not math.isinf(args.cache_miss) and not math.isinf(args.cache_miss_rate):
        parser.error(
            "--instructions-per-cache-miss and --cache-miss-rate are mutually exclusive"
        )
    if args.ipcm_values is not None and not math.isinf(args.cache_miss_rate):
        parser.error(
            "--ipcm-values and --cache-miss-rate are mutually exclusive"
        )

    ipcm_values = []
    if args.ipcm_values is not None:
        for raw in args.ipcm_values:
            if raw.lower() == "inf":
                ipcm_values.append(float("inf"))
            else:
                try:
                    val = float(raw)
                except ValueError:
                    parser.error(
                        f"--ipcm-values: invalid value '{raw}' "
                        "(expected a positive number or 'inf')"
                    )
                if val <= 0:
                    parser.error(
                        f"--ipcm-values: value '{raw}' must be > 0"
                    )
                ipcm_values.append(val)
        if not ipcm_values:
            parser.error("--ipcm-values: at least one value required")
    else:
        ipcm_values.append(args.cache_miss)

    instrs, arch, start, end = load_str_file(args.textfile)

    if not math.isinf(args.cache_miss_rate):
        result = analyze_str(instrs, arch, args.mcpu,
                             cache_latency=args.cache_latency,
                             cache_miss_mode=args.cache_miss_mode,
                             cache_miss_rate=args.cache_miss_rate)
        if result is None:
            return
        retired, elapsed_cycles, _ = result
        print(f"{args.cache_miss_rate:.2f},{retired},{elapsed_cycles}")
        return

    for ipcm in ipcm_values:
        result = analyze_str(instrs, arch, args.mcpu, ipcm,
                             args.cache_latency, args.cache_miss_mode)

        if result is None:
            return
        retired, elapsed_cycles, _ = result
        print(f"{ipcm:.2f},{retired},{elapsed_cycles}")


if __name__ == "__main__":
    main()
