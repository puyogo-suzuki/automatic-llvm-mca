#!/usr/bin/env python3
"""analyze_str.py: Estimate throughput for a dump-mode assembly file.

Accepts a text file in the format produced by ``analyze.py --dump`` (filename
``{start}_{end}.{arch}.txt``) and runs llvm-mca on its contents, reporting the
same CSV output as ``analyze.py``.

Usage::

    python3 analyze_str.py [--mcpu <cpu>] [--decode-width <W>] [--dependency <mode>] <textfile>
"""

import argparse
import os
import re
import sys

from analyze import (
    _run_mca,
    _compute_mlp,
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


def analyze_str(instrs, arch: ArchBase, mcpu: str = "", decode_width: int = 4, dependency: str = "none"):
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
    decode_width:
        The decode width used to compute Memory Level Parallelism.
    dependency:
        Dependency tracking mode for MLP estimation.

    Returns
    -------
    ``(retired_instructions, elapsed_cycles, load_instructions, mlp)`` on success,
    or ``None`` when llvm-mca produces no result.
    """
    mca_args = arch.mca_args
    if mcpu:
        mca_args = [a for a in mca_args if not a.startswith("-mcpu=")]
        mca_args = mca_args + [f"-mcpu={mcpu}"]

    result = _run_mca(instrs, mca_args, arch=arch)
    if result is None:
        return None
        
    retired, cycles, load_instrs = result
    mlp = _compute_mlp(instrs, decode_width, arch, dependency)
    return retired, cycles, load_instrs, mlp

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate throughput for a dump-mode assembly file generated by "
            "analyze.py --dump."
        ),
        usage="%(prog)s [--mcpu CPU] [--decode-width W] <textfile>"
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
        "--decode-width",
        type=int,
        default=4,
        metavar="W",
        help="The decode width used to compute Memory Level Parallelism (default: 4)."
    )
    parser.add_argument(
        "--dependency",
        choices=["none", "io", "ooo"],
        default="none",
        help=(
            "Dependency tracking mode for MLP estimation (default: none). "
            "none: no dependency tracking; "
            "io: in-order (distance to first use); "
            "ooo: out-of-order (independent loads in window)."
        )
    )
    args = parser.parse_args()

    if args.decode_width < 1:
        parser.error("--decode-width must be >= 1")

    if not os.path.isfile(args.textfile):
        parser.error(f"{args.textfile}: no such file")

    instrs, arch, start, end = load_str_file(args.textfile)

    # Single analysis pass.
    result = analyze_str(instrs, arch, args.mcpu, args.decode_width, args.dependency)
    if result is None:
        return
    retired, cycles, load_instrs, mlp = result

    print("start_address,end_address,retired_instructions,load_instructions,cycles,mlp")
    print(f"0x{start:x},0x{end:x},{retired},{load_instrs},{cycles},{mlp}")


if __name__ == "__main__":
    main()
