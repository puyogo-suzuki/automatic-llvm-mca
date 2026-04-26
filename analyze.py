#!/usr/bin/env python3
"""automatic-llvm-mca: Estimate throughput for ELF binaries using llvm-mca.


Procedure:
  1. Disassemble the ELF binary with objdump.
  2. Decompose each function into loops and (for non-loop code) basic blocks.
  3. Run llvm-mca on each region to obtain retired instructions, elapsed cycles,
     and the count of load instructions (those with the MayLoad attribute).
  4. Print a CSV with start address, end address, retired instructions, elapsed
     cycles, and load instructions for every region.

For nested loops the outer loop (including the inner loop body) and the inner
loop are reported separately:
  0x2,0x6,200,50,103
  0x0,0x8,960,120,800

Supported architectures: x86/x86-64, AArch64, 32-bit ARM, RISC-V (RV32IC, RV64IC).

Usage:
  python3 analyze.py [--mcpu <cpu>] <elf-binary>
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

from arch import (  # noqa: F401  (re-exported for backward compatibility)
    _RISCV_BRANCHES,
    _AARCH64_BRANCHES,
    _ARM_CONDS,
    _ARM_BRANCH_BASES,
    _ARM_BRANCHES,
    _ARM_CALLS,
    _parse_branch_target_candidates,
    ArchBase,
    X86Arch,
    AArch64Arch,
    ARMArch,
    RISCVArch,
    _find_cross_tool,
    _arch_from_platform,
    _detect_arch,
)


# ---------------------------------------------------------------------------
# Disassembly
# ---------------------------------------------------------------------------

def disassemble(binary: str, arch: ArchBase):
    """Disassemble *binary* using the objdump from *arch*.

    Returns a list of ``(func_name, instructions)`` pairs where
    *instructions* is a list of ``(addr, mnemonic, operands)`` triples.
    """
    proc = subprocess.run(
        [arch.objdump, "-d", "--no-show-raw-insn", binary],
        capture_output=True,
        text=True,
        check=True,
    )

    functions = []
    cur_name = None
    cur_instrs = []

    func_re = re.compile(r"^[0-9a-fA-F]+\s+<([^>]+)>:")
    instr_re = re.compile(r"^\s+([0-9a-fA-F]+):\s+(.+)")

    for line in proc.stdout.splitlines():
        fm = func_re.match(line)
        if fm:
            if cur_name is not None:
                functions.append((cur_name, cur_instrs))
            cur_name = fm.group(1)
            cur_instrs = []
            continue

        im = instr_re.match(line)
        if im and cur_name is not None:
            addr = int(im.group(1), 16)
            text = im.group(2)
            # Remove symbol annotations like <name+0x10>
            text = re.sub(r"<[^>]+>", "", text)
            # Remove inline comments using the arch-specific strategy.
            text = arch.strip_asm_comment(text)
            if not text:
                continue
            parts = text.split(None, 1)
            mnemonic = parts[0]
            operands = parts[1].strip() if len(parts) > 1 else ""
            cur_instrs.append((addr, mnemonic, operands))

    if cur_name is not None:
        functions.append((cur_name, cur_instrs))

    return functions


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

def _find_loops(instrs, arch: ArchBase):
    """Detect loops via backward branches.

    Returns a list of ``(start_addr, end_addr)`` pairs, one per detected
    loop.  Nested loops appear as separate entries — the inner loop has a
    smaller span.  The list is sorted so that inner loops come first
    (ascending end address; descending start address for equal end addresses).
    """
    addr_set = {a for a, _, _ in instrs}
    loops = []

    for addr, mnemonic, operands in instrs:
        if arch.is_branch(mnemonic):
            target = arch.get_branch_target(operands, mnemonic)
            if target is not None and target < addr and target in addr_set:
                loops.append((target, addr))

    # Inner loops first (smaller end address; for equal end, larger start = smaller span)
    loops.sort(key=lambda x: (x[1], -x[0]))
    return loops


def _in_any_loop(addr: int, loops) -> bool:
    return any(s <= addr <= e for s, e in loops)


# ---------------------------------------------------------------------------
# Assembly formatting for llvm-mca
# ---------------------------------------------------------------------------

def _replace_branch_target(operands: str, target: int,
                            replacement: str) -> str:
    """Replace the branch target address in *operands* with *replacement*.

    Works for both single-operand (``4016``) and multi-operand
    (``a4,a5,314``, ``x0, 2dc``) branch instructions.
    """
    hex_t = f"{target:x}"
    # Try 0x-prefixed form first, then plain hex.
    for pattern in (
        rf"(?<![#\w])0x{re.escape(hex_t)}\b",
        rf"(?<![#\w]){re.escape(hex_t)}\b",
    ):
        new_ops, n = re.subn(pattern, replacement, operands,
                              flags=re.IGNORECASE)
        if n:
            return new_ops
    return operands  # fallback: no replacement found


def _compute_labeled_addrs(instrs, arch: ArchBase) -> set:
    """Return the set of instruction addresses that need a local label.

    An address needs a label when it is the in-region target of at least one
    branch instruction.
    """
    addr_set = {a for a, _, _ in instrs}
    labeled: set = set()
    for _, mnemonic, operands in instrs:
        if arch.is_branch(mnemonic):
            t = arch.get_branch_target(operands, mnemonic)
            if t is not None and t in addr_set:
                labeled.add(t)
    return labeled


def _format_branch_instr(mnemonic: str, operands: str, addr_set: set,
                         arch: ArchBase, label_sfx: str = "") -> str:
    """Return a single formatted branch instruction line.

    The branch target is rewritten as follows:

    * In-region target → local label ``.Lmca_ADDR<label_sfx>``.
    * Out-of-region direct target → ``.Lmca_end``.
    * Indirect branch (register target, ``get_branch_target`` returns ``None``)
      → original operands kept verbatim.
    """
    t = arch.get_branch_target(operands, mnemonic)
    if t is not None and t in addr_set:
        new_operands = _replace_branch_target(
            operands, t, f".Lmca_{t:x}{label_sfx}"
        )
    elif t is not None:
        new_operands = _replace_branch_target(operands, t, ".Lmca_end")
    else:
        new_operands = operands

    # Apply architecture-specific operand formatting for LLVM-MCA
    new_operands = arch.format_operands_for_mca(mnemonic, new_operands)
    suffix = f" {new_operands}" if new_operands else ""
    return f"\t{mnemonic}{suffix}"


def _format_asm(instrs, arch: ArchBase) -> str:
    """Format *instrs* as assembly for llvm-mca.

    * Instructions whose address is a branch target within this region are
      preceded by a local label ``.Lmca_ADDR:``.
    * Branch instructions with an in-region target are rewritten to use the
      corresponding label (preserving any non-target operands for
      multi-operand architectures like RISC-V and AArch64).
    * Branch instructions with an out-of-region direct target are redirected
      to ``.Lmca_end``.
    * Indirect branches (no resolvable target) are kept as-is.
    * A ``.Lmca_end:`` label is appended after the last instruction so that
      all forward/external branch targets resolve without assembler errors.
    """
    addr_set = {a for a, _, _ in instrs}
    labeled = _compute_labeled_addrs(instrs, arch)

    lines = []
    for addr, mnemonic, operands in instrs:
        if addr in labeled:
            lines.append(f".Lmca_{addr:x}:")
        if arch.is_branch(mnemonic):
            lines.append(_format_branch_instr(mnemonic, operands, addr_set, arch))
        else:
            # Apply architecture-specific operand formatting for LLVM-MCA
            formatted_operands = arch.format_operands_for_mca(mnemonic, operands)
            tail = f" {formatted_operands}" if formatted_operands else ""
            lines.append(f"\t{mnemonic}{tail}")

    lines.append(".Lmca_end:")
    return "\n".join(lines) + "\n"


class Dumper:
    """Writes formatted assembly to disk.

    Each region produces one file named
    ``{start_address}_{end_address}.{arch}.txt``.

    Parameters
    ----------
    dump_dir:
        Directory where formatted assembly files are written.  Created on
        first use if it does not already exist.  Defaults to ``"dump"``.
    """

    def __init__(self, dump_dir: str = "dump"):
        self._dump_dir = dump_dir

    def dump(self, instrs, asm: str, arch: ArchBase) -> None:
        """Write the formatted assembly to disk.

        The formatted assembly is written to
        ``{dump_dir}/{start:x}_{end:x}.{arch.name}.txt`` as a side effect,
        where *start* and *end* are the addresses of the first and last
        instructions in *instrs*.

        Nothing is written when *instrs* is empty.
        """
        if instrs:
            os.makedirs(self._dump_dir, exist_ok=True)
            start = instrs[0][0]
            end = instrs[-1][0]
            filename = f"{start:x}_{end:x}.{arch.name}.txt"
            path = os.path.join(self._dump_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(asm)


# ---------------------------------------------------------------------------
# llvm-mca runner
# ---------------------------------------------------------------------------

def _find_llvm_mca() -> str:
    """Return the path to llvm-mca, trying versioned names as fallback."""
    for ver in range(30, 15, -1):
        name = f"llvm-mca-{ver}"
        if shutil.which(name):
            return name
    if shutil.which("llvm-mca"):
        return "llvm-mca"
    raise FileNotFoundError(
        "llvm-mca not found. Install LLVM (e.g. apt install llvm)."
    )


_LLVM_MCA = None  # resolved lazily


def _count_load_instructions(instrs, arch: ArchBase) -> int:
    """Return the count of load instructions in *instrs*.

    Uses :meth:`ArchBase.is_load_instruction` to identify loads.
    Returns 0 when *instrs* is empty.
    """
    if not instrs:
        return 0
    # Default llvm-mca implementation simulates 100 times.
    return sum(1 for _, mn, ops in instrs if arch.is_load_instruction(mn, ops)) * 100


def _compute_mlp(instrs, decode_width: int, arch: ArchBase, dependency: str = "none") -> float:
    """Compute the Memory Level Parallelism (MLP) for a block of instructions."""
    if not instrs:
        return 1.0
    n = len(instrs)
    is_load = [1 if arch.is_load_instruction(mn, ops) else 0 for _, mn, ops in instrs]
    io_regs = [arch.get_io_registers(mn, ops) for _, mn, ops in instrs]
    
    total_mlp = 0.0
    nonzero_count = 0
    if dependency == "io":
      for i in range(n):
        if not is_load[i]:
          continue
        _, outputs_i = io_regs[i]
        # distance to the first instruction that reads the destination register
        dist = n - 1 - i
        for j in range(i + 1, n):
          inputs_j, _ = io_regs[j]
          if outputs_i & inputs_j:
            dist = j - i
            break
        mlp_i = min(decode_width, dist)
        if mlp_i > 0:
          total_mlp += mlp_i
          nonzero_count += 1
    elif dependency == "none":
      for i in range(n):
        mlp_i = is_load[i]
        for j in range(i + 1, min(i + decode_width, n)):
          mlp_i += is_load[j]
        if mlp_i > 0:
          total_mlp += mlp_i
          nonzero_count += 1
    else:
      for i in range(n):
        mlp_i = is_load[i]
        for j in range(i + 1, min(i + decode_width, n)):
          nj_in, _ = io_regs[j]
          has_dep = False
          for m in range(i, j):
            _, mm_out = io_regs[m]
            if nj_in & mm_out:
              has_dep = True
              break # i.e., leave out fomr the outer loop
          if not has_dep:
            mlp_i += is_load[j]
        if mlp_i > 0:
          total_mlp += mlp_i
          nonzero_count += 1
    if nonzero_count == 0:
        return 1.0
    return total_mlp / nonzero_count


def _run_mca(instrs, mca_args=(), *, arch: ArchBase):
    """Run llvm-mca on *instrs* and return ``(retired, cycles, load_instructions)``, or None.

    Parameters
    ----------
    instrs:
        List of ``(addr, mnemonic, operands)`` triples to analyse.
    mca_args:
        Extra arguments forwarded to llvm-mca (e.g. ``-march=``, ``-mcpu=``).
    arch:
        Architecture object.
    """
    global _LLVM_MCA
    if not instrs:
        return None
    if _LLVM_MCA is None:
        try:
            _LLVM_MCA = _find_llvm_mca()
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    asm = _format_asm(instrs, arch)

    # Pass assembly via stdin (llvm-mca reads from stdin when given "-").
    cmd = [_LLVM_MCA, "--call-latency=0", *mca_args, "-"]
    proc = subprocess.run(cmd, input=asm, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    m_retired = re.search(r"\bInstructions:\s+(\d+)", proc.stdout)
    m_cycles = re.search(r"\bTotal Cycles:\s+(\d+)", proc.stdout)
    if m_retired is None or m_cycles is None:
        return None
    retired = int(m_retired.group(1))
    cycles = int(m_cycles.group(1))
    load_instructions = _count_load_instructions(instrs, arch)
    return retired, cycles, load_instructions


# ---------------------------------------------------------------------------
# Per-function analysis
# ---------------------------------------------------------------------------

def _analyze_function(instrs, mca_args=(), arch: ArchBase = None, dumper: Dumper = None, decode_width: int = 4, dependency: str = "none"):
    """Analyse one function's instructions.

    Yields ``(start_addr, end_addr, retired, cycles, load_instructions, mlp, kind)``
    tuples for every loop and every basic block that is not part of a loop.
    """
    if arch is None:
        arch = X86Arch()

    loops = _find_loops(instrs, arch)

    # --- Loops (including nested loops as separate entries) ---
    for ls, le in loops:
        region = [(a, m, o) for a, m, o in instrs if ls <= a <= le]
        result = _run_mca(region, mca_args, arch=arch)
        if result is not None:
            retired, cycles, load_instrs = result
            mlp = _compute_mlp(region, decode_width, arch, dependency)
            if dumper:
                dumper.dump(region, _format_asm(region, arch), arch)
            yield region[0][0], region[-1][0], retired, cycles, load_instrs, mlp, "loop"

    # --- Basic blocks outside loops ---
    non_loop = [(a, m, o) for a, m, o in instrs if not _in_any_loop(a, loops)]
    bb = []
    for instr in non_loop:
        addr, mnemonic, operands = instr
        bb.append(instr)
        if arch.ends_basic_block(mnemonic, operands):
            result = _run_mca(bb, mca_args, arch=arch)
            if result is not None:
                retired, cycles, load_instrs = result
                mlp = _compute_mlp(bb, decode_width, arch, dependency)
                if dumper:
                    dumper.dump(bb, _format_asm(bb, arch), arch)
                yield bb[0][0], bb[-1][0], retired, cycles, load_instrs, mlp, "block"
            bb = []
    if bb:
        result = _run_mca(bb, mca_args, arch=arch)
        if result is not None:
            retired, cycles, load_instrs = result
            mlp = _compute_mlp(bb, decode_width, arch, dependency)
            if dumper:
                dumper.dump(bb, _format_asm(bb, arch), arch)
            yield bb[0][0], bb[-1][0], retired, cycles, load_instrs, mlp, "block"


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------

def analyze(binary: str, mcpu: str = "", dump: bool = False, decode_width: int = 4, dependency: str = "none"):
    """Analyse *binary* and yield ``(start, end, retired, load_instructions, cycles, mlp)`` tuples.

    Parameters
    ----------
    binary:
        Path to the ELF binary to analyse.
    mcpu:
        If non-empty, overrides the default ``-mcpu`` value chosen by
        ``_detect_arch`` and is forwarded to llvm-mca.
    dump:
        When ``True``, the formatted assembly for each analysed region is
        written to a file in a ``"dump"`` directory.
    decode_width:
        The decode width used to compute Memory Level Parallelism.
    """
    arch = _detect_arch(binary)
    mca_args = arch.mca_args
    if mcpu:
        mca_args = [a for a in mca_args if not a.startswith("-mcpu=")]
        mca_args = mca_args + [f"-mcpu={mcpu}"]
    
    dumper = Dumper() if dump else None

    for _func_name, instrs in disassemble(binary, arch):
        for start, end, retired, cycles, load_instrs, mlp, _kind in _analyze_function(
                instrs, mca_args, arch, dumper, decode_width, dependency):
            yield start, end, retired, load_instrs, cycles, mlp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Estimate throughput for an ELF binary using llvm-mca.",
        usage="%(prog)s [--mcpu CPU] [--dump] [--decode-width W] <elf-binary>"
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
        "--dump",
        action="store_true",
        default=False,
        help=(
            "Write the formatted assembly for each analysed region to a text "
            "file inside a 'dump' directory.  Each file is named "
            "{start_address}_{end_address}.{arch}.txt."
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

    if not os.path.isfile(args.binary):
        parser.error(f"{args.binary}: no such file")

    # Single analysis pass over the binary.
    raw_results = list(analyze(
        binary=args.binary,
        mcpu=args.mcpu,
        dump=args.dump,
        decode_width=args.decode_width,
        dependency=args.dependency,
    ))

    # Deduplicate by (start, end), keeping the first occurrence.
    all_results: dict = {}
    for start, end, retired, load_instrs, cycles, mlp in raw_results:
        key = (start, end)
        if key not in all_results:
            all_results[key] = (start, end, retired, load_instrs, cycles, mlp)

    # Sort and print results.
    sorted_results = sorted(all_results.values(), key=lambda x: (x[1], -x[0]))

    print("start_address,end_address,retired_instructions,load_instructions,cycles,mlp")
    for start, end, retired, load_instrs, cycles, mlp in sorted_results:
        print(f"0x{start:x},0x{end:x},{retired},{load_instrs},{cycles},{mlp}")


if __name__ == "__main__":
    main()
