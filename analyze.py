#!/usr/bin/env python3
"""automatic-llvm-mca: Estimate throughput for ELF binaries using llvm-mca.

Procedure:
  1. Disassemble the ELF binary with objdump.
  2. Decompose each function into loops and (for non-loop code) basic blocks.
  3. Run llvm-mca on each region to estimate IPC.
  4. Print address range and estimated IPC for every region.

For nested loops the outer loop (including the inner loop body) and the inner
loop are reported separately:
  0x0-0x8 1.20
  0x2-0x6 1.00

Usage:
  python3 analyze.py <elf-binary>
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile


# ---------------------------------------------------------------------------
# Branch / jump detection
# ---------------------------------------------------------------------------

# Mnemonic prefixes that indicate an instruction with a branch target.
# "call" is included so its target operand is replaced when building the
# llvm-mca assembly snippet; however "call" does *not* end a basic block
# (execution continues at the return address).
_BRANCH_PREFIXES = ("j", "call", "loop")


def _is_branch(mnemonic: str) -> bool:
    """Return True if *mnemonic* is a branch/jump/call instruction."""
    m = mnemonic.lower()
    return any(m.startswith(p) for p in _BRANCH_PREFIXES)


def _ends_basic_block(mnemonic: str, operands: str = "") -> bool:
    """Return True if *mnemonic* (and *operands*) end a basic block."""
    m = mnemonic.lower()
    op = operands.lower()
    return (
        m.startswith("j")
        or m.startswith("loop")
        or m.startswith("ret")
        # "repz retq" / "repz retl" — branch-prediction padding idiom
        or (m == "repz" and op.startswith("ret"))
        or m in ("hlt", "ud2", "int3")
    )


def _get_branch_target(operands: str):
    """Return the numeric branch target from *operands*, or None.

    Handles:
    * Direct jumps:  ``4016``  or  ``0x4016``
    * Indirect jumps start with ``*`` in AT&T syntax → return None.
    """
    op = operands.strip()
    # Indirect jump/call: *%reg or *offset(%reg)
    if op.startswith("*"):
        return None
    # With explicit 0x prefix
    m = re.match(r"0x([0-9a-fA-F]+)", op)
    if m:
        return int(m.group(1), 16)
    # Plain hex address (objdump default — no 0x prefix)
    m = re.match(r"([0-9a-fA-F]{3,})\b", op)
    if m:
        try:
            return int(m.group(1), 16)
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Disassembly
# ---------------------------------------------------------------------------

def disassemble(binary: str):
    """Disassemble *binary* with objdump.

    Returns a list of ``(func_name, instructions)`` pairs where
    *instructions* is a list of ``(addr, mnemonic, operands)`` triples.
    """
    proc = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", binary],
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
            # Remove symbol annotations like <name+0x10> and inline comments
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"#.*$", "", text).strip()
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

def _find_loops(instrs):
    """Detect loops via backward branches.

    Returns a list of ``(start_addr, end_addr)`` pairs, one per detected
    loop.  Nested loops appear as separate entries — the outer loop has a
    larger span.  The list is sorted so that outer loops come first
    (ascending start address; descending span for equal start addresses).
    """
    addr_set = {a for a, _, _ in instrs}
    loops = []

    for addr, mnemonic, operands in instrs:
        if _is_branch(mnemonic):
            target = _get_branch_target(operands)
            if target is not None and target < addr and target in addr_set:
                loops.append((target, addr))

    # Outer loops first (smaller start, larger span)
    loops.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    return loops


def _in_any_loop(addr: int, loops) -> bool:
    return any(s <= addr <= e for s, e in loops)


# ---------------------------------------------------------------------------
# Assembly formatting for llvm-mca
# ---------------------------------------------------------------------------

def _format_asm(instrs) -> str:
    """Format *instrs* as AT&T assembly for llvm-mca.

    * Instructions whose address is a jump target within this region are
      preceded by a local label ``.Lmca_ADDR:``.
    * Branch instructions with an in-region target are rewritten to use the
      corresponding label.
    * Branch instructions with an out-of-region target (or an indirect
      target that couldn't be resolved) are redirected to ``.Lmca_end``.
    * A ``.Lmca_end:`` label is appended after the last instruction so that
      all forward/external branch targets resolve without assembler errors.
    """
    addr_set = {a for a, _, _ in instrs}

    # Determine which addresses need a label
    labeled: set = set()
    for addr, mnemonic, operands in instrs:
        if _is_branch(mnemonic):
            t = _get_branch_target(operands)
            if t is not None and t in addr_set:
                labeled.add(t)

    lines = []
    for addr, mnemonic, operands in instrs:
        if addr in labeled:
            lines.append(f".Lmca_{addr:x}:")
        if _is_branch(mnemonic):
            t = _get_branch_target(operands)
            if t is not None and t in addr_set:
                lines.append(f"\t{mnemonic} .Lmca_{t:x}")
            else:
                # Indirect or out-of-region target — redirect to end label
                lines.append(f"\t{mnemonic} .Lmca_end")
        else:
            tail = f" {operands}" if operands else ""
            lines.append(f"\t{mnemonic}{tail}")

    lines.append(".Lmca_end:")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# llvm-mca invocation
# ---------------------------------------------------------------------------

def _find_llvm_mca() -> str:
    """Return the path to llvm-mca, trying versioned names as fallback."""
    if shutil.which("llvm-mca"):
        return "llvm-mca"
    for ver in range(20, 10, -1):
        name = f"llvm-mca-{ver}"
        if shutil.which(name):
            return name
    raise FileNotFoundError(
        "llvm-mca not found. Install LLVM (e.g. apt install llvm)."
    )


_LLVM_MCA = None  # resolved lazily


def _run_mca(instrs, extra_args=()):
    """Run llvm-mca on *instrs* and return IPC as a float, or None."""
    global _LLVM_MCA
    if not instrs:
        return None
    if _LLVM_MCA is None:
        try:
            _LLVM_MCA = _find_llvm_mca()
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    asm = _format_asm(instrs)
    fd, path = tempfile.mkstemp(suffix=".s", prefix="llvm_mca_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(asm)
        cmd = [_LLVM_MCA, *extra_args, path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return None
        m = re.search(r"\bIPC:\s+([\d.]+)", proc.stdout)
        return float(m.group(1)) if m else None
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Per-function analysis
# ---------------------------------------------------------------------------

def _analyze_function(instrs, extra_args=()):
    """Analyse one function's instructions.

    Yields ``(start_addr, end_addr, ipc)`` triples for every loop and every
    basic block that is not part of a loop.
    """
    loops = _find_loops(instrs)

    # --- Loops (including nested loops as separate entries) ---
    for ls, le in loops:
        region = [(a, m, o) for a, m, o in instrs if ls <= a <= le]
        if region:
            ipc = _run_mca(region, extra_args)
            if ipc is not None:
                yield ls, le, ipc

    # --- Basic blocks outside loops ---
    non_loop = [(a, m, o) for a, m, o in instrs if not _in_any_loop(a, loops)]
    bb = []
    for instr in non_loop:
        addr, mnemonic, operands = instr
        bb.append(instr)
        if _ends_basic_block(mnemonic, operands):
            if bb:
                ipc = _run_mca(bb, extra_args)
                if ipc is not None:
                    yield bb[0][0], bb[-1][0], ipc
            bb = []
    if bb:
        ipc = _run_mca(bb, extra_args)
        if ipc is not None:
            yield bb[0][0], bb[-1][0], ipc


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------

def analyze(binary: str):
    """Analyse *binary* and yield ``(start, end, ipc)`` triples."""
    # Detect architecture-specific llvm-mca flags
    extra_args = []
    try:
        file_out = subprocess.run(
            ["file", binary], capture_output=True, text=True
        ).stdout.lower()
        if "aarch64" in file_out or "arm64" in file_out:
            extra_args = ["-march=aarch64"]
        elif re.search(r"\barm\b", file_out):
            extra_args = ["-march=arm"]
    except Exception:
        pass

    for _func_name, instrs in disassemble(binary):
        yield from _analyze_function(instrs, extra_args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <elf-binary>", file=sys.stderr)
        sys.exit(1)

    binary = sys.argv[1]
    if not os.path.isfile(binary):
        print(f"Error: {binary}: no such file", file=sys.stderr)
        sys.exit(1)

    # Sort: by start address ascending; for equal starts, larger span first
    # (outer loops before inner loops at the same start address).
    results = sorted(analyze(binary), key=lambda x: (x[0], -(x[1] - x[0])))
    for start, end, ipc in results:
        print(f"0x{start:x}-0x{end:x} {ipc:.2f}")


if __name__ == "__main__":
    main()
