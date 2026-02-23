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
import tempfile
from collections import namedtuple


# ---------------------------------------------------------------------------
# Architecture info
# ---------------------------------------------------------------------------

# _ArchInfo bundles all per-architecture settings needed during analysis.
#   name     : short arch tag used for branch-detection dispatch
#   objdump  : objdump executable that can disassemble this architecture
#   mca_args : extra arguments forwarded to llvm-mca
_ArchInfo = namedtuple("_ArchInfo", ["name", "objdump", "mca_args"])


def _find_cross_tool(preferred: str) -> str:
    """Return *preferred* tool if it exists on PATH, else fall back to plain objdump."""
    return preferred if shutil.which(preferred) else "objdump"


def _detect_arch(binary: str) -> "_ArchInfo":
    """Detect the architecture of *binary* and return an ``_ArchInfo``."""
    try:
        file_out = subprocess.run(
            ["file", binary], capture_output=True, text=True
        ).stdout.lower()
    except Exception:
        file_out = ""

    if "ucb risc-v" in file_out:
        objdump = _find_cross_tool("riscv64-linux-gnu-objdump")
        if "32-bit" in file_out:
            return _ArchInfo("riscv", objdump,
                             ["-march=riscv32", "-mcpu=sifive-e31"])
        return _ArchInfo("riscv", objdump,
                         ["-march=riscv64", "-mcpu=sifive-u74"])

    if "aarch64" in file_out or "arm64" in file_out:
        objdump = _find_cross_tool("aarch64-linux-gnu-objdump")
        return _ArchInfo("aarch64", objdump,
                         ["-march=aarch64", "-mcpu=cortex-a57"])

    if re.search(r"\barm\b", file_out):
        objdump = _find_cross_tool("arm-linux-gnueabihf-objdump")
        return _ArchInfo("arm", objdump, ["-march=arm"])

    return _ArchInfo("x86", "objdump", [])


# ---------------------------------------------------------------------------
# Branch / jump detection (architecture-aware)
# ---------------------------------------------------------------------------

# RISC-V branch/jump mnemonics (base ISA + standard pseudo-instructions).
# Using an explicit set avoids false positives from the B (bit-manipulation)
# extension whose instructions also start with "b" (bclr, bset, bext, …).
_RISCV_BRANCHES = frozenset({
    # RV32I/RV64I conditional branches
    "beq", "bne", "blt", "bge", "bltu", "bgeu",
    # Pseudo conditional branches (assembled from the above)
    "beqz", "bnez", "blez", "bgez", "bltz", "bgtz",
    "bgt", "ble", "bgtu", "bleu",
    # Unconditional jumps
    "j", "jr", "jal", "jalr",
    # Pseudo-call / pseudo-tail (resolve to auipc + jalr)
    "call", "tail",
    # Return pseudo-instruction (= jalr x0, 0(ra))
    "ret",
})

# AArch64 branch mnemonics.
# Conditional branches always use the "b." prefix (b.eq, b.ne, …) which is
# disjoint from non-branch instructions like bic, bfm, bfmlalt, etc.
_AARCH64_BRANCHES = frozenset({
    "b",       # unconditional branch
    "bl",      # branch with link (call)
    "br",      # branch to register
    "blr",     # branch with link to register (call)
    "ret",     # return
    "cbz",     # compare and branch if zero
    "cbnz",    # compare and branch if non-zero
    "tbz",     # test bit and branch if zero
    "tbnz",    # test bit and branch if non-zero
})


def _is_branch(mnemonic: str, arch: str = "x86") -> bool:
    """Return True if *mnemonic* is a branch/jump/call instruction.

    Recognises:
    * x86/x86-64 : ``j*``, ``call``, ``loop*``
    * AArch64    : explicit set + ``b.`` prefix (conditional branches like
      b.eq, b.ne, …)
    * RISC-V     : explicit set (avoids false positives from B-extension
      instructions bclr, bset, bext, …)
    """
    m = mnemonic.lower()
    if arch == "aarch64":
        return m in _AARCH64_BRANCHES or m.startswith("b.")
    if arch == "riscv":
        return m in _RISCV_BRANCHES
    # x86 / arm default
    return m.startswith(("j", "call", "loop"))


def _ends_basic_block(mnemonic: str, operands: str = "",
                      arch: str = "x86") -> bool:
    """Return True if this instruction ends a basic block."""
    m = mnemonic.lower()
    op = operands.lower()

    if arch == "aarch64":
        # bl / blr are calls (fall-through continues) — they do NOT end a BB.
        return (
            m in ("b", "br", "ret", "cbz", "cbnz", "tbz", "tbnz")
            or m.startswith("b.")  # b.eq, b.ne, b.lt, etc.
        )

    if arch == "riscv":
        # All conditional branches (beq/bne/blt/bge/bltu/bgeu/…) end a BB.
        # j/jr are unconditional pseudo-jumps; ret is a return.
        # jal/jalr: objdump typically shows `j target` (not `jal x0, target`)
        # for unconditional jumps, and `ret` (not `jalr x0, 0(ra)`) for
        # returns, so the distinction is handled by the pseudo-instruction
        # names rather than inspecting the destination register.
        return m in _RISCV_BRANCHES and m not in ("call", "tail")

    # x86 / arm
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

    Handles single-operand branches (x86: ``jne 4016``) as well as
    multi-operand branches (RISC-V: ``bne a4,a5,314``; AArch64:
    ``cbz x0, 2dc``; ``tbz x0, #0, 2dc``).

    The branch target is always the LAST address-like value in the operand
    string.  Values preceded by ``#`` (immediate field in tbz/tbnz etc.) are
    excluded so they are not mistaken for the target address.

    Returns None for indirect branches (operands starting with ``*``).
    """
    op = operands.strip()
    # Indirect jump/call in AT&T syntax: *%reg or *offset(%reg)
    if op.startswith("*"):
        return None

    candidates = []

    # 1. Explicit 0x-prefixed addresses not preceded by # (AArch64 immediates
    #    like #0x10 start with #, so the lookbehind filters them out).
    for mo in re.finditer(r"(?<![#\w])0x([0-9a-fA-F]+)\b", op):
        candidates.append(int(mo.group(1), 16))

    if not candidates:
        # 2. Plain hex addresses of at least 3 digits (to avoid matching
        #    short register names like a0, t0, x0, w0).
        #    ValueError from int(..., 16) cannot occur since the regex only
        #    captures [0-9a-fA-F]+ characters; the try/except is defensive.
        for mo in re.finditer(r"(?<![#\w])([0-9a-fA-F]{3,})\b", op):
            try:
                candidates.append(int(mo.group(1), 16))
            except ValueError:  # pragma: no cover
                pass

    # Return the last candidate — for multi-operand instructions the target
    # is always the final operand.
    return candidates[-1] if candidates else None


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


# ---------------------------------------------------------------------------
# Disassembly
# ---------------------------------------------------------------------------

def disassemble(binary: str, objdump: str = "objdump", arch: str = "x86"):
    """Disassemble *binary* with *objdump*.

    Returns a list of ``(func_name, instructions)`` pairs where
    *instructions* is a list of ``(addr, mnemonic, operands)`` triples.
    """
    proc = subprocess.run(
        [objdump, "-d", "--no-show-raw-insn", binary],
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
            # Remove inline comments.
            # AArch64 uses # for immediate operands (e.g. #0x1), so only
            # strip // style comments there.  All other architectures use
            # # style inline annotations (e.g. x86 "# addr", RISC-V "# addr").
            if arch == "aarch64":
                text = re.sub(r"//.*$", "", text).strip()
            else:
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

def _find_loops(instrs, arch: str = "x86"):
    """Detect loops via backward branches.

    Returns a list of ``(start_addr, end_addr)`` pairs, one per detected
    loop.  Nested loops appear as separate entries — the outer loop has a
    larger span.  The list is sorted so that outer loops come first
    (ascending start address; descending span for equal start addresses).
    """
    addr_set = {a for a, _, _ in instrs}
    loops = []

    for addr, mnemonic, operands in instrs:
        if _is_branch(mnemonic, arch):
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

def _format_asm(instrs, arch: str = "x86") -> str:
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

    # Determine which addresses need a label
    labeled: set = set()
    for addr, mnemonic, operands in instrs:
        if _is_branch(mnemonic, arch):
            t = _get_branch_target(operands)
            if t is not None and t in addr_set:
                labeled.add(t)

    lines = []
    for addr, mnemonic, operands in instrs:
        if addr in labeled:
            lines.append(f".Lmca_{addr:x}:")
        if _is_branch(mnemonic, arch):
            t = _get_branch_target(operands)
            if t is not None and t in addr_set:
                # In-region branch: replace target with a local label,
                # keeping any other operands (e.g. registers in cbz/bne).
                new_ops = _replace_branch_target(operands, t,
                                                 f".Lmca_{t:x}")
                lines.append(f"\t{mnemonic} {new_ops}")
            elif t is not None:
                # Out-of-region direct branch: redirect to the end label.
                new_ops = _replace_branch_target(operands, t, ".Lmca_end")
                lines.append(f"\t{mnemonic} {new_ops}")
            else:
                # Indirect branch (register target) — keep the original
                # instruction so the assembler does not error out.
                tail = f" {operands}" if operands else ""
                lines.append(f"\t{mnemonic}{tail}")
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


def _run_mca(instrs, mca_args=(), arch: str = "x86"):
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

    asm = _format_asm(instrs, arch)
    fd, path = tempfile.mkstemp(suffix=".s", prefix="llvm_mca_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(asm)
        cmd = [_LLVM_MCA, *mca_args, path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return None
        m = re.search(r"\bIPC:\s+([\d.]+)", proc.stdout)
        return float(m.group(1)) if m else None
    finally:
        os.unlink(path)


def _yield_mca_result(instrs, mca_args, arch):
    """Run llvm-mca on *instrs* and yield a ``(start, end, ipc)`` triple.

    Nothing is yielded when *instrs* is empty or llvm-mca returns no result.
    """
    if instrs:
        ipc = _run_mca(instrs, mca_args, arch)
        if ipc is not None:
            yield instrs[0][0], instrs[-1][0], ipc


# ---------------------------------------------------------------------------
# Per-function analysis
# ---------------------------------------------------------------------------

def _analyze_function(instrs, mca_args=(), arch: str = "x86"):
    """Analyse one function's instructions.

    Yields ``(start_addr, end_addr, ipc)`` triples for every loop and every
    basic block that is not part of a loop.
    """
    loops = _find_loops(instrs, arch)

    # --- Loops (including nested loops as separate entries) ---
    for ls, le in loops:
        region = [(a, m, o) for a, m, o in instrs if ls <= a <= le]
        yield from _yield_mca_result(region, mca_args, arch)

    # --- Basic blocks outside loops ---
    non_loop = [(a, m, o) for a, m, o in instrs if not _in_any_loop(a, loops)]
    bb = []
    for instr in non_loop:
        addr, mnemonic, operands = instr
        bb.append(instr)
        if _ends_basic_block(mnemonic, operands, arch):
            yield from _yield_mca_result(bb, mca_args, arch)
            bb = []
    yield from _yield_mca_result(bb, mca_args, arch)


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------

def analyze(binary: str, mcpu: str = ""):
    """Analyse *binary* and yield ``(start, end, ipc)`` triples.

    If *mcpu* is non-empty it overrides the default ``-mcpu`` value chosen by
    ``_detect_arch`` and is forwarded to llvm-mca.
    """
    arch_info = _detect_arch(binary)

    mca_args = arch_info.mca_args
    if mcpu:
        # Replace any existing -mcpu=... entry with the user-supplied value.
        mca_args = [a for a in mca_args if not a.startswith("-mcpu=")]
        mca_args = mca_args + [f"-mcpu={mcpu}"]

    for _func_name, instrs in disassemble(binary, arch_info.objdump,
                                          arch_info.name):
        yield from _analyze_function(instrs, mca_args, arch_info.name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Estimate throughput for an ELF binary using llvm-mca."
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
    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        parser.error(f"{args.binary}: no such file")

    # Sort: by start address ascending; for equal starts, larger span first
    # (outer loops before inner loops at the same start address).
    results = sorted(analyze(args.binary, args.mcpu),
                     key=lambda x: (x[0], -(x[1] - x[0])))
    for start, end, ipc in results:
        print(f"0x{start:x}-0x{end:x} {ipc:.2f}")


if __name__ == "__main__":
    main()
