#!/usr/bin/env python3
"""automatic-llvm-mca: Estimate throughput for ELF binaries using llvm-mca.


Procedure:
  1. Disassemble the ELF binary with objdump.
  2. Decompose each function into loops and (for non-loop code) basic blocks.
  3. Run llvm-mca on each region to estimate IPC and the proportion of load
     instructions (those with the MayLoad attribute).
  4. Print a CSV with start address, end address, estimated throughput and
     proportion of load instructions for every region.

For nested loops the outer loop (including the inner loop body) and the inner
loop are reported separately:
  0x2,0x6,1.00,0.2500
  0x0,0x8,1.20,0.1250

Supported architectures: x86/x86-64, AArch64, 32-bit ARM, RISC-V (RV32IC, RV64IC).

Usage:
  python3 analyze.py [--mcpu <cpu>] <elf-binary>
"""

import argparse
import math
import os
import platform
import re
import shutil
import subprocess
import sys


# ---------------------------------------------------------------------------
# Branch / jump detection constants (used by arch classes below)
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

# 32-bit ARM condition code suffixes (used to build branch mnemonic sets).
_ARM_CONDS = frozenset({
    "eq", "ne", "cs", "hs", "cc", "lo", "mi", "pl",
    "vs", "vc", "hi", "ls", "ge", "lt", "gt", "le", "al",
})

# ARM unconditional branch / call base forms.
_ARM_BRANCH_BASES = ("b", "bl", "bx", "blx", "bxj")

# All 32-bit ARM branch mnemonics (unconditional + conditional variants).
# Generated programmatically to avoid errors in the manual list.
_ARM_BRANCHES: frozenset = frozenset(
    list(_ARM_BRANCH_BASES)
    + [f"{base}{cond}" for base in _ARM_BRANCH_BASES for cond in _ARM_CONDS]
    + ["ret"]
)

# ARM call forms (bl / blx and their conditional variants) — these have a
# fall-through continuation and do NOT terminate a basic block.
_ARM_CALLS: frozenset = frozenset(
    ["bl", "blx"]
    + [f"bl{cond}" for cond in _ARM_CONDS]
    + [f"blx{cond}" for cond in _ARM_CONDS]
)


# ---------------------------------------------------------------------------
# Helper: branch-target candidate parser (shared regex logic)
# ---------------------------------------------------------------------------

def _parse_branch_target_candidates(op: str):
    """Parse branch-target address candidates from operand string *op*.

    Returns the last candidate (the actual branch target) or None.

    Pass 1: 0x-prefixed addresses not preceded by ``#`` (AArch64 immediates
    like ``#0x10`` start with ``#``, so the lookbehind filters them out).

    Pass 2 (fallback): plain hex addresses (1 or more hex digits).  The
    lookbehind ``(?<![#\\w])`` prevents matching ``#``-prefixed immediates and
    register names whose first hex-digit character is immediately preceded by a
    non-hex word character (e.g. AArch64 ``x0``, ``w5``; RISC-V ``t0``,
    ``s1``).

    For multi-operand instructions the target is always the **last** operand,
    so only the last candidate is returned.

    ``ValueError`` from ``int(..., 16)`` cannot occur since the regex only
    captures ``[0-9a-fA-F]+`` characters; the ``try/except`` is defensive.
    """
    candidates = []

    # 1. Explicit 0x-prefixed addresses.
    for mo in re.finditer(r"(?<![#\w])0x([0-9a-fA-F]+)\b", op):
        candidates.append(int(mo.group(1), 16))

    if not candidates:
        # 2. Plain hex addresses (1 or more hex digits).
        for mo in re.finditer(r"(?<![#\w])([0-9a-fA-F]+)\b", op):
            try:
                candidates.append(int(mo.group(1), 16))
            except ValueError:  # pragma: no cover
                pass

    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Architecture classes
# ---------------------------------------------------------------------------

class ArchBase:
    """Base class for architecture-specific analysis settings and logic.

    Each concrete subclass bundles all per-architecture knowledge:
    * Which objdump executable to use for disassembly.
    * Which extra arguments to pass to llvm-mca.
    * How to recognise branch, basic-block-terminating, and load instructions.
    * How to parse branch targets from disassembly operand strings.
    * How to strip inline assembly comments from disassembly output.

    The class-based design replaces the previous approach of passing an
    architecture tag string (``"x86"``, ``"aarch64"``, …) through every
    function and dispatching on it with ``if``/``elif`` chains.
    """

    #: Short architecture tag (set by subclasses as a class attribute).
    name: str = ""

    def __init__(self, objdump: str = "objdump", mca_args: list = None):
        self._objdump = objdump
        self._mca_args = list(mca_args) if mca_args is not None else []

    @property
    def objdump(self) -> str:
        """Path to the objdump executable for this architecture."""
        return self._objdump

    @property
    def mca_args(self) -> list:
        """Extra arguments to pass to llvm-mca for this architecture."""
        return list(self._mca_args)

    def is_branch(self, mnemonic: str) -> bool:
        """Return True if *mnemonic* is a branch/jump/call instruction."""
        raise NotImplementedError

    def ends_basic_block(self, mnemonic: str, operands: str) -> bool:
        """Return True if this instruction terminates a basic block."""
        raise NotImplementedError

    def is_load_instruction(self, mnemonic: str, operands: str) -> bool:
        """Return True if *mnemonic*/*operands* describe a memory-load."""
        raise NotImplementedError

    def get_branch_target(self, operands: str, mnemonic: str = ""):
        """Return the numeric branch target from *operands*, or None.

        Returns None for indirect branches (target in a register, not a static
        address).  The default implementation delegates to the shared regex
        parser; subclasses override to add architecture-specific guards (e.g.
        x86 ``*%reg`` indirect syntax, RISC-V ``jr``/``jalr`` single-register
        form).
        """
        return _parse_branch_target_candidates(operands.strip())

    def strip_asm_comment(self, text: str) -> str:
        """Strip an inline assembly comment from a disassembly text line.

        x86 and RISC-V use ``#``-style comments; AArch64 overrides this to
        strip ``//``-style comments instead (since ``#`` is used for
        immediates like ``#0x1``).
        """
        return re.sub(r"#.*$", "", text).strip()


class X86Arch(ArchBase):
    """x86 / x86-64 architecture."""

    name = "x86"

    def __init__(self):
        super().__init__("objdump", [])

    def is_branch(self, mnemonic: str) -> bool:
        m = mnemonic.lower()
        return m.startswith(("j", "call", "loop"))

    def ends_basic_block(self, mnemonic: str, operands: str) -> bool:
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

    def is_load_instruction(self, mnemonic: str, operands: str) -> bool:
        m = mnemonic.lower()
        if m in ("lea", "leal", "leaq"):
            return False
        # In AT&T syntax the source (first) operand comes first.
        # A memory operand contains '(' (e.g. ``(%rdi)``, ``8(%rsp)``).
        parts = operands.split(",") if operands else []
        src = parts[0].strip() if parts else operands.strip()
        return "(" in src

    def get_branch_target(self, operands: str, mnemonic: str = ""):
        op = operands.strip()
        # x86/x86-64 AT&T indirect branches: *%reg or *offset(%reg).
        if op.startswith("*"):
            return None
        return _parse_branch_target_candidates(op)


class AArch64Arch(ArchBase):
    """AArch64 (64-bit ARM) architecture."""

    name = "aarch64"

    def __init__(self, objdump: str = "objdump"):
        super().__init__(objdump, ["-march=aarch64", "-mcpu=cortex-a57"])

    def is_branch(self, mnemonic: str) -> bool:
        m = mnemonic.lower()
        return m in _AARCH64_BRANCHES or m.startswith("b.")

    def ends_basic_block(self, mnemonic: str, operands: str) -> bool:
        m = mnemonic.lower()
        # bl / blr are calls (fall-through continues) — they do NOT end a BB.
        return (
            m in ("b", "br", "ret", "cbz", "cbnz", "tbz", "tbnz")
            or m.startswith("b.")  # b.eq, b.ne, b.lt, etc.
        )

    def is_load_instruction(self, mnemonic: str, operands: str) -> bool:
        return mnemonic.lower().startswith("ld")

    def strip_asm_comment(self, text: str) -> str:
        # AArch64 uses # for immediate operands (e.g. #0x1), so only
        # strip // style comments.
        return re.sub(r"//.*$", "", text).strip()


class ARMArch(ArchBase):
    """32-bit ARM architecture."""

    name = "arm"

    def __init__(self, objdump: str = "objdump"):
        super().__init__(objdump, ["-march=arm"])

    def is_branch(self, mnemonic: str) -> bool:
        return mnemonic.lower() in _ARM_BRANCHES

    def ends_basic_block(self, mnemonic: str, operands: str) -> bool:
        m = mnemonic.lower()
        # bl / blx and their conditional forms are calls (fall-through) — they
        # do NOT terminate a basic block.  All other branch forms do.
        return m in _ARM_BRANCHES and m not in _ARM_CALLS

    def is_load_instruction(self, mnemonic: str, operands: str) -> bool:
        m = mnemonic.lower()
        return m.startswith("ld") or m == "pop"


class RISCVArch(ArchBase):
    """RISC-V architecture (RV32 and RV64)."""

    name = "riscv"

    def __init__(self, objdump: str = "objdump", mca_args: list = None):
        if mca_args is None:
            mca_args = ["-march=riscv64", "-mcpu=sifive-u74"]
        super().__init__(objdump, mca_args)

    def is_branch(self, mnemonic: str) -> bool:
        return mnemonic.lower() in _RISCV_BRANCHES

    def ends_basic_block(self, mnemonic: str, operands: str) -> bool:
        m = mnemonic.lower()
        # All conditional branches (beq/bne/blt/bge/bltu/bgeu/…) end a BB.
        # j/jr are unconditional pseudo-jumps; ret is a return.
        # call/tail are pseudo-calls (fall-through continues) — do NOT end a BB.
        return m in _RISCV_BRANCHES and m not in ("call", "tail")

    def is_load_instruction(self, mnemonic: str, operands: str) -> bool:
        return mnemonic.lower() in ("lb", "lh", "lw", "ld", "lbu", "lhu", "lwu")

    def get_branch_target(self, operands: str, mnemonic: str = ""):
        op = operands.strip()
        m = mnemonic.lower()
        # RISC-V: ``jr rs`` is the pseudo-instruction for ``jalr x0, rs, 0``
        # (unconditional indirect jump through a register).  ``jalr rs`` with a
        # single register operand (no comma) is also an indirect branch or call.
        # Both must return None rather than misinterpreting the register name as
        # an address — which would happen for a0–a7 since those names consist
        # entirely of hex digits.
        if m in ("jr", "jalr") and "," not in op:
            return None
        return _parse_branch_target_candidates(op)


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

def _find_cross_tool(preferred: str) -> str:
    """Return *preferred* tool if it exists on PATH, else fall back to plain objdump."""
    return preferred if shutil.which(preferred) else "objdump"


def _arch_from_platform() -> ArchBase:
    """Return an :class:`ArchBase` subclass based on the host machine's architecture.

    Used as a fallback when the ``file`` command is not available.
    ``platform.machine()`` provides the CPU type (e.g. ``x86_64``,
    ``aarch64``, ``riscv64``) and ``platform.architecture()`` provides the
    pointer-size bit width, which is used to distinguish RV32 from RV64.
    """
    machine = platform.machine().lower()
    bits, _ = platform.architecture()

    if "riscv" in machine:
        objdump = _find_cross_tool("riscv64-linux-gnu-objdump")
        mca_args = (
            ["-march=riscv32", "-mcpu=sifive-e31"]
            if bits == "32bit"
            else ["-march=riscv64", "-mcpu=sifive-u74"]
        )
        return RISCVArch(objdump, mca_args)

    if machine in ("aarch64", "arm64"):
        objdump = _find_cross_tool("aarch64-linux-gnu-objdump")
        return AArch64Arch(objdump)

    if machine.startswith("arm"):
        objdump = _find_cross_tool("arm-linux-gnueabihf-objdump")
        return ARMArch(objdump)

    return X86Arch()


def _detect_arch(binary: str) -> ArchBase:
    """Detect the architecture of *binary* and return an :class:`ArchBase` subclass."""
    if not shutil.which("file"):
        print(
            "Warning: 'file' command not found; falling back to host architecture.",
            file=sys.stderr,
        )
        return _arch_from_platform()
    try:
        file_out = subprocess.run(
            ["file", binary], capture_output=True, text=True
        ).stdout.lower()
    except Exception:
        file_out = ""

    if "ucb risc-v" in file_out:
        objdump = _find_cross_tool("riscv64-linux-gnu-objdump")
        mca_args = (
            ["-march=riscv32", "-mcpu=sifive-e31"]
            if "32-bit" in file_out
            else ["-march=riscv64", "-mcpu=sifive-u74"]
        )
        return RISCVArch(objdump, mca_args)

    if "aarch64" in file_out or "arm64" in file_out:
        objdump = _find_cross_tool("aarch64-linux-gnu-objdump")
        return AArch64Arch(objdump)

    if re.search(r"\barm\b", file_out):
        objdump = _find_cross_tool("arm-linux-gnueabihf-objdump")
        return ARMArch(objdump)

    return X86Arch()


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
    branch instruction.  Used by all three ``_format_asm*`` variants to avoid
    duplicating the label-discovery loop.
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

    *label_sfx* is appended to the local label name to make labels unique
    across repetitions in :func:`_format_asm_with_cache_miss` (e.g. ``_r0``).
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

    suffix = f" {new_operands}" if new_operands else ""
    return f"\t{mnemonic}{suffix}"

# Number of times the assembly block is repeated in stochastic cache-miss mode.
_CACHE_MISS_REPEAT = 100


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
            tail = f" {operands}" if operands else ""
            lines.append(f"\t{mnemonic}{tail}")

    lines.append(".Lmca_end:")
    return "\n".join(lines) + "\n"


def _format_asm_with_average_load_latency(instrs, arch: ArchBase,
                                          latency: int = 0) -> str:
    """Format *instrs* as assembly where every load gets a fixed latency override.

    Every load instruction is wrapped with ``# LLVM-MCA-LATENCY <latency>``
    and ``# LLVM-MCA-LATENCY`` directives, giving all loads the specified
    latency uniformly.  Unlike :func:`_format_asm_with_cache_miss`, no
    repetition is applied and all loads are treated identically.

    This implements the *average* cache-miss mode where all loads take
    ``cache_latency * cache_miss`` cycles, as opposed to a fraction of loads
    taking the full *cache_latency*.
    """
    addr_set = {a for a, _, _ in instrs}
    labeled = _compute_labeled_addrs(instrs, arch)

    lines = []
    for addr, mnemonic, operands in instrs:
        if addr in labeled:
            lines.append(f".Lmca_{addr:x}:")
        tail = f" {operands}" if operands else ""
        if arch.is_load_instruction(mnemonic, operands):
            lines.append(f"# LLVM-MCA-LATENCY {latency}")
            lines.append(f"\t{mnemonic}{tail}")
            lines.append("# LLVM-MCA-LATENCY")
        elif arch.is_branch(mnemonic):
            lines.append(_format_branch_instr(mnemonic, operands, addr_set, arch))
        else:
            lines.append(f"\t{mnemonic}{tail}")

    lines.append(".Lmca_end:")
    return "\n".join(lines) + "\n"


def _format_asm_with_cache_miss(instrs, arch: ArchBase,
                                 cache_miss: float = 0.0,
                                 cache_latency: int = 0) -> str:
    """Format *instrs* as assembly for llvm-mca with cache-miss simulation.

    *cache_miss* is the average number of cache misses per load instruction
    and may be any non-negative value, including values greater than 1.

    The code block is repeated ``_CACHE_MISS_REPEAT`` times.  The penalty is
    modelled in two parts:

    * **Base penalty** — every load receives
      ``round(cache_miss) * cache_latency`` cycles unconditionally, via an
      ``# LLVM-MCA-LATENCY`` override.

    * **Fractional adjustment** — let ``frac = cache_miss - round(cache_miss)``
      (in the range ``(-0.5, 0.5]``).  A fraction ``|frac|`` of load instances
      across the repeated block receives an additional ``±cache_latency``
      adjustment (``+`` when ``frac > 0``, ``−`` when ``frac < 0``), placed
      deterministically using the same evenly-spaced formula as the original
      stochastic logic.

    When ``cache_miss < 1`` the base penalty is zero and only the fractional
    stochastic component operates (identical to the original behaviour).

    Labels are made unique per repetition so that backward branches within a
    loop still resolve to the correct iteration-local target.
    """
    addr_set = {a for a, _, _ in instrs}
    labeled = _compute_labeled_addrs(instrs, arch)

    # Split cache_miss into a rounded base and a signed fractional remainder.
    base_misses = round(cache_miss)                   # integer base for all loads
    frac = cache_miss - base_misses                   # signed fraction in (-0.5, 0.5]
    base_latency = base_misses * cache_latency        # guaranteed latency
    extra = cache_latency if frac >= 0 else -cache_latency  # per-adjustment step

    # n: load instructions per repetition; b: total generated load count.
    # a: number of loads that receive the ±extra fractional adjustment.
    # Adjustment positions are floor(m * b / a) for m in 0..a-1.
    n = sum(
        1 for _, mn, ops in instrs if arch.is_load_instruction(mn, ops)
    )
    b = _CACHE_MISS_REPEAT * n
    a = round(abs(frac) * b) if b > 0 else 0
    load_counter = 0
    miss_counter = 0
    next_miss_position = 0

    def _emit_load(mnemonic: str, operands: str, lines: list) -> None:
        """Append a load instruction with the appropriate latency directive."""
        nonlocal load_counter, miss_counter, next_miss_position
        tail = f" {operands}" if operands else ""
        if miss_counter < a and load_counter == next_miss_position:
            lat = base_latency + extra
            miss_counter += 1
            next_miss_position = int(miss_counter * b / a)
        else:
            lat = base_latency
        if lat > 0:
            lines.append(f"# LLVM-MCA-LATENCY {lat}")
            lines.append(f"\t{mnemonic}{tail}")
            lines.append("# LLVM-MCA-LATENCY")
        else:
            lines.append(f"\t{mnemonic}{tail}")
        load_counter += 1

    lines = []
    for it in range(_CACHE_MISS_REPEAT):
        sfx = f"_r{it}"
        for addr, mnemonic, operands in instrs:
            if addr in labeled:
                lines.append(f".Lmca_{addr:x}{sfx}:")
            if arch.is_branch(mnemonic):
                lines.append(
                    _format_branch_instr(mnemonic, operands, addr_set, arch, sfx)
                )
            elif arch.is_load_instruction(mnemonic, operands):
                _emit_load(mnemonic, operands, lines)
            else:
                tail = f" {operands}" if operands else ""
                lines.append(f"\t{mnemonic}{tail}")

    lines.append(".Lmca_end:")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Cache-miss simulation mode classes
# ---------------------------------------------------------------------------

class _CacheMissMode:
    """Base class for cache-miss simulation strategies.

    Each subclass encapsulates the cache-miss parameters as instance
    variables and provides a ``format_asm`` method that produces the
    appropriate assembly for llvm-mca, as well as ``extra_mca_args`` for
    any extra flags required by that strategy.
    """

    def format_asm(self, instrs, arch: ArchBase) -> str:
        """Format *instrs* as assembly with the cache-miss strategy applied."""
        raise NotImplementedError

    def extra_mca_args(self) -> list:
        """Return extra arguments to add to the llvm-mca command."""
        return []


class _NoCacheMiss(_CacheMissMode):
    """No cache-miss simulation — plain assembly formatting."""

    def format_asm(self, instrs, arch: ArchBase) -> str:
        return _format_asm(instrs, arch)


class _StochasticCacheMiss(_CacheMissMode):
    """Stochastic cache-miss simulation based on instructions-per-cache-miss.

    The assembly is repeated ``_CACHE_MISS_REPEAT`` times with
    ``# LLVM-MCA-LATENCY`` directives inserted deterministically.  The
    effective miss penalty is derived from *instructions_per_cache_miss* and
    the region's instruction/load mix:

    * Let ``expected_misses = num_instructions / instructions_per_cache_miss``.
    * If ``num_loads >= expected_misses``: a fraction
      ``expected_misses / num_loads`` of loads receives the full
      *cache_latency* penalty (at most one miss per load).
    * If ``num_loads < expected_misses``: misses are distributed across loads
      using Bresenham-style rounding so that each load receives an integer
      number of miss penalties (either ``floor`` or ``ceil`` of the average).
      The block is still repeated ``_CACHE_MISS_REPEAT`` times with the same
      per-load latencies every iteration.

    ``-iterations=1`` is added to llvm-mca so that it does not add its own
    repetitions.
    """

    def __init__(self, instructions_per_cache_miss: float, cache_latency: int):
        self.instructions_per_cache_miss = instructions_per_cache_miss
        self.cache_latency = cache_latency

    def format_asm(self, instrs, arch: ArchBase) -> str:
        num_instrs = len(instrs)
        num_loads = sum(
            1 for _, mn, ops in instrs if arch.is_load_instruction(mn, ops)
        )
        if num_loads == 0:
            return _format_asm(instrs, arch)
        expected_misses = num_instrs / self.instructions_per_cache_miss
        miss_fraction = expected_misses / num_loads
        return _format_asm_with_cache_miss(instrs, arch, miss_fraction,
                                           self.cache_latency)

    def extra_mca_args(self) -> list:
        return ["-iterations=1"]


class _AverageCacheMiss(_CacheMissMode):
    """Average cache-miss simulation based on instructions-per-cache-miss.

    Each load instruction receives the same latency override derived from
    *instructions_per_cache_miss* and the region's instruction/load mix:

    * Let ``expected_misses = num_instructions / instructions_per_cache_miss``.
    * Every load gets ``round(expected_misses / num_loads * cache_latency)``
      cycles via an ``# LLVM-MCA-LATENCY`` directive.
    """

    def __init__(self, instructions_per_cache_miss: float, cache_latency: int):
        self.instructions_per_cache_miss = instructions_per_cache_miss
        self.cache_latency = cache_latency

    def format_asm(self, instrs, arch: ArchBase) -> str:
        num_instrs = len(instrs)
        num_loads = sum(
            1 for _, mn, ops in instrs if arch.is_load_instruction(mn, ops)
        )
        if num_loads == 0:
            return _format_asm(instrs, arch)
        expected_misses = num_instrs / self.instructions_per_cache_miss
        avg_latency = round(expected_misses / num_loads * self.cache_latency)
        return _format_asm_with_average_load_latency(instrs, arch, avg_latency)


def _build_cache_mode(instructions_per_cache_miss: float, cache_latency: int,
                      cache_miss_mode: str) -> _CacheMissMode:
    """Build and return the appropriate :class:`_CacheMissMode` instance.

    *instructions_per_cache_miss* must be positive and finite for a
    cache-miss simulation to be applied.  ``float('inf')``, zero, or negative
    values all result in :class:`_NoCacheMiss` (no simulation).
    """
    if math.isinf(instructions_per_cache_miss) or instructions_per_cache_miss <= 0:
        return _NoCacheMiss()
    if cache_miss_mode == "average":
        return _AverageCacheMiss(instructions_per_cache_miss, cache_latency)
    return _StochasticCacheMiss(instructions_per_cache_miss, cache_latency)


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


def _parse_load_proportion(mca_output: str) -> float:
    """Parse the Instruction Information section of llvm-mca output.

    Returns the proportion of instructions with the MayLoad attribute set,
    or 0.0 if the section is absent or no instructions are listed.

    llvm-mca emits a table like::

        [1]    [2]    [3]    [4]    [5]    [6]    Instructions:
         1      4     1.00    *                    mov rax, qword ptr [rdi]
         1      1     0.25                         inc rax

    where column [4] contains ``*`` when the instruction may perform a load.
    The column position is determined from the header line so that it remains
    correct regardless of the exact whitespace used by different llvm-mca
    versions.
    """
    lines = mca_output.splitlines()
    col4_pos = None
    in_table = False
    total = 0
    loads = 0
    for line in lines:
        if not in_table:
            # Locate the table header to determine the MayLoad column position.
            if "[4]" in line and "[1]" in line:
                m = re.search(r"\[4\]", line)
                if m:
                    col4_pos = m.start()
                    in_table = True
            continue
        # A blank line signals the end of the instruction-info table.
        if not line.strip():
            break
        # Instruction data rows start with optional whitespace then a digit.
        if re.match(r"^\s+\d", line):
            total += 1
            if col4_pos is not None:
                # Each column in the instruction-info table is 7 characters
                # wide, matching the header pattern "[N]    " (3 chars for the
                # bracket notation plus 4 spaces).  A '*' anywhere within that
                # range indicates a load instruction.
                field = line[col4_pos:col4_pos + 7] if len(line) > col4_pos else ""
                if "*" in field:
                    loads += 1
    return loads / total if total > 0 else 0.0


def _run_mca(instrs, mca_args=(), arch: ArchBase = None,
             cache_mode: _CacheMissMode = None):
    """Run llvm-mca on *instrs* and return ``(ipc, load_proportion)``, or None.

    Parameters
    ----------
    instrs:
        List of ``(addr, mnemonic, operands)`` triples to analyse.
    mca_args:
        Extra arguments forwarded to llvm-mca (e.g. ``-march=``, ``-mcpu=``).
    arch:
        Architecture object.  Defaults to :class:`X86Arch` when omitted.
    cache_mode:
        Cache-miss simulation strategy.  Defaults to :class:`_NoCacheMiss`
        (no simulation) when omitted.
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

    if arch is None:
        arch = X86Arch()
    if cache_mode is None:
        cache_mode = _NoCacheMiss()

    asm = cache_mode.format_asm(instrs, arch)
    extra = cache_mode.extra_mca_args()

    # Pass assembly via stdin (llvm-mca reads from stdin when given "-").
    # -instruction-info adds the per-instruction table used to extract MayLoad.
    cmd = [_LLVM_MCA, *mca_args, *extra,
           "--call-latency=0", "-instruction-info", "-"]
    proc = subprocess.run(cmd, input=asm, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    m = re.search(r"\bIPC:\s+([\d.]+)", proc.stdout)
    if m is None:
        return None
    ipc = float(m.group(1))
    load_proportion = _parse_load_proportion(proc.stdout)
    return ipc, load_proportion


def _yield_mca_result(instrs, mca_args, arch: ArchBase,
                      cache_mode: _CacheMissMode):
    """Run llvm-mca on *instrs* and yield a ``(start, end, ipc, load_proportion)`` tuple.

    Nothing is yielded when *instrs* is empty or llvm-mca returns no result.
    """
    if instrs:
        result = _run_mca(instrs, mca_args, arch, cache_mode)
        if result is not None:
            ipc, load_proportion = result
            yield instrs[0][0], instrs[-1][0], ipc, load_proportion


# ---------------------------------------------------------------------------
# Per-function analysis
# ---------------------------------------------------------------------------

def _analyze_function(instrs, mca_args=(), arch: ArchBase = None,
                      cache_mode: _CacheMissMode = None):
    """Analyse one function's instructions.

    Yields ``(start_addr, end_addr, ipc, load_proportion, kind)`` tuples for
    every loop and every basic block that is not part of a loop.  *kind* is
    ``"loop"`` for a loop region and ``"block"`` for a basic-block region.
    """
    if arch is None:
        arch = X86Arch()
    if cache_mode is None:
        cache_mode = _NoCacheMiss()

    loops = _find_loops(instrs, arch)

    # --- Loops (including nested loops as separate entries) ---
    for ls, le in loops:
        region = [(a, m, o) for a, m, o in instrs if ls <= a <= le]
        for start, end, ipc, lp in _yield_mca_result(region, mca_args, arch,
                                                      cache_mode):
            yield start, end, ipc, lp, "loop"

    # --- Basic blocks outside loops ---
    non_loop = [(a, m, o) for a, m, o in instrs if not _in_any_loop(a, loops)]
    bb = []
    for instr in non_loop:
        addr, mnemonic, operands = instr
        bb.append(instr)
        if arch.ends_basic_block(mnemonic, operands):
            for start, end, ipc, lp in _yield_mca_result(bb, mca_args, arch,
                                                          cache_mode):
                yield start, end, ipc, lp, "block"
            bb = []
    for start, end, ipc, lp in _yield_mca_result(bb, mca_args, arch,
                                                  cache_mode):
        yield start, end, ipc, lp, "block"


def _analyze_function_ipc(instrs, mca_args=(), arch: ArchBase = None,
                           cache_mode: _CacheMissMode = None):
    """Compute the IPC estimate for one function.

    IPC_f = max { IPC_l  for l in loops inside f }

    If the function contains no loops, falls back to:

    IPC_f = max { IPC_b  for b in basic blocks inside f }

    Returns a ``(start_addr, end_addr, ipc, load_proportion)`` tuple where the
    address range spans the whole function, *ipc* is the highest IPC found, and
    *load_proportion* belongs to that highest-IPC region.  Returns ``None``
    when llvm-mca produces no results for any region in the function.
    """
    if not instrs:
        return None

    if arch is None:
        arch = X86Arch()
    if cache_mode is None:
        cache_mode = _NoCacheMiss()

    loop_candidates = []
    block_candidates = []
    for _, _, ipc, lp, kind in _analyze_function(instrs, mca_args, arch,
                                                  cache_mode):
        if ipc > 0:
            if kind == "loop":
                loop_candidates.append((ipc, lp))
            else:
                block_candidates.append((ipc, lp))

    candidates = loop_candidates or block_candidates
    if not candidates:
        return None
    best_ipc, best_lp = max(candidates, key=lambda r: r[0])
    return instrs[0][0], instrs[-1][0], best_ipc, best_lp


# ---------------------------------------------------------------------------
# Analyzer class — encapsulates optional analysis parameters
# ---------------------------------------------------------------------------

class Analyzer:
    """Analyses an ELF binary using llvm-mca.

    Optional analysis parameters (CPU override, analysis mode, and cache-miss
    simulation settings) are stored as instance variables rather than being
    threaded through every function call as positional/keyword arguments.

    Parameters
    ----------
    binary:
        Path to the ELF binary to analyse.
    mcpu:
        If non-empty, overrides the default ``-mcpu`` value chosen by
        :func:`_detect_arch` and is forwarded to llvm-mca.
    mode:
        ``"blocks"`` (default) — yield ``(start, end, ipc, load_proportion)``
        for every loop and non-loop basic block.

        ``"functions"`` — yield one tuple per function where *ipc* is the
        maximum IPC across all loops (or basic blocks when no loops exist).
    cache_miss:
        Number of retired instructions per cache miss (``instructions_per_cache_miss``).
        Use ``float('inf')`` (default) for no cache-miss simulation.
    cache_latency:
        Cache-miss penalty in cycles.  Only used when *instructions_per_cache_miss*
        is finite.
    cache_miss_mode:
        ``"stochastic"`` (default) or ``"average"``.  See :func:`analyze`
        for a full description.
    """

    def __init__(self, binary: str, mcpu: str = "", mode: str = "blocks",
                 cache_miss: float = float("inf"), cache_latency: int = 0,
                 cache_miss_mode: str = "stochastic"):
        self.binary = binary
        self.mcpu = mcpu
        self.mode = mode
        self.cache_miss = cache_miss
        self.cache_latency = cache_latency
        self.cache_miss_mode = cache_miss_mode

    def _effective_mca_args(self, arch: ArchBase) -> list:
        """Return mca_args for *arch*, applying any :attr:`mcpu` override."""
        mca_args = arch.mca_args
        if self.mcpu:
            mca_args = [a for a in mca_args if not a.startswith("-mcpu=")]
            mca_args = mca_args + [f"-mcpu={self.mcpu}"]
        return mca_args

    def run(self):
        """Analyse :attr:`binary` and yield ``(start, end, ipc, load_proportion)`` tuples."""
        arch = _detect_arch(self.binary)
        mca_args = self._effective_mca_args(arch)
        cache_mode = _build_cache_mode(self.cache_miss, self.cache_latency,
                                       self.cache_miss_mode)

        for _func_name, instrs in disassemble(self.binary, arch):
            if self.mode == "functions":
                result = _analyze_function_ipc(instrs, mca_args, arch,
                                               cache_mode)
                if result is not None:
                    yield result
            else:
                for start, end, ipc, lp, _kind in _analyze_function(
                        instrs, mca_args, arch, cache_mode):
                    yield start, end, ipc, lp


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------

def analyze(binary: str, mcpu: str = "", mode: str = "blocks",
            cache_miss: float = float("inf"), cache_latency: int = 0,
            cache_miss_mode: str = "stochastic"):
    """Analyse *binary* and yield result tuples.

    This is a convenience wrapper around :class:`Analyzer`.  All parameters
    are forwarded as instance variables; see :class:`Analyzer` for details.

    Parameters
    ----------
    binary:
        Path to the ELF binary to analyse.
    mcpu:
        If non-empty, overrides the default ``-mcpu`` value chosen by
        ``_detect_arch`` and is forwarded to llvm-mca.
    mode:
        ``"blocks"`` (default) — yield ``(start, end, ipc, load_proportion)``
        for every loop and non-loop basic block, as in the original behaviour.

        ``"functions"`` — yield ``(start, end, ipc, load_proportion)`` for
        every function, where *start*/*end* span the whole function and
        ``ipc = max { IPC_l  for all loops l in f }`` (or the max over basic
        blocks when the function has no loops).
    cache_miss:
        Number of retired instructions per cache miss
        (``instructions_per_cache_miss``).  Use ``float('inf')`` (default) for
        no cache-miss simulation.  Interpretation depends on *cache_miss_mode*.
    cache_latency:
        Cache-miss penalty in cycles.  Only used when *cache_miss* is finite.
        Default is 0.
    cache_miss_mode:
        ``"stochastic"`` (default) — the code block is repeated 100 times with
        ``# LLVM-MCA-LATENCY`` directives distributed across load instructions
        according to the effective miss fraction derived from *cache_miss*;
        llvm-mca is run with ``-iterations=1``.

        ``"average"`` — every load instruction receives a fixed latency derived
        from ``(num_instructions / cache_miss / num_loads) * cache_latency``
        cycles; models the average cost of cache misses uniformly across all
        loads.
    """
    return Analyzer(
        binary=binary,
        mcpu=mcpu,
        mode=mode,
        cache_miss=cache_miss,
        cache_latency=cache_latency,
        cache_miss_mode=cache_miss_mode,
    ).run()


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
    parser.add_argument(
        "--mode",
        choices=["blocks", "functions"],
        default="blocks",
        help=(
            "Analysis mode. 'blocks' (default) reports each basic block and "
            "loop separately with its IPC. 'functions' reports one IPC "
            "estimate per function (address range = whole function; "
            "IPC = max IPC across all loops in the function, or max IPC "
            "across basic blocks if the function has no loops)."
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
            "when --instructions-per-cache-miss is finite."
        ),
    )
    parser.add_argument(
        "--cache-miss-mode",
        choices=["stochastic", "average"],
        default="stochastic",
        dest="cache_miss_mode",
        help=(
            "Cache-miss simulation mode (default: stochastic). "
            "'stochastic': the code block is repeated 100 times and load "
            "instructions receive the full --cache-latency penalty according "
            "to the effective miss fraction derived from "
            "--instructions-per-cache-miss; llvm-mca is run with "
            "-iterations=1. "
            "'average': all load instructions receive a fixed latency derived "
            "from (num_instructions / --instructions-per-cache-miss / "
            "num_loads) * --cache-latency cycles, modelling the average cost "
            "of cache misses uniformly across all loads."
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        parser.error(f"{args.binary}: no such file")

    if args.cache_miss <= 0:
        parser.error("--instructions-per-cache-miss must be > 0")
    if args.cache_latency < 0:
        parser.error("--cache-latency must be >= 0")

    runner = Analyzer(
        binary=args.binary,
        mcpu=args.mcpu,
        mode=args.mode,
        cache_miss=args.cache_miss,
        cache_latency=args.cache_latency,
        cache_miss_mode=args.cache_miss_mode,
    )

    if args.mode == "functions":
        # Function mode: one row per function, same output format as block mode.
        print("start_address,end_address,throughput,load_proportion")
        for start, end, ipc, load_proportion in runner.run():
            print(f"0x{start:x},0x{end:x},{ipc:.2f},{load_proportion:.4f}")
    else:
        # Block mode (default): existing behaviour — one row per loop/BB, sorted.
        results = sorted(
            runner.run(),
            key=lambda x: (x[1], -x[0]))
        print("start_address,end_address,throughput,load_proportion")
        for start, end, ipc, load_proportion in results:
            print(f"0x{start:x},0x{end:x},{ipc:.2f},{load_proportion:.4f}")


if __name__ == "__main__":
    main()
