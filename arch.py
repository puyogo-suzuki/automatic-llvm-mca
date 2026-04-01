"""Architecture classes and detection for automatic-llvm-mca.

This module contains all architecture-specific constants, helper functions,
classes, and detection logic used by analyze.py.

Supported architectures: x86/x86-64, AArch64, 32-bit ARM, RISC-V (RV32IC, RV64IC).
"""

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
