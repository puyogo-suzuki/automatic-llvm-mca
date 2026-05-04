"""Integration and unit tests for analyze.py — AMD64 (x86-64) and AArch64.

Integration tests compile a small C function that contains a loop and memory
loads, then run ``analyze.analyze()`` on the resulting ELF object file and
validate the output.

Unit tests exercise internal helpers such as ``_get_branch_target`` and
``_find_loops`` to document expected behaviour and guard against regressions.

Run with::

    python3 -m pytest tests/test_analyze.py -v
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Make the package root importable when running pytest from any directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze  # noqa: E402  (import after sys.path manipulation)


# ---------------------------------------------------------------------------
# Simple C source used by all integration tests.
# It contains a loop (backward branch) and a memory load so that both
# loop-detection can be verified.
# ---------------------------------------------------------------------------
_C_SOURCE = """\
long sum(long *a, int n) {
    long s = 0;
    for (int i = 0; i < n; i++)
        s += a[i];
    return s;
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _have(*tools: str) -> bool:
    """Return True if every tool in *tools* is found on PATH."""
    return all(shutil.which(t) for t in tools)


def _llvm_mca_available() -> bool:
    """Return True if any version of llvm-mca is available on PATH."""
    if shutil.which("llvm-mca"):
        return True
    for ver in range(30, 10, -1):
        if shutil.which(f"llvm-mca-{ver}"):
            return True
    return False


def _compile(source: str, compiler: str, extra_flags=()) -> str:
    """Compile *source* to an ELF object file and return its path.

    The caller is responsible for deleting the file when done.
    """
    fd_src, src_path = tempfile.mkstemp(suffix=".c")
    try:
        with os.fdopen(fd_src, "w") as f:
            f.write(source)
        fd_obj, obj_path = tempfile.mkstemp(suffix=".o")
        os.close(fd_obj)
        subprocess.run(
            [compiler, "-O2", "-c", *extra_flags, "-o", obj_path, src_path],
            check=True,
            capture_output=True,
        )
        return obj_path
    finally:
        os.unlink(src_path)


# ---------------------------------------------------------------------------
# skip markers
# ---------------------------------------------------------------------------

_NEED_MCA = pytest.mark.skipif(
    not _llvm_mca_available(),
    reason="llvm-mca not found — install llvm",
)

_NEED_X86_GCC = pytest.mark.skipif(
    not _have("x86_64-linux-gnu-gcc") and not _have("gcc"),
    reason="x86-64 gcc not found",
)

_NEED_AARCH64 = pytest.mark.skipif(
    not _have("aarch64-linux-gnu-gcc", "aarch64-linux-gnu-objdump"),
    reason="aarch64-linux-gnu-gcc / aarch64-linux-gnu-objdump not found",
)


# ---------------------------------------------------------------------------
# Module-scoped fixtures so the compilation runs only once per architecture.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def x86_obj():
    """Compile the test C source to an x86-64 ELF object file."""
    if not _have("x86_64-linux-gnu-gcc") and not _have("gcc"):
        pytest.skip("x86-64 gcc not found")
    compiler = (
        "x86_64-linux-gnu-gcc"
        if shutil.which("x86_64-linux-gnu-gcc")
        else "gcc"
    )
    obj = _compile(_C_SOURCE, compiler)
    yield obj
    os.unlink(obj)


@pytest.fixture(scope="module")
def aarch64_obj():
    """Compile the test C source to an AArch64 ELF object file."""
    if not _have("aarch64-linux-gnu-gcc", "aarch64-linux-gnu-objdump"):
        pytest.skip("aarch64-linux-gnu-gcc / aarch64-linux-gnu-objdump not found")
    obj = _compile(_C_SOURCE, "aarch64-linux-gnu-gcc")
    yield obj
    os.unlink(obj)


# ---------------------------------------------------------------------------
# Unit tests — internal helpers
# ---------------------------------------------------------------------------

class TestGetBranchTarget:
    """Unit tests for arch.get_branch_target() on each architecture class.

    The tests are grouped to document three distinct categories:

    (A) Correct direct-branch cases — the method returns the right target.
    (B) Harmless false-positive candidates — earlier hex-like register names
        are matched by the plain-hex fallback but the last-candidate heuristic
        still returns the correct address.
    (C) Previously broken indirect-branch false positives — RISC-V ``jr``/
        ``jalr`` with a bare register operand whose name is all-hex (``a0``–
        ``a7``) were misidentified as direct branches to an address.  This is
        now handled by the architecture class's ``get_branch_target`` method.
    """

    # ------------------------------------------------------------------
    # (A) Correct direct-branch parsing
    # ------------------------------------------------------------------

    # x86-64 style (AT&T syntax, plain hex, possibly short)
    def test_x86_plain_hex_3digits(self):
        assert analyze.X86Arch().get_branch_target("4016") == 0x4016

    def test_x86_plain_hex_2digits(self):
        """Two-digit hex targets (common in object files) must be parsed."""
        assert analyze.X86Arch().get_branch_target("20") == 0x20

    def test_x86_plain_hex_1digit(self):
        """Single-digit hex targets must be parsed."""
        assert analyze.X86Arch().get_branch_target("8") == 0x8

    # AArch64 style
    def test_aarch64_bne_2digit_target(self):
        """b.ne with a 2-digit hex target (object-file address)."""
        assert analyze.AArch64Arch().get_branch_target("18", "b.ne") == 0x18

    def test_aarch64_ble_2digit_target(self):
        assert analyze.AArch64Arch().get_branch_target("2c", "b.le") == 0x2c

    def test_aarch64_cbz_register_plus_target(self):
        """cbz xN, addr — target is the last operand."""
        assert analyze.AArch64Arch().get_branch_target("x0, 2dc", "cbz") == 0x2dc

    def test_aarch64_tbz_register_imm_target(self):
        """tbz xN, #imm, addr — the #-prefixed imm must NOT be taken as target."""
        assert analyze.AArch64Arch().get_branch_target("x0, #0, 2dc", "tbz") == 0x2dc

    def test_aarch64_0x_prefix(self):
        assert analyze.AArch64Arch().get_branch_target("0x400808", "b") == 0x400808

    # ------------------------------------------------------------------
    # (B) Harmless false-positive candidates
    # These registers (a0, a5…) are all-hex so they appear in the
    # candidate list, but the LAST candidate — the real target address —
    # is what gets returned.
    # ------------------------------------------------------------------

    def test_riscv_bne_fp_candidates_last_is_target(self):
        """bne a4,a5,314 — a4 and a5 are false-positive candidates,
        but the last candidate (0x314) is the correct branch target."""
        assert analyze.RISCVArch().get_branch_target("a4,a5,314", "bne") == 0x314

    def test_riscv_beqz_fp_candidate_last_is_target(self):
        """beqz a0,100 — a0 is a false-positive candidate,
        but the last candidate (0x100) is the correct branch target."""
        assert analyze.RISCVArch().get_branch_target("a0,100", "beqz") == 0x100

    # ------------------------------------------------------------------
    # (C) Indirect-branch false positives — must return None
    # jr/jalr with a bare register operand (no comma) is an indirect
    # jump; a0–a5 are all-hex so they would be mistaken for addresses
    # without the mnemonic/arch guard in RISCVArch.get_branch_target.
    # ------------------------------------------------------------------

    def test_riscv_jr_indirect_a0(self):
        """jr a0 is an indirect jump; must return None, not 0xa0."""
        assert analyze.RISCVArch().get_branch_target("a0", "jr") is None

    def test_riscv_jr_indirect_a5(self):
        """jr a5 is an indirect jump; must return None, not 0xa5."""
        assert analyze.RISCVArch().get_branch_target("a5", "jr") is None

    def test_riscv_jalr_indirect_a1(self):
        """jalr a1 (single-register operand) is indirect; must return None."""
        assert analyze.RISCVArch().get_branch_target("a1", "jalr") is None

    def test_riscv_jalr_with_comma_bypasses_indirect_guard(self):
        """jalr with multiple operands (comma present) is NOT blocked by the
        indirect-branch guard — the guard only fires for single-register form.
        Normal parsing applies: a0 and 0 are candidates; last one (0) is returned.

        Note: multi-operand jalr (e.g. 'jalr ra, a0, 0') is architecturally an
        indirect branch too, but the operand format doesn't trigger the guard.
        This edge case is documented here rather than fixed, as binutils typically
        disassembles indirect RISC-V calls in single-register pseudo form ('jalr a0').
        """
        # "jalr ra, a0, 0" — last candidate is 0
        assert analyze.RISCVArch().get_branch_target("ra, a0, 0", "jalr") == 0x0

    def test_riscv_jr_ra_returns_none_due_to_nonhex_char(self):
        """jr ra — 'ra' contains a non-hex character ('r'), so the plain-hex
        regex finds no match and returns None independently of the mnemonic
        guard.  Verified here via _parse_branch_target_candidates to show
        the shared regex alone handles non-hex register names correctly."""
        assert analyze._parse_branch_target_candidates("ra") is None

    # Indirect branches — architecture-agnostic check (x86 AT&T prefix)
    def test_indirect_x86_star_prefix(self):
        """Indirect branch in AT&T syntax (*%reg) returns None."""
        assert analyze.X86Arch().get_branch_target("*%rax") is None


class TestFindLoops:
    """Unit tests for analyze._find_loops."""

    def _make_instrs(self, pairs):
        """Build a minimal instruction list from (addr, mnemonic, operands) triples."""
        return [(a, m, o) for a, m, o in pairs]

    def test_x86_simple_loop(self):
        """A backward jne is detected as a loop in x86 mode."""
        instrs = [
            (0x20, "add", "%eax,%edx"),
            (0x22, "add", "$0x2,%eax"),
            (0x25, "cmp", "%eax,%edi"),
            (0x27, "jne", "20"),  # backward branch → loop 0x20..0x27
        ]
        loops = analyze._find_loops(instrs, analyze.X86Arch())
        assert (0x20, 0x27) in loops

    def test_x86_no_loop_forward_branch(self):
        """A forward branch does not create a loop."""
        instrs = [
            (0x10, "test", "%edi,%edi"),
            (0x12, "jle", "30"),  # forward branch → no loop
            (0x14, "xor", "%eax,%eax"),
            (0x30, "ret", ""),
        ]
        loops = analyze._find_loops(instrs, analyze.X86Arch())
        assert loops == []

    def test_aarch64_simple_loop_short_addresses(self):
        """AArch64 object-file addresses (< 3 hex digits) are detected as loops."""
        instrs = [
            (0x18, "add", "w0, w0, w1"),
            (0x1c, "add", "w1, w1, #0x1"),
            (0x20, "cmp", "w2, w1"),
            (0x24, "b.ne", "18"),  # backward branch → loop 0x18..0x24
        ]
        loops = analyze._find_loops(instrs, analyze.AArch64Arch())
        assert (0x18, 0x24) in loops

    def test_aarch64_no_loop_forward_branch(self):
        """A forward AArch64 branch does not create a loop."""
        instrs = [
            (0x0, "mov", "w2, w0"),
            (0x4, "cmp", "w0, #0x0"),
            (0x8, "b.le", "2c"),  # forward branch → no loop
            (0x2c, "ret", ""),
        ]
        loops = analyze._find_loops(instrs, analyze.AArch64Arch())
        assert loops == []

    def test_riscv_jr_a0_not_a_loop(self):
        """jr a0 is an indirect jump; it must NOT produce a false loop even
        when 0xa0 (= decimal 160) coincidentally equals an instruction
        address earlier in the same function."""
        instrs = [
            (0xa0, "add", "a0, a0, a1"),   # address 0xa0 is in the function
            (0xa4, "addi", "a1, a1, 1"),
            (0xb0, "jr", "a0"),             # indirect — operand is a register,
                                             # NOT the address 0xa0
        ]
        loops = analyze._find_loops(instrs, analyze.RISCVArch())
        assert loops == [], (
            "jr a0 is an indirect branch and must not create a loop; "
            f"got loops={loops}"
        )


class TestFormatAsm:
    """Unit tests for analyze._format_asm indirect-branch handling."""

    def test_riscv_jr_kept_as_indirect(self):
        """_format_asm must emit 'jr a0' verbatim, not 'jr .Lmca_end'.

        Previously, RISCVArch.get_branch_target("a0") returned 0xa0 instead
        of None, causing _format_asm to rewrite the indirect branch as
        ``jr .Lmca_end`` — invalid RISC-V assembly that made llvm-mca fail
        and silently drop the region.
        """
        instrs = [
            (0x0,  "add", "a0, a0, a1"),
            (0x4,  "jr",  "a0"),
        ]
        asm = analyze._format_asm(instrs, analyze.RISCVArch())
        assert "jr a0" in asm, (
            f"Expected 'jr a0' to be preserved as an indirect branch. Got:\n{asm}"
        )
        assert ".Lmca_end" not in asm.split("\n")[1], (
            f"jr a0 must NOT be rewritten to use .Lmca_end. Got:\n{asm}"
        )

    def test_riscv_jalr_indirect_kept(self):
        """jalr a1 (single register, no comma) must also be kept as-is."""
        instrs = [
            (0x0, "addi", "a1, a1, 4"),
            (0x4, "jalr", "a1"),
        ]
        asm = analyze._format_asm(instrs, analyze.RISCVArch())
        assert "jalr a1" in asm, (
            f"Expected 'jalr a1' to be preserved. Got:\n{asm}"
        )


class TestComputeMLP:
    """Unit tests for loop-aware MLP computation."""

    def test_enable_loop_wraps_load_window(self):
        """With enable_loop=True, the window wraps from tail to head."""
        arch = analyze.X86Arch()
        instrs = [
            (0x0, "movq", "(%rdi), %rax"),   # load
            (0x4, "addq", "%rax, %rcx"),
            (0x8, "movq", "(%rsi), %rbx"),   # load (tail)
        ]

        no_loop = analyze._compute_mlp(
            instrs, decode_width=2, arch=arch, dependency="none", enable_loop=False
        )
        looped = analyze._compute_mlp(
            instrs, decode_width=2, arch=arch, dependency="none", enable_loop=True
        )

        assert no_loop == 1.0
        assert looped == 1.5

    def test_enable_loop_wraps_io_dependency_distance(self):
        """With enable_loop=True, io dependency search wraps to block head."""
        arch = analyze.X86Arch()
        instrs = [
            (0x0, "addq", "%rbx, %rax"),      # consumes %rbx at loop head
            (0x4, "addq", "%rax, %rcx"),
            (0x8, "movq", "(%rsi), %rbx"),    # load produces %rbx at tail
        ]

        no_loop = analyze._compute_mlp(
            instrs, decode_width=4, arch=arch, dependency="io", enable_loop=False
        )
        looped = analyze._compute_mlp(
            instrs, decode_width=4, arch=arch, dependency="io", enable_loop=True
        )

        assert no_loop == 1.0
        assert looped == 1.0

    def test_io_allows_loads_until_first_use_barrier(self):
        """io mode counts loads issued before the first use of anchor-load output."""
        arch = analyze.X86Arch()
        instrs = [
            (0x0, "movq", "(%rdi), %rax"),  # anchor load
            (0x4, "movq", "(%rsi), %rbx"),  # independent load before first use
            (0x8, "addq", "$1, %rcx"),
            (0xc, "addq", "%rax, %rdx"),    # first use of anchor output -> barrier
            (0x10, "movq", "(%r8), %r9"),   # after barrier, not counted for anchor
        ]

        mlp = analyze._compute_mlp(
            instrs, decode_width=5, arch=arch, dependency="io", enable_loop=False
        )

        assert mlp == 5.0 / 3.0

    def test_io_stops_on_use_of_any_prior_issued_load(self):
        """io mode stops when an instruction uses any issued load output."""
        arch = analyze.X86Arch()
        instrs = [
            (0x0, "movq", "(%rdi), %rax"),  # load L1
            (0x4, "movq", "(%rsi), %rbx"),  # load L2
            (0x8, "addq", "%rbx, %rcx"),    # uses L2 output -> barrier
            (0xc, "movq", "(%r8), %r9"),    # should not be counted for L1
            (0x10, "addq", "%rax, %rdx"),   # later use of L1 output
        ]

        mlp = analyze._compute_mlp(
            instrs, decode_width=5, arch=arch, dependency="io", enable_loop=False
        )

        assert mlp == 4.0 / 3.0


# ---------------------------------------------------------------------------
# Integration tests — AMD64 (x86-64)
# ---------------------------------------------------------------------------

class TestAMD64:
    """Integration tests using a compiled x86-64 ELF object file."""

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_results_nonempty(self, x86_obj):
        """analyze() produces at least one result for an x86-64 binary."""
        results = list(analyze.analyze(x86_obj))
        assert results, "Expected at least one result from analyze()"

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_dependency_modes(self, x86_obj):
        """Analyze runs successfully with different dependency modes."""
        res_none = list(analyze.analyze(x86_obj, dependency="none"))
        res_io = list(analyze.analyze(x86_obj, dependency="io"))
        res_ooo = list(analyze.analyze(x86_obj, dependency="ooo"))
        assert res_none
        assert res_io
        assert res_ooo

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_ipc_positive(self, x86_obj):
        """Retired instructions and elapsed cycles for x86-64 are strictly positive."""
        results = list(analyze.analyze(x86_obj))
        assert results
        for start, end, retired, load_instrs, cycles, mlp in results:
            assert retired > 0, f"retired should be positive, got {retired} for region 0x{start:x}–0x{end:x}"
            assert cycles > 0, f"cycles should be positive, got {cycles} for region 0x{start:x}–0x{end:x}"
            assert mlp >= 0, f"mlp should be non-negative, got {mlp} for region 0x{start:x}–0x{end:x}"

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_loop_detected(self, x86_obj):
        """The backward branch in sum() is detected as a loop (start < end)."""
        results = list(analyze.analyze(x86_obj))
        loops = [(s, e, retired, cycles) for s, e, retired, _li, cycles, _mlp in results if s < e]
        assert loops, (
            "Expected at least one loop region (start < end). "
            f"All results: {results}"
        )

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_addresses_are_nonneg_ints(self, x86_obj):
        """Start and end addresses are non-negative integers."""
        results = list(analyze.analyze(x86_obj))
        for start, end, _retired, _li, _cycles, _mlp in results:
            assert isinstance(start, int) and start >= 0
            assert isinstance(end, int) and end >= 0


# ---------------------------------------------------------------------------
# Integration tests — AArch64
# ---------------------------------------------------------------------------

class TestAArch64:
    """Integration tests using a compiled AArch64 ELF object file."""

    @_NEED_MCA
    @_NEED_AARCH64
    def test_results_nonempty(self, aarch64_obj):
        """analyze() produces at least one result for an AArch64 binary."""
        results = list(analyze.analyze(aarch64_obj))
        assert results, "Expected at least one result from analyze()"

    @_NEED_MCA
    @_NEED_AARCH64
    def test_ipc_positive(self, aarch64_obj):
        """Retired instructions and elapsed cycles for AArch64 are strictly positive."""
        results = list(analyze.analyze(aarch64_obj))
        assert results
        for start, end, retired, load_instrs, cycles, mlp in results:
            assert retired > 0, f"retired should be positive, got {retired} for region 0x{start:x}–0x{end:x}"
            assert cycles > 0, f"cycles should be positive, got {cycles} for region 0x{start:x}–0x{end:x}"
            assert mlp >= 0, f"mlp should be non-negative, got {mlp} for region 0x{start:x}–0x{end:x}"

    @_NEED_MCA
    @_NEED_AARCH64
    def test_loop_detected(self, aarch64_obj):
        """The backward branch in sum() is detected as a loop (start < end)."""
        results = list(analyze.analyze(aarch64_obj))
        loops = [(s, e, retired, cycles) for s, e, retired, _li, cycles, _mlp in results if s < e]
        assert loops, (
            "Expected at least one loop region (start < end). "
            f"All results: {results}"
        )

    @_NEED_MCA
    @_NEED_AARCH64
    def test_addresses_are_nonneg_ints(self, aarch64_obj):
        """Start and end addresses are non-negative integers."""
        results = list(analyze.analyze(aarch64_obj))
        for start, end, _retired, _li, _cycles, _mlp in results:
            assert isinstance(start, int) and start >= 0
            assert isinstance(end, int) and end >= 0


# ---------------------------------------------------------------------------
# Unit tests — _is_load_instruction
# ---------------------------------------------------------------------------

class TestIsLoadInstruction:
    """Unit tests for arch.is_load_instruction() on each architecture class."""

    # ------------------------------------------------------------------
    # x86 AT&T syntax
    # ------------------------------------------------------------------

    def test_x86_mov_from_memory_is_load(self):
        """mov (%edi), %eax reads from memory — load."""
        assert analyze.X86Arch().is_load_instruction("mov", "(%edi),%eax")

    def test_x86_add_from_memory_is_load(self):
        """add (%rdi), %rax reads from memory — load."""
        assert analyze.X86Arch().is_load_instruction("add", "(%rdi),%rax")

    def test_x86_mov_to_memory_not_load(self):
        """mov %eax, (%edi) writes to memory — not a load (source is a reg)."""
        assert not analyze.X86Arch().is_load_instruction("mov", "%eax,(%edi)")

    def test_x86_reg_to_reg_not_load(self):
        """add %eax, %edx is register-to-register — not a load."""
        assert not analyze.X86Arch().is_load_instruction("add", "%eax,%edx")

    def test_x86_lea_not_a_load(self):
        """lea computes an address without reading memory."""
        assert not analyze.X86Arch().is_load_instruction("lea", "0x10(%rip),%rax")

    def test_x86_leal_not_a_load(self):
        """leal (32-bit variant) also computes address without reading memory."""
        assert not analyze.X86Arch().is_load_instruction("leal", "(%rdi,%rsi),%ecx")

    def test_x86_leaq_not_a_load(self):
        """leaq (64-bit variant) also computes address without reading memory."""
        assert not analyze.X86Arch().is_load_instruction("leaq", "0x10(%rip),%rax")

    def test_x86_offset_memory_source_is_load(self):
        """mov 8(%rsp), %rax — offset memory source."""
        assert analyze.X86Arch().is_load_instruction("mov", "8(%rsp),%rax")

    def test_x86_no_operands_not_load(self):
        """Instruction with no operands is not a load."""
        assert not analyze.X86Arch().is_load_instruction("ret", "")

    # ------------------------------------------------------------------
    # AArch64
    # ------------------------------------------------------------------

    def test_aarch64_ldr_is_load(self):
        assert analyze.AArch64Arch().is_load_instruction("ldr", "x0, [x1]")

    def test_aarch64_ldp_is_load(self):
        assert analyze.AArch64Arch().is_load_instruction("ldp", "x0, x1, [sp]")

    def test_aarch64_str_not_load(self):
        assert not analyze.AArch64Arch().is_load_instruction("str", "x0, [x1]")

    def test_aarch64_add_not_load(self):
        assert not analyze.AArch64Arch().is_load_instruction("add", "x0, x1, x2")

    # ------------------------------------------------------------------
    # ARM (32-bit)
    # ------------------------------------------------------------------

    def test_arm_ldr_is_load(self):
        assert analyze.ARMArch().is_load_instruction("ldr", "r0, [r1]")

    def test_arm_pop_is_load(self):
        assert analyze.ARMArch().is_load_instruction("pop", "{r0, r1}")

    def test_arm_str_not_load(self):
        assert not analyze.ARMArch().is_load_instruction("str", "r0, [r1]")

    # ------------------------------------------------------------------
    # RISC-V
    # ------------------------------------------------------------------

    def test_riscv_lw_is_load(self):
        assert analyze.RISCVArch().is_load_instruction("lw", "a0, 0(a1)")

    def test_riscv_ld_is_load(self):
        assert analyze.RISCVArch().is_load_instruction("ld", "a0, 8(sp)")

    def test_riscv_lb_is_load(self):
        assert analyze.RISCVArch().is_load_instruction("lb", "a0, 0(a1)")

    def test_riscv_sw_not_load(self):
        assert not analyze.RISCVArch().is_load_instruction("sw", "a0, 0(a1)")

    def test_riscv_add_not_load(self):
        assert not analyze.RISCVArch().is_load_instruction("add", "a0, a1, a2")


# ---------------------------------------------------------------------------
# Unit tests — _run_mca plumbing (monkeypatched)
# ---------------------------------------------------------------------------

class TestRunMcaPlumbing:
    """Verify that _run_mca uses the right formatter and mca flags."""

    def test_call_latency_flag_is_passed(self, monkeypatch):
        """_run_mca must pass ``--call-latency=0`` to llvm-mca.

        This flag suppresses artificially inflated cycle counts that llvm-mca
        would otherwise attribute to CALL instructions.  It is required for
        accurate IPC estimates on any block that contains calls.
        """
        captured_cmd = {}
        monkeypatch.setattr(analyze, "_LLVM_MCA", "llvm-mca")

        class FakeProc:
            returncode = 0
            stdout = "Instructions: 100\nTotal Cycles: 100\n"
            stderr = ""

        def fake_run(cmd, **kw):
            captured_cmd["cmd"] = cmd
            return FakeProc()

        monkeypatch.setattr(subprocess, "run", fake_run)

        analyze._run_mca([(0x0, "nop", "")], arch=analyze.X86Arch())
        assert "--call-latency=0" in captured_cmd.get("cmd", []), (
            "--call-latency=0 must be passed to llvm-mca"
        )


class TestDumper:
    """Tests for the Dumper class."""

    _INSTRS = [
        (0x10, "nop", ""),
        (0x11, "ret", ""),
    ]
    _ASM = "  nop\n  ret\n.Lmca_end:\n"
    _ARCH = analyze.X86Arch()

    def test_creates_dump_directory(self, tmp_path):
        """Dumper creates the dump directory when it does not exist."""
        dump_dir = tmp_path / "dump"
        assert not dump_dir.exists()
        dumper = analyze.Dumper(dump_dir=str(dump_dir))
        dumper.dump(self._INSTRS, self._ASM, self._ARCH)
        assert dump_dir.is_dir()

    def test_writes_file_with_correct_name(self, tmp_path):
        """Dumper writes {start}_{end}.{arch}.txt using hex addresses."""
        dump_dir = tmp_path / "dump"
        dumper = analyze.Dumper(dump_dir=str(dump_dir))
        dumper.dump(self._INSTRS, self._ASM, self._ARCH)
        # start=0x10, end=0x11, arch.name="x86"
        expected_name = "10_11.x86.txt"
        assert (dump_dir / expected_name).is_file()

    def test_written_file_content_matches_asm(self, tmp_path):
        """The content of the dump file equals the provided assembly."""
        dump_dir = tmp_path / "dump"
        dumper = analyze.Dumper(dump_dir=str(dump_dir))
        dumper.dump(self._INSTRS, self._ASM, self._ARCH)
        written = (dump_dir / "10_11.x86.txt").read_text(encoding="utf-8")
        assert written == self._ASM

    def test_no_file_written_for_empty_instrs(self, tmp_path):
        """Dumper does not create any file when instrs is empty."""
        dump_dir = tmp_path / "dump"
        dumper = analyze.Dumper(dump_dir=str(dump_dir))
        dumper.dump([], self._ASM, self._ARCH)
        # The dump directory should not have been created at all.
        assert not dump_dir.exists()
