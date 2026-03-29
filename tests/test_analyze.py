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
# loop-detection and load_proportion can be verified.
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
    for ver in range(20, 10, -1):
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
    """Unit tests for analyze._get_branch_target.

    The tests are grouped to document three distinct categories:

    (A) Correct direct-branch cases — the regex returns the right target.
    (B) Harmless false-positive candidates — earlier hex-like register names
        are matched by the plain-hex fallback but the last-candidate heuristic
        still returns the correct address.
    (C) Previously broken indirect-branch false positives — RISC-V ``jr``/
        ``jalr`` with a bare register operand whose name is all-hex (``a0``–
        ``a7``) were misidentified as direct branches to an address.  This is
        now fixed by the ``mnemonic``/``arch`` parameters.
    """

    # ------------------------------------------------------------------
    # (A) Correct direct-branch parsing
    # ------------------------------------------------------------------

    # x86-64 style (AT&T syntax, plain hex, possibly short)
    def test_x86_plain_hex_3digits(self):
        assert analyze._get_branch_target("4016") == 0x4016

    def test_x86_plain_hex_2digits(self):
        """Two-digit hex targets (common in object files) must be parsed."""
        assert analyze._get_branch_target("20") == 0x20

    def test_x86_plain_hex_1digit(self):
        """Single-digit hex targets must be parsed."""
        assert analyze._get_branch_target("8") == 0x8

    # AArch64 style
    def test_aarch64_bne_2digit_target(self):
        """b.ne with a 2-digit hex target (object-file address)."""
        assert analyze._get_branch_target("18", "b.ne", "aarch64") == 0x18

    def test_aarch64_ble_2digit_target(self):
        assert analyze._get_branch_target("2c", "b.le", "aarch64") == 0x2c

    def test_aarch64_cbz_register_plus_target(self):
        """cbz xN, addr — target is the last operand."""
        assert analyze._get_branch_target("x0, 2dc", "cbz", "aarch64") == 0x2dc

    def test_aarch64_tbz_register_imm_target(self):
        """tbz xN, #imm, addr — the #-prefixed imm must NOT be taken as target."""
        assert analyze._get_branch_target("x0, #0, 2dc", "tbz", "aarch64") == 0x2dc

    def test_aarch64_0x_prefix(self):
        assert analyze._get_branch_target("0x400808", "b", "aarch64") == 0x400808

    # ------------------------------------------------------------------
    # (B) Harmless false-positive candidates
    # These registers (a0, a5…) are all-hex so they appear in the
    # candidate list, but the LAST candidate — the real target address —
    # is what gets returned.
    # ------------------------------------------------------------------

    def test_riscv_bne_fp_candidates_last_is_target(self):
        """bne a4,a5,314 — a4 and a5 are false-positive candidates,
        but the last candidate (0x314) is the correct branch target."""
        assert analyze._get_branch_target("a4,a5,314", "bne", "riscv") == 0x314

    def test_riscv_beqz_fp_candidate_last_is_target(self):
        """beqz a0,100 — a0 is a false-positive candidate,
        but the last candidate (0x100) is the correct branch target."""
        assert analyze._get_branch_target("a0,100", "beqz", "riscv") == 0x100

    # ------------------------------------------------------------------
    # (C) Indirect-branch false positives — must return None
    # jr/jalr with a bare register operand (no comma) is an indirect
    # jump; a0–a5 are all-hex so they would be mistaken for addresses
    # without the mnemonic/arch guard.
    # ------------------------------------------------------------------

    def test_riscv_jr_indirect_a0(self):
        """jr a0 is an indirect jump; must return None, not 0xa0."""
        assert analyze._get_branch_target("a0", "jr", "riscv") is None

    def test_riscv_jr_indirect_a5(self):
        """jr a5 is an indirect jump; must return None, not 0xa5."""
        assert analyze._get_branch_target("a5", "jr", "riscv") is None

    def test_riscv_jalr_indirect_a1(self):
        """jalr a1 (single-register operand) is indirect; must return None."""
        assert analyze._get_branch_target("a1", "jalr", "riscv") is None

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
        assert analyze._get_branch_target("ra, a0, 0", "jalr", "riscv") == 0x0

    def test_riscv_jr_ra_returns_none_due_to_nonhex_char(self):
        """jr ra — 'ra' contains a non-hex character ('r'), so the plain-hex
        regex finds no match and returns None independently of the mnemonic
        guard.  Verified here without arch/mnemonic to show the regex alone
        handles non-hex register names correctly."""
        assert analyze._get_branch_target("ra") is None

    # Indirect branches — architecture-agnostic check (x86 AT&T prefix)
    def test_indirect_x86_star_prefix(self):
        """Indirect branch in AT&T syntax (*%reg) returns None."""
        assert analyze._get_branch_target("*%rax") is None


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
        loops = analyze._find_loops(instrs, "x86")
        assert (0x20, 0x27) in loops

    def test_x86_no_loop_forward_branch(self):
        """A forward branch does not create a loop."""
        instrs = [
            (0x10, "test", "%edi,%edi"),
            (0x12, "jle", "30"),  # forward branch → no loop
            (0x14, "xor", "%eax,%eax"),
            (0x30, "ret", ""),
        ]
        loops = analyze._find_loops(instrs, "x86")
        assert loops == []

    def test_aarch64_simple_loop_short_addresses(self):
        """AArch64 object-file addresses (< 3 hex digits) are detected as loops."""
        instrs = [
            (0x18, "add", "w0, w0, w1"),
            (0x1c, "add", "w1, w1, #0x1"),
            (0x20, "cmp", "w2, w1"),
            (0x24, "b.ne", "18"),  # backward branch → loop 0x18..0x24
        ]
        loops = analyze._find_loops(instrs, "aarch64")
        assert (0x18, 0x24) in loops

    def test_aarch64_no_loop_forward_branch(self):
        """A forward AArch64 branch does not create a loop."""
        instrs = [
            (0x0, "mov", "w2, w0"),
            (0x4, "cmp", "w0, #0x0"),
            (0x8, "b.le", "2c"),  # forward branch → no loop
            (0x2c, "ret", ""),
        ]
        loops = analyze._find_loops(instrs, "aarch64")
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
        loops = analyze._find_loops(instrs, "riscv")
        assert loops == [], (
            "jr a0 is an indirect branch and must not create a loop; "
            f"got loops={loops}"
        )


class TestFormatAsm:
    """Unit tests for analyze._format_asm indirect-branch handling."""

    def test_riscv_jr_kept_as_indirect(self):
        """_format_asm must emit 'jr a0' verbatim, not 'jr .Lmca_end'.

        Previously, _get_branch_target("a0") returned 0xa0 instead of None,
        causing _format_asm to rewrite the indirect branch as
        ``jr .Lmca_end`` — invalid RISC-V assembly that made llvm-mca fail
        and silently drop the region.
        """
        instrs = [
            (0x0,  "add", "a0, a0, a1"),
            (0x4,  "jr",  "a0"),
        ]
        asm = analyze._format_asm(instrs, "riscv")
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
        asm = analyze._format_asm(instrs, "riscv")
        assert "jalr a1" in asm, (
            f"Expected 'jalr a1' to be preserved. Got:\n{asm}"
        )


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
    def test_ipc_positive(self, x86_obj):
        """IPC values for x86-64 are strictly positive."""
        results = list(analyze.analyze(x86_obj))
        assert results
        for start, end, ipc, _lp in results:
            assert ipc > 0, f"IPC should be positive, got {ipc} for region 0x{start:x}–0x{end:x}"

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_load_proportion_in_range(self, x86_obj):
        """load_proportion values are in [0, 1]."""
        results = list(analyze.analyze(x86_obj))
        for start, end, _ipc, load_proportion in results:
            assert 0.0 <= load_proportion <= 1.0, (
                f"load_proportion {load_proportion} out of range for "
                f"region 0x{start:x}–0x{end:x}"
            )

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_loop_detected(self, x86_obj):
        """The backward branch in sum() is detected as a loop (start < end)."""
        results = list(analyze.analyze(x86_obj))
        loops = [(s, e, ipc, lp) for s, e, ipc, lp in results if s < e]
        assert loops, (
            "Expected at least one loop region (start < end). "
            f"All results: {results}"
        )

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_addresses_are_nonneg_ints(self, x86_obj):
        """Start and end addresses are non-negative integers."""
        results = list(analyze.analyze(x86_obj))
        for start, end, _ipc, _lp in results:
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
        """IPC values for AArch64 are strictly positive."""
        results = list(analyze.analyze(aarch64_obj))
        assert results
        for start, end, ipc, _lp in results:
            assert ipc > 0, f"IPC should be positive, got {ipc} for region 0x{start:x}–0x{end:x}"

    @_NEED_MCA
    @_NEED_AARCH64
    def test_load_proportion_in_range(self, aarch64_obj):
        """load_proportion values are in [0, 1]."""
        results = list(analyze.analyze(aarch64_obj))
        for start, end, _ipc, load_proportion in results:
            assert 0.0 <= load_proportion <= 1.0, (
                f"load_proportion {load_proportion} out of range for "
                f"region 0x{start:x}–0x{end:x}"
            )

    @_NEED_MCA
    @_NEED_AARCH64
    def test_loop_detected(self, aarch64_obj):
        """The backward branch in sum() is detected as a loop (start < end)."""
        results = list(analyze.analyze(aarch64_obj))
        loops = [(s, e, ipc, lp) for s, e, ipc, lp in results if s < e]
        assert loops, (
            "Expected at least one loop region (start < end). "
            f"All results: {results}"
        )

    @_NEED_MCA
    @_NEED_AARCH64
    def test_addresses_are_nonneg_ints(self, aarch64_obj):
        """Start and end addresses are non-negative integers."""
        results = list(analyze.analyze(aarch64_obj))
        for start, end, _ipc, _lp in results:
            assert isinstance(start, int) and start >= 0
            assert isinstance(end, int) and end >= 0


# ---------------------------------------------------------------------------
# Unit tests — _analyze_function_ipc
# ---------------------------------------------------------------------------

class TestAnalyzeFunctionIpc:
    """Unit tests for analyze._analyze_function_ipc."""

    def test_empty_instrs_returns_none(self):
        """An empty instruction list must return None."""
        assert analyze._analyze_function_ipc([]) is None

    def test_loop_ipc_returned(self, monkeypatch):
        """IPC returned equals the llvm-mca IPC of the single loop."""
        # Backward branch at 0x2 → 0x0 forms a loop.
        instrs = [
            (0x0, "add", "%eax,%edx"),
            (0x2, "jne", "0"),
        ]
        monkeypatch.setattr(analyze, "_run_mca",
                            lambda *a, **kw: (2.0, 0.25))
        result = analyze._analyze_function_ipc(instrs)
        assert result is not None
        start, end, ipc, lp = result
        assert start == 0x0
        assert end == 0x2
        assert abs(ipc - 2.0) < 1e-9
        assert abs(lp - 0.25) < 1e-9

    def test_max_loop_ipc_selected(self, monkeypatch):
        """The loop with the highest IPC (hottest loop) is chosen."""
        # Two non-overlapping loops + one basic block outside loops.
        # Loop 1: [0x0, 0x2], loop 2: [0x10, 0x12], basic block: [0x20].
        instrs = [
            (0x0,  "add", "%eax,%edx"),
            (0x2,  "jne", "0"),            # loop 1: backward branch to 0x0
            (0x10, "xor", "%eax,%eax"),
            (0x12, "jne", "10"),           # loop 2: backward branch to 0x10
            (0x20, "ret", ""),             # basic block outside loops
        ]

        def fake_run_mca(region, *a, **kw):
            if region[0][0] == 0x0:
                return (4.0, 0.10)   # loop 1 (high IPC — hottest)
            if region[0][0] == 0x10:
                return (1.0, 0.50)   # loop 2 (low IPC)
            return (2.0, 0.00)       # basic block — must be ignored

        monkeypatch.setattr(analyze, "_run_mca", fake_run_mca)
        result = analyze._analyze_function_ipc(instrs)
        assert result is not None
        start, end, ipc, lp = result
        # Address range spans all instructions
        assert start == 0x0
        assert end == 0x20
        # IPC_f = max(4.0, 1.0) = 4.0; load_proportion from hottest loop
        assert abs(ipc - 4.0) < 1e-9
        assert abs(lp - 0.10) < 1e-9

    def test_no_loops_falls_back_to_blocks(self, monkeypatch):
        """A function with no loops falls back to the max-IPC basic block."""
        instrs = [
            (0x0,  "add", "%eax,%edx"),
            (0x2,  "jmp", "10"),           # forward branch — no loop
            (0x10, "xor", "%eax,%eax"),
            (0x12, "ret", ""),
        ]

        def fake_run_mca(region, *a, **kw):
            if region[0][0] == 0x0:
                return (3.0, 0.20)   # first BB
            return (1.5, 0.40)       # second BB

        monkeypatch.setattr(analyze, "_run_mca", fake_run_mca)
        result = analyze._analyze_function_ipc(instrs)
        assert result is not None
        start, end, ipc, lp = result
        assert start == 0x0
        assert end == 0x12
        # Falls back to max over blocks: max(3.0, 1.5) = 3.0
        assert abs(ipc - 3.0) < 1e-9
        assert abs(lp - 0.20) < 1e-9

    def test_no_mca_results_returns_none(self, monkeypatch):
        """If llvm-mca returns None for every region, return None."""
        instrs = [
            (0x0, "add", "%eax,%edx"),
            (0x2, "ret", ""),
        ]
        monkeypatch.setattr(analyze, "_run_mca", lambda *a, **kw: None)
        assert analyze._analyze_function_ipc(instrs) is None


# ---------------------------------------------------------------------------
# Integration tests — function mode (x86-64)
# ---------------------------------------------------------------------------

class TestAMD64FunctionMode:
    """Integration tests for analyze() in 'functions' mode (x86-64)."""

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_results_nonempty(self, x86_obj):
        """analyze() in 'functions' mode produces at least one result."""
        results = list(analyze.analyze(x86_obj, mode="functions"))
        assert results, "Expected at least one result from analyze(mode='functions')"

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_one_row_per_function(self, x86_obj):
        """Function mode yields at most one row for the compiled source file."""
        results = list(analyze.analyze(x86_obj, mode="functions"))
        # The test binary has a single function (sum); allow for compiler-injected
        # helpers but there should be far fewer rows than in block mode.
        block_results = list(analyze.analyze(x86_obj))
        assert len(results) <= len(block_results), (
            "Function mode should yield no more rows than block mode"
        )

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_ipc_positive(self, x86_obj):
        """IPC values are strictly positive."""
        results = list(analyze.analyze(x86_obj, mode="functions"))
        assert results
        for start, end, ipc, _lp in results:
            assert ipc > 0, (
                f"IPC should be positive, got {ipc} for function "
                f"0x{start:x}–0x{end:x}"
            )

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_load_proportion_in_range(self, x86_obj):
        """load_proportion values are in [0, 1]."""
        results = list(analyze.analyze(x86_obj, mode="functions"))
        for start, end, _ipc, lp in results:
            assert 0.0 <= lp <= 1.0, (
                f"load_proportion {lp} out of range for function "
                f"0x{start:x}–0x{end:x}"
            )

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_addresses_are_nonneg_ints(self, x86_obj):
        """Start and end addresses are non-negative integers."""
        results = list(analyze.analyze(x86_obj, mode="functions"))
        for start, end, _ipc, _lp in results:
            assert isinstance(start, int) and start >= 0
            assert isinstance(end, int) and end >= 0

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_ipc_le_max_block_ipc(self, x86_obj):
        """Function IPC must be <= global max block IPC."""
        block_results = list(analyze.analyze(x86_obj))
        func_results = list(analyze.analyze(x86_obj, mode="functions"))
        assert func_results
        # IPC_f is the max over loops (or blocks if no loops); either way it is
        # drawn from the same pool of regions as block mode, so it cannot exceed
        # the global maximum block IPC.
        max_block_ipc = max(ipc for _, _, ipc, _ in block_results if ipc > 0)
        for _start, _end, func_ipc, _lp in func_results:
            assert func_ipc <= max_block_ipc + 1e-9, (
                f"Function IPC {func_ipc:.4f} exceeds maximum block IPC "
                f"{max_block_ipc:.4f}; expected IPC_f = max of all loop IPCs "
                f"(or block IPCs when no loops exist)"
            )


# ---------------------------------------------------------------------------
# Unit tests — _is_load_instruction
# ---------------------------------------------------------------------------

class TestIsLoadInstruction:
    """Unit tests for analyze._is_load_instruction."""

    # ------------------------------------------------------------------
    # x86 AT&T syntax
    # ------------------------------------------------------------------

    def test_x86_mov_from_memory_is_load(self):
        """mov (%edi), %eax reads from memory — load."""
        assert analyze._is_load_instruction("mov", "(%edi),%eax", "x86")

    def test_x86_add_from_memory_is_load(self):
        """add (%rdi), %rax reads from memory — load."""
        assert analyze._is_load_instruction("add", "(%rdi),%rax", "x86")

    def test_x86_mov_to_memory_not_load(self):
        """mov %eax, (%edi) writes to memory — not a load (source is a reg)."""
        assert not analyze._is_load_instruction("mov", "%eax,(%edi)", "x86")

    def test_x86_reg_to_reg_not_load(self):
        """add %eax, %edx is register-to-register — not a load."""
        assert not analyze._is_load_instruction("add", "%eax,%edx", "x86")

    def test_x86_lea_not_a_load(self):
        """lea computes an address without reading memory."""
        assert not analyze._is_load_instruction("lea", "0x10(%rip),%rax", "x86")

    def test_x86_leal_not_a_load(self):
        """leal (32-bit variant) also computes address without reading memory."""
        assert not analyze._is_load_instruction("leal", "(%rdi,%rsi),%ecx", "x86")

    def test_x86_leaq_not_a_load(self):
        """leaq (64-bit variant) also computes address without reading memory."""
        assert not analyze._is_load_instruction("leaq", "0x10(%rip),%rax", "x86")

    def test_x86_offset_memory_source_is_load(self):
        """mov 8(%rsp), %rax — offset memory source."""
        assert analyze._is_load_instruction("mov", "8(%rsp),%rax", "x86")

    def test_x86_no_operands_not_load(self):
        """Instruction with no operands is not a load."""
        assert not analyze._is_load_instruction("ret", "", "x86")

    # ------------------------------------------------------------------
    # AArch64
    # ------------------------------------------------------------------

    def test_aarch64_ldr_is_load(self):
        assert analyze._is_load_instruction("ldr", "x0, [x1]", "aarch64")

    def test_aarch64_ldp_is_load(self):
        assert analyze._is_load_instruction("ldp", "x0, x1, [sp]", "aarch64")

    def test_aarch64_str_not_load(self):
        assert not analyze._is_load_instruction("str", "x0, [x1]", "aarch64")

    def test_aarch64_add_not_load(self):
        assert not analyze._is_load_instruction("add", "x0, x1, x2", "aarch64")

    # ------------------------------------------------------------------
    # ARM (32-bit)
    # ------------------------------------------------------------------

    def test_arm_ldr_is_load(self):
        assert analyze._is_load_instruction("ldr", "r0, [r1]", "arm")

    def test_arm_pop_is_load(self):
        assert analyze._is_load_instruction("pop", "{r0, r1}", "arm")

    def test_arm_str_not_load(self):
        assert not analyze._is_load_instruction("str", "r0, [r1]", "arm")

    # ------------------------------------------------------------------
    # RISC-V
    # ------------------------------------------------------------------

    def test_riscv_lw_is_load(self):
        assert analyze._is_load_instruction("lw", "a0, 0(a1)", "riscv")

    def test_riscv_ld_is_load(self):
        assert analyze._is_load_instruction("ld", "a0, 8(sp)", "riscv")

    def test_riscv_lb_is_load(self):
        assert analyze._is_load_instruction("lb", "a0, 0(a1)", "riscv")

    def test_riscv_sw_not_load(self):
        assert not analyze._is_load_instruction("sw", "a0, 0(a1)", "riscv")

    def test_riscv_add_not_load(self):
        assert not analyze._is_load_instruction("add", "a0, a1, a2", "riscv")


# ---------------------------------------------------------------------------
# Unit tests — _format_asm_with_cache_miss
# ---------------------------------------------------------------------------

class TestFormatAsmWithCacheMiss:
    """Unit tests for analyze._format_asm_with_cache_miss."""

    _LOAD_INSTRS = [
        (0x0, "mov", "(%edi),%eax"),
        (0x2, "add", "%eax,%edx"),
        (0x4, "ret", ""),
    ]

    def test_no_cache_miss_no_latency_directives(self):
        """With cache_miss=0 no LLVM-MCA-LATENCY directives must appear."""
        asm = analyze._format_asm_with_cache_miss(
            self._LOAD_INSTRS, "x86", cache_miss=0.0, cache_latency=100)
        assert "LLVM-MCA-LATENCY" not in asm

    def test_full_cache_miss_all_loads_get_latency(self):
        """With cache_miss=1.0 every load must be wrapped with directives."""
        asm = analyze._format_asm_with_cache_miss(
            self._LOAD_INSTRS, "x86", cache_miss=1.0, cache_latency=100)
        assert "# LLVM-MCA-LATENCY 100" in asm
        assert "# LLVM-MCA-LATENCY\n" in asm or asm.endswith("# LLVM-MCA-LATENCY")

    def test_non_load_not_wrapped(self):
        """Non-load instructions must never have an opening latency directive."""
        asm = analyze._format_asm_with_cache_miss(
            self._LOAD_INSTRS, "x86", cache_miss=1.0, cache_latency=100)
        lines = asm.splitlines()
        _open_re = re.compile(r"#\s+LLVM-MCA-LATENCY\s+\d+")
        for i, line in enumerate(lines):
            if "add\t%eax,%edx" in line or "\tadd %eax,%edx" in line:
                if i > 0:
                    assert not _open_re.search(lines[i - 1]), (
                        f"Non-load 'add' must not be preceded by an opening "
                        f"LLVM-MCA-LATENCY directive. Line before: {lines[i-1]!r}"
                    )

    def test_code_repeated_100_times(self):
        """The instruction block must appear 100 times in the output."""
        instrs = [(0x0, "add", "%eax,%edx"), (0x2, "ret", "")]
        asm = analyze._format_asm_with_cache_miss(
            instrs, "x86", cache_miss=0.0, cache_latency=0)
        # The non-branch instruction appears once per iteration.
        assert asm.count("add %eax,%edx") == 100

    def test_labels_are_unique_per_iteration(self):
        """Backward-branch labels must be unique per repetition."""
        instrs = [
            (0x0, "add", "%eax,%edx"),
            (0x2, "jne", "0"),
        ]
        asm = analyze._format_asm_with_cache_miss(
            instrs, "x86", cache_miss=0.0, cache_latency=0)
        # Per-iteration label suffix: _r0, _r1, …, _r99
        assert ".Lmca_0_r0:" in asm
        assert ".Lmca_0_r99:" in asm
        # No bare .Lmca_0: label (which would be duplicated)
        assert not re.search(r"^\.Lmca_0:$", asm, re.MULTILINE)

    def test_lmca_end_label_present(self):
        """The closing .Lmca_end: label must be present exactly once."""
        asm = analyze._format_asm_with_cache_miss(
            self._LOAD_INSTRS, "x86", cache_miss=0.0, cache_latency=0)
        assert asm.count(".Lmca_end:") == 1

    def test_deterministic_exact_miss_count(self):
        """Cache miss count must equal round(cache_miss * loads_per_block) per repetition."""
        # Block has one load instruction (mov), repeated 100 times.
        # cache_miss=0.3 → a=round(0.3*1)=0 → no misses expected.
        # Use a block with 10 loads so we can check various rates precisely.
        instrs = [(i * 2, "mov", f"({i})(%edi),%eax") for i in range(10)]
        for rate, expected_per_rep in [(0.0, 0), (0.3, 3), (0.5, 5), (1.0, 10)]:
            asm = analyze._format_asm_with_cache_miss(
                instrs, "x86", cache_miss=rate, cache_latency=100)
            # Count opening latency directives (one per cache-miss event).
            total_misses = asm.count("# LLVM-MCA-LATENCY 100")
            assert total_misses == expected_per_rep * 100, (
                f"cache_miss={rate}: expected {expected_per_rep * 100} misses "
                f"total, got {total_misses}"
            )

    def test_deterministic_miss_positions_uniform_early_bias(self):
        """Miss positions must follow floor(m*b/a) formula within each repetition.

        For b=10, a=3 the expected miss positions (0-indexed among loads) are
        0, 3, 6 — i.e. the 1st, 4th, and 7th load in each repetition.
        """
        # Build 10 loads with distinct operands so we can identify them.
        instrs = [(i * 2, "mov", f"({i})(%edi),%eax") for i in range(10)]
        asm = analyze._format_asm_with_cache_miss(
            instrs, "x86", cache_miss=0.3, cache_latency=999)

        lines = asm.splitlines()
        miss_indices = []  # 0-indexed load positions that are misses, within one rep
        load_idx = 0
        for i, line in enumerate(lines):
            if "mov" in line and "(%edi)" in line:
                # Check if the previous non-empty line is the opening directive.
                prev = lines[i - 1] if i > 0 else ""
                if "# LLVM-MCA-LATENCY 999" in prev:
                    miss_indices.append(load_idx % 10)
                load_idx += 1

        # Each repetition should contribute misses at positions 0, 3, 6.
        expected = {0, 3, 6}
        assert set(miss_indices) == expected, (
            f"Expected miss positions {expected}, got {set(miss_indices)}"
        )

    def test_deterministic_no_randomness(self):
        """Two calls with the same arguments must produce identical output."""
        instrs = [(i * 2, "mov", f"({i})(%edi),%eax") for i in range(5)]
        asm1 = analyze._format_asm_with_cache_miss(
            instrs, "x86", cache_miss=0.4, cache_latency=200)
        asm2 = analyze._format_asm_with_cache_miss(
            instrs, "x86", cache_miss=0.4, cache_latency=200)
        assert asm1 == asm2, "Output must be deterministic across calls"


# ---------------------------------------------------------------------------
# Unit tests — _run_mca cache-miss plumbing (monkeypatched)
# ---------------------------------------------------------------------------

class TestRunMcaCacheMissPlumbing:
    """Verify that _run_mca uses the right formatter and mca flags."""

    def test_cache_miss_zero_uses_format_asm(self, monkeypatch):
        """With cache_miss=0, _format_asm (not _format_asm_with_cache_miss) is used."""
        called = {}

        def fake_format_asm(instrs, arch):
            called["format_asm"] = True
            return "\tnop\n.Lmca_end:\n"

        def fake_format_asm_with_cache_miss(instrs, arch, cache_miss, cache_latency):
            called["format_asm_with_cache_miss"] = True
            return "\tnop\n.Lmca_end:\n"

        monkeypatch.setattr(analyze, "_format_asm", fake_format_asm)
        monkeypatch.setattr(analyze, "_format_asm_with_cache_miss",
                            fake_format_asm_with_cache_miss)
        monkeypatch.setattr(analyze, "_LLVM_MCA", "llvm-mca")

        class FakeProc:
            returncode = 0
            stdout = "IPC: 1.00\n"
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeProc())

        analyze._run_mca([(0x0, "nop", "")], cache_miss=0.0)
        assert called.get("format_asm"), "Expected _format_asm to be called"
        assert not called.get("format_asm_with_cache_miss"), (
            "Did not expect _format_asm_with_cache_miss to be called"
        )

    def test_cache_miss_nonzero_uses_format_asm_with_cache_miss(self, monkeypatch):
        """With cache_miss>0, _format_asm_with_cache_miss is used and
        -iterations=0 is added to the llvm-mca command."""
        captured_cmd = {}

        def fake_format_asm_with_cache_miss(instrs, arch, cache_miss, cache_latency):
            return "\tnop\n.Lmca_end:\n"

        monkeypatch.setattr(analyze, "_format_asm_with_cache_miss",
                            fake_format_asm_with_cache_miss)
        monkeypatch.setattr(analyze, "_LLVM_MCA", "llvm-mca")

        class FakeProc:
            returncode = 0
            stdout = "IPC: 1.00\n"
            stderr = ""

        def fake_run(cmd, **kw):
            captured_cmd["cmd"] = cmd
            return FakeProc()

        monkeypatch.setattr(subprocess, "run", fake_run)

        analyze._run_mca([(0x0, "nop", "")], cache_miss=0.5, cache_latency=100)
        assert "-iterations=0" in captured_cmd.get("cmd", []), (
            "Expected -iterations=0 in llvm-mca command when cache_miss > 0"
        )
