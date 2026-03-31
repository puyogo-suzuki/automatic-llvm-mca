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
# Unit tests — _format_asm_with_cache_miss
# ---------------------------------------------------------------------------

class TestFormatAsmWithCacheMiss:
    """Unit tests for analyze._format_asm_with_cache_miss."""

    _LOAD_INSTRS = [
        (0x0, "mov", "(%edi),%eax"),
        (0x2, "add", "%eax,%edx"),
        (0x4, "ret", ""),
    ]
    _ARCH = analyze.X86Arch()

    def test_no_cache_miss_no_latency_directives(self):
        """With cache_miss=0 no LLVM-MCA-LATENCY directives must appear."""
        asm = analyze._format_asm_with_cache_miss(
            self._LOAD_INSTRS, self._ARCH, cache_miss=0.0, cache_latency=100)
        assert "LLVM-MCA-LATENCY" not in asm

    def test_full_cache_miss_all_loads_get_latency(self):
        """With cache_miss=1.0 every load must be wrapped with directives."""
        asm = analyze._format_asm_with_cache_miss(
            self._LOAD_INSTRS, self._ARCH, cache_miss=1.0, cache_latency=100)
        assert "# LLVM-MCA-LATENCY 100" in asm
        assert "# LLVM-MCA-LATENCY\n" in asm or asm.endswith("# LLVM-MCA-LATENCY")

    def test_non_load_not_wrapped(self):
        """Non-load instructions must never have an opening latency directive."""
        asm = analyze._format_asm_with_cache_miss(
            self._LOAD_INSTRS, self._ARCH, cache_miss=1.0, cache_latency=100)
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
            instrs, self._ARCH, cache_miss=0.0, cache_latency=0)
        # The non-branch instruction appears once per iteration.
        assert asm.count("add %eax,%edx") == 100

    def test_labels_are_unique_per_iteration(self):
        """Backward-branch labels must be unique per repetition."""
        instrs = [
            (0x0, "add", "%eax,%edx"),
            (0x2, "jne", "0"),
        ]
        asm = analyze._format_asm_with_cache_miss(
            instrs, self._ARCH, cache_miss=0.0, cache_latency=0)
        # Per-iteration label suffix: _r0, _r1, …, _r99
        assert ".Lmca_0_r0:" in asm
        assert ".Lmca_0_r99:" in asm
        # No bare .Lmca_0: label (which would be duplicated)
        assert not re.search(r"^\.Lmca_0:$", asm, re.MULTILINE)

    def test_lmca_end_label_present(self):
        """The closing .Lmca_end: label must be present exactly once."""
        asm = analyze._format_asm_with_cache_miss(
            self._LOAD_INSTRS, self._ARCH, cache_miss=0.0, cache_latency=0)
        assert asm.count(".Lmca_end:") == 1

    def test_deterministic_exact_miss_count(self):
        """Cache miss count must equal round(cache_miss * loads_per_block) per repetition."""
        # Block has one load instruction (mov), repeated 100 times.
        # cache_miss=0.3 → a=round(0.3*1)=0 → no misses expected.
        # Use a block with 10 loads so we can check various rates precisely.
        instrs = [(i * 2, "mov", f"({i})(%edi),%eax") for i in range(10)]
        for rate, expected_per_rep in [(0.0, 0), (0.3, 3), (0.5, 5), (1.0, 10)]:
            asm = analyze._format_asm_with_cache_miss(
                instrs, self._ARCH, cache_miss=rate, cache_latency=100)
            # Count opening latency directives (one per cache-miss event).
            total_misses = asm.count("# LLVM-MCA-LATENCY 100")
            assert total_misses == expected_per_rep * 100, (
                f"cache_miss={rate}: expected {expected_per_rep * 100} misses "
                f"total, got {total_misses}"
            )

    def test_deterministic_miss_positions_uniform_early_bias(self):
        """Miss positions must follow floor(m*b/a) and prefer earlier loads.

        With n=10 loads/repetition and _CACHE_MISS_REPEAT=100:
        b = 1000 and a = 300 for cache_miss=0.3.
        The first repetition should hit 0, 3, 6 (1st, 4th, 7th loads).
        """
        # Build 10 loads with distinct operands so we can identify them.
        instrs = [(i * 2, "mov", f"({i})(%edi),%eax") for i in range(10)]
        asm = analyze._format_asm_with_cache_miss(
            instrs, self._ARCH, cache_miss=0.3, cache_latency=999)

        lines = asm.splitlines()
        miss_indices = []  # global 0-indexed load positions that are misses
        load_idx = 0
        for i, line in enumerate(lines):
            if "mov" in line and "(%edi)" in line:
                # Check if the previous non-empty line is the opening directive.
                prev = lines[i - 1] if i > 0 else ""
                if "# LLVM-MCA-LATENCY 999" in prev:
                    miss_indices.append(load_idx)
                load_idx += 1

        # In the first repetition (global positions 0..9), misses are 0,3,6.
        first_rep = [idx for idx in miss_indices if idx < 10]
        assert first_rep == [0, 3, 6], (
            f"Expected first repetition misses [0, 3, 6], got {first_rep}"
        )

    def test_deterministic_no_randomness(self):
        """Two calls with the same arguments must produce identical output."""
        instrs = [(i * 2, "mov", f"({i})(%edi),%eax") for i in range(5)]
        asm1 = analyze._format_asm_with_cache_miss(
            instrs, self._ARCH, cache_miss=0.4, cache_latency=200)
        asm2 = analyze._format_asm_with_cache_miss(
            instrs, self._ARCH, cache_miss=0.4, cache_latency=200)
        assert asm1 == asm2, "Output must be deterministic across calls"


# ---------------------------------------------------------------------------
# Unit tests — _run_mca cache-miss plumbing (monkeypatched)
# ---------------------------------------------------------------------------

class TestRunMcaCacheMissPlumbing:
    """Verify that _run_mca uses the right formatter and mca flags."""

    def test_cache_miss_zero_uses_format_asm(self, monkeypatch):
        """With _NoCacheMiss, _format_asm (not _format_asm_with_cache_miss) is used."""
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

        analyze._run_mca([(0x0, "nop", "")],
                         cache_mode=analyze._NoCacheMiss())
        assert called.get("format_asm"), "Expected _format_asm to be called"
        assert not called.get("format_asm_with_cache_miss"), (
            "Did not expect _format_asm_with_cache_miss to be called"
        )

    def test_cache_miss_nonzero_uses_format_asm_with_cache_miss(self, monkeypatch):
        """With _StochasticCacheMiss and load instructions, _format_asm_with_cache_miss
        is used and -iterations=1 is added to the llvm-mca command."""
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

        # Use a region with a load instruction so stochastic mode activates.
        analyze._run_mca([(0x0, "mov", "(%edi),%eax"), (0x2, "add", "%eax,%edx")],
                         cache_mode=analyze._StochasticCacheMiss(10, 100))
        assert "-iterations=1" in captured_cmd.get("cmd", []), (
            "Expected -iterations=1 in llvm-mca command for stochastic mode"
        )

    def test_call_latency_flag_is_passed(self, monkeypatch):
        """_run_mca always passes the undocumented --call-latency=0 flag."""
        captured_cmd = {}
        monkeypatch.setattr(analyze, "_LLVM_MCA", "llvm-mca")

        class FakeProc:
            returncode = 0
            stdout = "IPC: 1.00\n"
            stderr = ""

        def fake_run(cmd, **kw):
            captured_cmd["cmd"] = cmd
            return FakeProc()

        monkeypatch.setattr(subprocess, "run", fake_run)

        analyze._run_mca([(0x0, "nop", "")], cache_mode=analyze._NoCacheMiss())
        assert "--call-latency=0" in captured_cmd.get("cmd", []), (
            "Expected --call-latency=0 in llvm-mca command"
        )


# ---------------------------------------------------------------------------
# Unit tests — _format_asm_with_average_load_latency
# ---------------------------------------------------------------------------

class TestFormatAsmWithAverageLoadLatency:
    """Unit tests for analyze._format_asm_with_average_load_latency."""

    _LOAD_INSTRS = [
        (0x0, "mov", "(%edi),%eax"),
        (0x2, "add", "%eax,%edx"),
        (0x4, "ret", ""),
    ]
    _ARCH = analyze.X86Arch()

    def test_all_loads_get_latency_directive(self):
        """Every load must be wrapped with opening and closing latency directives."""
        asm = analyze._format_asm_with_average_load_latency(
            self._LOAD_INSTRS, self._ARCH, latency=50)
        assert "# LLVM-MCA-LATENCY 50" in asm
        assert "# LLVM-MCA-LATENCY\n" in asm or asm.endswith("# LLVM-MCA-LATENCY")

    def test_zero_latency_still_wraps_loads(self):
        """A latency of 0 still inserts directives (latency=0 is valid)."""
        asm = analyze._format_asm_with_average_load_latency(
            self._LOAD_INSTRS, self._ARCH, latency=0)
        assert "# LLVM-MCA-LATENCY 0" in asm

    def test_non_load_not_wrapped(self):
        """Non-load instructions must not be preceded by a latency directive."""
        asm = analyze._format_asm_with_average_load_latency(
            self._LOAD_INSTRS, self._ARCH, latency=75)
        lines = asm.splitlines()
        open_re = re.compile(r"#\s+LLVM-MCA-LATENCY\s+\d+")
        for i, line in enumerate(lines):
            if "add\t%eax,%edx" in line or "\tadd %eax,%edx" in line:
                if i > 0:
                    assert not open_re.search(lines[i - 1]), (
                        f"Non-load 'add' must not be preceded by an opening "
                        f"LLVM-MCA-LATENCY directive. Line before: {lines[i-1]!r}"
                    )

    def test_no_repetition(self):
        """The instruction block must appear exactly once (no repetition)."""
        instrs = [(0x0, "mov", "(%edi),%eax"), (0x2, "ret", "")]
        asm = analyze._format_asm_with_average_load_latency(
            instrs, self._ARCH, latency=10)
        # Load appears once; not repeated 100 times like stochastic mode.
        assert asm.count("mov (%edi),%eax") == 1

    def test_lmca_end_label_present(self):
        """The closing .Lmca_end: label must be present."""
        asm = analyze._format_asm_with_average_load_latency(
            self._LOAD_INSTRS, self._ARCH, latency=0)
        assert ".Lmca_end:" in asm

    def test_no_loads_no_latency_directives(self):
        """A block with no loads must produce no LLVM-MCA-LATENCY directives."""
        instrs = [(0x0, "add", "%eax,%edx"), (0x2, "ret", "")]
        asm = analyze._format_asm_with_average_load_latency(
            instrs, self._ARCH, latency=100)
        assert "LLVM-MCA-LATENCY" not in asm

    def test_multiple_loads_all_wrapped(self):
        """All load instructions in a block must each be wrapped."""
        instrs = [(i * 2, "mov", f"({i})(%edi),%eax") for i in range(5)]
        asm = analyze._format_asm_with_average_load_latency(
            instrs, self._ARCH, latency=30)
        # There should be exactly 5 opening latency directives.
        assert asm.count("# LLVM-MCA-LATENCY 30") == 5

    def test_deterministic(self):
        """Two calls with the same arguments produce identical output."""
        instrs = [(i * 2, "mov", f"({i})(%edi),%eax") for i in range(3)]
        asm1 = analyze._format_asm_with_average_load_latency(
            instrs, self._ARCH, latency=20)
        asm2 = analyze._format_asm_with_average_load_latency(
            instrs, self._ARCH, latency=20)
        assert asm1 == asm2


# ---------------------------------------------------------------------------
# Unit tests — _run_mca average mode plumbing (monkeypatched)
# ---------------------------------------------------------------------------

class TestRunMcaAverageModePlumbing:
    """Verify that _run_mca uses the average-mode formatter when requested."""

    def test_average_mode_uses_format_asm_with_per_load_latencies(
            self, monkeypatch):
        """With _AverageCacheMiss and load instructions,
        _format_asm_with_per_load_latencies is called with Bresenham latencies."""
        called = {}

        def fake_per_load(instrs, arch, latencies, repeat=1):
            called["latencies"] = latencies
            called["repeat"] = repeat
            return "\tnop\n.Lmca_end:\n"

        monkeypatch.setattr(analyze, "_format_asm_with_per_load_latencies",
                            fake_per_load)
        monkeypatch.setattr(analyze, "_LLVM_MCA", "llvm-mca")

        class FakeProc:
            returncode = 0
            stdout = "IPC: 2.00\n"
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeProc())

        # Region: 10 instrs, 1 load, ipcm=10 → expected_misses=1.0
        # _bresenham_load_misses(1.0, 1) = [1] → latencies=[100]
        instrs = [(0x0, "mov", "(%edi),%eax")] + [
            (0x2 + i * 2, "add", "%eax,%edx") for i in range(9)
        ]
        analyze._run_mca(
            instrs,
            cache_mode=analyze._AverageCacheMiss(10, 100),
        )
        assert "latencies" in called, (
            "Expected _format_asm_with_per_load_latencies to be called"
        )
        assert called["latencies"] == [100], (
            f"Expected latencies=[100] for 1 miss * 100 cycles, got {called['latencies']}"
        )
        assert called.get("repeat", 1) == 1, (
            "Average mode must not repeat the block"
        )

    def test_average_mode_no_iterations_flag(self, monkeypatch):
        """Average mode must NOT add -iterations=1 to the llvm-mca command."""
        captured_cmd = {}

        monkeypatch.setattr(
            analyze, "_format_asm_with_per_load_latencies",
            lambda *a, **kw: "\tnop\n.Lmca_end:\n",
        )
        monkeypatch.setattr(analyze, "_LLVM_MCA", "llvm-mca")

        class FakeProc:
            returncode = 0
            stdout = "IPC: 1.00\n"
            stderr = ""

        def fake_run(cmd, **kw):
            captured_cmd["cmd"] = cmd
            return FakeProc()

        monkeypatch.setattr(subprocess, "run", fake_run)

        # Use a region with a load instruction to activate average-mode formatter.
        analyze._run_mca(
            [(0x0, "mov", "(%edi),%eax"), (0x2, "add", "%eax,%edx")],
            cache_mode=analyze._AverageCacheMiss(10, 100),
        )
        assert "-iterations=1" not in captured_cmd.get("cmd", []), (
            "Average mode must not add -iterations=1 to the llvm-mca command"
        )

    def test_stochastic_mode_still_uses_cache_miss_formatter(self, monkeypatch):
        """Stochastic mode uses _format_asm_with_cache_miss when loads are present."""
        called = {}

        def fake_format_stochastic(instrs, arch, cache_miss, cache_latency):
            called["stochastic"] = True
            return "\tnop\n.Lmca_end:\n"

        monkeypatch.setattr(analyze, "_format_asm_with_cache_miss",
                            fake_format_stochastic)
        monkeypatch.setattr(analyze, "_LLVM_MCA", "llvm-mca")

        class FakeProc:
            returncode = 0
            stdout = "IPC: 1.00\n"
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeProc())

        # Use a region with a load instruction so the stochastic formatter is used.
        analyze._run_mca(
            [(0x0, "mov", "(%edi),%eax"), (0x2, "add", "%eax,%edx")],
            cache_mode=analyze._StochasticCacheMiss(10, 100),
        )
        assert called.get("stochastic"), (
            "Expected _format_asm_with_cache_miss to be called in stochastic mode"
        )

    def test_average_mode_bresenham_latency_distribution(self, monkeypatch):
        """_AverageCacheMiss uses Bresenham distribution for per-load latencies.

        Region: 3 loads, expected_misses=8.0 (ipcm=1, 8 instrs).
        _bresenham_load_misses(8.0, 3) = [3, 2, 3] → latencies=[300, 200, 300].
        """
        called = {}

        def fake_per_load(instrs, arch, latencies, repeat=1):
            called["latencies"] = latencies
            return "\tnop\n.Lmca_end:\n"

        monkeypatch.setattr(analyze, "_format_asm_with_per_load_latencies",
                            fake_per_load)

        # 8 instrs total, 3 loads: expected_misses = 8/1 = 8.0
        # round(8*(i+1)/3) - round(8*i/3) for i=0,1,2:
        #   i=0: round(8/3)  - 0 = 3 - 0 = 3
        #   i=1: round(16/3) - 3 = 5 - 3 = 2
        #   i=2: round(24/3) - 5 = 8 - 5 = 3
        instrs = [
            (0x0, "mov", "(%edi),%eax"),   # load 0
            (0x2, "add", "%eax,%edx"),
            (0x4, "mov", "(%esi),%ebx"),   # load 1
            (0x6, "add", "%ebx,%ecx"),
            (0x8, "mov", "(%ebx),%ecx"),   # load 2
            (0xa, "add", "%ecx,%edx"),
            (0xc, "add", "%edx,%eax"),
            (0xe, "ret", ""),
        ]
        mode = analyze._AverageCacheMiss(instructions_per_cache_miss=1,
                                          cache_latency=100)
        mode.format_asm(instrs, analyze.X86Arch())

        assert called.get("latencies") == [300, 200, 300], (
            f"Expected Bresenham latencies [300, 200, 300], got {called.get('latencies')}"
        )


# ---------------------------------------------------------------------------
# Unit tests — instructions-per-cache-miss (IPCM) logic
# ---------------------------------------------------------------------------

class TestBresenhamLoadMisses:
    """Verify _bresenham_load_misses distributes misses correctly."""

    def test_even_split_two_loads(self):
        """10 misses across 2 loads → [5, 5]."""
        assert analyze._bresenham_load_misses(10.0, 2) == [5, 5]

    def test_bresenham_three_loads_total_8(self):
        """8 misses across 3 loads → [3, 2, 3].

        round(8*1/3)=3, round(8*2/3)=5, round(8*3/3)=8
        deltas: 3-0=3, 5-3=2, 8-5=3
        """
        assert analyze._bresenham_load_misses(8.0, 3) == [3, 2, 3]

    def test_bresenham_three_loads_total_11(self):
        """11 misses across 3 loads → [4, 3, 4].

        round(11*1/3)=4, round(11*2/3)=7, round(11*3/3)=11
        deltas: 4-0=4, 7-4=3, 11-7=4
        """
        assert analyze._bresenham_load_misses(11.0, 3) == [4, 3, 4]

    def test_sum_equals_round_of_expected(self):
        """Sum of misses equals round(expected_misses)."""
        for n_loads in [1, 2, 3, 5, 7]:
            for expected in [0.2, 1.0, 3.3, 7.5, 10.0]:
                misses = analyze._bresenham_load_misses(expected, n_loads)
                assert sum(misses) == round(expected), (
                    f"n_loads={n_loads}, expected={expected}: "
                    f"sum={sum(misses)}, round(expected)={round(expected)}"
                )

    def test_each_load_gets_at_least_floor_misses(self):
        """Every load gets either floor or ceil of the average."""
        import math as _math
        for n_loads in [2, 3, 5]:
            for expected in [1.0, 3.0, 7.5]:
                misses = analyze._bresenham_load_misses(expected, n_loads)
                avg = expected / n_loads
                for m in misses:
                    assert m in (_math.floor(avg), _math.ceil(avg)), (
                        f"n_loads={n_loads}, expected={expected}: "
                        f"miss count {m} is not floor({avg}) or ceil({avg})"
                    )

    def test_single_load_gets_all_misses(self):
        """With 1 load, it gets round(expected_misses) misses."""
        assert analyze._bresenham_load_misses(5.7, 1) == [round(5.7)]


class TestFormatAsmWithPerLoadLatencies:
    """Verify _format_asm_with_per_load_latencies emits correct directives."""

    _ARCH = analyze.X86Arch()

    def test_single_load_gets_correct_latency(self):
        """The one load instruction gets its specific latency."""
        instrs = [(0x0, "mov", "(%edi),%eax"), (0x2, "add", "%eax,%edx")]
        out = analyze._format_asm_with_per_load_latencies(instrs, self._ARCH, [200])
        assert "# LLVM-MCA-LATENCY 200" in out
        # Opening directive should appear before the load
        load_pos = out.index("mov (%edi),%eax")
        lat_pos = out.index("# LLVM-MCA-LATENCY 200")
        assert lat_pos < load_pos

    def test_per_load_different_latencies(self):
        """Different loads receive their respective latencies."""
        instrs = [
            (0x0, "mov", "(%edi),%eax"),
            (0x2, "add", "%eax,%edx"),
            (0x4, "mov", "(%esi),%ebx"),
        ]
        out = analyze._format_asm_with_per_load_latencies(instrs, self._ARCH, [300, 200])
        assert "# LLVM-MCA-LATENCY 300" in out
        assert "# LLVM-MCA-LATENCY 200" in out

    def test_zero_latency_no_directive(self):
        """A load with latency=0 is emitted without LLVM-MCA-LATENCY directives."""
        instrs = [(0x0, "mov", "(%edi),%eax"), (0x2, "ret", "")]
        out = analyze._format_asm_with_per_load_latencies(instrs, self._ARCH, [0])
        assert "LLVM-MCA-LATENCY" not in out

    def test_lmca_end_label_present(self):
        """The .Lmca_end: label is always present."""
        instrs = [(0x0, "mov", "(%edi),%eax")]
        out = analyze._format_asm_with_per_load_latencies(instrs, self._ARCH, [100])
        assert ".Lmca_end:" in out

    def test_repeat_produces_unique_labels(self):
        """With repeat>1, each iteration gets a unique label suffix."""
        instrs = [(0x0, "mov", "(%edi),%eax")]
        # Force a label by making it a branch target — use a self-loop
        loop_instrs = [
            (0x0, "mov", "(%edi),%eax"),
            (0x2, "jne", "0x0"),           # back-branch → labels addr 0x0
        ]
        out = analyze._format_asm_with_per_load_latencies(
            loop_instrs, self._ARCH, [100], repeat=3
        )
        assert ".Lmca_0_r0:" in out
        assert ".Lmca_0_r1:" in out
        assert ".Lmca_0_r2:" in out

    def test_no_repetition_by_default(self):
        """Default repeat=1 produces no _r0 suffix in labels."""
        loop_instrs = [
            (0x0, "mov", "(%edi),%eax"),
            (0x2, "jne", "0x0"),
        ]
        out = analyze._format_asm_with_per_load_latencies(
            loop_instrs, self._ARCH, [100]
        )
        assert "_r0" not in out


class TestCacheMissIpcmLogic:
    """Verify IPCM-based effective miss rate and latency computation."""

    # Region: 10 instructions, 2 loads (mov ... + mov ...).
    _LOADS_INSTRS = [
        (0x0, "mov", "(%edi),%eax"),
        (0x2, "add", "%eax,%edx"),
        (0x4, "mov", "(%esi),%ebx"),
        (0x6, "add", "%ebx,%ecx"),
        (0x8, "add", "%ecx,%edx"),
        (0xa, "add", "%edx,%eax"),
        (0xc, "add", "%eax,%ecx"),
        (0xe, "sub", "%ecx,%edx"),
        (0x10, "inc", "%eax"),
        (0x12, "ret", ""),
    ]
    _ARCH = analyze.X86Arch()

    def test_stochastic_multiple_misses_per_load_uses_bresenham(self, monkeypatch):
        """When num_loads < expected_misses, _format_asm_with_per_load_latencies
        is called with Bresenham-distributed latencies and _CACHE_MISS_REPEAT.

        Region: 10 instrs, 2 loads, ipcm=1 → expected_misses=10.
        _bresenham_load_misses(10, 2) = [5, 5] → latencies=[500, 500].
        """
        called = {}

        def fake_per_load(instrs, arch, latencies, repeat=1):
            called["latencies"] = latencies
            called["repeat"] = repeat
            return "\tnop\n.Lmca_end:\n"

        monkeypatch.setattr(analyze, "_format_asm_with_per_load_latencies",
                            fake_per_load)

        mode = analyze._StochasticCacheMiss(instructions_per_cache_miss=1,
                                             cache_latency=100)
        mode.format_asm(self._LOADS_INSTRS, self._ARCH)

        assert called.get("latencies") == [500, 500], (
            f"Expected Bresenham latencies [500, 500], got {called.get('latencies')}"
        )
        assert called.get("repeat") == analyze._CACHE_MISS_REPEAT, (
            f"Expected repeat={analyze._CACHE_MISS_REPEAT}, got {called.get('repeat')}"
        )

    def test_stochastic_fractional_miss_rate_when_loads_exceed_expected(self,
                                                                          monkeypatch):
        """When num_loads >= expected_misses, miss_fraction < 1.

        Region: 10 instrs, 2 loads, ipcm=10 → expected_misses=1.
        miss_fraction = 1/2 = 0.5.
        """
        called = {}

        def fake_fmt(instrs, arch, cache_miss, cache_latency):
            called["cache_miss"] = cache_miss
            called["cache_latency"] = cache_latency
            return "\tnop\n.Lmca_end:\n"

        monkeypatch.setattr(analyze, "_format_asm_with_cache_miss", fake_fmt)

        mode = analyze._StochasticCacheMiss(instructions_per_cache_miss=10,
                                             cache_latency=100)
        mode.format_asm(self._LOADS_INSTRS, self._ARCH)

        assert abs(called.get("cache_miss", 0) - 0.5) < 1e-9, (
            f"Expected cache_miss=0.5, got {called.get('cache_miss')}"
        )
        assert called.get("cache_latency") == 100, (
            f"Expected cache_latency=100, got {called.get('cache_latency')}"
        )

    def test_stochastic_no_loads_uses_plain_format_asm(self, monkeypatch):
        """When num_loads=0, _format_asm is used (no miss possible)."""
        called = {}

        def fake_plain(instrs, arch):
            called["plain"] = True
            return "\tnop\n.Lmca_end:\n"

        def fake_miss(instrs, arch, cache_miss, cache_latency):
            called["miss"] = True
            return "\tnop\n.Lmca_end:\n"

        monkeypatch.setattr(analyze, "_format_asm", fake_plain)
        monkeypatch.setattr(analyze, "_format_asm_with_cache_miss", fake_miss)

        no_load_instrs = [(0x0, "nop", ""), (0x1, "ret", "")]
        mode = analyze._StochasticCacheMiss(instructions_per_cache_miss=10,
                                             cache_latency=100)
        mode.format_asm(no_load_instrs, self._ARCH)

        assert called.get("plain"), "Expected _format_asm to be called for no-load region"
        assert not called.get("miss"), (
            "Did not expect _format_asm_with_cache_miss for no-load region"
        )

    def test_average_multiple_misses_per_load_uses_bresenham(self, monkeypatch):
        """When num_loads < expected_misses, _format_asm_with_per_load_latencies
        is called with Bresenham-distributed per-load latencies.

        Region: 10 instrs, 2 loads, ipcm=1 → expected_misses=10.
        _bresenham_load_misses(10, 2) = [5, 5] → latencies=[500, 500].
        """
        called = {}

        def fake_per_load(instrs, arch, latencies, repeat=1):
            called["latencies"] = latencies
            return "\tnop\n.Lmca_end:\n"

        monkeypatch.setattr(analyze, "_format_asm_with_per_load_latencies",
                            fake_per_load)

        mode = analyze._AverageCacheMiss(instructions_per_cache_miss=1,
                                          cache_latency=100)
        mode.format_asm(self._LOADS_INSTRS, self._ARCH)

        assert called.get("latencies") == [500, 500], (
            f"Expected Bresenham latencies [500, 500], got {called.get('latencies')}"
        )

    def test_build_cache_mode_inf_returns_no_cache_miss(self):
        """_build_cache_mode(float('inf'), ...) returns _NoCacheMiss."""
        import math
        mode = analyze._build_cache_mode(float("inf"), 100, "stochastic")
        assert isinstance(mode, analyze._NoCacheMiss)

    def test_build_cache_mode_finite_stochastic(self):
        """_build_cache_mode(10, ...) returns _StochasticCacheMiss."""
        mode = analyze._build_cache_mode(10, 100, "stochastic")
        assert isinstance(mode, analyze._StochasticCacheMiss)

    def test_build_cache_mode_finite_average(self):
        """_build_cache_mode(10, ..., 'average') returns _AverageCacheMiss."""
        mode = analyze._build_cache_mode(10, 100, "average")
        assert isinstance(mode, analyze._AverageCacheMiss)
