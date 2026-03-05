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
    """Unit tests for analyze._get_branch_target."""

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
        assert analyze._get_branch_target("18") == 0x18

    def test_aarch64_ble_2digit_target(self):
        assert analyze._get_branch_target("2c") == 0x2c

    def test_aarch64_cbz_register_plus_target(self):
        """cbz xN, addr — target is the last operand."""
        assert analyze._get_branch_target("x0, 2dc") == 0x2dc

    def test_aarch64_tbz_register_imm_target(self):
        """tbz xN, #imm, addr — the #-prefixed imm must NOT be taken as target."""
        assert analyze._get_branch_target("x0, #0, 2dc") == 0x2dc

    def test_aarch64_0x_prefix(self):
        assert analyze._get_branch_target("0x400808") == 0x400808

    # RISC-V style — register names like a0/a5 precede the actual target
    def test_riscv_bne_register_register_target(self):
        """bne a4,a5,314 — last candidate is the branch target."""
        assert analyze._get_branch_target("a4,a5,314") == 0x314

    def test_riscv_beqz_register_target(self):
        """beqz a0,100 — last candidate is the branch target."""
        assert analyze._get_branch_target("a0,100") == 0x100

    # Indirect branches
    def test_indirect_returns_none(self):
        """Indirect branch (AT&T *%reg) returns None."""
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
