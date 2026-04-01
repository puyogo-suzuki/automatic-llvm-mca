"""Tests for ipc_relate.py — unit and integration tests.

Integration tests compile a small C function that contains a loop and memory
loads, then run ``ipc_relate.ipc_relate()`` on the resulting ELF object file
and validate the output format and basic properties.

Run with::

    python3 -m pytest tests/test_ipc_relate.py -v
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
import analyze        # noqa: E402
import ipc_relate     # noqa: E402


# ---------------------------------------------------------------------------
# Simple C source used by integration tests.
# Contains a loop (backward branch) and a memory load so that both
# loop detection and load_proportion can be verified.
# ---------------------------------------------------------------------------
_C_SOURCE = """\
long sum(long *a, int n) {
    long s = 0;
    for (int i = 0; i < n; i++)
        s += a[i];
    return s;
}
"""

_EXPECTED_COLS = 3 + len(ipc_relate._INSTRUCTIONS_PER_CACHE_MISS)  # start, end, lp, cpi*


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _have(*tools: str) -> bool:
    return all(shutil.which(t) for t in tools)


def _llvm_mca_available() -> bool:
    if shutil.which("llvm-mca"):
        return True
    for ver in range(20, 10, -1):
        if shutil.which(f"llvm-mca-{ver}"):
            return True
    return False


def _compile(source: str, compiler: str, extra_flags=()) -> str:
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
# Skip markers
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
# Module-scoped fixtures
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
# Unit tests — _CACHE_MISS_RATES constant
# ---------------------------------------------------------------------------

class TestInstructionsPerCacheMiss:
    """Sanity checks on the _INSTRUCTIONS_PER_CACHE_MISS constant."""

    def test_length(self):
        """There are 6 instructions-per-cache-miss steps (1, 2, 5, 10, 1000, inf)."""
        assert len(ipc_relate._INSTRUCTIONS_PER_CACHE_MISS) == 6

    def test_values(self):
        """The values are 1, 2, 5, 10, 1000, inf."""
        assert ipc_relate._INSTRUCTIONS_PER_CACHE_MISS == [1, 2, 5, 10, 1000, float("inf")]

    def test_strictly_increasing(self):
        rates = ipc_relate._INSTRUCTIONS_PER_CACHE_MISS
        assert all(rates[i] < rates[i + 1] for i in range(len(rates) - 1))


# ---------------------------------------------------------------------------
# Unit tests — _region_cpis with monkeypatching
# ---------------------------------------------------------------------------

class TestRegionCpis:
    """Unit tests for ipc_relate._region_cpis."""

    def _dummy_region(self):
        return [(0x0, "add", "%eax,%edx"), (0x2, "jne", "0")]

    def test_returns_tuple_of_cpis_and_load_proportion(self, monkeypatch):
        """_region_cpis returns (cpis, load_proportion) on success."""
        monkeypatch.setattr(analyze, "_run_mca", lambda *a, **kw: (2.0, 0.25))
        result = ipc_relate._region_cpis(
            self._dummy_region(), [], "x86", 100
        )
        assert result is not None
        cpis, load_proportion = result
        assert len(cpis) == len(ipc_relate._INSTRUCTIONS_PER_CACHE_MISS)
        assert abs(load_proportion - 0.25) < 1e-9

    def test_cpi_is_reciprocal_of_ipc(self, monkeypatch):
        """CPI values are 1/IPC."""
        monkeypatch.setattr(analyze, "_run_mca", lambda *a, **kw: (4.0, 0.5))
        cpis, _ = ipc_relate._region_cpis(
            self._dummy_region(), [], "x86", 100
        )
        for cpi in cpis:
            assert abs(cpi - 0.25) < 1e-9

    def test_returns_none_when_mca_fails(self, monkeypatch):
        """_region_cpis returns None if _run_mca returns None for any rate."""
        monkeypatch.setattr(analyze, "_run_mca", lambda *a, **kw: None)
        result = ipc_relate._region_cpis(
            self._dummy_region(), [], "x86", 100
        )
        assert result is None

    def test_load_proportion_from_first_run(self, monkeypatch):
        """load_proportion is taken from the zero-miss-rate run (first call)."""
        call_count = [0]

        def fake_run_mca(*args, **kwargs):
            lp = 0.1 * (call_count[0] + 1)  # changes per call
            call_count[0] += 1
            return (2.0, lp)

        monkeypatch.setattr(analyze, "_run_mca", fake_run_mca)
        _, load_proportion = ipc_relate._region_cpis(
            self._dummy_region(), [], "x86", 100
        )
        # load_proportion should come from the first (miss_rate=0) call.
        assert abs(load_proportion - 0.1) < 1e-9

    def test_custom_instructions_per_cache_miss(self, monkeypatch):
        """_region_cpis respects a custom instructions_per_cache_miss list."""
        monkeypatch.setattr(analyze, "_run_mca", lambda *a, **kw: (2.0, 0.5))
        custom = [1, 50, float("inf")]
        cpis, _ = ipc_relate._region_cpis(
            self._dummy_region(), [], "x86", 100,
            instructions_per_cache_miss=custom,
        )
        assert len(cpis) == len(custom)

    def test_single_ipcm_value(self, monkeypatch):
        """_region_cpis works with a single-element instructions_per_cache_miss."""
        monkeypatch.setattr(analyze, "_run_mca", lambda *a, **kw: (5.0, 0.0))
        cpis, _ = ipc_relate._region_cpis(
            self._dummy_region(), [], "x86", 100,
            instructions_per_cache_miss=[float("inf")],
        )
        assert len(cpis) == 1
        assert abs(cpis[0] - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# Unit tests — ipc_relate() function with custom IPCM values
# ---------------------------------------------------------------------------

class TestIpcRelateCustomIpcm:
    """Unit tests for ipc_relate.ipc_relate() with custom instructions_per_cache_miss."""

    def _dummy_region(self):
        return [(0x0, "add", "%eax,%edx"), (0x2, "jne", "0")]

    def test_custom_ipcm_tuple_length(self, monkeypatch):
        """ipc_relate() yields tuples with one CPI per custom IPCM value."""
        custom = [1, 50, float("inf")]

        def fake_run_mca(region, mca_args, arch, cache_mode):
            return (2.0, 0.25)

        monkeypatch.setattr(analyze, "_run_mca", fake_run_mca)
        monkeypatch.setattr(analyze, "_detect_arch", lambda b: analyze.X86Arch())
        monkeypatch.setattr(
            analyze, "disassemble",
            lambda binary, arch: [("func", self._dummy_region())],
        )
        monkeypatch.setattr(analyze, "_find_loops", lambda instrs, arch: [])
        monkeypatch.setattr(analyze, "_in_any_loop", lambda addr, loops: False)

        results = list(
            ipc_relate.ipc_relate(
                "dummy.o", instructions_per_cache_miss=custom,
            )
        )
        assert results, "Expected at least one result"
        for row in results:
            # start, end, load_proportion + one CPI per IPCM value
            assert len(row) == 3 + len(custom)


# ---------------------------------------------------------------------------
# Unit tests — main() CLI argument --ipcm-values
# ---------------------------------------------------------------------------

class TestMainIpcmValues:
    """Unit tests for the --ipcm-values CLI argument in main()."""

    def _run_main(self, argv, monkeypatch, fake_ipc_relate=None):
        """Helper: run ipc_relate.main() with patched sys.argv."""
        import io

        if fake_ipc_relate is None:
            def fake_ipc_relate(*a, **kw):
                return iter([])

        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(ipc_relate, "ipc_relate", fake_ipc_relate)
        # Patch os.path.isfile to avoid needing a real binary.
        monkeypatch.setattr(os.path, "isfile", lambda p: True)

        captured = io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(captured):
            ipc_relate.main()
        return captured.getvalue()

    def test_default_ipcm_header(self, monkeypatch):
        """main() emits the default IPCM column headers when --ipcm-values is not given."""
        output = self._run_main(
            ["ipc_relate.py", "dummy.o"], monkeypatch
        )
        header = output.splitlines()[0]
        assert header == (
            "start_address,end_address,load_proportion,"
            "cpi_ipcm1,cpi_ipcm2,cpi_ipcm5,cpi_ipcm10,cpi_ipcm1000,cpi_ipcm_inf"
        )

    def test_custom_ipcm_header(self, monkeypatch):
        """main() emits custom IPCM column headers when --ipcm-values is given."""
        output = self._run_main(
            ["ipc_relate.py", "dummy.o", "--ipcm-values", "1", "50", "inf"],
            monkeypatch,
        )
        header = output.splitlines()[0]
        assert header == (
            "start_address,end_address,load_proportion,"
            "cpi_ipcm1,cpi_ipcm50,cpi_ipcm_inf"
        )

    def test_custom_ipcm_passed_to_ipc_relate(self, monkeypatch):
        """main() passes the parsed --ipcm-values to ipc_relate()."""
        captured_ipcm = []

        def fake_ipc_relate(binary, mcpu="", cache_latency=100,
                             cache_miss_mode="stochastic",
                             instructions_per_cache_miss=None):
            captured_ipcm.append(instructions_per_cache_miss)
            return iter([])

        self._run_main(
            ["ipc_relate.py", "dummy.o", "--ipcm-values", "5", "100", "inf"],
            monkeypatch,
            fake_ipc_relate=fake_ipc_relate,
        )
        assert len(captured_ipcm) == 1
        assert captured_ipcm[0] == [5.0, 100.0, float("inf")]


# ---------------------------------------------------------------------------
# Integration tests — AMD64 (x86-64)
# ---------------------------------------------------------------------------

class TestAMD64:
    """Integration tests using a compiled x86-64 ELF object file."""

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_results_nonempty(self, x86_obj):
        """ipc_relate() produces at least one result for an x86-64 binary."""
        results = list(ipc_relate.ipc_relate(x86_obj))
        assert results, "Expected at least one result from ipc_relate()"

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_tuple_length(self, x86_obj):
        """Each result tuple has start, end, load_proportion + one CPI per rate."""
        results = list(ipc_relate.ipc_relate(x86_obj))
        assert results
        for row in results:
            assert len(row) == _EXPECTED_COLS, (
                f"Expected {_EXPECTED_COLS} columns, got {len(row)}: {row}"
            )

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_addresses_are_nonneg_ints(self, x86_obj):
        """Start and end addresses are non-negative integers."""
        for row in ipc_relate.ipc_relate(x86_obj):
            start, end = row[0], row[1]
            assert isinstance(start, int) and start >= 0
            assert isinstance(end, int) and end >= 0

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_load_proportion_in_range(self, x86_obj):
        """load_proportion values are in [0, 1]."""
        for row in ipc_relate.ipc_relate(x86_obj):
            load_proportion = row[2]
            assert 0.0 <= load_proportion <= 1.0, (
                f"load_proportion {load_proportion} out of range"
            )

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_cpi_values_positive(self, x86_obj):
        """All CPI values are strictly positive."""
        for row in ipc_relate.ipc_relate(x86_obj):
            cpis = row[3:]
            for cpi in cpis:
                assert cpi > 0, f"CPI should be positive, got {cpi}"

    @_NEED_MCA
    @_NEED_X86_GCC
    def test_loop_detected(self, x86_obj):
        """The backward branch in sum() is detected as a loop (start < end)."""
        results = list(ipc_relate.ipc_relate(x86_obj))
        loops = [row for row in results if row[0] < row[1]]
        assert loops, (
            "Expected at least one loop region (start < end). "
            f"All results: {results}"
        )


# ---------------------------------------------------------------------------
# Integration tests — AArch64
# ---------------------------------------------------------------------------

class TestAArch64:
    """Integration tests using a compiled AArch64 ELF object file."""

    @_NEED_MCA
    @_NEED_AARCH64
    def test_results_nonempty(self, aarch64_obj):
        """ipc_relate() produces at least one result for an AArch64 binary."""
        results = list(ipc_relate.ipc_relate(aarch64_obj))
        assert results, "Expected at least one result from ipc_relate()"

    @_NEED_MCA
    @_NEED_AARCH64
    def test_tuple_length(self, aarch64_obj):
        """Each result tuple has start, end, load_proportion + one CPI per rate."""
        results = list(ipc_relate.ipc_relate(aarch64_obj))
        assert results
        for row in results:
            assert len(row) == _EXPECTED_COLS, (
                f"Expected {_EXPECTED_COLS} columns, got {len(row)}: {row}"
            )

    @_NEED_MCA
    @_NEED_AARCH64
    def test_load_proportion_in_range(self, aarch64_obj):
        """load_proportion values are in [0, 1]."""
        for row in ipc_relate.ipc_relate(aarch64_obj):
            load_proportion = row[2]
            assert 0.0 <= load_proportion <= 1.0
