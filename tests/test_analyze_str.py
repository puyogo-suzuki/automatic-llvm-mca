"""Unit and integration tests for analyze_str.py.

Unit tests cover filename parsing, arch detection from name, and instruction
parsing from dump-format text.  Integration tests require llvm-mca to be
installed; they are skipped automatically when it is not available.

Run with::

    python3 -m pytest tests/test_analyze_str.py -v
"""

import os
import shutil
import sys
import tempfile
import io
import contextlib

import pytest

# ---------------------------------------------------------------------------
# Make the package root importable when running pytest from any directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze_str  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llvm_mca_available() -> bool:
    if shutil.which("llvm-mca"):
        return True
    for ver in range(30, 10, -1):
        if shutil.which(f"llvm-mca-{ver}"):
            return True
    return False


_NEED_MCA = pytest.mark.skipif(
    not _llvm_mca_available(),
    reason="llvm-mca not found — install llvm",
)


# ---------------------------------------------------------------------------
# Unit tests — _parse_filename
# ---------------------------------------------------------------------------

class TestParseFilename:

    def test_basic_x86(self):
        start, end, arch = analyze_str._parse_filename("1000_1080.x86.txt")
        assert start == 0x1000
        assert end == 0x1080
        assert arch == "x86"

    def test_aarch64(self):
        start, end, arch = analyze_str._parse_filename("0_1c.aarch64.txt")
        assert start == 0x0
        assert end == 0x1c
        assert arch == "aarch64"

    def test_arm(self):
        start, end, arch = analyze_str._parse_filename("a0_b4.arm.txt")
        assert start == 0xa0
        assert end == 0xb4
        assert arch == "arm"

    def test_riscv(self):
        start, end, arch = analyze_str._parse_filename("100_11c.riscv.txt")
        assert start == 0x100
        assert end == 0x11c
        assert arch == "riscv"

    def test_full_path_uses_basename(self):
        start, end, arch = analyze_str._parse_filename(
            "/some/dir/1000_1080.x86.txt"
        )
        assert start == 0x1000
        assert end == 0x1080
        assert arch == "x86"

    def test_invalid_filename_raises(self):
        with pytest.raises(ValueError, match="Cannot parse filename"):
            analyze_str._parse_filename("badname.txt")

    def test_missing_arch_raises(self):
        with pytest.raises(ValueError, match="Cannot parse filename"):
            analyze_str._parse_filename("1000_1080.txt")

    def test_non_hex_raises(self):
        with pytest.raises(ValueError, match="Cannot parse filename"):
            analyze_str._parse_filename("xyz_abc.x86.txt")


# ---------------------------------------------------------------------------
# Unit tests — _arch_from_name
# ---------------------------------------------------------------------------

class TestArchFromName:

    def test_x86(self):
        arch = analyze_str._arch_from_name("x86")
        assert arch.name == "x86"

    def test_aarch64(self):
        arch = analyze_str._arch_from_name("aarch64")
        assert arch.name == "aarch64"

    def test_arm(self):
        arch = analyze_str._arch_from_name("arm")
        assert arch.name == "arm"

    def test_riscv(self):
        arch = analyze_str._arch_from_name("riscv")
        assert arch.name == "riscv"

    def test_case_insensitive_X86(self):
        arch = analyze_str._arch_from_name("X86")
        assert arch.name == "x86"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            analyze_str._arch_from_name("mips")


# ---------------------------------------------------------------------------
# Unit tests — _parse_instrs_from_dump
# ---------------------------------------------------------------------------

class TestParseInstrsFromDump:

    def test_basic_block_no_labels(self):
        """A plain basic block (no internal labels) is parsed correctly."""
        text = (
            "\tmovq (%rdi,%rax,8), %rax\n"
            "\taddq %rax, %rcx\n"
            "\tincq %rax\n"
            ".Lmca_end:\n"
        )
        instrs = analyze_str._parse_instrs_from_dump(text)
        assert len(instrs) == 3
        _, mn0, ops0 = instrs[0]
        assert mn0 == "movq"
        assert "(%rdi,%rax,8)" in ops0

    def test_loop_with_label(self):
        """A loop with a .Lmca_xx: label is parsed and the address is preserved."""
        text = (
            ".Lmca_1c:\n"
            "\tmovq (%rdi,%rax,8), %rax\n"
            "\taddq %rax, %rcx\n"
            "\tjne .Lmca_1c\n"
            ".Lmca_end:\n"
        )
        instrs = analyze_str._parse_instrs_from_dump(text)
        assert len(instrs) == 3
        addr0, mn0, _ = instrs[0]
        assert addr0 == 0x1c
        assert mn0 == "movq"

    def test_branch_label_replaced_with_hex(self):
        """Branch operands referencing .Lmca_xxx are replaced with hex addresses."""
        text = (
            ".Lmca_1c:\n"
            "\taddq %rax, %rcx\n"
            "\tjne .Lmca_1c\n"
            ".Lmca_end:\n"
        )
        instrs = analyze_str._parse_instrs_from_dump(text)
        # Last instruction is 'jne'; its operand should be '1c' (hex), not '.Lmca_1c'.
        _, mn_br, ops_br = instrs[-1]
        assert mn_br == "jne"
        assert ".Lmca_1c" not in ops_br
        assert "1c" in ops_br

    def test_llvm_mca_latency_directives_are_skipped(self):
        """# LLVM-MCA-LATENCY directives in the dump file are not parsed as instructions."""
        text = (
            "# LLVM-MCA-LATENCY 200\n"
            "\tmovq (%rdi), %rax\n"
            "# LLVM-MCA-LATENCY\n"
            "\taddq %rax, %rcx\n"
            ".Lmca_end:\n"
        )
        instrs = analyze_str._parse_instrs_from_dump(text)
        assert len(instrs) == 2
        mnemonics = [mn for _, mn, _ in instrs]
        assert mnemonics == ["movq", "addq"]

    def test_empty_text(self):
        """An empty dump file produces an empty instruction list."""
        instrs = analyze_str._parse_instrs_from_dump("")
        assert instrs == []

    def test_only_labels_and_directives(self):
        """A file with only labels/directives produces an empty instruction list."""
        text = ".Lmca_0:\n.Lmca_end:\n# LLVM-MCA-LATENCY 0\n"
        instrs = analyze_str._parse_instrs_from_dump(text)
        assert instrs == []

    def test_unlabeled_instructions_get_sequential_addresses(self):
        """Without labels, instructions receive sequential addresses 0, 1, 2, …"""
        text = "\tnop\n\tnop\n\tnop\n.Lmca_end:\n"
        instrs = analyze_str._parse_instrs_from_dump(text)
        assert [addr for addr, _, _ in instrs] == [0, 1, 2]

    def test_labeled_followed_by_unlabeled(self):
        """After a labeled instruction, auto_addr continues from labeled_addr + 1."""
        text = (
            ".Lmca_10:\n"
            "\tnop\n"      # addr=0x10, auto_addr becomes 0x11
            "\tnop\n"      # addr=0x11 (sequential)
            ".Lmca_end:\n"
        )
        instrs = analyze_str._parse_instrs_from_dump(text)
        assert instrs[0][0] == 0x10
        assert instrs[1][0] == 0x11

    def test_multiple_labels(self):
        """Multiple .Lmca_ labels are each assigned to their respective instructions."""
        text = (
            ".Lmca_0:\n"
            "\taddq %rax, %rcx\n"
            ".Lmca_4:\n"
            "\taddq %rdx, %rcx\n"
            ".Lmca_end:\n"
        )
        instrs = analyze_str._parse_instrs_from_dump(text)
        assert instrs[0][0] == 0x0
        assert instrs[1][0] == 0x4


# ---------------------------------------------------------------------------
# Integration tests — analyze_str() with real llvm-mca
# ---------------------------------------------------------------------------

# A minimal x86-64 basic block in dump format (no loops, one load).
_X86_BASIC_BLOCK_DUMP = (
    "\tmovq (%rdi), %rax\n"
    "\taddq %rax, %rsi\n"
    ".Lmca_end:\n"
)

# A minimal x86-64 loop in dump format.
_X86_LOOP_DUMP = (
    ".Lmca_0:\n"
    "\tmovq (%rdi,%rax,8), %rax\n"
    "\taddq %rax, %rcx\n"
    "\taddq $0x8, %rdi\n"
    "\tjne .Lmca_0\n"
    ".Lmca_end:\n"
)


def _write_dump_file(tmp_path, name: str, content: str) -> str:
    """Write *content* to a temp file named *name* and return its path."""
    path = str(tmp_path / name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


class TestAnalyzeStrIntegration:

    @_NEED_MCA
    def test_basic_block_returns_result(self, tmp_path):
        """analyze_str() returns a result for an x86-64 basic block."""
        path = _write_dump_file(tmp_path, "0_8.x86.txt", _X86_BASIC_BLOCK_DUMP)
        instrs, arch, start, end = analyze_str.load_str_file(path)
        result = analyze_str.analyze_str(instrs, arch)
        assert result is not None, "Expected a result"

    @_NEED_MCA
    def test_ipc_positive(self, tmp_path):
        """Retired instructions and elapsed cycles from the basic block dump are strictly positive."""
        path = _write_dump_file(tmp_path, "0_8.x86.txt", _X86_BASIC_BLOCK_DUMP)
        instrs, arch, start, end = analyze_str.load_str_file(path)
        result = analyze_str.analyze_str(instrs, arch)
        assert result is not None
        retired, cycles, _ = result
        assert retired > 0, f"Expected positive retired instructions, got {retired}"
        assert cycles > 0, f"Expected positive elapsed cycles, got {cycles}"

    @_NEED_MCA
    def test_load_instructions_nonzero(self, tmp_path):
        """The basic block has one load, so load_instructions must be > 0."""
        path = _write_dump_file(tmp_path, "0_8.x86.txt", _X86_BASIC_BLOCK_DUMP)
        instrs, arch, start, end = analyze_str.load_str_file(path)
        result = analyze_str.analyze_str(instrs, arch)
        assert result is not None
        _, _, load_instrs = result
        assert load_instrs > 0, (
            f"Expected load_instrs > 0 (block has a load), got {load_instrs}"
        )

    @_NEED_MCA
    def test_addresses_from_filename(self, tmp_path):
        """start/end addresses are taken from the filename, not from the instructions."""
        path = _write_dump_file(
            tmp_path, "1000_100c.x86.txt", _X86_BASIC_BLOCK_DUMP
        )
        instrs, arch, start, end = analyze_str.load_str_file(path)
        assert start == 0x1000
        assert end == 0x100c

    @_NEED_MCA
    def test_loop_dump_returns_result(self, tmp_path):
        """analyze_str() handles a loop dump file (with .Lmca_xx: labels)."""
        path = _write_dump_file(tmp_path, "0_18.x86.txt", _X86_LOOP_DUMP)
        instrs, arch, start, end = analyze_str.load_str_file(path)
        result = analyze_str.analyze_str(instrs, arch)
        assert result is not None, "Expected a result for loop dump"
        retired, cycles, _ = result
        assert retired > 0
        assert cycles > 0

    @_NEED_MCA
    def test_unknown_arch_in_filename_raises(self, tmp_path):
        """load_str_file() raises ValueError for an unknown arch in the filename."""
        path = _write_dump_file(
            tmp_path, "0_8.mips.txt", _X86_BASIC_BLOCK_DUMP
        )
        with pytest.raises(ValueError, match="Unknown architecture"):
            analyze_str.load_str_file(path)

    @_NEED_MCA
    def test_bad_filename_format_raises(self, tmp_path):
        """load_str_file() raises ValueError for a filename that does not match the pattern."""
        path = _write_dump_file(tmp_path, "badname.txt", _X86_BASIC_BLOCK_DUMP)
        with pytest.raises(ValueError, match="Cannot parse filename"):
            analyze_str.load_str_file(path)


# ---------------------------------------------------------------------------
# Integration test — round-trip via Dumper + analyze_str
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Test that analyze.py --dump output is correctly re-analysed by analyze_str."""

    @_NEED_MCA
    def test_dump_and_reanalyze_produces_same_counts(self, tmp_path):
        """A region dumped by analyze.Dumper and re-analyzed by analyze_str
        produces the same retired/cycles counts as the original _run_mca call."""
        import analyze as _analyze

        instrs = [
            (0x10, "movq", "(%rdi,%rax,8), %rax"),
            (0x14, "addq", "%rax, %rcx"),
            (0x18, "incq", "%rax"),
        ]
        arch = _analyze.X86Arch()
        mca_args = arch.mca_args

        # Run llvm-mca directly.
        original = _analyze._run_mca(instrs, mca_args, arch=arch)
        if original is None:
            pytest.skip("llvm-mca returned None for the original run")
        orig_retired, orig_cycles, orig_li = original

        # Dump to a temp file using Dumper.
        dump_dir = str(tmp_path / "dump")
        dumper = _analyze.Dumper(dump_dir=dump_dir)
        dumper.dump(instrs, _analyze._format_asm(instrs, arch), arch)

        # Locate the written file.
        start = instrs[0][0]
        end = instrs[-1][0]
        dump_path = os.path.join(dump_dir, f"{start:x}_{end:x}.{arch.name}.txt")
        assert os.path.isfile(dump_path), f"Dump file not created: {dump_path}"

        # Re-analyse with analyze_str (loading done by the caller, as in main()).
        new_instrs, new_arch, new_start, new_end = analyze_str.load_str_file(dump_path)
        result = analyze_str.analyze_str(new_instrs, new_arch)
        assert result is not None, "analyze_str returned None"
        new_retired, new_cycles, new_li = result

        assert new_start == start
        assert new_end == end
        assert new_retired == orig_retired, (
            f"retired mismatch: original={orig_retired}, re-analysed={new_retired}"
        )
        assert new_cycles == orig_cycles, (
            f"cycles mismatch: original={orig_cycles}, re-analysed={new_cycles}"
        )
        assert new_li == orig_li, (
            f"load_instructions mismatch: original={orig_li}, re-analysed={new_li}"
        )
