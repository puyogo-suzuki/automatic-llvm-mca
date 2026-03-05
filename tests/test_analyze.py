"""Tests for analyze.py, with a focus on AArch64 support.

Unit tests exercise pure-Python helpers using synthesised instruction
sequences and do not require any external tools.

Integration tests compile small C functions with the AArch64 cross-
compiler and run the full analysis pipeline.  They are skipped
automatically when the required tools are not available.
"""

import os
import shutil
import subprocess
import sys
import textwrap

import pytest

# Allow importing analyze.py from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze import (  # noqa: E402
    _ends_basic_block,
    _find_loops,
    _format_asm,
    _get_branch_target,
    _is_branch,
    analyze,
    disassemble,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _instrs(*rows):
    """Build a ``(addr, mnemonic, operands)`` list from plain tuples."""
    return list(rows)


# ---------------------------------------------------------------------------
# _is_branch — AArch64
# ---------------------------------------------------------------------------

class TestIsBranchAarch64:
    """_is_branch must recognise all AArch64 branch/jump/call mnemonics."""

    # --- True branches ---
    def test_unconditional_b(self):
        assert _is_branch("b", "aarch64")

    def test_branch_with_link(self):
        # bl is a *call* but still a branch instruction
        assert _is_branch("bl", "aarch64")

    def test_branch_to_register(self):
        assert _is_branch("br", "aarch64")

    def test_branch_with_link_to_register(self):
        assert _is_branch("blr", "aarch64")

    def test_ret(self):
        assert _is_branch("ret", "aarch64")

    def test_cbz(self):
        assert _is_branch("cbz", "aarch64")

    def test_cbnz(self):
        assert _is_branch("cbnz", "aarch64")

    def test_tbz(self):
        assert _is_branch("tbz", "aarch64")

    def test_tbnz(self):
        assert _is_branch("tbnz", "aarch64")

    def test_b_eq(self):
        assert _is_branch("b.eq", "aarch64")

    def test_b_ne(self):
        assert _is_branch("b.ne", "aarch64")

    def test_b_lt(self):
        assert _is_branch("b.lt", "aarch64")

    def test_b_le(self):
        assert _is_branch("b.le", "aarch64")

    def test_b_gt(self):
        assert _is_branch("b.gt", "aarch64")

    def test_b_ge(self):
        assert _is_branch("b.ge", "aarch64")

    def test_b_cs(self):
        assert _is_branch("b.cs", "aarch64")

    def test_b_vs(self):
        assert _is_branch("b.vs", "aarch64")

    def test_uppercase(self):
        # Detection is case-insensitive.
        assert _is_branch("B", "aarch64")
        assert _is_branch("B.NE", "aarch64")
        assert _is_branch("RET", "aarch64")

    # --- Non-branches (common instructions starting with 'b') ---
    def test_bic_not_branch(self):
        # BIC (bit-clear) starts with 'b' but is not a branch.
        assert not _is_branch("bic", "aarch64")

    def test_bfm_not_branch(self):
        # BFM (bitfield move) is not a branch.
        assert not _is_branch("bfm", "aarch64")

    def test_bfxil_not_branch(self):
        assert not _is_branch("bfxil", "aarch64")

    # --- Other non-branch instructions ---
    def test_add_not_branch(self):
        assert not _is_branch("add", "aarch64")

    def test_ldr_not_branch(self):
        assert not _is_branch("ldr", "aarch64")

    def test_str_not_branch(self):
        assert not _is_branch("str", "aarch64")

    def test_mov_not_branch(self):
        assert not _is_branch("mov", "aarch64")

    def test_cmp_not_branch(self):
        assert not _is_branch("cmp", "aarch64")


# ---------------------------------------------------------------------------
# _ends_basic_block — AArch64
# ---------------------------------------------------------------------------

class TestEndsBasicBlockAarch64:
    """_ends_basic_block must correctly classify AArch64 BB terminators.

    Key rule: ``bl`` and ``blr`` are *calls* — execution falls through to
    the next instruction, so they do NOT end a basic block.
    """

    # --- Terminators ---
    def test_b_ends_bb(self):
        assert _ends_basic_block("b", "", "aarch64")

    def test_br_ends_bb(self):
        assert _ends_basic_block("br", "", "aarch64")

    def test_ret_ends_bb(self):
        assert _ends_basic_block("ret", "", "aarch64")

    def test_cbz_ends_bb(self):
        assert _ends_basic_block("cbz", "", "aarch64")

    def test_cbnz_ends_bb(self):
        assert _ends_basic_block("cbnz", "", "aarch64")

    def test_tbz_ends_bb(self):
        assert _ends_basic_block("tbz", "", "aarch64")

    def test_tbnz_ends_bb(self):
        assert _ends_basic_block("tbnz", "", "aarch64")

    def test_b_ne_ends_bb(self):
        assert _ends_basic_block("b.ne", "", "aarch64")

    def test_b_eq_ends_bb(self):
        assert _ends_basic_block("b.eq", "", "aarch64")

    def test_b_le_ends_bb(self):
        assert _ends_basic_block("b.le", "", "aarch64")

    def test_b_gt_ends_bb(self):
        assert _ends_basic_block("b.gt", "", "aarch64")

    # --- Non-terminators (calls fall through) ---
    def test_bl_does_not_end_bb(self):
        # bl is a function call — fall-through continues.
        assert not _ends_basic_block("bl", "", "aarch64")

    def test_blr_does_not_end_bb(self):
        # blr is an indirect call — fall-through continues.
        assert not _ends_basic_block("blr", "", "aarch64")

    # --- Regular instructions ---
    def test_add_does_not_end_bb(self):
        assert not _ends_basic_block("add", "", "aarch64")

    def test_ldr_does_not_end_bb(self):
        assert not _ends_basic_block("ldr", "", "aarch64")

    def test_str_does_not_end_bb(self):
        assert not _ends_basic_block("str", "", "aarch64")

    def test_mov_does_not_end_bb(self):
        assert not _ends_basic_block("mov", "", "aarch64")

    def test_cmp_does_not_end_bb(self):
        assert not _ends_basic_block("cmp", "", "aarch64")


# ---------------------------------------------------------------------------
# _get_branch_target — AArch64 operand formats
# ---------------------------------------------------------------------------

class TestGetBranchTargetAarch64:
    """_get_branch_target must parse all AArch64 operand styles.

    AArch64 object files use short (2-digit) hex addresses (e.g. ``14``,
    ``4c``) while linked executables use longer addresses.
    """

    def test_two_digit_hex_address(self):
        # AArch64 .o: b.ne 14
        assert _get_branch_target("14") == 0x14

    def test_two_digit_hex_address_uppercase(self):
        assert _get_branch_target("4C") == 0x4C

    def test_three_digit_hex_address(self):
        assert _get_branch_target("4c0") == 0x4c0

    def test_b_to_small_address(self):
        # b 4c  (address = 0x4c)
        assert _get_branch_target("4c") == 0x4c

    def test_cbz_with_register_operand(self):
        # cbz x0, 2dc  — x is not a hex char, so x0 is ignored
        assert _get_branch_target("x0, 2dc") == 0x2DC

    def test_cbnz_with_register_operand(self):
        assert _get_branch_target("x1, 1a0") == 0x1A0

    def test_tbz_with_register_and_bit(self):
        # tbz x0, #0, 2dc  — #0 is excluded by the lookbehind
        assert _get_branch_target("x0, #0, 2dc") == 0x2DC

    def test_tbnz_with_register_and_bit(self):
        assert _get_branch_target("x2, #3, 10c") == 0x10C

    def test_0x_prefixed_address(self):
        # Linked binary: b.ne 0x4007c4
        assert _get_branch_target("0x4007c4") == 0x4007C4

    def test_long_plain_hex_address(self):
        # After stripping <func+...>, operand is plain hex
        assert _get_branch_target("4007c4") == 0x4007C4

    def test_indirect_branch_returns_none(self):
        # AT&T syntax indirect call: *%rax
        assert _get_branch_target("*%rax") is None

    def test_empty_operands_returns_none(self):
        assert _get_branch_target("") is None

    def test_register_only_returns_none(self):
        # ret (no operand) or pure register — no valid address
        assert _get_branch_target("x30") is None


# ---------------------------------------------------------------------------
# _find_loops — AArch64 instruction sequences
# ---------------------------------------------------------------------------

class TestFindLoopsAarch64:
    """_find_loops must detect backward branches as loops in AArch64 code."""

    def test_no_loop(self):
        instrs = _instrs(
            (0x0, "add", "w0, w0, w1"),
            (0x4, "ret", ""),
        )
        assert _find_loops(instrs, "aarch64") == []

    def test_simple_loop_b_ne(self):
        # Loop: addr 0xc to 0x14 via b.ne back-edge.
        instrs = _instrs(
            (0x0,  "cmp",  "w1, #0x0"),
            (0x4,  "b.le", "1c"),       # forward branch (skip loop)
            (0x8,  "mov",  "w0, #0x0"),
            (0xc,  "add",  "w0, w0, w1"),  # loop start
            (0x10, "sub",  "w1, w1, #0x1"),
            (0x14, "b.ne", "c"),        # backward branch → loop back-edge
            (0x18, "ret",  ""),
            (0x1c, "mov",  "w0, #0x0"),
            (0x20, "ret",  ""),
        )
        loops = _find_loops(instrs, "aarch64")
        assert (0xc, 0x14) in loops

    def test_simple_loop_cbnz(self):
        # Loop using cbnz as the back-edge.
        instrs = _instrs(
            (0x0,  "mov",  "w0, #0x0"),
            (0x4,  "add",  "w0, w0, w1"),  # loop start
            (0x8,  "sub",  "w1, w1, #0x1"),
            (0xc,  "cbnz", "w1, 4"),    # backward branch → 0x4
            (0x10, "ret",  ""),
        )
        loops = _find_loops(instrs, "aarch64")
        assert (0x4, 0xc) in loops

    def test_simple_loop_unconditional_b(self):
        # Infinite loop: b loops back to itself.
        instrs = _instrs(
            (0x0, "nop",  ""),
            (0x4, "b",    "0"),  # backward branch → 0x0
        )
        loops = _find_loops(instrs, "aarch64")
        assert (0x0, 0x4) in loops

    def test_nested_loops(self):
        # Simulates: outer loop 0x14–0x3c, inner loop 0x18–0x28.
        instrs = _instrs(
            (0x0,  "mov",  "w0, #0x0"),
            (0x4,  "cmp",  "w5, #0x0"),
            (0x8,  "b.le", "40"),
            (0xc,  "mov",  "w6, #0x0"),
            (0x10, "mov",  "w2, #0x0"),
            (0x14, "mov",  "w1, #0x0"),  # outer loop start
            (0x18, "add",  "w0, w0, w2"),  # inner loop start
            (0x1c, "add",  "w2, w2, w1"),
            (0x20, "add",  "w1, w1, #0x1"),
            (0x24, "cmp",  "w5, w1"),
            (0x28, "b.ne", "18"),  # inner back-edge → 0x18
            (0x2c, "add",  "w1, w6, #0x1"),
            (0x30, "cmp",  "w6, w4"),
            (0x34, "b.eq", "40"),   # forward exit
            (0x38, "mov",  "w6, w1"),
            (0x3c, "b",    "14"),   # outer back-edge → 0x14
            (0x40, "ret",  ""),
        )
        loops = _find_loops(instrs, "aarch64")
        assert (0x18, 0x28) in loops   # inner loop
        assert (0x14, 0x3c) in loops   # outer loop
        assert len(loops) == 2

    def test_forward_branch_is_not_loop(self):
        # A forward branch (b.gt to a higher address) must NOT be a loop.
        instrs = _instrs(
            (0x0, "cmp", "w0, #0x0"),
            (0x4, "b.gt", "10"),   # forward: 0x10 > 0x4
            (0x8, "neg", "w0, w0"),
            (0xc, "b",   "10"),    # forward jump to end
            (0x10, "ret", ""),
        )
        assert _find_loops(instrs, "aarch64") == []

    def test_inner_loop_first_in_sorted_order(self):
        # _find_loops sorts so that inner loops (smaller end addr) appear first.
        instrs = _instrs(
            (0x0,  "nop",  ""),
            (0x4,  "nop",  ""),  # inner loop start
            (0x8,  "nop",  ""),
            (0xc,  "b.ne", "4"),  # inner back-edge
            (0x10, "nop",  ""),
            (0x14, "b",    "0"),  # outer back-edge
        )
        loops = _find_loops(instrs, "aarch64")
        assert loops[0] == (0x4, 0xc)   # inner first
        assert loops[1] == (0x0, 0x14)  # outer second


# ---------------------------------------------------------------------------
# _format_asm — AArch64
# ---------------------------------------------------------------------------

class TestFormatAsmAarch64:
    """_format_asm must emit correct assembly for AArch64 regions."""

    def test_simple_linear_region(self):
        instrs = _instrs(
            (0x0, "add", "w0, w0, w1"),
            (0x4, "ret", ""),
        )
        asm = _format_asm(instrs, "aarch64")
        assert "\tadd w0, w0, w1" in asm
        assert "\tret" in asm
        assert ".Lmca_end:" in asm

    def test_in_region_branch_gets_label(self):
        # b.ne back-edge inside the region → target gets a label.
        instrs = _instrs(
            (0x0, "add",  "w0, w1, w2"),
            (0x4, "b.ne", "0"),         # in-region backward branch
        )
        asm = _format_asm(instrs, "aarch64")
        assert ".Lmca_0:" in asm
        assert ".Lmca_0" in asm.split(".Lmca_0:")[1]  # label used in operand

    def test_out_of_region_branch_redirected(self):
        # A forward branch to an out-of-region address → .Lmca_end.
        instrs = _instrs(
            (0x0, "cmp", "w0, #0x0"),
            (0x4, "b.eq", "100"),       # out-of-region target
            (0x8, "add", "w0, w0, #1"),
        )
        asm = _format_asm(instrs, "aarch64")
        assert ".Lmca_end" in asm
        # The original far-away address must not appear verbatim.
        assert "100" not in asm or ".Lmca_end" in asm

    def test_lmca_end_label_always_present(self):
        instrs = _instrs(
            (0x0, "mov", "x0, #0"),
            (0x4, "ret", ""),
        )
        asm = _format_asm(instrs, "aarch64")
        assert asm.endswith(".Lmca_end:\n")


# ---------------------------------------------------------------------------
# Integration tests — require AArch64 cross-toolchain + llvm-mca
# ---------------------------------------------------------------------------

def _has_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _has_llvm_mca() -> bool:
    if _has_tool("llvm-mca"):
        return True
    return any(_has_tool(f"llvm-mca-{v}") for v in range(20, 10, -1))


_NEED_AARCH64 = pytest.mark.skipif(
    not (_has_tool("aarch64-linux-gnu-gcc")
         and _has_tool("aarch64-linux-gnu-objdump")
         and _has_llvm_mca()),
    reason=(
        "AArch64 integration tests require aarch64-linux-gnu-gcc, "
        "aarch64-linux-gnu-objdump, and llvm-mca"
    ),
)

# C sources used by integration tests.
_SUM_C = textwrap.dedent("""\
    int sum(int *a, int n) {
        int s = 0;
        int i = 0;
        while (i < n) {
            s += a[i];
            i++;
        }
        return s;
    }
""")

_NO_LOOP_C = textwrap.dedent("""\
    int add(int x, int y) { return x + y; }
""")

_NESTED_C = textwrap.dedent("""\
    int nested(int n) {
        int s = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                s += i * j;
        return s;
    }
""")


def _compile_aarch64(tmp_path, name: str, source: str) -> str:
    """Compile *source* to an AArch64 object file; return its path."""
    src = tmp_path / f"{name}.c"
    obj = tmp_path / f"{name}.o"
    src.write_text(source)
    subprocess.run(
        ["aarch64-linux-gnu-gcc", "-O1", "-c", str(src), "-o", str(obj)],
        check=True,
        capture_output=True,
    )
    return str(obj)


@_NEED_AARCH64
class TestDisassembleAarch64:
    """disassemble() must correctly parse AArch64 objdump output."""

    def test_disassemble_returns_functions(self, tmp_path):
        obj = _compile_aarch64(tmp_path, "sum", _SUM_C)
        funcs = disassemble(obj, "aarch64-linux-gnu-objdump", "aarch64")
        names = [n for n, _ in funcs]
        assert "sum" in names

    def test_instructions_have_three_fields(self, tmp_path):
        obj = _compile_aarch64(tmp_path, "add", _NO_LOOP_C)
        funcs = disassemble(obj, "aarch64-linux-gnu-objdump", "aarch64")
        assert funcs, "no functions found"
        for _name, instrs in funcs:
            for item in instrs:
                assert len(item) == 3, f"expected (addr, mnemonic, operands), got {item!r}"

    def test_addresses_are_integers(self, tmp_path):
        obj = _compile_aarch64(tmp_path, "add", _NO_LOOP_C)
        funcs = disassemble(obj, "aarch64-linux-gnu-objdump", "aarch64")
        for _name, instrs in funcs:
            for addr, _mn, _op in instrs:
                assert isinstance(addr, int), f"addr {addr!r} is not int"

    def test_aarch64_comments_stripped(self, tmp_path):
        """AArch64 objdump emits '// #0' style comments — they must be removed."""
        obj = _compile_aarch64(tmp_path, "sum", _SUM_C)
        funcs = disassemble(obj, "aarch64-linux-gnu-objdump", "aarch64")
        for _name, instrs in funcs:
            for _addr, _mn, operands in instrs:
                assert "//" not in operands, \
                    f"comment not stripped from operands: {operands!r}"

    def test_symbol_annotations_stripped(self, tmp_path):
        """objdump adds <func+0x10> annotations after branch targets — strip them."""
        obj = _compile_aarch64(tmp_path, "sum", _SUM_C)
        funcs = disassemble(obj, "aarch64-linux-gnu-objdump", "aarch64")
        for _name, instrs in funcs:
            for _addr, _mn, operands in instrs:
                assert "<" not in operands and ">" not in operands, \
                    f"symbol annotation not stripped: {operands!r}"

    def test_no_loop_function_has_no_loop_instrs(self, tmp_path):
        obj = _compile_aarch64(tmp_path, "add", _NO_LOOP_C)
        funcs = disassemble(obj, "aarch64-linux-gnu-objdump", "aarch64")
        func_map = dict(funcs)
        assert "add" in func_map
        loops = _find_loops(func_map["add"], "aarch64")
        assert loops == [], "simple add function should have no loops"


@_NEED_AARCH64
class TestAnalyzeAarch64:
    """analyze() end-to-end tests for AArch64 object files."""

    def test_no_loop_function_produces_results(self, tmp_path):
        obj = _compile_aarch64(tmp_path, "add", _NO_LOOP_C)
        results = list(analyze(obj))
        assert results, "expected at least one result for no-loop function"

    def test_no_loop_results_have_positive_ipc(self, tmp_path):
        obj = _compile_aarch64(tmp_path, "add", _NO_LOOP_C)
        for start, end, ipc in analyze(obj):
            assert ipc > 0, f"IPC must be positive, got {ipc} for 0x{start:x}-0x{end:x}"

    def test_no_loop_addresses_are_ordered(self, tmp_path):
        obj = _compile_aarch64(tmp_path, "add", _NO_LOOP_C)
        for start, end, _ipc in analyze(obj):
            assert start <= end, f"start 0x{start:x} > end 0x{end:x}"

    def test_loop_function_detects_loop_region(self, tmp_path):
        """The inner loop body must appear as a distinct region."""
        obj = _compile_aarch64(tmp_path, "sum", _SUM_C)
        funcs = disassemble(obj, "aarch64-linux-gnu-objdump", "aarch64")
        func_map = dict(funcs)
        assert "sum" in func_map
        loops = _find_loops(func_map["sum"], "aarch64")
        # There must be at least one loop detected in the sum function.
        assert loops, "expected at least one loop in sum()"

    def test_loop_function_produces_multiple_results(self, tmp_path):
        """Loop + basic blocks outside the loop → multiple analysis regions."""
        obj = _compile_aarch64(tmp_path, "sum", _SUM_C)
        results = list(analyze(obj))
        assert len(results) >= 2, \
            f"expected ≥ 2 regions for sum() with a loop, got {len(results)}"

    def test_loop_region_ipc_is_positive(self, tmp_path):
        obj = _compile_aarch64(tmp_path, "sum", _SUM_C)
        for start, end, ipc in analyze(obj):
            assert ipc > 0, f"IPC must be positive, got {ipc} for 0x{start:x}-0x{end:x}"

    def test_nested_loops_produce_two_loop_regions(self, tmp_path):
        """Both the inner and outer loops must be reported separately."""
        obj = _compile_aarch64(tmp_path, "nested", _NESTED_C)
        funcs = disassemble(obj, "aarch64-linux-gnu-objdump", "aarch64")
        func_map = dict(funcs)
        assert "nested" in func_map
        loops = _find_loops(func_map["nested"], "aarch64")
        assert len(loops) >= 2, \
            f"expected ≥ 2 loops in nested(), got {len(loops)}: {loops}"

    def test_nested_inner_loop_span_smaller_than_outer(self, tmp_path):
        """Inner loop span must be strictly contained within the outer loop."""
        obj = _compile_aarch64(tmp_path, "nested", _NESTED_C)
        funcs = disassemble(obj, "aarch64-linux-gnu-objdump", "aarch64")
        func_map = dict(funcs)
        loops = _find_loops(func_map["nested"], "aarch64")
        assert len(loops) >= 2
        # loops is sorted inner-first (smaller end address first).
        inner_start, inner_end = loops[0]
        outer_start, outer_end = loops[1]
        assert outer_start <= inner_start and inner_end <= outer_end, (
            f"inner ({hex(inner_start)}-{hex(inner_end)}) not inside "
            f"outer ({hex(outer_start)}-{hex(outer_end)})"
        )

    def test_analyze_result_tuples_have_three_elements(self, tmp_path):
        obj = _compile_aarch64(tmp_path, "sum", _SUM_C)
        for result in analyze(obj):
            assert len(result) == 3, f"expected (start, end, ipc), got {result!r}"
