#!/usr/bin/env python3
"""automatic-llvm-mca: Estimate throughput for ELF binaries using llvm-mca.


Procedure:
  1. Disassemble the ELF binary with objdump.
  2. Decompose each function into loops and (for non-loop code) basic blocks.
  3. Run llvm-mca on each region to obtain retired instructions, elapsed cycles,
     and the proportion of load instructions (those with the MayLoad attribute).
  4. Print a CSV with start address, end address, retired instructions, elapsed
     cycles, and proportion of load instructions for every region.

For nested loops the outer loop (including the inner loop body) and the inner
loop are reported separately:
  0x2,0x6,200,103,0.2500
  0x0,0x8,960,800,0.1250

Supported architectures: x86/x86-64, AArch64, 32-bit ARM, RISC-V (RV32IC, RV64IC).

Usage:
  python3 analyze.py [--mcpu <cpu>] <elf-binary>
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys

from arch import (  # noqa: F401  (re-exported for backward compatibility)
    _RISCV_BRANCHES,
    _AARCH64_BRANCHES,
    _ARM_CONDS,
    _ARM_BRANCH_BASES,
    _ARM_BRANCHES,
    _ARM_CALLS,
    _parse_branch_target_candidates,
    ArchBase,
    X86Arch,
    AArch64Arch,
    ARMArch,
    RISCVArch,
    _find_cross_tool,
    _arch_from_platform,
    _detect_arch,
)


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

    # Apply architecture-specific operand formatting for LLVM-MCA
    new_operands = arch.format_operands_for_mca(mnemonic, new_operands)
    suffix = f" {new_operands}" if new_operands else ""
    return f"\t{mnemonic}{suffix}"

# Number of times the assembly block is repeated in stochastic cache-miss mode.
_CACHE_MISS_REPEAT = 100


def _format_asm(instrs, arch: ArchBase) -> tuple[str, list]:
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

    Returns a ``(asm, [])`` pair — no extra llvm-mca arguments are needed
    for plain formatting.
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
            # Apply architecture-specific operand formatting for LLVM-MCA
            formatted_operands = arch.format_operands_for_mca(mnemonic, operands)
            tail = f" {formatted_operands}" if formatted_operands else ""
            lines.append(f"\t{mnemonic}{tail}")

    lines.append(".Lmca_end:")
    return "\n".join(lines) + "\n", []


def _format_asm_with_average_load_latency(instrs, arch: ArchBase,
                                          latency: int = 0) -> tuple[str, list]:
    """Format *instrs* as assembly where every load gets a fixed latency override.

    Every load instruction is wrapped with ``# LLVM-MCA-LATENCY <latency>``
    and ``# LLVM-MCA-LATENCY`` directives, giving all loads the specified
    latency uniformly.  Unlike :func:`_format_asm_with_cache_miss`, no
    repetition is applied and all loads are treated identically.

    This implements the *average* cache-miss mode where all loads take
    ``cache_latency * cache_miss`` cycles, as opposed to a fraction of loads
    taking the full *cache_latency*.

    Returns a ``(asm, [])`` pair — no extra llvm-mca arguments are needed
    for average-latency formatting.
    """
    addr_set = {a for a, _, _ in instrs}
    labeled = _compute_labeled_addrs(instrs, arch)

    lines = []
    for addr, mnemonic, operands in instrs:
        if addr in labeled:
            lines.append(f".Lmca_{addr:x}:")
        # Apply architecture-specific operand formatting for LLVM-MCA
        formatted_operands = arch.format_operands_for_mca(mnemonic, operands)
        tail = f" {formatted_operands}" if formatted_operands else ""
        if arch.is_load_instruction(mnemonic, operands):
            lines.append(f"# LLVM-MCA-LATENCY {latency}")
            lines.append(f"\t{mnemonic}{tail}")
            lines.append("# LLVM-MCA-LATENCY")
        elif arch.is_branch(mnemonic):
            lines.append(_format_branch_instr(mnemonic, operands, addr_set, arch))
        else:
            lines.append(f"\t{mnemonic}{tail}")

    lines.append(".Lmca_end:")
    return "\n".join(lines) + "\n", []


def _format_asm_with_cache_miss(instrs, arch: ArchBase,
                                 cache_miss: float = 0.0,
                                 cache_latency: int = 0,
                                 front_loaded: bool = False) -> tuple[str, list]:
    """Format *instrs* as assembly for llvm-mca with cache-miss simulation.

    *cache_miss* is the average number of cache misses per load instruction
    and may be any non-negative value, including values greater than 1.

    The code block is repeated ``_CACHE_MISS_REPEAT`` times.  The penalty is
    modelled in two parts:

    * **Base penalty** — every load receives
      ``floor(cache_miss) * cache_latency`` cycles unconditionally, via an
      ``# LLVM-MCA-LATENCY`` override.

    * **Fractional adjustment** — let ``frac = cache_miss - floor(cache_miss)``
      (in the range ``[0, 1)``).  A fraction ``frac`` of load instances across
      the repeated block receives an additional ``+cache_latency`` penalty.
      When *front_loaded* is ``False`` (default), penalties are placed
      deterministically using an evenly-spaced (Bresenham) formula.
      When *front_loaded* is ``True``, the first ``a`` loads receive the
      penalty and the remaining loads do not.

    When ``cache_miss < 1`` the base penalty is zero and only the fractional
    adjustment operates (identical to the original behaviour for the default
    placement).

    Labels are made unique per repetition so that backward branches within a
    loop still resolve to the correct iteration-local target.

    Returns ``(asm, ["-iterations=1"])`` when the block is repeated (loads
    are present), or ``(asm, [])`` via :func:`_format_asm` when there are no
    load instructions in *instrs*.
    """
    n = sum(
        1 for _, mn, ops in instrs if arch.is_load_instruction(mn, ops)
    )
    if n == 0:
        return _format_asm(instrs, arch)

    addr_set = {a for a, _, _ in instrs}
    labeled = _compute_labeled_addrs(instrs, arch)

    # Split cache_miss into an integer floor and a non-negative fractional part.
    base_misses = int(cache_miss)                     # floor for non-negative cache_miss
    frac = cache_miss - base_misses                   # always in [0, 1)
    base_latency = base_misses * cache_latency        # guaranteed latency per load

    # n: load instructions per repetition; b: total generated load count.
    # a: number of loads that receive the +cache_latency fractional adjustment.
    # Adjustment positions are floor(m * b / a) for m in 0..a-1 (stochastic),
    # or 0..a-1 (front-loaded).
    b = _CACHE_MISS_REPEAT * n
    a = round(frac * b) if b > 0 else 0
    load_counter = 0
    miss_counter = 0
    next_miss_position = 0

    def _emit_load(mnemonic: str, operands: str, lines: list) -> None:
        """Append a load instruction with the appropriate latency directive."""
        nonlocal load_counter, miss_counter, next_miss_position
        # Apply architecture-specific operand formatting for LLVM-MCA
        formatted_operands = arch.format_operands_for_mca(mnemonic, operands)
        tail = f" {formatted_operands}" if formatted_operands else ""
        if front_loaded:
            is_miss = load_counter < a
        else:
            is_miss = miss_counter < a and load_counter == next_miss_position
        if is_miss:
            lat = base_latency + cache_latency
            miss_counter += 1
            if not front_loaded:
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
                # Apply architecture-specific operand formatting for LLVM-MCA
                formatted_operands = arch.format_operands_for_mca(mnemonic, operands)
                tail = f" {formatted_operands}" if formatted_operands else ""
                lines.append(f"\t{mnemonic}{tail}")

    lines.append(".Lmca_end:")
    return "\n".join(lines) + "\n", ["-iterations=1"]


# ---------------------------------------------------------------------------
# Cache-miss simulation mode classes
# ---------------------------------------------------------------------------

class _CacheMissMode:
    """Base class for cache-miss simulation strategies.

    Each subclass encapsulates the cache-miss parameters as instance
    variables and provides a ``format_asm`` method that produces the
    appropriate assembly for llvm-mca along with any extra llvm-mca flags
    required by that strategy.
    """

    def format_asm(self, instrs, arch: ArchBase) -> tuple[str, list]:
        """Format *instrs* as assembly with the cache-miss strategy applied.

        Returns a ``(asm, extra_args)`` pair where *asm* is the formatted
        assembly string and *extra_args* is a list of additional arguments to
        pass to llvm-mca (e.g. ``["-iterations=1"]``).
        """
        raise NotImplementedError


class _NoCacheMiss(_CacheMissMode):
    """No cache-miss simulation — plain assembly formatting."""

    def format_asm(self, instrs, arch: ArchBase) -> tuple[str, list]:
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

    def format_asm(self, instrs, arch: ArchBase) -> tuple[str, list]:
        num_instrs = len(instrs)
        num_loads = sum(
            1 for _, mn, ops in instrs if arch.is_load_instruction(mn, ops)
        )
        expected_misses = num_instrs / self.instructions_per_cache_miss
        miss_fraction = expected_misses / num_loads if num_loads > 0 else 0.0
        return _format_asm_with_cache_miss(instrs, arch, miss_fraction,
                                           self.cache_latency)


class _EarlyCacheMiss(_CacheMissMode):
    """Early (front-loaded) cache-miss simulation based on instructions-per-cache-miss.

    Like :class:`_StochasticCacheMiss`, the assembly is repeated
    ``_CACHE_MISS_REPEAT`` times with ``# LLVM-MCA-LATENCY`` directives
    inserted deterministically.  The difference is in *placement*: all cache
    misses occur on the **first** loads in the repeated block rather than being
    spread uniformly.

    For example, with 5 loads and 3 expected misses the pattern is::

        miss miss miss hit hit

    instead of the uniform stochastic pattern::

        miss hit miss hit miss

    ``-iterations=1`` is added to llvm-mca so that it does not add its own
    repetitions.
    """

    def __init__(self, instructions_per_cache_miss: float, cache_latency: int):
        self.instructions_per_cache_miss = instructions_per_cache_miss
        self.cache_latency = cache_latency

    def format_asm(self, instrs, arch: ArchBase) -> tuple[str, list]:
        num_instrs = len(instrs)
        num_loads = sum(
            1 for _, mn, ops in instrs if arch.is_load_instruction(mn, ops)
        )
        expected_misses = num_instrs / self.instructions_per_cache_miss
        miss_fraction = expected_misses / num_loads if num_loads > 0 else 0.0
        return _format_asm_with_cache_miss(instrs, arch, miss_fraction,
                                           self.cache_latency,
                                           front_loaded=True)


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

    def format_asm(self, instrs, arch: ArchBase) -> tuple[str, list]:
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
    if cache_miss_mode == "early":
        return _EarlyCacheMiss(instructions_per_cache_miss, cache_latency)
    return _StochasticCacheMiss(instructions_per_cache_miss, cache_latency)


class _StochasticCacheMissRate(_CacheMissMode):
    """Stochastic cache-miss simulation based on cache-miss rate.

    Like :class:`_StochasticCacheMiss`, but the miss rate is given directly
    as *cache_miss_rate* (cache misses per retired load instruction) rather
    than being derived from ``instructions_per_cache_miss``.  The rate may be
    greater than 1 (e.g. 1.5 means each load causes an average of 1.5 misses).

    The assembly is repeated ``_CACHE_MISS_REPEAT`` times with
    ``# LLVM-MCA-LATENCY`` directives distributed deterministically using the
    Bresenham-style formula in :func:`_format_asm_with_cache_miss`.

    ``-iterations=1`` is added to llvm-mca so that it does not add its own
    repetitions.
    """

    def __init__(self, cache_miss_rate: float, cache_latency: int):
        self.cache_miss_rate = cache_miss_rate
        self.cache_latency = cache_latency

    def format_asm(self, instrs, arch: ArchBase) -> tuple[str, list]:
        return _format_asm_with_cache_miss(instrs, arch, self.cache_miss_rate,
                                           self.cache_latency)


class _EarlyCacheMissRate(_CacheMissMode):
    """Early (front-loaded) cache-miss simulation based on cache-miss rate.

    Like :class:`_EarlyCacheMiss`, but the miss rate is given directly as
    *cache_miss_rate* (cache misses per retired load instruction).  All
    cache-miss penalties are placed on the **first** loads in the repeated
    block rather than distributed uniformly.

    ``-iterations=1`` is added to llvm-mca so that it does not add its own
    repetitions.
    """

    def __init__(self, cache_miss_rate: float, cache_latency: int):
        self.cache_miss_rate = cache_miss_rate
        self.cache_latency = cache_latency

    def format_asm(self, instrs, arch: ArchBase) -> tuple[str, list]:
        return _format_asm_with_cache_miss(instrs, arch, self.cache_miss_rate,
                                           self.cache_latency,
                                           front_loaded=True)


class _AverageCacheMissRate(_CacheMissMode):
    """Average cache-miss simulation based on cache-miss rate.

    Like :class:`_AverageCacheMiss`, but the miss rate is given directly as
    *cache_miss_rate* (cache misses per retired load instruction).  Every load
    receives ``round(cache_miss_rate * cache_latency)`` cycles via an
    ``# LLVM-MCA-LATENCY`` directive.
    """

    def __init__(self, cache_miss_rate: float, cache_latency: int):
        self.cache_miss_rate = cache_miss_rate
        self.cache_latency = cache_latency

    def format_asm(self, instrs, arch: ArchBase) -> tuple[str, list]:
        num_loads = sum(
            1 for _, mn, ops in instrs if arch.is_load_instruction(mn, ops)
        )
        if num_loads == 0:
            return _format_asm(instrs, arch)
        avg_latency = round(self.cache_miss_rate * self.cache_latency)
        return _format_asm_with_average_load_latency(instrs, arch, avg_latency)


def _build_cache_mode_from_rate(cache_miss_rate: float, cache_latency: int,
                                cache_miss_mode: str) -> _CacheMissMode:
    """Build and return the appropriate rate-based :class:`_CacheMissMode` instance.

    *cache_miss_rate* is the number of cache misses per retired load
    instruction.  It must be positive and finite for a cache-miss simulation
    to be applied.  ``float('inf')``, zero, or negative values all result in
    :class:`_NoCacheMiss` (no simulation).
    """
    if math.isinf(cache_miss_rate) or cache_miss_rate <= 0:
        return _NoCacheMiss()
    if cache_miss_mode == "average":
        return _AverageCacheMissRate(cache_miss_rate, cache_latency)
    if cache_miss_mode == "early":
        return _EarlyCacheMissRate(cache_miss_rate, cache_latency)
    return _StochasticCacheMissRate(cache_miss_rate, cache_latency)


class Dumper(_CacheMissMode):
    """A :class:`_CacheMissMode` wrapper that writes formatted assembly to disk.

    ``Dumper`` delegates all formatting logic to an inner :class:`_CacheMissMode`
    instance and, after formatting, writes the result to a text file inside
    *dump_dir*.  Each region produces one file named
    ``{start_address}_{end_address}.{arch}.txt``.

    Parameters
    ----------
    inner:
        The underlying :class:`_CacheMissMode` instance that performs the
        actual assembly formatting (e.g. :class:`_NoCacheMiss`,
        :class:`_StochasticCacheMiss`, :class:`_AverageCacheMiss`).
    dump_dir:
        Directory where formatted assembly files are written.  Created on
        first use if it does not already exist.  Defaults to ``"dump"``.
    """

    def __init__(self, inner: _CacheMissMode, dump_dir: str = "dump"):
        self._inner = inner
        self._dump_dir = dump_dir

    def format_asm(self, instrs, arch: ArchBase) -> tuple[str, list]:
        """Format *instrs* by delegating to the inner mode and write to disk.

        The formatted assembly is written to
        ``{dump_dir}/{start:x}_{end:x}.{arch.name}.txt`` as a side effect,
        where *start* and *end* are the addresses of the first and last
        instructions in *instrs*.  The formatted string is also returned so
        that the caller (typically :func:`_run_mca`) can pass it to llvm-mca.

        Nothing is written when *instrs* is empty.
        """
        result, extra = self._inner.format_asm(instrs, arch)
        if instrs:
            os.makedirs(self._dump_dir, exist_ok=True)
            start = instrs[0][0]
            end = instrs[-1][0]
            filename = f"{start:x}_{end:x}.{arch.name}.txt"
            path = os.path.join(self._dump_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(result)
        return result, extra


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


def _count_load_proportion(instrs, arch: ArchBase) -> float:
    """Return the proportion of *instrs* that are load instructions.

    Uses :meth:`ArchBase.is_load_instruction` to identify loads, so the result
    is based on the disassembled instruction text rather than llvm-mca metadata.
    Returns 0.0 when *instrs* is empty.
    """
    if not instrs:
        return 0.0
    loads = sum(1 for _, mn, ops in instrs if arch.is_load_instruction(mn, ops))
    return loads / len(instrs)


def _run_mca(instrs, mca_args=(), *, arch: ArchBase, cache_mode: _CacheMissMode):
    """Run llvm-mca on *instrs* and return ``(retired, cycles, load_proportion)``, or None.

    Parameters
    ----------
    instrs:
        List of ``(addr, mnemonic, operands)`` triples to analyse.
    mca_args:
        Extra arguments forwarded to llvm-mca (e.g. ``-march=``, ``-mcpu=``).
    arch:
        Architecture object.
    cache_mode:
        Cache-miss simulation strategy.
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

    asm, extra = cache_mode.format_asm(instrs, arch)

    # Pass assembly via stdin (llvm-mca reads from stdin when given "-").
    cmd = [_LLVM_MCA, *mca_args, *extra, "-"]
    proc = subprocess.run(cmd, input=asm, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    m_retired = re.search(r"\bInstructions:\s+(\d+)", proc.stdout)
    m_cycles = re.search(r"\bTotal Cycles:\s+(\d+)", proc.stdout)
    if m_retired is None or m_cycles is None:
        return None
    retired = int(m_retired.group(1))
    cycles = int(m_cycles.group(1))
    load_proportion = _count_load_proportion(instrs, arch)
    return retired, cycles, load_proportion


def _yield_mca_result(instrs, mca_args, arch: ArchBase,
                      cache_mode: _CacheMissMode):
    """Run llvm-mca on *instrs* and yield a ``(start, end, retired, cycles, load_proportion)`` tuple.

    Nothing is yielded when *instrs* is empty or llvm-mca returns no result.
    """
    if instrs:
        result = _run_mca(instrs, mca_args, arch=arch, cache_mode=cache_mode)
        if result is not None:
            retired, cycles, load_proportion = result
            yield instrs[0][0], instrs[-1][0], retired, cycles, load_proportion


# ---------------------------------------------------------------------------
# Per-function analysis
# ---------------------------------------------------------------------------

def _analyze_function(instrs, mca_args=(), arch: ArchBase = None,
                      cache_mode: _CacheMissMode = None):
    """Analyse one function's instructions.

    Yields ``(start_addr, end_addr, retired, cycles, load_proportion, kind)`` tuples for
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
        for start, end, retired, cycles, lp in _yield_mca_result(region, mca_args, arch,
                                                                  cache_mode):
            yield start, end, retired, cycles, lp, "loop"

    # --- Basic blocks outside loops ---
    non_loop = [(a, m, o) for a, m, o in instrs if not _in_any_loop(a, loops)]
    bb = []
    for instr in non_loop:
        addr, mnemonic, operands = instr
        bb.append(instr)
        if arch.ends_basic_block(mnemonic, operands):
            for start, end, retired, cycles, lp in _yield_mca_result(bb, mca_args, arch,
                                                                      cache_mode):
                yield start, end, retired, cycles, lp, "block"
            bb = []
    for start, end, retired, cycles, lp in _yield_mca_result(bb, mca_args, arch,
                                                              cache_mode):
        yield start, end, retired, cycles, lp, "block"


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------

def analyze(binary: str, mcpu: str = "",
            cache_miss: float = 0.0, cache_latency: int = 0,
            cache_miss_mode: str = "stochastic",
            cache_miss_rate: float = 0.0,
            dump: bool = False):
    """Analyse *binary* and yield ``(start, end, retired, cycles, load_proportion)`` tuples.

    Parameters
    ----------
    binary:
        Path to the ELF binary to analyse.
    mcpu:
        If non-empty, overrides the default ``-mcpu`` value chosen by
        ``_detect_arch`` and is forwarded to llvm-mca.
    cache_miss:
        Number of retired instructions per cache miss
        (``instructions_per_cache_miss``).  Use ``0`` (default) for
        no cache-miss simulation.  Interpretation depends on *cache_miss_mode*.
        Mutually exclusive with *cache_miss_rate*.
    cache_latency:
        Cache-miss penalty in cycles.  Only used when *cache_miss* or
        *cache_miss_rate* is finite.  Default is 0.
    cache_miss_mode:
        ``"stochastic"`` (default) — the code block is repeated 100 times with
        ``# LLVM-MCA-LATENCY`` directives distributed across load instructions
        according to the effective miss fraction; llvm-mca is run with
        ``-iterations=1``.

        ``"average"`` — every load instruction receives a fixed latency derived
        from the miss fraction multiplied by *cache_latency* cycles.

        ``"early"`` — like ``"stochastic"`` but all cache-miss penalties are
        placed on the first loads in the repeated block rather than distributed
        uniformly (e.g. ``miss miss miss hit hit`` instead of
        ``miss hit miss hit miss``).
    cache_miss_rate:
        Cache misses per retired load instruction.  Use ``0.0``
        (default) for no cache-miss simulation.  May be greater than 1.
        When non-zero, this takes precedence over *cache_miss*.
        Mutually exclusive with *cache_miss*.
    dump:
        When ``True``, the formatted assembly for each analysed region is
        written to a file in a ``"dump"`` directory.  Filename format:
        ``{start_address}_{end_address}.{arch}.txt``.
    """
    arch = _detect_arch(binary)
    mca_args = arch.mca_args
    if mcpu:
        mca_args = [a for a in mca_args if not a.startswith("-mcpu=")]
        mca_args = mca_args + [f"-mcpu={mcpu}"]
    if cache_miss_rate > 0.0:
        cache_mode = _build_cache_mode_from_rate(cache_miss_rate, cache_latency,
                                                 cache_miss_mode)
    else:
        cache_mode = _build_cache_mode(cache_miss, cache_latency, cache_miss_mode)
    if dump:
        cache_mode = Dumper(cache_mode)

    for _func_name, instrs in disassemble(binary, arch):
        for start, end, retired, cycles, lp, _kind in _analyze_function(
                instrs, mca_args, arch, cache_mode):
            yield start, end, retired, cycles, lp


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
            "when --instructions-per-cache-miss or --cache-miss-rate is finite."
        ),
    )
    parser.add_argument(
        "--cache-miss-mode",
        choices=["stochastic", "average", "early"],
        default="stochastic",
        dest="cache_miss_mode",
        help=(
            "Cache-miss simulation mode (default: stochastic). "
            "'stochastic': the code block is repeated 100 times and load "
            "instructions receive the full --cache-latency penalty according "
            "to the effective miss fraction; llvm-mca is run with "
            "-iterations=1. "
            "'average': all load instructions receive a fixed latency derived "
            "from the miss fraction multiplied by --cache-latency cycles. "
            "'early': like 'stochastic' but all cache misses are placed on "
            "the first loads in the repeated block rather than distributed "
            "uniformly (e.g. miss miss miss hit hit instead of "
            "miss hit miss hit miss)."
        ),
    )
    parser.add_argument(
        "--cache-miss-rate",
        type=float,
        default=0.0,
        metavar="R",
        dest="cache_miss_rate",
        help=(
            "Cache misses per retired load instruction (>0, default 0.0 "
            "meaning no cache-miss simulation). "
            "May be greater than 1 (e.g. 1.5 means each load causes an "
            "average of 1.5 misses). "
            "Cannot be combined with --instructions-per-cache-miss. "
            "Interpretation depends on --cache-miss-mode."
        ),
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        default=False,
        help=(
            "Write the formatted assembly for each analysed region to a text "
            "file inside a 'dump' directory.  Each file is named "
            "{start_address}_{end_address}.{arch}.txt."
        ),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        parser.error(f"{args.binary}: no such file")

    if args.cache_miss <= 0:
        parser.error("--instructions-per-cache-miss must be > 0")
    if args.cache_latency < 0:
        parser.error("--cache-latency must be >= 0")
    if args.cache_miss_rate < 0:
        parser.error("--cache-miss-rate must be >= 0")
    if not math.isinf(args.cache_miss) and args.cache_miss_rate > 0:
        parser.error(
            "--instructions-per-cache-miss and --cache-miss-rate are mutually exclusive"
        )

    results = sorted(
        analyze(
            binary=args.binary,
            mcpu=args.mcpu,
            cache_miss=args.cache_miss,
            cache_latency=args.cache_latency,
            cache_miss_mode=args.cache_miss_mode,
            dump=args.dump,
            cache_miss_rate=args.cache_miss_rate,
        ),
        key=lambda x: (x[1], -x[0]))
    print("start_address,end_address,retired_instructions,elapsed_cycles,load_proportion")
    for start, end, retired, elapsed_cycles, load_proportion in results:
        print(f"0x{start:x},0x{end:x},{retired},{elapsed_cycles},{load_proportion:.4f}")


if __name__ == "__main__":
    main()
