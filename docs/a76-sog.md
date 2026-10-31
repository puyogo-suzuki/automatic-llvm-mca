# Cortex-A76: the model against the Software Optimization Guide

> **Who wrote this, and how far to trust it.** This document was written on **2026-10-31** by
> **Claude** (Anthropic's Claude Opus model, running as a sub-agent of Claude Code), after reading
> the Cortex-A76 Software Optimization Guide and the contents of this repository's working tree on
> that day. **It is not an Arm document**, and it has not been reviewed by Arm. Every claim about
> the guide is tied to a section, a page and (where useful) a line of the text extraction, so that
> it can be checked; it may still contain errors of reading or transcription. **Check against the
> original guide before relying on any statement here.** Statements that could not be checked are
> marked "not verified".

Contents

1. [About this document](#1-about-this-document)
2. [Implementation overview](#2-implementation-overview)
3. [Compatible with the SOG (C)](#3-compatible-with-the-sog-c)
4. [Not compatible with the SOG (N)](#4-not-compatible-with-the-sog-n)
5. [Deliberate deviations and interpretations (D)](#5-deliberate-deviations-and-interpretations-d)
6. [Unclear in the SOG (U)](#6-unclear-in-the-sog-u)
7. [Inconsistent within the SOG (I)](#7-inconsistent-within-the-sog-i)
8. [Outside the A76 path: shared code (X)](#8-outside-the-a76-path-shared-code-x)
9. [Taken over from the in-order cores (B)](#9-taken-over-from-the-in-order-cores-b)
10. [Reproducing the cycle counts quoted here](#10-reproducing-the-cycle-counts-quoted-here)
11. [Maintaining this document](#11-maintaining-this-document)

| Category | Items |
|---|---|
| C, compatible | [C-1](#c-1) to [C-23](#c-23) |
| N, not compatible | none open |
| D, deliberate deviation or interpretation | [D-1](#d-1) to [D-9](#d-9) |
| U, unclear in the SOG | [U-1](#u-1) to [U-12](#u-12) |
| I, inconsistent within the SOG | [I-1](#i-1) to [I-8](#i-8) |
| X, outside the A76 path | [X-1](#x-1) to [X-2](#x-2) |
| B, taken over from the Cortex-A55 / Cortex-A520 (not SOG-based) | [B-1](#b-1) |

---

## 1. About this document

### 1.1 Source

* **Guide**: "Arm® Cortex®-A76 Software Optimization Guide", document number
  **PJDOC-466751330-7215**, **Version 10.0**, Copyright © 2019-2021 Arm Limited (or its
  affiliates), **Non-Confidential**, product status "Final" (title page and p.2-3). The guide prints
  no release date; the PDF's creation date is 2021-12-21. Called "the SOG" below. Page numbers are
  those of the guide's footers ("Page N of 46"); "l.N" is line N of the text extraction
  `tmp/a76sog.txt` (not tracked).

### 1.2 What "the implementation" is

The state of the working tree on 2026-10-31 (last commit `515f915`, plus uncommitted changes of
this and other sessions):

* `ModifiedTarget/AArch64/AArch64SchedNeoverseN1.td`: the TableGen model `NeoverseN1Model` (LLVM's
  Neoverse N1 model with local changes), compiled into the tool's own scheduling tables (see
  [docs.md](docs.md) section 5). `src/cpu/sched_model.cpp` (`installLocalSchedModel`, `case
  CpuModel::A76`) installs it **only** for `cortex-a76` and `cortex-a76ae` (`src/cpu/cpu_traits.cpp`).
  `neoverse-n1` and `cortex-a77`, which `AArch64Processors.td` also maps to `NeoverseN1Model`, have
  no local model in `cpu_traits.cpp` and keep libLLVM's stock model (checked: their
  `mca-insts-info` output is unchanged by this audit). The file was therefore edited in place; no
  separate A76 model was needed.
* `src/cpu/cpu_traits.cpp`: the A76 entry (`UopDispatchWidth = 8`, `MopDispatchWidth = 4`, no
  fusion table, `ZeroLatencyMov = false`, no N2 corrections, no MLP window resources).
* `src/facile/facile.cpp`: the Facile predictor, which produces the A76 cycle counts (the research
  runs `mca_tool --merge-same-header --dependency ooo --mcpu cortex-a76 --facile`; with `--facile`
  the MCA simulation is not run, the CSV `cycles` column is the rounded Facile estimate).
* `tests/a76_sog_test.cpp`: unit tests that pin the values below (added by this audit, not yet
  committed).

Implementation facts are given by file and identifier (TableGen `def` names, C++ function names),
not by line number. The model marks the changes of this audit with tags; [section 2.4](#24-code-tags-and-the-items-of-this-document)
maps them to the items. No C++ file was changed by this audit.

### 1.3 Categories

| ID | Category | Meaning |
|---|---|---|
| **C-n** | compatible | The implementation does what the SOG states. Items fixed by this audit are C items marked *fixed 2026-10-31*. |
| **N-n** | not compatible | The SOG states something and the implementation does not follow it. Each item says why it is still open. |
| **D-n** | deliberate deviation or interpretation | The implementation knowingly departs from the literal SOG, or fills a gap with an interpretation. Each item gives the SOG text, the implementation and the reason. |
| **U-n** | unclear | The SOG does not decide the question; the implementation makes an assumption or leaves it open. Each item gives the SOG text, what is unclear, the current handling, the expected impact, and how to settle it. |
| **I-n** | inconsistent | Two places of the SOG disagree, or the SOG contradicts itself or has a misprint. Each item says which reading the implementation follows. |
| **X-n** | outside the A76 path | Problems in shared code or in LLVM, found during this audit. Not changed. |
| **B-n** | taken over from another core | The A76 SOG says nothing on the question, and the treatment the project decided for the in-order cores (Cortex-A55, Cortex-A520) is applied to the A76 for consistency. **Not SOG-based.** |

### 1.4 Reading the SOG instruction tables

Sec. 3.1 (p.8, l.286-296):

* **Exec latency** is "the minimum latency seen by an operation dependent on an instruction in the
  described group". A number in parentheses is the accumulate latency of a forwarding path (notes of
  sec. 3.6, 3.11, 3.15, 3.16), except for "Multiply high" (5(3)), where it is a pipeline stall
  (sec. 3.6 note 5, l.500-501).
* **Execution throughput** is "the maximum throughput (in instructions per cycle) of the specified
  instruction group that can be achieved in the entirety of the Cortex-A76 microarchitecture". The
  model holds a resource of `n` units for `n / T` cycles in total ([D-2](#d-2)).
* **Utilized pipelines**, legend of sec. 3.2 (p.8, l.299-309, checked in the PDF): B branch, S
  integer single cycle 0/1, I "Integer single Cycle 0/1 and single/multicycle", M integer
  single/multicycle, **D "Integer single Cycle 1 and Integer multicycle"**, L load/store 0/1, V
  FP/ASIMD 0/1, V0, V1.
* Ranges ("5 to 12", "1/12 to 1/5") are data dependent; the model takes the best value
  ([D-1](#d-1)).
* The AArch32 tables are not compared (the research binaries are AArch64; [D-9](#d-9)).

---

## 2. Implementation overview

### 2.1 Machine model (`NeoverseN1Model`)

`IssueWidth = 4` (not used by Facile, whose widths come from `cpu_traits.cpp`), `MicroOpBufferSize
= 128` (the MLP window width, [U-1](#u-1)), `LoadLatency = 4`, `MispredictPenalty = 11`,
`LoopMicroOpBufferSize = 16` (neither used by the A76 path), `CompleteModel = 1`;
`UnsupportedFeatures` = PAuth, SME, SVE, MTE, CSSC, MOPS.

| Resource | Units | Role |
|---|---|---|
| `N1UnitB` | 1 | Branch |
| `N1UnitS0`, `N1UnitS1` | 1, 1 | Integer single cycle 0, 1 (split by this audit, [C-2](#c-2)) |
| `N1UnitM` | 1 | Integer single/multicycle |
| `N1UnitL` | 2 | Load/Store 0/1 |
| `N1UnitV0`, `N1UnitV1` | 1, 1 | FP/ASIMD 0, 1 |
| `N1UnitI` | group S0, S1, M | "I" |
| `N1UnitD` | group S1, M | "D", store data of integer stores (a group since this audit, [C-2](#c-2)) |
| `N1UnitV` | group V0, V1 | "V" |
| `N1UnitIDiv` | 1 | not a pipeline: the integer divider inside M ([D-3](#d-3)) |
| `N1UnitFDiv` | 1 | not a pipeline: the FP divide / square root unit inside V0 ([D-3](#d-3)) |

TableGen lists a use of a pipeline also under every group that contains it (a use of M appears as
M, I and D), so `mca-insts-info` and the tests show, for example, `[N1UnitD:1 N1UnitI:1 N1UnitL:1]`
for `str x1, [x2]`.

### 2.2 The Facile path (`src/facile/facile.cpp`)

For `--facile`, every region gets `EstimatedCycles = max(IssueBound, PortBound, PrecedenceBound)`:

* `calculateIssueBound`: `max(uops / 8, instructions / 4)` (uops = the model's `NumMicroOps`, one
  Mop per instruction; [C-1](#c-1), [U-2](#u-2)).
* `calculatePortUsageBound`: for every resource (pipelines and groups), the sum of the resource
  cycles divided by its units; the A78-only corrections (`n2_corrections.cpp`) do not apply.
* `calculatePrecedenceBound`: the maximum cycle ratio of the dependence graph. Register edges have
  the latency of the producing def minus the consumer's `ReadAdvance`, floored at 1 (`getEdgeLatency`,
  [C-5](#c-5)); a store with a base update has a second node for the update ([C-6](#c-6)); store ->
  load memory edges have the store's latency ([D-8](#d-8)).

The MLP columns use a window of `MicroOpBufferSize` = 128 uops ([U-1](#u-1)); a change of an
instruction's uop count therefore also changes the `mlp` / `mlp_r` columns of the regions around it.

### 2.3 What changed in this audit

The model before the audit is the working-tree file of 2026-10-31 (upstream LLVM's N1 model plus the
project's earlier changes: the multiply and FP multiply-accumulate forwarding, the CRC forwarding,
the best-case FP divide occupancy, `IssueWidth = 4`). This audit changed only the `.td` file (and
added the tests); every change is listed in the items.

### 2.4 Code tags and the items of this document

| Tag in `AArch64SchedNeoverseN1.td` | Items |
|---|---|
| `[a76_store_data_pipes]` | [C-2](#c-2), [C-12](#c-12) |
| `[a76_blr_lr]` | [C-7](#c-7), [X-2](#x-2) |
| `[a76_logic_flagset]` | [C-8](#c-8) |
| `[a76_div_best]` | [C-9](#c-9), [D-1](#d-1) |
| `[a76_dividers]` | [D-3](#d-3) |
| `[a76_ldp_hi]`, `[a76_reg_offset_h]` | [C-11](#c-11), [C-15](#c-15) |
| `[a76_store_rows]` | [C-12](#c-12), [C-16](#c-16), [I-4](#i-4) |
| `[a76_store_wb]` | [D-6](#d-6), [I-3](#i-3) |
| `[a76_fmul_fwd]` | [C-13](#c-13) |
| `[a76_fp_cvt]` | [C-14](#c-14), [C-18](#c-18) |
| `[a76_fp_cvt_scalar]` | [D-4](#d-4) |
| `[a76_stpq_post]` | [C-16](#c-16), [I-5](#i-5) |
| `[a76_asimd_forms]` | [C-17](#c-17) |
| `[a76_asimd_scalar]` | [D-5](#d-5) |
| `[a76_ls_throughput]` | [C-20](#c-20), [C-21](#c-21), [D-7](#d-7) |
| `[a76_from_a55_store_data]` | [B-1](#b-1) |

The rows without a tag are the upstream model's, with the SOG group names as comments.

---

## 3. Compatible with the SOG (C)

Values are "latency / throughput / pipelines" as printed. "Model" values were read with
`mca-insts-info --mtriple=aarch64 --mcpu=cortex-a76` and, for variant classes, with the unit tests
(`tests/a76_sog_test.cpp`, suite `A76SogTest`). Cycles per iteration are Facile estimates of small
loops that include the loop branch ([section 10](#10-reproducing-the-cycle-counts-quoted-here)).

### 3.1 Dispatch, pipelines and forwarding

<a id="c-1"></a>
**C-1. Four Mops and eight uops per cycle.**
SOG sec. 4.1 (p.41, l.2044-2045): "The dispatch stage can process up to 4 Mops per cycle and
dispatch up to 8 uops per cycle"; sec. 2.1 (p.7, l.224-225): instructions are decoded into Mops, "A
Mop can be split further into two Uops at dispatch stage". Implementation: `cpu_traits.cpp`
`UopDispatchWidth = 8`, `MopDispatchWidth = 4`; `calculateIssueBound` takes the larger of uops / 8
and instructions / 4. Measured: four `nop` + branch 1.25 (five Mops). Test:
`DispatchAndPipelines`. Instructions with more than two uops: [U-2](#u-2).

<a id="c-2"></a>
**C-2. Eight pipelines; store data on S1 and M.**
*Fixed 2026-10-31* (`[a76_store_data_pipes]`).
SOG sec. 2.1 (p.7, l.225-226 and the diagram): uops "issue out-of-order to one of eight execution
pipelines. Each execution pipeline can accept and complete one uop per cycle": Branch, Integer
Single-Cycle 0, 1, Integer Single/Multi-Cycle, FP/ASIMD 0, 1, Load/Store 0, 1. Legend (p.8,
l.305): D, the pipelines of the store data of integer stores, is "Integer single Cycle 1 and Integer
multicycle". Before: the model had two store-data pipelines of its own (`N1UnitD :
ProcResource<2>`), so integer stores took no integer-pipe bandwidth. Now S is split into S0 and S1,
I = {S0, S1, M}, D = {S1, M}. The SOG's own figures need this reading: "Store register, register
offset, scaled by 2 STRH 2 3/2 I, L, D" (l.769-770) is 3/2 per cycle exactly when the I uop and the
D uop share the three integer pipelines. Measured: `add; add; str; str` + branch 1.25 -> 1.33
cycles/iteration (now four integer-pipe uops on three pipelines). The per-pipeline dispatch limits
of sec. 4.1 (l.2047-2051: two uops per cycle for B, each V and each L; four for S; two for M) are
not modelled; none of them is below the issue rate of the pipelines concerned, so none can bind in
a steady state. Test: `Stores`, `DispatchAndPipelines`.

<a id="c-3"></a>
**C-3. NOP takes no execution resource.**
The SOG has no row for NOP or the other hints. Implementation (upstream): `WriteHint` has no
resource; a NOP still counts as one Mop and one uop. This is the decision made for the Cortex-A55
and Cortex-A520 ([a55-sog.md](a55-sog.md), [a520-sog.md](a520-sog.md) D-14), and the model already
followed it. The research binaries contain 348,224 NOPs (mostly padding; `mca_tool` skips all-NOP
regions). Test: `DispatchAndPipelines`.

<a id="c-4"></a>
**C-4. No zero-latency moves, no instruction fusion.**
The A76 SOG has no section on zero-latency moves or on instruction fusion (chapter 4, p.41-46; the
AES pairs of sec. 4.7 are a forwarding effect, [U-6](#u-6)). Implementation: `ZeroLatencyMov =
false`, `Fusion = FusionTable::None` for `cortex-a76`; MOV (`ORR` register) is 1 / 3 / I like any
logical instruction. Test: `DispatchAndPipelines`.

<a id="c-5"></a>
**C-5. Latency seen by dependent instructions.**
SOG sec. 3.1 (l.292-293): the latency is the one "seen by an operation dependent on an instruction".
Implementation: `getEdgeLatency` uses the latency of the def that the consumer reads (so the base
update of a post-index load has 1, not the load's 4) and subtracts the consumer's `ReadAdvance`
(the accumulator paths of [C-9](#c-9), [C-13](#c-13), [C-17](#c-17), [C-18](#c-18), [C-23](#c-23)),
with a floor of 1 cycle, which no A76 row goes below. The address operands have `ReadAdrBase`
advance 0: an ALU result is used as an address after its own latency, no extra cycle (the
decision of the Cortex-A55 and Cortex-A520 audits). Measured: `ldr x1, [x0], #8` chains through
x0 at 1 cycle.

<a id="c-6"></a>
**C-6. Stores are split into address and data uops; store-to-load forwarding.**
SOG sec. 3.10 (p.16, l.738-739): "Stores uops are split into address and data uops at dispatch
time"; sec. 4.6 (p.42, l.2120-2129): store-to-load forwarding needs the load to start at the start
or the middle of the older store. Implementation (shared Facile code, [docs.md](docs.md) sections
4.F and 4.G): the base update of a store is a node of its own that does not wait for the data, and
a store -> load edge is left out when the two register-offset accesses are provably disjoint. The
latency of the forwarding is not stated ([D-8](#d-8)).

### 3.2 Instruction tables, integer

<a id="c-7"></a>
**C-7. Sec. 3.3, branches (p.8, l.313-323).**

| SOG group | SOG | Model |
|---|---|---|
| B; BR, RET; CBZ, CBNZ, TBZ, TBNZ | 1 / 1 / B | `N1Write_1c_1B` |
| BL; BLR (reg != lr) | 1 / 1 / I, B | `N1Write_1c_1B_1I` (two uops) |
| **BLR (reg == lr)** | 2 / 1 / I, B | `N1WriteBLR`: `N1IsBlrLR` -> `N1Write_2c_1B_1I` (*fixed 2026-10-31*, `[a76_blr_lr]`; was 1) |

B.cond is not listed; it has `WriteBr` like B. The BLR x30 latency does not reach the predictions:
MCA's `InstrBuilder` gives every call latency 0 ([X-2](#x-2)). No `blr x30` occurs in the research
binaries. Test: `Branches`.

<a id="c-8"></a>
**C-8. Sec. 3.4, arithmetic and logical (p.9, l.348-362).**

| SOG group | SOG | Model |
|---|---|---|
| Arithmetic, basic: ADD{S}, ADC{S}, SUB{S}, SBC{S} | 1 / 3 / I | `WriteI` 1, I |
| Arithmetic, extend and shift | 2 / 1 / M | `WriteIEReg` 2, M ([U-4](#u-4)) |
| Arithmetic, LSL shift <= 4 | 1 / 3 / I | `N1WriteISReg`, `IsCheapLSL` -> 1, I |
| Arithmetic, LSR/ASR/ROR shift or LSL > 4 | 2 / 1 / M | `N1WriteISReg` otherwise -> 2, M |
| Conditional compare; conditional select | 1 / 3 / I | `WriteI` |
| Logical, basic: AND{S}, BIC{S}, EON, EOR, ORN, ORR | 1 / 3 / I | `WriteI`; **ANDS/BICS register with shift 0** `N1WriteLogicFlagset` -> 1, I (*fixed 2026-10-31*, `[a76_logic_flagset]`; was 2, M) |
| Logical, shift, no flagset | 1 / 3 / I | `N1Write_1c_1I` |
| Logical, shift, flagset: ANDS, BICS | 2 / 1 / M | `N1WriteLogicFlagset` (`RegShiftedPred`) -> 2, M |

ANDS/BICS: every register form, also the unshifted ones such as `tst x1, x2`, used to be 2 on M.
The research binaries contain 8,526 `tst` register and 427 `ands`/`bics` register without a shift
(19 with). Measured: `tst x1, x2; csel x1, x1, x3, eq` + branch 3.0 -> 2.0. Test:
`ArithmeticAndLogical`.

<a id="c-9"></a>
**C-9. Sec. 3.6, divide and multiply (p.10-12, l.425-501).**

| SOG group | SOG | Model |
|---|---|---|
| **Divide, W-form; X-form: SDIV, UDIV** | 5 to 12; 5 to 20 / 1/12 to 1/5; 1/20 to 1/5 / M (note 1) | `N1Write_5c5_1M_IDiv`: 5, M 1 cycle, divider 5 cycles (*fixed 2026-10-31*, `[a76_div_best]`, [D-1](#d-1), [D-3](#d-3); was 12 / 20 with M held 5) |
| Multiply accumulate, W-form: MADD, MSUB | 2(1) / 1 / M (note 2) | `N1Wr_IM32` 2 + `N1Rd_IMA` (advance 1) |
| Multiply accumulate, X-form | 4(3) / 1/3 / M (notes 2, 4) | `N1Wr_IM64` 4, M 3 cycles + `N1Rd_IMA` |
| Multiply accumulate long: SMADDL, SMSUBL, UMADDL, UMSUBL | 2(1) / 1 / M | `N1Wr_IM32` + `N1Rd_IMA` |
| Multiply high: SMULH, UMULH | 5(3) / 1/4 / M (note 5) | `N1Wr_MH` 5, M 4 cycles |

MUL, MNEG, SMULL, ... are the MADD family with the zero register and take the same rows. Note 4
("X-form multiply accumulates stall the multiplier pipeline for 2 extra cycles") and note 5 ("Multiply
high operations stall the multiplier pipeline for N extra cycles before any other type M uop can be
issued") are the M occupancies 3 and 4. The "(3)" of multiply high is that stall, not an accumulate
latency; the model's `N1Rd_MH` (advance 2) is attached to a fourth operand that SMULH does not have,
so it has no effect (checked: `smulh x1, x2, x1` and `smulh x1, x1, x2` both 5.0). Forwarding
between W, X and long forms: [U-5](#u-5). Measured: `sdiv w1, w1, w2` + branch 12.0 -> 5.0; `udiv
x1, x2, x3` + three `mul x` + branch 14.0 -> 10.0 (M-bound: 1 + 3 x 3). The research binaries contain
8,803 SDIV/UDIV and 461 SMULH/UMULH. Test: `DivideAndMultiply`.

<a id="c-10"></a>
**C-10. Sec. 3.8, miscellaneous data processing (p.13, l.548-569).**

| SOG group | SOG | Model |
|---|---|---|
| Address generation: ADR, ADRP | 1 / 3 / I | `WriteI` |
| Bitfield extract, one reg (ROR alias); two regs: EXTR | 1 / 3 / I; 3 / 1 / I, M | `N1WriteExtr`: `IsRORImmIdiomPred` -> 1, I; else `N1Write_3c_1I_1M` |
| Bitfield move, basic: SBFM, UBFM; insert: BFM | 1 / 3 / I; 2 / 1 / M | `WriteIS` 1, I; `N1Write_2c_1M` |
| Count leading; move immed; reverse; variable shift | 1 / 3 / I | `WriteI` / `WriteImm` / `WriteIS` 1, I |

Test: `MiscDataProcessing`.

<a id="c-11"></a>
**C-11. Sec. 3.9, loads (p.14-15, l.601-660).**

| SOG group | SOG | Model |
|---|---|---|
| Load register, literal, unscaled, unprivileged, unsigned immed; register offset basic, scale by 4/8, extend, extend scale by 4/8 (incl. PRFM) | 4 / 2 / L | `WriteLD` / `WriteLDIdx` 4, L |
| Load register, immed post-index / pre-index | 4 / 2 / L, I | `WriteAdr` (1, I) + `WriteLD` |
| **Load register, register offset, scale by 2; extend, scale by 2: LDRH, LDRSH** | 5 / 2 / I, L | `N1WriteLDRH`: `N1IsMemScaled` -> `N1Write_5c_1I_1L`, else 4, L (*fixed 2026-10-31*, `[a76_reg_offset_h]`; was 4, L) |
| Load pair, W-form: LDP, LDNP | 4 / 2 / L | `N1Write_4c_1L` + **`N1Write_4c_0Z`** (second register, *fixed*, `[a76_ldp_hi]`) |
| Load pair, X-form: LDP, LDNP | 4 / 1 / L | `WriteLD` + `WriteLDHi` (two L uops) |
| **Load pair, signed words: LDPSW (offset, pre, post)** | 5 / 1 / I, L | `N1Write_5c_1I_2L` (L held 2 cycles) + `N1Write_5c_0Z` (*fixed*; was throughput 2 and the second register 0) |
| Load pair, post-index / pre-index, normal: LDP | 4 / 1 / L, I | `WriteAdr` + `WriteLD` + `WriteLDHi` |

The second destination of LDP W and LDPSW used to be `N1Write_0c_0Z` (latency 0), which Facile
floors to 1: `ldp w1, w2, [x0]; add x0, x0, x2` + branch 2.0 -> 5.0 cycles/iteration. "Scaled" is the
S bit of the addressing mode (`AArch64_AM::getMemDoShift`), so `ldrh w1, [x0, x1]` and `ldrh w1,
[x0, w1, uxtw]` stay 4. Measured: `ldrh w1, [x0, x1, lsl #1]` + branch 4.0 -> 5.0; two LDPSW +
branch 1.0 -> 2.0. The research binaries contain 3,542 LDRH/LDRSH scaled by 2, 18,492 LDP W and 758
LDPSW. Acquire and exclusive loads: [U-7](#u-7). Test: `Loads`.

<a id="c-12"></a>
**C-12. Sec. 3.10, stores (p.16-17, l.737-787).**

| SOG group | SOG | Model |
|---|---|---|
| Store register: unscaled, unprivileged, unsigned immed; register offset basic, scaled 4/8, extend, extend scale 4/8 | 1 / 2 / L, D | `WriteST` / `WriteSTIdx` = `N1Write_1c_1L_1D` |
| Store register, immed post-index / pre-index | 1 / 2 / L, D | `N1Write_1c_STWB` (no pipeline, [D-6](#d-6)) + `WriteST` |
| **Store register, register offset, scaled by 2; extend, scale by 1 (sic, [I-4](#i-4)): STRH** | 2 / 3/2 / I, L, D | `N1WriteSTRH`: scaled -> `N1Write_2c_1I_1L_1D`, else `N1Write_1c_1L_1D` (*fixed 2026-10-31*, `[a76_store_rows]`; was 1, L, D) |
| Store pair, immed offset, W-form: STP, **STNP** | 1 / 2 / L, D | `N1Write_1c_1L_1D` (*fixed*: STNP W used to take two L uops, throughput 1) |
| Store pair, immed offset, X-form; post/pre-index W and X | 1 / 1 / L, D | `WriteSTP` = two L uops and D; pre/post with `N1Write_1c_STWB` |

D is S1 + M ([C-2](#c-2)), so an integer store now takes integer-pipe bandwidth. Measured: two
`str x, [x], #8` + branch 1.0 (unchanged; two per cycle on D and L); three `strh w1, [x0, xN, lsl
#1]` + branch 1.5 -> 2.0. The research binaries contain 2,015 STRH scaled by 2. The data operand
reads `ReadST` ([B-1](#b-1)). Release and exclusive stores: [U-7](#u-7). Test: `Stores`.

### 3.3 Instruction tables, FP and ASIMD

<a id="c-13"></a>
**C-13. Sec. 3.11, FP data processing (p.18-20, l.846-930).**

| SOG group | SOG | Model |
|---|---|---|
| FABS; FADD, FSUB; FMIN/FMAX(NM); FNEG; FCSEL | 2 / 2 / V | `WriteF` 2, V |
| FCMP{E}, FCCMP{E} | 2 / 1 / V0 | `WriteFCmp` 2, V0 |
| FDIV H; S; D | 7 / 4/7; 7 to 10 / 4/9 to 4/7; 7 to 15 / 1/7 to 2/7 / V0 (note 1) | `N1WriteFDS_7c2`, `_7c2`, `_7c4`: 7, V0 1 cycle, FP divide unit 2, 2, 4 cycles ([D-1](#d-1), [D-3](#d-3)) |
| FSQRT H; S; D | 7 / 4/7; 7 to 10 / 4/9 to 4/7; 7 to 17 / 1/8 to 2/7 / V0 | `N1WriteFDS_7c2`, `_7c2`, `_7c4` |
| FMUL, FNMUL | 3 / 2 / V (note 2) | **`N1Wr_FMul`** 3, V (*fixed 2026-10-31*, `[a76_fmul_fwd]`) |
| FMADD, FMSUB, FNMADD, FNMSUB | 4(2) / 2 / V (note 3) | `N1Wr_FMA` 4 + `N1Rd_FMA` (advance 2 from FMUL and FMA) |
| FRINT* | 3 / 1 / V0 | `N1Write_3c_1V0` |

Note 2 (l.925-927): an FP multiply-accumulate "can potentially be issued 1 cycle after the FP
multiply uop has been issued", i.e. FMUL -> accumulator 1. The advance `N1Rd_FMA` named `WriteFMul`,
which the model maps with `SchedAlias` to `N1Write_3c_1V`; a `ReadAdvance` does not see through an
alias, so the scalar FMUL -> FMADD accumulator edge kept 3 cycles (the vector forms, whose FMUL has
the explicit write `N1Wr_VFM`, were right). The scalar FMUL, FNMUL, FMULX now have `N1Wr_FMul`, named
in the advance. Measured: `fmul d0, d1, d2; fmadd d3, d4, d5, d0; fmov d1, d3` + branch 9.0 -> 7.0
(1 + 4 + 2); the A78, X1 and A720 models declare `WriteFMul` with `WriteRes` and give 7 already. The
research binaries contain 164,030 scalar FMUL/FNMUL and 49,675 FDIV/FSQRT. Test: `FpScalar`.

<a id="c-14"></a>
**C-14. Sec. 3.12, FP miscellaneous (p.20, l.933-945).**

| SOG group | SOG | Model |
|---|---|---|
| **FP convert, from vec to vec reg: FCVT, FCVTXN** | 3 / 1 / V0 | `N1Write_3c_1V0` (*fixed 2026-10-31*, `[a76_fp_cvt]`; FCVT was 3 on V, scalar FCVTXN 2 on V) |
| FP convert, from gen to vec reg: SCVTF, UCVTF | 6 / 1 / M, V0 | `N1Write_6c_1M_1V0` (integer and fixed-point) |
| FP convert, from vec to gen reg: FCVT[AMNPZ][SU] | 4 / 1 / V0, V1 | `N1Write_4c_1V0_1V1`, **also the fixed-point FCVTZS/FCVTZU** (*fixed*; they were 3 on V) |
| FP move, immed; register | 2 / 2 / V | `WriteFImm`; `N1Write_2c_1V` |
| FP transfer, gen to vec; vec to gen: FMOV | 3 / 1 / M; 2 / 1 / V1 | `N1Write_3c_1M`; `WriteFCopy` = `N1Write_2c_1V1` |

The research binaries contain 22,072 scalar FCVT and 63 fixed-point FCVTZ* to a GPR. Measured: two
`fcvt s, d` + branch 1.0 -> 2.0. Conversions with the operand and the result in FP registers:
[D-4](#d-4). Test: `FpScalar`.

<a id="c-15"></a>
**C-15. Sec. 3.13, FP loads (p.21-22, l.995-1045).**

| SOG group | SOG | Model |
|---|---|---|
| Load vector reg, literal | (blank) / 2 / L | 5, L ([I-7](#i-7)) |
| Load vector reg, unscaled immed | 5 / 2 / L | `N1Write_5c_1L` |
| Load vector reg, post/pre-index; unsigned immed | 5 / 2 / L, I | `WriteAdr` + `N1Write_5c_1L`; `N1Write_5c_1I_1L` ([U-11](#u-11)) |
| Load vector reg, register offset basic, extend; scale S/D; extend scale S/D | 5 / 2 / L, I | `N1Write_5c_1I_1L` |
| **Load vector reg, register offset, scale, H/Q-form; extend, scale, H/Q-form** | 6 / 2 / I, L | `N1WriteLDRHQ`: scaled -> 6, else 5 (*fixed 2026-10-31*, `[a76_reg_offset_h]`; every H/Q register-offset load was 6) |
| Load vector pair, immed offset, S/D-form | 5 / 1 / L, I | `N1Write_5c_1I_1L` + **`N1Write_5c_1L`** (*fixed*, `[a76_ldp_hi]`) |
| **Load vector pair, immed offset, Q-form: LDP, LDNP** | 7 / 1 / L | `N1Write_7c_1L` x 2 (*fixed*: there was an I uop, and LDNP Q, missed by the regular expression `^LDPN?[HQ]i$`, had 4 and 4) |
| Load vector pair, post/pre-index, S/D-form; Q-form | 5; 7 / 1 / I, L | `WriteAdr` + `N1Write_5c_1L` x 2; `N1Write_7c_1L` x 2 (*fixed*) |

The second register of every FP load pair used to be `WriteLDHi` (4, the integer load). The
research binaries contain 56,217 LDP S/D, 24,425 LDP Q and 2,326 `ldr q, [x, x]` without a scale.
Test: `FpLoadsAndStores`.

<a id="c-16"></a>
**C-16. Sec. 3.14, FP stores (p.23-24, l.1076-1149).**

| SOG group | SOG | Model |
|---|---|---|
| Store vector reg, unscaled immed, B/H/S/D; Q | 2 / 2; 2 / 1 / L, I | `N1Write_2c_1I_1L`; `N1Write_2c_2I_2L` ([U-11](#u-11)) |
| Store vector reg, post/pre-index, unsigned immed, B/H/S/D; Q | 2 / 2; 2 / 1 / L, V | `N1Write_2c_1L_1V`; `N1Write_2c_2L_2V` (+ `WriteAdr`) |
| Store vector reg, register offset basic, extend: B/H/S/D; Q | 2 / 2; 2 / 1 / L, V | `N1WriteSTRHro` / `N1WriteSTRQro` unscaled, `N1Write_2c_1L_1V` (B/S/D) |
| **Store vector reg, register offset, scale and extend scale: H; Q** | 2 / 2; 2 / 1 / I, L, V | scaled -> `N1Write_2c_1I_1L_1V`; `N1Write_2c_1I_2L_2V` (*fixed 2026-10-31*, `[a76_store_rows]`: H used to have the I uop always, Q never) |
| Store vector reg, register offset, scale and extend scale: S/D | 2 / 2 / L, V | `N1Write_2c_1L_1V` |
| Store vector pair, immed offset: S; D; Q | 2 / 2; 2 / 1; 3 / 1/2 / L, V | `N1Write_2c_1L_1V`; `N1Write_2c_2L_2V`; `N1Write_3c_4L_2V` |
| **Store vector pair, post/pre-index, S-form** | 2 / 1 / L, V | `WriteAdr` + `N1WriteLS<2, 2, 2, 1>` (L 2 cycles; *fixed*, was throughput 2) |
| Store vector pair, post/pre-index, D-form | 2 / 1 / L, V | `WriteAdr` + `N1Write_2c_2L_2V` |
| **Store vector pair, post-index, Q-form**; pre-index Q-form | 3 / 1; 3 / 1/2 / L, V | `WriteAdr` + `N1WriteLS<3, 6, 2, 2>` (*fixed*, `[a76_stpq_post]`, [I-5](#i-5)); `N1Write_3c_4L_2V` |

`N1Write_2c_1I_1L_1V` had latency 7 (a typo; the name and the SOG say 2), `N1Write_3c_2L_2V`
latency 2 (name and SOG 3); both are corrected. A store's latency matters only on Facile's
store -> load edges ([D-8](#d-8)). The writeback uop "(1) +I" of the FP stores (note 4, l.1165-1166)
is `WriteAdr`. The research binaries contain 254 STP Q post-index and 68 STP S pre/post-index.
Test: `FpLoadsAndStores`.

<a id="c-17"></a>
**C-17. Sec. 3.15, ASIMD integer (p.25-27, l.1179-1263).**

| SOG row | SOG | Model |
|---|---|---|
| Absolute diff; arith basic, complex, pair-wise; compare; dot product; logical; max/min basic and pair-wise; absolute diff long | 2 / 2 / V | `WriteVd` / `WriteVq` = `N1Write_2c_1V` |
| SABA, UABA; SABAL(2), UABAL(2) | 4(1) / 1 / V1 (note 2) | `N1Wr_VA` + `N1Rd_VA` (advance 3) |
| Reduce 4H/4S; 8B/8H; 16B: ADDV, SADDLV, UADDLV and SMAXV, SMINV, UMAXV, UMINV | 3 / 1 / V1; 5 / 1 / V1, V; 6 / 1/2 / V1 | `N1Write_3c_1V1`; `N1Write_5c_1V1_1V`; `N1Write_6c_2V1` |
| **Multiply, D-form; Q-form: MUL, SQDMULH, SQRDMULH** | 4 / 1 / V0; 5 / 1/2 / V0 | `N1Write_4c_1V0`; `N1Write_5c_2V0`, now **also the by-element forms** (*fixed 2026-10-31*, `[a76_asimd_forms]`; they were 2 on V) |
| **Multiply accumulate, D-form; Q-form: MLA, MLS** | 4(1) / 1 / V0; 5(2) / 1/2 / V0 (note 1) | `N1Wr_VMA`; `N1Wr_VMAQ` + `N1Rd_VMA` (advance 3), also by element (*fixed*) |
| **Multiply accumulate high, D; Q: SQRDMLAH, SQRDMLSH** | 4 / 1 / V0; 5 / 1/2 / V0 | `N1Write_4c_1V0`; `N1Write_5c_2V0`, also by element (*fixed*) |
| Multiply accumulate long: [SU]MLAL(2), [SU]MLSL(2) | 4(1) / 1 / V0 | `N1Wr_VMAL` + `N1Rd_VMAL` (advance 3), incl. by element |
| Multiply accumulate saturating long; multiply long | 4 / 1 / V0 | `N1Write_4c_1V0` |
| PMUL, PMULL(2) 8x8, D; Q | 3 / 1; 4 / 1/2 / V0 | `N1Write_3c_1V0`; `N1Write_4c_2V0` |
| SADALP, UADALP; SSRA, SRSRA, USRA, URSRA | 4(1) / 1 / V1 (note 2) | `N1Wr_VPA` / `N1Wr_VSA` + advance 3 |
| Shift by immed basic; and insert; by register basic | 2 / 1 / V1 | `N1Write_2c_1V1` |
| **Shift by immed, complex; by register, complex** | 4 / 1 / V1 | `N1Write_4c_1V1`, now **also SQSHRUN** (*fixed*: `^SQSHU?RNv` cannot match SQSHRUN, which was 2 on V) |

The by-element forms (`..._indexed`) are the same mnemonics as the rows; the regular expressions of
the model named only the vector-by-vector forms. Measured: `mul v0.4s, v0.4s, v1.s[1]` + branch 2.0
-> 5.0. The research binaries contain 11 integer by-element multiplies and no SQSHRUN. Scalar forms:
[D-5](#d-5). Test: `Asimd`.

<a id="c-18"></a>
**C-18. Sec. 3.16, ASIMD FP (p.28-30, l.1360-1509).**

| SOG row | SOG | Model |
|---|---|---|
| FABS, FABD, FADD, FSUB, FADDP, compare, max/min normal and pairwise, FNEG | 2 / 2 / V | `WriteVd` / `WriteVq` |
| FCVTL, FCVTN: F16<->F32; F32<->F64, FCVTXN | 4 / 1/2; 3 / 1 / V0 | `N1Write_4c_2V0`; `N1Write_3c_1V0` |
| Convert other, D F32 and Q F64; D F16 and Q F32; Q F16 | 3 / 1; 4 / 1/2; 6 / 1/4 / V0 | `N1Write_3c_1V0`; `N1Write_4c_2V0`; `N1Write_6c_4V0`, **also the fixed-point forms** (`..._shift`, *fixed*, `[a76_fp_cvt]`; they were 2 on V) |
| FDIV, FSQRT (all forms) | ranges / V0 (note 3) | `N1WriteFDS_*`, best values ([D-1](#d-1), [D-3](#d-3)): D F16 7/7, D F32 7/4, Q F16 10/10 (FSQRT 11/11), Q F32 7/7, Q F64 7/7 (latency / divider cycles) |
| FMAXV etc. reduce; Q-form F16 | 5 / 2; 8 / 2/3 / V | `N1Write_5c_1V`; `N1Write_8c_3V` |
| FMUL, FMULX; FMLA, FMLS; FMLAL(2), FMLSL(2) | 3 / 2; 4(2) / 2; 5(2) / 2 / V | `N1Wr_VFM`; `N1Wr_VFMA` + `N1Rd_VFMA`; `N1Wr_VFMAL` + `N1Rd_VFMAL` ([U-12](#u-12)) |
| FRINT*: D F32 and Q F64; D F16 and Q F32; Q F16 | 3 / 1; 4 / 1/2; 6 / 1/4 / V0 | `N1Write_3c_1V0`; `N1Write_4c_2V0`; `N1Write_6c_4V0` |

Measured: `fmul v0.4s; fmla v0.4s (accumulator v0); mov v1.16b, v0.16b` 7.0 (unchanged). Test:
`Asimd`, `FpScalar`.

<a id="c-19"></a>
**C-19. Sec. 3.17, ASIMD miscellaneous (p.31-32, l.1514-1571).**

| SOG row | SOG | Model |
|---|---|---|
| RBIT; BIF, BIT, BSL; CLS, CLZ, CNT; DUP element; EXT; XTN; INS element; FMOV, MOVI, MVNI; REV; TBL 1-2 regs; TBX 1 reg; TRN; UZP, ZIP | 2 / 2 / V | default `N1Write_2c_1V` |
| DUP gen reg | 3 / 1 / M | `N1Write_3c_1M` |
| SQXTN(2), SQXTUN(2), UQXTN(2) | 4 / 1 / V1 | `N1Write_4c_1V1` |
| Reciprocal estimate D F32/F64; D F16 and Q F32; Q F16 | 3 / 1; 4 / 1/2; 6 / 1/4 / V0 | `N1Write_3c_1V0`; `N1Write_4c_2V0`; `N1Write_6c_4V0` ([U-8](#u-8)) |
| FRECPS, FRSQRTS | 4 / 2 / V | `N1Write_4c_1V` |
| TBL 3 regs; 4 regs | 4 / 1/2; 4 / 2/3 / V | `N1Write_4c_4V`; `N1Write_4c_3V` |
| TBX 2; 3; 4 regs | 4 / 1/2; 6 / 2/3; 6 / 2/5 / V | `N1Write_4c_4V`; `N1Write_6c_3V`; `N1Write_6c_5V` |
| UMOV, SMOV; INS gen reg | 2 / 1 / V1; 5 / 1 / M, V | `N1Write_2c_1V1`; `N1Write_5c_1M_1V` |

<a id="c-20"></a>
**C-20. Sec. 3.18, ASIMD loads (p.33-36, l.1632-1748).**
Checked row by row: LD1 1-4 registers (5 / 2, 5 / 1, 6 / 2/3, 6 / 1/2, L), LD1 one lane and LD1R (7 /
2 / L, V), LD2 multiple, one lane, LD2R (7 / 1 / L, V), LD4 one lane and LD4R (8 / 1/2) match the
upstream writes. **LD3 multiple (8 / 1/2), LD3 one lane and LD3R (7 / 1/2), LD4 multiple D (8 / 2/7)
and Q (10 / 1/5)** used one L uop per cycle and reached 2/3, 2/3 and 1/2 per cycle; they are now
`N1WriteLS` with the L pipes held 2/T cycles in total (4, 4, 7, 10; *fixed 2026-10-31*,
`[a76_ls_throughput]`, [D-7](#d-7)), uop counts unchanged. Writeback forms add `WriteAdr` (1, I),
as "(1) +I" (l.1748). Measured: `ld3 {v0.16b-v2.16b}, [x0]` + branch 1.5 -> 2.0. The research
binaries contain 31 LD3/LD4/ST3/ST4 in all. Test: `AsimdLoadsAndStores`.

<a id="c-21"></a>
**C-21. Sec. 3.19, ASIMD stores (p.37-38, l.1826-1900).**
Checked row by row. Unchanged: ST1 1 register D (2 / 2), 2 registers D (2 / 1), 3 registers D (3 /
2/3). The following were up to twice as fast as the SOG (one L uop per cycle) and are now
`N1WriteLS` (L held 2/T cycles; *fixed 2026-10-31*, `[a76_ls_throughput]`, [D-7](#d-7)): ST1 1
register Q (1), 2 registers Q (1/2, latency 3), 3 registers Q (1/3), 4 registers D (1/2, latency
3) and Q (1/4); ST1 one lane (1); ST2 D (1), Q (1/2), one lane (1); ST3 D (1/2), Q (1/3), one lane
(1/2); ST4 D (1/3), Q B/H/S (1/6), Q D (1/4). Together they amount to one 128-bit store per cycle,
which sec. 4.4 states (l.2069-2070: "two 128 bit load uops and one 128 bit store uop every cycle").
ST4 one lane is partly blank in the SOG ([I-6](#i-6)) and keeps its write. Writeback "(1) Add I"
(l.1899): `WriteAdr`. Measured: two `st1 {vN.16b}, [xN]` + branch 1.0 -> 2.0. The research binaries
contain 287 ST1/ST2. Test: `AsimdLoadsAndStores`.

<a id="c-22"></a>
**C-22. Sec. 3.20, cryptography (p.39, l.1955-1970).**
AESD, AESE, AESIMC, AESMC 2 / 1 / V0: `N1WriteVC` / `N1Write_2c_1V0`; PMULL 64x64, SHA1H, SHA1SU0,
SHA1SU1, SHA256SU0, SHA256SU1 2 / 1 / V0: `N1Write_2c_1V0`; SHA1C/M/P, SHA256H/H2 4 / 1 / V0:
`N1Write_4c_1V0`. Sec. 4.7 (l.2133-2134): one AES instruction per cycle, latency 2. AES pairs:
[U-6](#u-6). These instructions decode for `cortex-a76` in LLVM 22 (checked with `llvm-mc`); none
occurs in the research binaries. Test: `CryptoAndCrc`.

<a id="c-23"></a>
**C-23. Sec. 3.21, CRC (p.40, l.2009-2024).**
CRC32, CRC32C 2 / 1 / M, note 1: "late forwarding of the result from a producer uop to a consumer
uop. This results in a 1 cycle reduction in latency": `N1Wr_CRC` 2, M + `N1Rd_CRC` (advance 1) on
the accumulator. Measured (test): `crc32x w1, w1, x3` 1.0 per iteration. Test: `MiscDataProcessing`,
`CryptoAndCrc`.

---

## 4. Not compatible with the SOG (N)

None open. The differences found by this audit where the SOG states a value unambiguously were
fixed; they are the C items marked *fixed 2026-10-31*.

---

## 5. Deliberate deviations and interpretations (D)

<a id="d-1"></a>
**D-1. Best case: shortest latency, best throughput, L1 hits, aligned accesses.**
SOG: integer divide "5 to 12" / "5 to 20", "Early termination is possible, depending upon the data
values" (l.428-429, l.482-483); FP divide and square root ranges (l.852-854, l.879-881,
l.1403-1408, l.1431-1453); load latencies assume an L1 hit (l.602, l.996, l.1633); unaligned
penalties (sec. 4.5, l.2110-2117). Implementation (project policy: a data-dependent value takes its
best value; occupancy = best throughput rounded up): integer divide latency 5 and divider 5 cycles
for both forms (`[a76_div_best]`, before: 12 / 20); FP divide / square root the shortest latency
and `ceil(1 / best throughput)` cycles (as before, [docs.md](docs.md) section 4.I); L1 hits;
aligned accesses. Effect: optimistic for divide-heavy code (a divide can take up to 12 or 20
cycles).

<a id="d-2"></a>
**D-2. Throughput as resource occupancy.**
SOG: throughput in instructions per cycle for the whole core (l.295-296). Implementation: a row
with throughput T on `n` pipelines holds them `n / T` cycles in total (ceil where needed); e.g.
MADD X 3 cycles of M (T = 1/3), LDPSW 2 L-cycles (T = 1), ST4 Q B/H/S 12 L-cycles (T = 1/6).
Facile divides the cycles of each resource by its units, so another instruction may use the other
unit meanwhile, which the SOG neither states nor excludes.

<a id="d-3"></a>
**D-3. The dividers are resources of their own.**
*Introduced 2026-10-31* (`[a76_dividers]`).
SOG: "Integer divides are performed using a iterative algorithm and block any subsequent divide
operations until complete" (sec. 3.6 note 1, l.482); "FP divide and square root operations are
performed using an iterative algorithm and block subsequent similar operations to the same pipeline
until complete" (sec. 3.11 note 1, l.922-923; sec. 3.16 note 3, l.1508-1509); "Each execution
pipeline can accept and complete one uop per cycle" (l.226). Notes 4 and 5 of sec. 3.6, by contrast,
say explicitly that X-form multiply-accumulates and multiply-high operations stall the multiplier
pipeline for other M uops. Implementation: a divide takes M (integer) or V0 (FP) for one cycle and
holds `N1UnitIDiv` / `N1UnitFDiv` for its occupancy; before, it held M or V0 itself, blocking every
other M or V0 uop (shifts with flags, multiplies, store data, FP converts, AES, ...). Reason: the SOG
names what a divide blocks (later divides / similar operations), and the reading that also blocks
the pipeline has no SOG basis; the lower estimate is preferred (project policy). Measured: `fdiv d0,
d2, d1; fcvtzs w1, d3; frinta d4, d5` + branch 6.0 -> 4.0; `udiv x1, x2, x3` + three `mul x` 14.0 ->
10.0 (with the latency change of D-1). Test: `DivideAndMultiply`, `FpScalar`.

<a id="d-4"></a>
**D-4. FP conversions with operand and result in FP registers: 3 / 1 / V0.**
*Introduced 2026-10-31* (`[a76_fp_cvt_scalar]`).
SOG: no row names the scalar forms of FCVT[AMNPZ][SU], SCVTF, UCVTF whose operand and result are FP
registers (`scvtf d0, d1`, `fcvtzs s0, s1`, also with `#fbits`). "FP convert, from vec to vec reg"
is "3 1 V0" (l.936, mnemonics FCVT, FCVTXN); the ASIMD rows of the same mnemonics give 3 / 1 / V0
for two lanes, 4 / 1/2 for four and 6 / 1/4 for eight (l.1377-1402). Implementation:
`N1Write_3c_1V0`, i.e. one lane is taken as the two-lane row, the lowest the SOG gives for these
mnemonics; both readings give the same value. Before: the stock class, 2 on V, which no row supports.
The research binaries contain 10,420 such conversions (integer forms) and 14 fixed-point ones.
Measured: `scvtf d0, d0` + branch 2.0 -> 3.0. Test: `FpScalar`.

<a id="d-5"></a>
**D-5. Scalar ASIMD forms take the row of their mnemonic.**
*Introduced 2026-10-31* (`[a76_asimd_scalar]`).
SOG: the ASIMD rows list mnemonics (SHL, SSHR, USHR, SLI, SRI, SRSHR, URSHR, SSRA, USRA, SRSRA,
URSRA, SQDMLAL, SQDMLSL, SQDMULL, SQSHRN, SQRSHRN, ...), without saying whether the scalar forms
(`shl d0, d1, #3`, `sqdmlal s0, h1, h2`, `sqshrn b0, h1, #3`) belong to them. Implementation: they
take the row of their mnemonic (the D-form row where the row distinguishes D and Q). Before: the
stock default, 2 on V. The research binaries contain 56 scalar shifts by immediate (D form). Test:
`Asimd`.

<a id="d-6"></a>
**D-6. Base update of integer stores: one uop, no pipeline, latency 1.**
*Introduced 2026-10-31* (`[a76_store_wb]`).
SOG: the pre- and post-index integer stores are "1 2 L, D" (STR, l.744-747) and "1 1 L, D" (STP,
l.780-787), without I; the FP and ASIMD store notes say "Writeback forms of store instructions
require an extra uop to update the base address ... (update latency shown in parentheses)" with
"(1) +I" (l.1158, l.1165-1166, l.1899, l.1951-1952) ([I-3](#i-3)). Implementation: the base update
of STR(B/H/W/X) and STP(W/X) pre/post is `N1Write_1c_STWB`: latency 1, one uop, no pipeline (it was
`WriteAdr`, 1 cycle on I). The FP and ASIMD stores keep `WriteAdr`. Reason: with the store data on
S1 and M ([C-2](#c-2)), an I uop for the update would make two STR post-index per cycle impossible
(four integer uops on three pipelines), against the row's throughput 2; the integer table is taken
as printed. Measured: two `str x, [x], #8` + branch 1.0 (it would be 1.33 with the I uop). Test:
`Stores`.

<a id="d-7"></a>
**D-7. Load/store throughput below the uop count: expressed on the L pipelines.**
SOG: the rows of [C-20](#c-20), [C-21](#c-21) and STP S / Q post-index ([C-16](#c-16)) name "L, V"
and a throughput below what their uops give on two L and two V pipelines; which of the two limits is
not stated. Implementation: `N1WriteLS<lat, uops, lcyc, vcyc>` keeps the uop count and the V cycles
of the write it replaces and holds L for `lcyc = 2 / T` cycles in total. Reason: sec. 4.4 describes
the limit as a load/store one ("two 128 bit load uops and one 128 bit store uop every cycle"); one
rule for all rows. Consequence: a mix of such an instruction with other L users is limited
together, with V users it is not.

<a id="d-8"></a>
**D-8. Store-to-load forwarding latency.**
SOG sec. 4.6 (l.2120-2129) gives the conditions of forwarding but no latency; the store tables give
store latencies (1 for integer stores, 2 or 3 for FP stores, 2-9 for ASIMD stores, "the minimum
latency seen by an operation dependent on" the store). Implementation (shared Facile code,
`addMemoryDependencies`, [docs.md](docs.md) section 4): a store -> load edge has the store's latency,
then the load's own latency applies; i.e. forwarding is taken to be no slower than an L1 hit.

<a id="d-9"></a>
**D-9. Not modelled: AArch32, dispatch stall, branch alignment.**
The AArch32 tables, IT blocks (sec. 4.13) and the T32/A32 rows are not compared (the research
binaries are AArch64). The dispatch stall of sec. 4.2 (p.41, l.2057-2060: a V uop with more than one
Q source partly written as S registers stalls three cycles, "only on the first such instance") is
not modelled: it does not repeat in a loop. The branch alignment advice of sec. 4.9 (p.44,
l.2219-2231) gives no cost; not modelled.

---

## 6. Unclear in the SOG (U)

<a id="u-1"></a>
**U-1. ROB size and the MLP window.**
The SOG gives no reorder-buffer size or scheduler sizes. Current handling: `MicroOpBufferSize =
128` (an upstream value, commented "Copied from Cortex-A76"), which is the MLP window (128 uops) and
not used otherwise by the A76 path; `LoopMicroOpBufferSize = 16`, `MispredictPenalty = 11` and
`IssueWidth = 4` are not used by the A76 path either. Impact: the MLP columns. How to settle: a
documented source (whether the A76 Technical Reference Manual gives it was not checked).
Test: `DispatchAndPipelines`.

<a id="u-2"></a>
**U-2. Mops per instruction.**
SOG l.224-225: instructions are decoded into Mops, "A Mop can be split further into two Uops at
dispatch stage"; sec. 4.1: 4 Mops and 8 uops per cycle. The SOG does not say how many Mops an
instruction with more than two uops decodes into (the model gives LD4 eight uops, ST4 Q twelve, STR Q
four). Current handling: one Mop per instruction, the model's uop count for the uop bound. Impact:
optimistic Mop bound for such instructions (rare). How to settle: hardware measurement.

<a id="u-3"></a>
**U-3. Region-based fast forwarding.**
SOG sec. 4.8 (p.43, l.2159-2216): "The effective latency of FP and ASIMD instructions as described
in section 3 is increased by one cycle if the producer and consumer instructions are not part of
the same forwarding region", with three regions, exclusions and special notes (references to "table
40" and to sections 3.14 / 3.16 that do not match the numbering, [I-8](#i-8)). Unclear: the exact
membership (e.g. "ASIMD miscellaneous instructions" are in regions 1 and 2, "can only be consumers"
of region 2; FP convert and rounding belong to no region, so any edge into or out of them would cost
one more cycle). Current handling: not modelled (every FP/ASIMD edge has the table latency).
Impact: optimistic by one cycle per cross-region edge on FP/ASIMD dependence chains. How to settle:
an owner decision on a classification, or hardware measurement.

<a id="u-4"></a>
**U-4. ADD/SUB (extended register) with an identity extend.**
SOG "Arithmetic, extend and shift ADD{S}, SUB{S} 2 1 M" (l.353). Unclear: whether the extended
register form with a zero shift and a full-width extend (UXTX/SXTX for X, UXTW/SXTW for W), which
assemblers write as `add x0, sp, x1`, counts as "extend". Current handling: 2 on M (all extended
forms). Impact: the research binaries contain 25,199 `add`/`sub` with SP and a register written this
way; at 1 / 3 / I they would be cheaper. How to settle: hardware measurement or an owner decision.

<a id="u-5"></a>
**U-5. Multiply-accumulate forwarding between forms.**
SOG sec. 3.6 note 2 (l.485-486): "late-forwarding of accumulate operands from similar uops".
Unclear: whether a W-form MADD forwards to an X-form or long accumulator and vice versa. Current
handling: `N1Rd_IMA` (advance 1) from both `N1Wr_IM32` and `N1Wr_IM64` to the accumulator of every
MADD/MSUB and SMADDL-family instruction, i.e. any form forwards to any form (the lower estimate).
How to settle: hardware measurement.

<a id="u-6"></a>
**U-6. AESE/AESMC and AESD/AESIMC pairs.**
SOG sec. 4.7 (l.2155-2156): dependent pairs "are higher performance when they are adjacent in the
program code and both instructions use the same destination register"; sec. 3.20 note 1. No
number. Current handling (upstream): `N1ReadVC` gives AESMC/AESIMC an advance of 2 from
AESE/AESD (`N1WriteVC`), i.e. latency 0, which Facile floors to 1; adjacency is not checked. No AES
instruction occurs in the research binaries. How to settle: hardware measurement.

<a id="u-7"></a>
**U-7. Exclusives, acquire/release, LSE atomics.**
The SOG has no rows for LDAR/LDAPR, STLR, LDXR/LDAXR/LDXP, STXR/STLXR/STXP/STLXP or the LSE
atomics (the A76 implements LSE; `llvm-mc -mcpu=cortex-a76` accepts LDADD). Current handling:
stock classes (LDAR 4 on L; STLXR `WriteSTX` = store + load, latency 5); `WriteAtomic` is
`Unsupported`, so InstrBuilder drops every LSE atomic with a `[WARN]` ([docs.md](docs.md) section
2.F). The research binaries contain 139 LDAR/STLR/LDXR/STXR-family instructions and no LSE atomic;
the only dropped opcode in the A76 runs is UDF. How to settle: a project decision (e.g. the
treatment the A55/A520 received) or hardware measurement.

<a id="u-8"></a>
**U-8. FRECPE, FRSQRTE, Q-form F64.**
SOG sec. 3.17 (l.1530-1535): rows for D-form F32 and F64, D-form F16 and Q-form F32, Q-form F16; none
for Q-form F64 (`frecpe v0.2d`). Current handling: the stock class (2, V). None in the research
binaries.

<a id="u-9"></a>
**U-9. Special-register access.**
SOG sec. 4.11 (p.44-45, l.2239-2306): reads and writes of most special registers are
non-speculative and/or in order, FPCR writes may act as a barrier; no cycle counts. Current handling:
MRS, MSR, barriers and other system instructions have no pipeline and latency 1 (upstream
`WriteSys`, `WriteBarrier`); no serialisation. NZCV and SP "are fully renamed" (l.2279), which
matches. The research binaries contain 10 MRS/MSR.

<a id="u-10"></a>
**U-10. Register forwarding hazards (S written, Q read).**
SOG sec. 4.12 (p.46, l.2320-2346): when a uop reads a Q register recently written as S registers,
"Cortex-A76 performs slightly worse"; no number. Not modelled.

<a id="u-11"></a>
**U-11. An I uop on FP LDR (unsigned immed) and FP STUR.**
SOG: "Load vector reg, unsigned immed LDR 5 2 L, I" (l.1009-1010) while the unscaled form LDUR is
"L" only (l.1003-1004); "Store vector reg, unscaled immed STUR 2 2 L, I" and Q "2 1 L, I"
(l.1083-1086) while every other vector store is "L, V". Unclear: whether these I are real or
misprints (the store's data has to come from a V pipeline). Current handling: as printed
(`N1Write_5c_1I_1L`, `N1Write_2c_1I_1L`, `N1Write_2c_2I_2L`). Impact: the research binaries contain
293,015 FP LDR with an immediate offset and 66,093 FP STUR, each taking an integer-pipe slot.

<a id="u-12"></a>
**U-12. FMLAL, FMLSL.**
SOG sec. 3.16 (l.1419-1420) has a row "ASIMD FP multiply accumulate long FMLAL(2), FMLSL(2) 5(2) 2 V",
and the model has `N1Wr_VFMAL`. LLVM 22's `cortex-a76` has no FP16FML feature, so these instructions
are rejected by the assembler and the disassembler (checked with `llvm-mc`). Whether the A76
implements them was not verified; the row cannot apply to the research binaries.

---

## 7. Inconsistent within the SOG (I)

<a id="i-1"></a>
**I-1. "Branch 0/1".**
The legend (p.8, l.301) names "Branch 0/1"; sec. 2.1 (p.7, l.225-264) describes eight pipelines with
one branch pipeline, and every branch row has throughput 1. Implementation: one branch pipeline.

<a id="i-2"></a>
**I-2. FP divide: "until complete" against the throughput column.**
Notes l.922-923 and l.1508-1509: divides "block subsequent similar operations to the same pipeline
until complete", which would make the throughput 1/latency; the tables print e.g. scalar H 7 / 4/7,
scalar D 7 to 15 / 1/7 to 2/7. Implementation: the throughput column ([D-1](#d-1)). For the
integer divide the note and the table agree (5 to 12 / 1/12 to 1/5).

<a id="i-3"></a>
**I-3. Base update of stores.**
FP/ASIMD notes (l.1158, l.1165-1166, l.1899-1900, l.1951-1952): writeback forms "require an extra
uop to update the base address", "+I". AArch64 integer store rows (l.744-747, l.780-787): "L, D",
STR pre/post throughput 2, which leaves no integer pipeline for such a uop ([C-2](#c-2)); the
AArch32 store rows show "I, L, D" for some writeback forms (l.805: throughput 3/2; l.821: 1). The FP
rows (l.1087-1094) print "L, V" without I. Implementation: integer stores without I
([D-6](#d-6)), FP and ASIMD stores with I.

<a id="i-4"></a>
**I-4. STRH "extend, scale by 1".**
Sec. 3.10 (l.775-776): "Store register, register offset, extend, scale by 1 STRH 2 3/2 I, L, D";
the non-extended row says "scaled by 2" (l.769-770), and the load table "extend, scale by 2" for
LDRH (l.629-630). Implementation: scale by 2.

<a id="i-5"></a>
**I-5. STP Q: post-index throughput 1, offset and pre-index 1/2.**
Sec. 3.14 (l.1136-1149): immediate offset Q "3 1/2", post-index Q "3 1", pre-index Q "3 1/2".
Implementation: each row as printed (`[a76_stpq_post]`); before, the post-index form had the 1/2 of
the others. The research binaries contain 254 STP Q post-index and no pre-index.

<a id="i-6"></a>
**I-6. ST4 one lane: blank cells.**
Sec. 3.19 (l.1893-1898): B/H "5" with no throughput, S no latency and "2/3", D neither. Current
handling: the upstream writes (B/H/S 5 on three L and three V uops, D 4), i.e. throughput 2/3.

<a id="i-7"></a>
**I-7. FP LDR literal: no latency.**
Sec. 3.13 (l.1001-1002): "Load vector reg, literal, S/D/Q forms LDR (blank) 2 L". The section text
(l.996-997) says an FP load needs one cycle more than an integer load (4), and every other FP load
row is 5. Implementation: 5. None occurs in the research binaries.

<a id="i-8"></a>
**I-8. Misprints and references.**
"VCVTAU" in AArch64 rows (l.1395, l.1399; FCVTAU meant); pipelines "IL" (l.713), "LI" (l.1155), "LV"
(l.1777), "I, L D" (l.769); note reference "1.2" (l.668); "Q2F" (l.2346); sec. 4.8 refers to "table
40" and to sections "3.14" / "3.16" for ASIMD shift and narrow instructions (the table is "Table 1",
the sections are 3.15 / 3.17), and has an unfinished sentence ("instructions belonging to
different.", l.2200); sec. 3.20 note 1 refers to "Section 4.5" for AES (4.7 meant). Implementation:
the evident meaning.

---

## 8. Outside the A76 path: shared code (X)

<a id="x-1"></a>
**X-1. The test harness gives `cortex-a76` the stock model.**
`AArch64TestContext` (`tests/test_support.h`) installs the locally generated scheduling model only
for icestorm, firestorm, cortex-a720(ae), cortex-x1(c), cortex-a55 and cortex-a520. `cortex-a76`,
the default CPU of the context and of most MLP, splitter and Facile tests, runs on libLLVM's stock
`NeoverseN1Model`, not on the model `mca_tool` uses; the A76 "control" cases of
`tests/facile_test.cpp` therefore test the stock model. The header's comment notes that adding
cortex-a76 would change an MLP test. Not changed; `tests/a76_sog_test.cpp` builds its own context
(`A76Context`) the way `frontend.cpp` does.

<a id="x-2"></a>
**X-2. MCA gives every call latency 0.**
LLVM MCA's `InstrBuilder` prints "call instructions are not correctly modeled. Assume a latency of
0cy" and sets the latency of BL, BLR (and other calls) to 0, whatever the scheduling model says;
Facile then floors the LR edge to 1. So the SOG's BL / BLR latency 1 is met by the floor, but the
"reg == lr" BLR latency 2 ([C-7](#c-7)) does not reach the predictions. Shared LLVM behaviour; not
changed (no `blr x30` in the research binaries).

---

## 9. Taken over from the in-order cores (B)

<a id="b-1"></a>
**B-1. Store data is read one cycle late.**
*Introduced 2026-10-31* (`[a76_from_a55_store_data]`). **Not SOG-based.**
A76 SOG: stores are split into address and data uops at dispatch (l.738-739, l.1077, l.1827); no
statement on when the data operand is needed; store latencies 1 (integer), 2-3 (FP), 2-9 (ASIMD).
The project decided for the Cortex-A55 (SOG-based there: data "required at wr", [a55-sog.md](a55-sog.md)
C-18) and for the Cortex-A520 (borrowed, [a520-sog.md](a520-sog.md) B-2) that store data is read one
cycle late, and asked for the same treatment wherever the A76 SOG is silent on a point decided
there. Implementation: `ReadAdvance<ReadST, 1>` (was 0), and `ReadST` on the data operands of every
store form (integer STR/STUR/STTR/STLR/STLLR, register offset, pre/post, STP/STNP, STLXR/STLXP; FP
STR/STUR/STP of every addressing mode; the `Vt` of ST1-ST4, with the lane index of the one-lane
forms reading `ReadDefault` in between), as on the A520; STXR/STXP have none. The writes are
unchanged. Effect: only Facile's store -> load memory edges, and only for producers of latency 2 or
more (Facile floors an edge at 1). Measured: `ldr x1, [x0]; str x1, [x0]` + branch 5.0 -> 4.0;
`ldr x1, [x0]; add x1, x1, #1; str x1, [x0]` 6.0 (unchanged). To be revisited when an A76 statement
or measurement appears. Test: `BorrowedStoreDataNeededLater`.

---

## 10. Reproducing the cycle counts quoted here

The cycles/iteration figures that are not from `tests/a76_sog_test.cpp` were measured by the author
on 2026-10-31 with private builds (CMake, Release; not `build/`): "before" is the working tree before
this audit, "after" the working tree with its changes. Each loop is assembled, linked and analysed
with the invocation of the research for the A76, plus `--fractional-cycles` (the unrounded Facile
estimate) and `--facile-reason`:

```sh
# body: the instructions of one iteration, e.g. "tst x1, x2" "csel x1, x1, x3, eq"
{ echo '.text'; echo '.globl f'; echo '.type f,%function'; echo 'f:'; echo '1:';
  printf '  %s\n' "${body[@]}"; echo '  b 1b'; echo '  ret'; } > t.s
aarch64-linux-gnu-gcc -march=armv8.2-a+crypto+lse+crc+dotprod+fp16+rcpc -nostdlib -static -Wl,-e,f t.s -o t
mca_tool --mcpu=cortex-a76 --mtriple=aarch64 --dependency ooo --facile --fractional-cycles --facile-reason t
# cycles/iteration = the cycles column of the first CSV row (the loop); the last column says
# which bound decided it (inst, exec, prec)
```

The loop branch `b 1b` is part of every body. The research invocation (without the last two flags)
is `mca_tool --merge-same-header --dependency ooo --mcpu cortex-a76 --facile <binary>`.
Instruction counts "in the research binaries" are mnemonic/operand counts of `objdump -d` over the
46 binaries. Re-measure before quoting a number.

---

## 11. Maintaining this document

* Update an item when the SOG reading or the implementation changes. IDs run from 1 without gaps
  within each category, in the order of the items; an item whose difference has been fixed becomes a
  C item, and the code, tests and other documents that name an ID are updated with it.
* When a deviation is introduced on purpose, add a D item, and tag the code with a comment that names
  the ID.
* When a behaviour is taken over from another core's model by a project decision, without an A76
  SOG statement, add a B item and tag the code `[a76_from_<core>_<topic>]`.
* `NeoverseN1Model` is installed only for `cortex-a76` / `cortex-a76ae`. If another CPU is ever given
  this local model (e.g. `neoverse-n1`), its own SOG must be compared first; the A76 changes above
  are not N1 statements.
* Evaluation results (accuracy metrics, totals over benchmark sets) are not recorded in this document;
  cycles/iteration only when reproduced with `mca_tool` as in section 10.
* [docs.md](docs.md) holds the rest of the technical documentation; [a55-sog.md](a55-sog.md) and
  [a520-sog.md](a520-sog.md) are the Cortex-A55 and Cortex-A520 counterparts of this document.
  [README.md](README.md) lists the documents.
