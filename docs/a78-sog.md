# Cortex-A78: the model against the Software Optimization Guide

> **Who wrote this, and how far to trust it.** This document was written on **2026-10-31** by
> **Claude** (Anthropic's Claude Opus model, running as a sub-agent of Claude Code), after reading
> the Cortex-A78 Core Software Optimization Guide and the contents of this repository's working tree
> on that day. **It is not an Arm document**, and it has not been reviewed by Arm. Every claim about
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
8. [Outside the A78 path: shared code (X)](#8-outside-the-a78-path-shared-code-x)
9. [Taken over from other cores (B)](#9-taken-over-from-other-cores-b)
10. [Reproducing the cycle counts quoted here](#10-reproducing-the-cycle-counts-quoted-here)
11. [Maintaining this document](#11-maintaining-this-document)

| Category | Items |
|---|---|
| C, compatible | [C-1](#c-1) to [C-24](#c-24) |
| N, not compatible | none open |
| D, deliberate deviation or interpretation | [D-1](#d-1) to [D-11](#d-11) |
| U, unclear in the SOG | [U-1](#u-1) to [U-12](#u-12) |
| I, inconsistent within the SOG | [I-1](#i-1) to [I-7](#i-7) |
| X, outside the A78 path | [X-1](#x-1) to [X-2](#x-2) |
| B, taken over from the Cortex-A55 / Cortex-A520 / Cortex-A76 (not SOG-based) | [B-1](#b-1) |

---

## 1. About this document

### 1.1 Source

* **Guide**: "Arm® Cortex®-A78 Core Software Optimization Guide", document number
  **PJDOC-466751330-9691**, **Issue 4.0** of 28 April 2021 ("First release for r1p2"), revision
  r1p2, Copyright © 2019-2021 Arm Limited (or its affiliates), **Non-Confidential**, product status
  "Final" (title page and p.2-3, l.1-23, l.89-90). Called "the SOG" below. Page numbers are those of
  the guide's footers ("Page N of 59"); "l.N" is line N of the text extraction `tmp/a78sog.txt` (not
  tracked). The rows of the tables quoted here were checked against the PDF for the FP load and
  store tables (p.30-33); the other rows were read from the text extraction only.

### 1.2 What "the implementation" is

The state of the working tree on 2026-10-31 (last commit `e7a7195`, plus uncommitted changes of
this and other sessions):

* `ModifiedTarget/AArch64/AArch64SchedNeoverseN2.td`: the TableGen model `NeoverseN2Model` (LLVM's
  Neoverse N2 model with local changes), compiled into the tool's own scheduling tables (see
  [docs.md](docs.md) section 5). `src/cpu/sched_model.cpp` (`installLocalSchedModel`, `case
  CpuModel::A78`) installs it **only** for `cortex-a78`, `cortex-a78ae` and `cortex-a78c`
  (`src/cpu/cpu_traits.cpp`). `neoverse-n2`, `cortex-a710` and `cortex-a715`, which LLVM also maps
  to `NeoverseN2Model`, have `CpuModel::None` and keep libLLVM's stock model (checked: their
  `mca-insts-info` output is unchanged by this audit). The file was therefore edited in place; no
  separate A78 model was needed.
* `src/cpu/cpu_traits.cpp`: the A78 entry (`UopDispatchWidth = 12`, `MopDispatchWidth = 6`,
  `Fusion = FusionTable::A78`, `ZeroLatencyMov = true`, no MLP window resources).
* `src/facile/facile.cpp` and `src/facile/fusion.cpp`: the Facile predictor and the A78 fusion
  table, which produce the A78 cycle counts. The research runs `mca_tool --merge-same-header
  --dependency ooo --mcpu cortex-a78 --facile` (CSV `mca.a78.csv`) and the same with
  `--disable-always-hit-loads-heuristic` (`mca.a78_2.csv`); with `--facile` the MCA simulation is
  not run, the CSV `cycles` column is the rounded Facile estimate.
* `tests/a78_sog_test.cpp`: unit tests that pin the values below (added by this audit, not yet
  committed).

Implementation facts are given by file and identifier (TableGen `def` names, C++ function names),
not by line number. The model marks the changes of this audit with tags; [section 2.4](#24-code-tags-and-the-items-of-this-document)
maps them to the items.

### 1.3 Categories

| ID | Category | Meaning |
|---|---|---|
| **C-n** | compatible | The implementation does what the SOG states. Items fixed by this audit are C items marked *fixed 2026-10-31*. |
| **N-n** | not compatible | The SOG states something and the implementation does not follow it. Each item says why it is still open. |
| **D-n** | deliberate deviation or interpretation | The implementation knowingly departs from the literal SOG, or fills a gap with an interpretation. Each item gives the SOG text, the implementation and the reason. |
| **U-n** | unclear | The SOG does not decide the question; the implementation makes an assumption or leaves it open. Each item gives the SOG text, what is unclear, the current handling, the expected impact, and how to settle it. |
| **I-n** | inconsistent | Two places of the SOG disagree, or the SOG contradicts itself or has a misprint. Each item says which reading the implementation follows. |
| **X-n** | outside the A78 path | Problems in shared code or in LLVM, found during this audit. Not changed. |
| **B-n** | taken over from another core | The A78 SOG says nothing on the question, and the treatment the project decided for the Cortex-A55 / Cortex-A520 and applied to the Cortex-A76 is applied to the A78 for consistency. **Not SOG-based.** |

### 1.4 Reading the SOG instruction tables

Sec. 3.1 (p.14, l.488-500):

* **Execution latency** is "the minimum latency seen by an operation dependent on an instruction in
  the described group". A number in parentheses is the accumulate latency of a forwarding path
  (notes of sec. 3.6, 3.11, 3.15, 3.16).
* **Execution throughput** is "the maximum throughput (in instructions per cycle) of the specified
  instruction group that can be achieved in the entirety of the Cortex-A78 microarchitecture". The
  model holds a resource of `n` units for `n / T` cycles in total ([D-2](#d-2)).
* **Utilized pipelines**, Table 3-1 (p.14, l.505-516): B branch 0/1, S integer single cycle 0/1,
  I "Integer single Cycle 0/1 and single/multicycle 0/1", M integer single/multicycle 0/1, M0
  integer multicycle 0, L "Load/Store 0/1 and Load 2", D store data 0/1, V FP/ASIMD 0/1, V0, V1.
  The store tables also use L01 (load/store 0/1).
* Ranges ("5 to 12", "1/12 to 1/5") are data dependent; the model takes the best value
  ([D-1](#d-1)).
* The AArch32 tables are not compared (the research binaries are AArch64; [D-8](#d-8)); they are
  quoted only where they disagree with an AArch64 row ([I-1](#i-1)).

---

## 2. Implementation overview

### 2.1 Machine model (`NeoverseN2Model`)

`IssueWidth = 5` (not used by Facile, whose widths come from `cpu_traits.cpp`), `MicroOpBufferSize
= 160` (the MLP window width, [U-1](#u-1)), `LoadLatency = 4`, `MispredictPenalty = 10`,
`LoopMicroOpBufferSize = 16` (neither used by the A78 path), `CompleteModel = 1`;
`UnsupportedFeatures` = SME, SVE2p1, SVE B16B16, PAuthLR, CPA, CSSC, MOPS.

| Resource | Units | Role |
|---|---|---|
| `N2UnitB` | 2 | Branch 0/1 |
| `N2UnitS` | 2 | Integer single cycle 0/1 |
| `N2UnitM0`, `N2UnitM1` | 1, 1 | Integer single/multicycle 0, 1 |
| `N2UnitL01` | 2 | Load/Store 0/1 |
| `N2UnitL2` | 1 | Load 2 |
| `N2UnitD` | 2 | Store data 0/1 |
| `N2UnitV0`, `N2UnitV1` | 1, 1 | FP/ASIMD 0, 1 |
| `N2UnitI` | group S, M0, M1 (4) | "I" |
| `N2UnitM` | group M0, M1 (2) | "M" |
| `N2UnitL` | group L01, L2 (3) | "L" |
| `N2UnitV` | group V0, V1 (2) | "V" |
| `N2UnitFlg` | 3 | not a pipeline: flag-setting ALU operations ([D-9](#d-9)) |
| `A78UnitIDiv` | 1 | not a pipeline: the integer divider inside M0 ([D-3](#d-3)) |
| `A78UnitFDiv` | 1 | not a pipeline: the FP divide / square root unit inside V0 ([D-3](#d-3)) |

TableGen lists a use of a pipeline also under every group that contains it (a use of M0 appears as
M0, M and I), so `mca-insts-info` and the tests show, for example, `[N2UnitI:1 N2UnitM:1
N2UnitM0:1]` for `madd x1, x2, x3, x4`.

### 2.2 The Facile path (`src/facile/facile.cpp`)

For `--facile`, every region gets `EstimatedCycles = max(IssueBound, PortBound, PrecedenceBound)`:

* `calculateIssueBound`: `max(uops / 12, Mops / 6)` (uops = the model's `NumMicroOps`, one Mop per
  instruction; the absorbed half of a fused pair is not counted; [C-1](#c-1), [C-5](#c-5),
  [U-2](#u-2)).
* `calculatePortUsageBound`: for every resource (pipelines, groups, `N2UnitFlg`, the dividers), the
  sum of the resource cycles divided by its units; the absorbed half of a fused pair is not counted.
* `calculatePrecedenceBound`: the maximum cycle ratio of the dependence graph. Register edges have
  the latency of the producing def minus the consumer's `ReadAdvance`, floored at 1, and 0 for the
  zero-latency MOVs (`getEdgeLatency`, [C-4](#c-4), [C-6](#c-6)); a store with a base update has a
  second node for the update ([C-7](#c-7)); store -> load memory edges have the store's latency
  ([D-7](#d-7)).

The MLP columns use a window of `MicroOpBufferSize` = 160 uops ([U-1](#u-1)); a change of an
instruction's uop count therefore also changes the `mlp` / `mlp_r` columns of the regions around it.

### 2.3 What changed in this audit

The model before the audit is the working-tree file of 2026-10-31: upstream LLVM's N2 model plus the
project's earlier changes (the multiply-accumulate, FP and ASIMD accumulator forwarding, the CRC
forwarding, the best-case FP divide latency and occupancy, and uncommitted A78 FP load rows of
another session, which this audit kept and corrected, [C-16](#c-16)). Four A78 corrections were
applied on top of the model by Facile C++ code (`src/facile/n2_corrections.cpp`, switched on by
`CpuTraits::N2ModelCorrections`): no I µOP on register-offset loads and stores, one M0 cycle for
SCVTF/UCVTF, FMOV and DUP from a GPR, I + 2 L for LDPSW, and a divide occupancy of 5. Since the
local `NeoverseN2Model` is the A78's alone, this audit moved these four corrections into the `.td`
file ([C-10](#c-10), [C-12](#c-12), [C-13](#c-13), [C-15](#c-15), [C-20](#c-20)) and removed the
C++ code, the flag and the files; Facile has no A78-specific port or uop correction now. All other
changes are in the `.td` file and are listed in the items.

### 2.4 Code tags and the items of this document

| Tag in `AArch64SchedNeoverseN2.td` | Items |
|---|---|
| `[a78_dividers]` | [D-3](#d-3) |
| `[a78_div_best]` | [C-10](#c-10), [D-1](#d-1) |
| `[a78_maddl]` | [C-10](#c-10), [U-5](#u-5) |
| `[a78_extr]` | [C-11](#c-11) |
| `[a78_reg_offset]` | [C-12](#c-12), [C-13](#c-13) |
| `[a78_reg_offset_h]` | [C-12](#c-12), [C-16](#c-16), [I-4](#i-4) |
| `[a78_ldpsw]` | [C-12](#c-12) |
| `[a78_ldp_hi]` | [D-10](#d-10) |
| `[a78_store_rows]` | [C-13](#c-13), [C-17](#c-17), [I-3](#i-3), [I-5](#i-5) |
| `[a78_fdiv_best]` | [C-14](#c-14), [C-19](#c-19), [D-1](#d-1) |
| `[a78_frint]` | [C-14](#c-14) |
| `[a78_m0_pipelined]` | [C-15](#c-15), [C-20](#c-20) |
| `[a78_fp_cvt]` | [C-15](#c-15), [C-19](#c-19), [C-20](#c-20) |
| `[a78_fp_cvt_scalar]` | [D-4](#d-4) |
| `[a78_zero_mov]` | [C-4](#c-4) |
| `[a78_fp_ld]` | [C-16](#c-16) |
| `[a78_fp_st]` | [C-17](#c-17), [I-6](#i-6) |
| `[a78_asimd_acc]` | [C-18](#c-18) |
| `[a78_asimd_mull]` | [C-18](#c-18) |
| `[a78_asimd_scalar]` | [D-5](#d-5) |
| `[a78_v_throughput]` | [C-18](#c-18), [C-19](#c-19), [C-20](#c-20) |
| `[a78_asimd_ld]` | [C-21](#c-21) |
| `[a78_asimd_st]` | [C-22](#c-22) |
| `[a78_ls_throughput]` | [C-21](#c-21), [C-22](#c-22), [D-6](#d-6) |
| `[a78_from_a55_store_data]` | [B-1](#b-1) |

The rows without a tag are the upstream model's, with the SOG group names as comments.

---

## 3. Compatible with the SOG (C)

Values are "latency / throughput / pipelines" as printed. "Model" values were read with
`mca-insts-info --mtriple=aarch64 --mcpu=cortex-a78` and, for variant classes, with the unit tests
(`tests/a78_sog_test.cpp`, suite `A78SogTest`). Cycles per iteration are Facile estimates of small
loops that include the loop branch ([section 10](#10-reproducing-the-cycle-counts-quoted-here)).
Instruction counts "in the research binaries" are explained in section 10.

### 3.1 Dispatch, pipelines and forwarding

<a id="c-1"></a>
**C-1. Six Mops and twelve uops per cycle.**
SOG sec. 4.1 (p.51, l.2455): "The dispatch stage can process up to 6 MOPs per cycle and dispatch up
to 12 µOPs per cycle"; sec. 2.1 (p.11, l.386-389): instructions are decoded into MOPs, "A MOP can be
split into two Micro-OPerations (µOPs) further down the pipeline after the decode stage".
Implementation: `cpu_traits.cpp` `UopDispatchWidth = 12`, `MopDispatchWidth = 6`;
`calculateIssueBound` takes the larger of uops / 12 and instructions / 6. The per-type limits of
sec. 4.1 (l.2457-2462: four µOPs for S or B, four for M, two each for M0, V0, V1, six for L) are not
modelled; none of them is below the issue rate of the pipelines concerned (four S/B, two M, one M0,
one V0, one V1, three L), so none can bind in a steady state. Test: `DispatchAndPipelines`.
Instructions with more than two µOPs: [U-2](#u-2).

<a id="c-2"></a>
**C-2. Thirteen issue pipelines.**
SOG sec. 2.1 (p.11-13, l.388-390, Figure 2-1, Table 2-1 l.462-473): µOPs "issue out-of-order to one
of thirteen issue pipelines. Each issue pipeline can accept one µOP per cycle": Branch 0, 1, Integer
Single-Cycle 0, 1, Integer Single/Multi-Cycle 0, 1, FP/ASIMD 0, 1, Load/Store 0, 1, Load 2, Store
data 0, 1. Implementation: the resources of [section 2.1](#21-machine-model-neoverse-n2model); the
groups I, M, L and V are those of Table 3-1. Integer store data take their own pipes D (unlike the
Cortex-A76), so integer stores use no integer pipe. Measured: two `str` and four `add` + branch 1.0.
Test: `DispatchAndPipelines`, `Stores`.

<a id="c-3"></a>
**C-3. NOP takes no execution resource.**
The SOG has no row for NOP or the other hints; it lists "NOP + Any instruction" as a fused pair
(sec. 4.14, l.2817, [C-5](#c-5)). Implementation (upstream): `WriteHint` has no resource; a NOP that
is not fused counts as one Mop and one µOP. This is the decision made for the Cortex-A55, Cortex-A520
and Cortex-A76 ([a76-sog.md](a76-sog.md) C-3). The research binaries contain 348,224 NOPs. Test:
`DispatchAndPipelines`.

<a id="c-4"></a>
**C-4. Zero-latency MOVs.**
*Fixed 2026-10-31* (`[a78_zero_mov]`).
SOG sec. 4.15 (p.58-59, l.2829-2858): "MOV Xd, #0 / MOV Xd, XZR / MOV Wd, #0 / MOV Wd, WZR / MOV
Wd, Wn / MOV Xd, Xn" (and two AArch32 forms) "are executed with zero latency. These instructions do
not utilize the scheduling and execution resources of the machine." Implementation: the stock
`NeoverseZeroMove` variant on `MOVZ[WX]i` and `ORR[WX]rs` gives `N2Write_0c` (latency 0, no
resource), and `CpuTraits::ZeroLatencyMov` lets Facile use the latency 0 ([docs.md](docs.md) 4.B).
The same predicate also covers FMOV Hd/Sd/Dd from WZR/XZR and MOVI Dd, #0 / MOVI Vd.2D, #0, which
the N2 model had bound to it; the A78 list does not name them. They now take their rows: FMOV from
a GPR 3 1 M0 (l.1310), MOVI 2 2 V (l.1947). The research binaries contain 16,642 `movi d/2d, #0`
and no FMOV from a zero register. MOV Vd, Vn (ORR vector) is not bound to the variant and is "ASIMD
logical" 2 2 V. "The last 3 instructions may not be executed with zero latency under certain
conditions" (l.2858): [D-1](#d-1). Test: `ZeroLatencyMovs`.

<a id="c-5"></a>
**C-5. Instruction fusion.**
SOG sec. 4.14 (p.58, l.2805-2825), compared row by row with `computeA78FusedMask` /
`isA78FusiblePair` (`src/facile/fusion.cpp`; the earlier comparison is [docs.md](docs.md) 4.C):

| SOG row | Implementation |
|---|---|
| 1, 2. CMP/CMN (immediate), (register) + B.cond | SUBS/ADDS (any form) with a ZR destination, then `Bcc` |
| 3, 4. CMP (immediate), (register) + CSEL | SUBS with a ZR destination, then CSEL (not CMN) |
| 5, 6. CMP (immediate), (register) + CSET | SUBS with a ZR destination, then CSINC Rd, ZR, ZR (CSET; not CSETM) |
| 7, 8. TST (immediate), (register) + B.cond | ANDS with a ZR destination, then `Bcc` |
| 9. BICS (register) + B.cond | any BICS (also with a real destination), then `Bcc` |
| 10. NOP + Any instruction | `NOP` or `HINT #0`, then any instruction |
| AESE + AESMC, AESD + AESIMC | not in the mask ([D-11](#d-11), [U-6](#u-6)) |

Pairs must be adjacent (l.2825); pairing is greedy and non-overlapping. The "(register)" forms:
[U-11](#u-11); accounting of a fused pair: [D-11](#d-11). Test: `Fusion`.

<a id="c-6"></a>
**C-6. Latency seen by dependent instructions.**
SOG sec. 3.1 (l.495-496). Implementation: `getEdgeLatency` uses the latency of the def that the
consumer reads (so the base update of a post-index load has 1, not the load's 4) and subtracts the
consumer's `ReadAdvance` (the accumulator paths of [C-10](#c-10), [C-14](#c-14), [C-18](#c-18),
[C-19](#c-19), [C-24](#c-24)), with a floor of 1 cycle except for the zero-latency MOVs. The address
operands have `ReadAdrBase` advance 0: an ALU result is used as an address after its own latency,
no extra cycle (the decision of the Cortex-A55, Cortex-A520 and Cortex-A76 audits). Test:
`ZeroLatencyMovs`, `DivideAndMultiply`.

<a id="c-7"></a>
**C-7. Stores are split into address and data µOPs; store-to-load forwarding.**
SOG sec. 3.10 (p.25, l.1102-1104): "Stores µOPs are split into address and data µOPs"; sec. 3.14
and 3.19 (l.1455, l.2245): "split into store address and store data µOPs at dispatch time"; sec. 4.6
(p.53, l.2554-2564): forwarding needs the load to start at the start or the middle of the older
store (not for LDP of two 32-bit registers), loads larger than 8 bytes from at most two stores,
smaller ones from one. Implementation (shared Facile code, [docs.md](docs.md) sections 4.F and 4.G):
the base update of a store is a node of its own that does not wait for the data, and a store ->
load edge is left out when the two register-offset accesses are provably disjoint. The size rules
are not checked; the latency of the forwarding is not stated ([D-7](#d-7)).

### 3.2 Instruction tables, integer

<a id="c-8"></a>
**C-8. Sec. 3.3, branches (p.15, l.528-537).**

| SOG group | SOG | Model |
|---|---|---|
| B; BR, RET; CBZ, CBNZ, TBZ, TBNZ | 1 / 2 / B | `N2Write_1c_1B` (two branch pipes) |
| BL; BLR | 1 / 2 / B, S | `N2Write_1c_1B_1S` (two µOPs) |

B.cond is not listed; it has `WriteBr` like B. MCA gives every call latency 0 ([X-2](#x-2));
Facile floors the edge to 1. Test: `Branches`.

<a id="c-9"></a>
**C-9. Sec. 3.4, arithmetic and logical (p.15-16, l.553-586).**

| SOG group | SOG | Model |
|---|---|---|
| ALU, basic: ADD, ADC, AND, BIC, EON, EOR, ORN, ORR, SUB, SBC | 1 / 4 / I | `WriteI` = `N2Write_1c_1I` |
| ALU, basic, flagset: ADDS, ADCS, ANDS, BICS, SUBS, SBCS | 1 / 3 / I | `N2Write_1c_1I_1Flg` ([D-9](#d-9)); ANDS/BICS register via `N2Write_Logical` (`NeoverseNoLSL`) |
| ALU, extend and shift: ADD{S}, SUB{S} | 2 / 2 / M | `WriteIEReg` = `N2Write_2c_1M`; flag-setting `N2Write_2c_1M_1Flg` ([U-4](#u-4)) |
| Arithmetic, LSL shift <= 4 (flagset or not) | 1 / 4; 1 / 3 / I | `N2Write_Arith` / `N2Write_ArithI`, `IsCheapLSL` |
| Arithmetic, LSR/ASR/ROR shift or LSL > 4 | 2 / 2 / M | the same variants otherwise |
| Conditional compare; conditional select | 1 / 3; 1 / 4 / I | `N2Write_1c_1I_1Flg`; `WriteI` |
| Logical, shift, no flagset; flagset (ANDS, BICS) | 1 / 4 / I; 2 / 2 / M | `N2Write_1c_1I` (ORR: zero-move variant); `N2Write_Logical` shifted -> `N2Write_2c_1M_1Flg` |

Test: `ArithmeticAndLogical`, `DispatchAndPipelines`.

<a id="c-10"></a>
**C-10. Sec. 3.6, divide and multiply (p.18-19, l.695-791).**

| SOG group | SOG | Model |
|---|---|---|
| **Divide, W-form; X-form: SDIV, UDIV** | 5 to 12; 5 to 20 / 1/12 to 1/5; 1/20 to 1/5 / M0 (note 1) | `A78Write_5c_1M0_IDiv`: 5, M0 1 cycle, divider 5 cycles (*fixed 2026-10-31*, `[a78_div_best]`, [D-1](#d-1), [D-3](#d-3)) |
| Multiply: MUL, MNEG | 2 / 2 / M | `N2Wr_IMUL` with a zero accumulator -> `N2Wr_IM` 2, M |
| Multiply accumulate, W-form; X-form: MADD, MSUB | 2(1) / 1 / M0 (note 2) | `N2Wr_IMA` 2, M0 + `N2Rd_IMA` (advance 1) |
| **Multiply accumulate long: SMADDL, SMSUBL, UMADDL, UMSUBL** | 2(1) / 2 / M | `A78Wr_IMULL` -> `A78Wr_IMAL` 2, M + `N2Rd_IMA` (*fixed*, `[a78_maddl]`; was M0, 1 per cycle) |
| Multiply high: SMULH, UMULH | 3 / 2 / M | `N2Write_3c_1M` |
| Multiply long: SMNEGL, SMULL, UMNEGL, UMULL | 2 / 2 / M | zero accumulator -> `N2Wr_IM` |

The divide used to be the worst case, latency 12 / 20 with M0 held 12 / 20 cycles, of which a
Facile C++ correction capped only the occupancy at 5. Note 2 (l.789-791), "Accumulator forwarding is
not supported for consumers of 64 bit multiply high operations": `N2Rd_IMA` names only the
multiply-accumulate writes. Measured: `sdiv w1, w1, w2` + branch 12.0 -> 5.0; `udiv x1, x2, x3` +
two `madd` + branch 7.0 -> 5.0; four independent `smaddl` + branch 4.0 -> 2.0. The research binaries
contain 8,803 SDIV/UDIV and 15,992 SMADDL-family instructions with an accumulator. Forwarding
between forms: [U-5](#u-5). Test: `DivideAndMultiply`.

<a id="c-11"></a>
**C-11. Sec. 3.8, miscellaneous data processing (p.21, l.881-899).**

| SOG group | SOG | Model |
|---|---|---|
| Address generation: ADR, ADRP | 1 / 4 / I | `WriteI` |
| **Bitfield extract, one reg; two regs: EXTR** | 1 / 4 / I; 3 / 2 / I, M (note 1) | `N2Write_Extr`: `A78IsExtrOneRegPred` (Rn == Rm **or imm == 0**) -> `N2Write_1c_1I`, else `N2Write_3c_1I_1M` (*fixed 2026-10-31*, `[a78_extr]`; the stock predicate checked only Rn == Rm) |
| Bitfield move, basic: SBFM, UBFM; insert: BFM | 1 / 4 / I; 2 / 2 / M | `WriteIS`; `N2Write_2c_1M` |
| Count leading; move immed; reverse; variable shift | 1 / 4 / I | `WriteI` / `WriteImm` / `WriteIS` 1, I |

Measured: `extr x1, x1, x2, #0` + branch 3.0 -> 1.0. The research binaries contain 34 EXTR in all.
Test: `MiscDataProcessing`.

<a id="c-12"></a>
**C-12. Sec. 3.9, loads (p.23-24, l.956-1020).**

| SOG group | SOG | Model |
|---|---|---|
| Load register, literal, unscaled, unprivileged, unsigned immed | 4 / 3 / L | `WriteLD` = `N2Write_4c_1L` |
| Load register, immed post-index / pre-index | 4 / 3 / L, I | `WriteAdr` (1, I) + `WriteLD` |
| **Load register, register offset: basic; scale by 4/8; extend; extend, scale by 4/8** | 4 / 3 / L | `WriteLDIdx` = `N2Write_4c_1L` (`[a78_reg_offset]`; the stock class has an I µOP, which a Facile C++ correction used to remove) |
| **Load register, register offset, scale by 2; extend, scale by 2: LDRH, LDRSH** | 5 / 3 / I, L | `A78WriteLDRH`: `A78IsMemScaled` -> `A78Write_5c_1I_1L`, else `N2Write_4c_1L` (*fixed 2026-10-31*, `[a78_reg_offset_h]`; was 4, I, L) |
| Load pair, W-form; X-form: LDP, LDNP | 4 / 3; 4 / 3/2 / L | `WriteLD` / `N2Write_4c_2L` + `WriteLDHi` (4, no µOP, [D-10](#d-10)) |
| **Load pair, signed words: LDPSW (offset, pre, post)** | 5 / 3/2 / I, L | `A78Write_5c_1I_2L` (L held 2 cycles) + `A78Write_5c_0Z` (*fixed*, `[a78_ldpsw]`; was M0 held 5 cycles and no load pipe, ports corrected by the Facile C++, second register 4) |
| Load pair, post-index / pre-index, W; X | 4 / 3; 4 / 3/2 / L, I | `WriteAdr` + `WriteLD` / `N2Write_4c_2L` + `WriteLDHi` |

"Scaled" is the S bit of the addressing mode (`AArch64_AM::getMemDoShift`), so `ldrh w1, [x0, x1]`
and `ldrh w1, [x0, w1, uxtw]` stay 4. The pre/post-index LDPSW also has `WriteAdr`, i.e. two I µOPs
for "I, L" ([I-7](#i-7)). Measured: `ldrh w1, [x0, x1, lsl #1]` + branch 4.0 -> 5.0; `ldpsw x1, x2,
[x0]; add x0, x0, x2` + branch 5.0 -> 6.0. The research binaries contain 411,894 integer
register-offset loads and stores of the forms without an I µOP, 3,542 LDRH/LDRSH scaled by 2 and 758
LDPSW. Acquire and exclusive loads: [U-7](#u-7). Test: `Loads`.

<a id="c-13"></a>
**C-13. Sec. 3.10, stores (p.25-26, l.1101-1141).**

| SOG group | SOG | Model |
|---|---|---|
| Store register, unscaled; unprivileged; unsigned immed | 1 / 2 / L01, D | `WriteST` = `N2Write_1c_1L01_1D` |
| Store register, immed post-index / pre-index | 1 / 2 / L01, D, I | `WriteAdr` + `WriteST` |
| **Store register, register offset: basic; scaled by 4/8; extend; extend, scale by 4/8** | 1 / 2 / L01, D | `WriteSTIdx` = `N2Write_1c_1L01_1D` (`[a78_reg_offset]`; same history as C-12) |
| **Store register, register offset, scaled by 2; extend, scale by 1 (sic, [I-3](#i-3)): STRH** | 2 / 2 / I, L01, D | `A78WriteSTRH`: scaled -> `A78Write_2c_1I_1L01_1D`, else `N2Write_1c_1L01_1D` (*fixed 2026-10-31*, `[a78_store_rows]`; latency was 1) |
| Store pair, immed offset; post-index; pre-index: STP, STNP | 1 / 2 / L01, D (+ I) | `WriteSTP` = `N2Write_1c_1L01_1D` (+ `WriteAdr`) |

A store's latency matters only on Facile's store -> load edges ([D-7](#d-7)). The data operand reads
`ReadST` ([B-1](#b-1)). The research binaries contain 2,015 STRH scaled by 2. Release and exclusive
stores: [U-7](#u-7). Test: `Stores`.

### 3.3 Instruction tables, FP and ASIMD

<a id="c-14"></a>
**C-14. Sec. 3.11, FP data processing (p.27-28, l.1205-1293).**

| SOG group | SOG | Model |
|---|---|---|
| FABS; FADD, FSUB; FMIN/FMAX(NM); FNEG; FCSEL | 2 / 2 / V | `WriteF` = `N2Write_2c_1V` |
| FCMP{E}, FCCMP{E} | 2 / 1 / V0 | `WriteFCmp` = `N2Write_2c_1V0` |
| FDIV H; S; D | 7 / 4/7; 7 to 10 / 4/9 to 4/7; 7 to 15 / 1/7 to 2/7 / V0 (note 1) | `N2WriteFDS_7c2_7`, `_7c2_10`, `_7c4_15`: 7, V0 1 cycle, FP divide unit 2, 2, 4 cycles (`[a78_fdiv_best]`, [D-1](#d-1), [D-3](#d-3)) |
| FSQRT H; S; D | 7 / 4/7; 7 to 9 / 1/2 to 4/7; 7 to 16 / 2/15 to 2/7 / V0 | `N2WriteFDS_7c2_7`, `_7c2_9`, `_7c4_16` |
| FMUL, FNMUL | 3 / 2 / V (note 2) | `WriteFMul` 3, V |
| FMADD, FMSUB, FNMADD, FNMSUB | 4 (2) / 2 / V (note 3) | `N2Wr_FMA` 4 + `N2Rd_FMA` (advance 2 from `WriteFMul` and `N2Wr_FMA`) |
| **FRINT*** | 2 / 1 / V0 | `N2Write_2c_1V0` (*fixed 2026-10-31*, `[a78_frint]`; was 3) |

Note 2 (l.1288-1290): an FP multiply-accumulate "can potentially be issued 1 cycle after the FP
multiply µOP has been issued", i.e. FMUL -> accumulator 1: `N2Rd_FMA` names `WriteFMul`, which this
model declares with `WriteRes`, so the advance applies. Measured: `fmul d0, d1, d2; fmadd d3, d4,
d5, d0; fmov d1, d3` + branch 7.0 (1 + 4 + 2); `frinta d0, d0` + branch 3.0 -> 2.0; `fdiv d0, d2, d1;
fcvtzs w1, d3; frinta d4, d5` + branch 5.0 -> 4.0. The research binaries contain 49,675 FDIV/FSQRT
and 557 scalar FRINT. Test: `FpScalar`.

<a id="c-15"></a>
**C-15. Sec. 3.12, FP miscellaneous (p.29, l.1297-1314).**

| SOG group | SOG | Model |
|---|---|---|
| **FP convert, from vec to vec reg: FCVT, FCVTXN** | 2 / 1 / V0 | `N2Write_2c_1V0` (*fixed 2026-10-31*, `[a78_fp_cvt]`; FCVT was 3, scalar FCVTXN 2 on V) |
| FP convert, from gen to vec reg: SCVTF, UCVTF | 3 / 1 / M0 | `A78Write_3c_1M0`, M0 one cycle (`[a78_m0_pipelined]`; the N2 class holds M0 3 cycles, a Facile C++ correction used to cap it) |
| **FP convert, from vec to gen reg: FCVT[AMNPZ][SU]** | 3 / 1 / V0 | `N2Write_3c_1V0` (*fixed*, `[a78_fp_cvt]`; the integer forms were on V); fixed-point forms `WriteFCvt` = `N2Write_3c_1V0` |
| FP move, immed; register | 2 / 2 / V | `WriteFImm`; `N2Write_2c_1V` |
| FP transfer, gen to low half: FMOV | 3 / 1 / M0 | `A78Write_3c_1M0` (`[a78_m0_pipelined]`, `[a78_zero_mov]`, [C-4](#c-4)) |
| FP transfer, gen to high half: FMOV | 5 / 1 / M0, V | `N2Write_5c_1M0_1V` |
| **FP transfer, vec to gen reg: FMOV** | 2 / 1 / V1 | `WriteFCopy` = `N2Write_2c_1V1` (*fixed*, `[a78_fp_cvt]`; was V) |

The research binaries contain 22,072 scalar FCVT, 7,847 FCVT[AMNPZ][SU] to a GPR, 2,790 FMOV to a
GPR, 17,757 SCVTF/UCVTF and 29,564 FMOV from a GPR. Measured: `fcvt s0, d0; fcvt d0, s0` + branch 6.0
-> 4.0; two `fcvtzs w, d` + branch 1.0 -> 2.0; two `fmov x, d` + branch 1.0 -> 2.0. Conversions with
the operand and the result in FP registers: [D-4](#d-4). Test: `FpScalar`.

<a id="c-16"></a>
**C-16. Sec. 3.13, FP loads (p.30-31, l.1366-1419; checked in the PDF).**

| SOG group | SOG | Model |
|---|---|---|
| Load vector reg, literal; unscaled immed; unsigned immed | 5 / 3 / L | `A78Write_5c_1L` |
| **Load vector reg, immed post-index / pre-index** | 5 / 3 / L, I | `WriteAdr` + `A78Write_5c_1L` (*fixed 2026-10-31*, `[a78_fp_ld]`; the load took a second I µOP) |
| **Load vector reg, register offset, basic** | 5 / 3 / L, I | X index without the scale: `A78Write_5c_1I_1L` (*fixed*, `[a78_reg_offset_h]`, [I-4](#i-4); B/S/D had no I, H/Q were 6) |
| Load vector reg, register offset, scale, S/D-form; extend; extend, scale, S/D-form | 5 / 3 / L | `A78Write_5c_1L` (**extend, H/Q** *fixed*: was 6, I, L) |
| Load vector reg, register offset, scale, H/Q-form; extend, scale, H/Q-form | 6 / 3 / I, L | `A78Write_6c_1I_1L` |
| Load vector pair, immed offset, S/D; Q: LDP, LDNP | 5 / 3; 5 / 3/2 / L | `A78Write_5c_1L`; `A78Write_5c_2L` + `A78Write_5c_0Z` (*fixed*, [D-10](#d-10): the second register was 4) |
| **Load vector pair, post/pre-index, S/D; Q** | 5 / 3; 5 / 3/2 / L, I | `WriteAdr` + `A78Write_5c_1L` / `A78Write_5c_2L` + `A78Write_5c_0Z` (*fixed*; the load took one or two more I µOPs) |

The Neoverse N2 rows had latency 6 (7 for H/Q register offset); an uncommitted change of another
session had set the A78 latencies, which this audit kept, and its rows for the pre/post-index and
register-offset forms were corrected as above. The variants are `A78WriteLDRSDroX` (S/D, X index),
`A78WriteLDRHQroX`, `A78WriteLDRHQroW`; B has no "scale" row, so LDR B with an X index is "basic"
and with a W index "extend". The research binaries contain 71,860 B/S/D and 2,342 H/Q loads with an
unscaled X index, 79,965 FP LDP/LDNP with an immediate offset and 677 pre/post-index FP LDP.
Measured: three `ldr q, [xN], #16` + branch 1.5 -> 1.0; `ldp d0, d1, [x0]; fadd d1, d1, d1; fmov x0,
d1` + branch 8.0 -> 9.0; three `ldr d, [x0, xN]` and three `add` + branch 1.17 -> 1.5. Test:
`FpLoadsAndStores`.

<a id="c-17"></a>
**C-17. Sec. 3.14, FP stores (p.32-33, l.1454-1523; checked in the PDF).**

| SOG group | SOG | Model |
|---|---|---|
| Store vector reg, unscaled immed; unsigned immed, B/H/S/D/Q | 2 / 2 / L01, V | `N2Write_2c_1L01_1V` |
| **Store vector reg, immed post-index / pre-index, B/H/S/D/Q** | 2 / 2 / L01, V, I | `WriteAdr` + `N2Write_2c_1L01_1V` (*fixed 2026-10-31*, `[a78_fp_st]`; the store took a second I µOP) |
| Store vector reg, register offset: basic; extend (B/H/S/D/Q); scale S/D, Q; extend, scale S/D | 2 / 2 / L01, V | `N2Write_2c_1L01_1V` |
| **Store vector reg, register offset, scale, H; extend, scale, H; extend, scale, Q** | 2 / 2 / I, L01, V | `A78WriteSTRHro`: scaled -> `A78Write_2c_1I_1L01_1V` (*fixed*, `[a78_store_rows]`, [I-5](#i-5); had no I) |
| Store vector pair, immed offset, S; D | 2 / 2 / L01, V | `N2Write_2c_1L01_1V` |
| **Store vector pair, immed offset, Q-form** | 2 / 2 / L01, V | `N2Write_2c_1L01_1V` (*fixed*, `[a78_fp_st]`, [I-6](#i-6); was two V µOPs, 1 per cycle) |
| **Store vector pair, post/pre-index, S; D** | 2 / 2 / I, L01, V | `WriteAdr` + `N2Write_2c_1L01_1V` (*fixed*; two I µOPs) |
| **Store vector pair, post/pre-index, Q** | 2 / 1 / I, L01, V | `WriteAdr` + `N2Write_2c_1L01_2V` (*fixed*: the post-index form gave the base register latency 2, the pre-index form had two I µOPs) |

The base-update latency 1 is the "(1)" of note 3 (l.1537-1542). The data operands read `ReadST`
([B-1](#b-1)). The research binaries contain 79,856 STP/STNP Q with an immediate offset, 254 STP Q
pre/post-index, 5,684 FP STR pre/post-index and 4 scaled STR H or Q with a W index. Measured: two
`stp q, q, [xN]` + branch 2.0 -> 1.0. Test: `FpLoadsAndStores`.

<a id="c-18"></a>
**C-18. Sec. 3.15, ASIMD integer (p.34-37, l.1554-1679, notes l.1752-1759).**

| SOG row | SOG | Model |
|---|---|---|
| Absolute diff; arith basic, complex, pair-wise; compare; logical; max/min basic and pair-wise; absolute diff long | 2 / 2 / V | `WriteVd` / `WriteVq` = `N2Write_2c_1V` |
| **SABA, UABA; SABAL(2), UABAL(2)** | 3(1) / 1 / V1 (note 2) | `N2Wr_VA` 3 + `N2Rd_VA` (advance 2) (*fixed 2026-10-31*, `[a78_asimd_acc]`, [I-1](#i-1); was 4(1)) |
| Reduce 4H/4S; 8B/8H: ADDV, SADDLV, UADDLV and SMAXV, ... | 2 / 1 / V1; 4 / 1 / V1, V | `N2Write_2c_1V1`; `N2Write_4c_1V1_1V` |
| **Reduce 16B: ADDV, SADDLV, UADDLV** | 4 / 1/2 / V1 | `A78Write_4c_1V1_2cyc` (*fixed*, `[a78_v_throughput]`; throughput was 1) |
| Max/min reduce 16B | 4 / 1/2 / V1 | `N2Write_4c_2V1` |
| **Dot product: SDOT, UDOT** | 2(1) / 2 / V (note 2) | `N2Wr_VDOT` 2 + `N2Rd_VDOT` (advance 1) (*fixed*, `[a78_asimd_acc]`; was 3(1)) |
| Multiply: MUL, SQDMULH, SQRDMULH | 4 / 1 / V0 | `N2Write_4c_1V0` (also by element) |
| Multiply accumulate: MLA, MLS; long: [SU]MLAL(2), [SU]MLSL(2) | 4(1) / 1 / V0 (note 1) | `N2Wr_VMA` / `N2Wr_VMAL` + advance 3 |
| **Multiply accumulate high: SQRDMLAH, SQRDMLSH** | 4 / 1 / V0 | `N2Write_4c_1V0` (*fixed*, `[a78_asimd_acc]`: the N2 row forwarded the accumulator, advance 2; the SOG gives no accumulate latency and no note) |
| Multiply accumulate saturating long: SQDMLAL(2), SQDMLSL(2) | 4 / 1 / V0 | `N2Write_4c_1V0` |
| PMUL, PMULL(2) 8x8, D; Q | 3 / 1 / V0 (note 3) | `N2Write_3c_1V0` |
| **Multiply long: SMULL(2), UMULL(2), SQDMULL(2)** | 3 / 1 / V0 | `N2Write_3c_1V0` (*fixed*, `[a78_asimd_mull]`; was V, 2 per cycle) |
| SADALP, UADALP; SSRA, SRSRA, USRA, URSRA | 4(1) / 1 / V1 (note 2) | `N2Wr_VPA` / `N2Wr_VSA` + advance 3 |
| Shift by immed basic; and insert; by register basic | 2 / 1 / V1 | `N2Write_2c_1V1` |
| Shift by immed, complex; by register, complex | 4 / 1 / V1 | `N2Write_4c_1V1` |

The research binaries contain 18 SABA/UABA(L), 4 vector SMULL/UMULL/SQDMULL and no SDOT/UDOT,
SQRDMLAH or ADDV 16B. Scalar forms: [D-5](#d-5). Measured (test): two dependent `saba` 2.0 per
iteration (1 + 1). Test: `Asimd`.

<a id="c-19"></a>
**C-19. Sec. 3.16, ASIMD FP (p.38-40, l.1771-1861, notes l.1912-1920).**

| SOG row | SOG | Model |
|---|---|---|
| FABS, FABD, FADD, FSUB, FADDP, compare, max/min normal and pairwise, FNEG | 2 / 2 / V | `WriteVd` / `WriteVq` |
| FCVTL, FCVTN: F16<->F32; F32<->F64, FCVTXN | 4 / 1/2; 3 / 1 / V0 | `N2Write_4c_2V0`; `N2Write_3c_1V0` |
| Convert other, D F32 and Q F64; D F16 and Q F32; Q F16 | 3 / 1; 4 / 1/2; 6 / 1/4 / V0 | `N2Write_3c_1V0`; `N2Write_4c_2V0`; `N2Write_6c_4V0`, **also the fixed-point forms** (`..._shift`, *fixed 2026-10-31*, `[a78_fp_cvt]`; they were 2 on V) |
| FDIV, FSQRT (all forms) | ranges / V0 (note 3) | `N2WriteFDS_*`, best values ([D-1](#d-1), [D-3](#d-3)): D F16 7/7, D F32 7/4, Q F16 10/10 (FSQRT 11/11), Q F32 7/7, Q F64 7/7 (latency / divide unit cycles) |
| **FMAXV etc. reduce, F32 and D-form F16; Q-form F16** | 4 / 1; 6 / 2/3 / V | `A78WriteV<4, 1, 2>`; `A78WriteV<6, 2, 3>` (*fixed*, `[a78_v_throughput]`; throughput was 2 and 1) |
| FMUL, FMULX; FMLA, FMLS | 3 / 2; 4 (2) / 2 / V (notes 1, 2) | `N2Wr_VFM`; `N2Wr_VFMA` + `N2Rd_VFMA` (from `N2Wr_VFM` and `N2Wr_VFMA`) |
| FRINT*: D F32 and Q F64; D F16 and Q F32; Q F16 | 3 / 1; 4 / 1/2; 6 / 1/4 / V0 | `N2Write_3c_1V0`; `N2Write_4c_2V0`; `N2Write_6c_4V0` |

No FMAXV-family instruction occurs in the research binaries. FMLAL: [U-12](#u-12). Test: `Asimd`.

<a id="c-20"></a>
**C-20. Sec. 3.17, ASIMD miscellaneous (p.41-42, l.1930-1994).**

| SOG row | SOG | Model |
|---|---|---|
| RBIT; BIF, BIT, BSL; CLS, CLZ, CNT; DUP element; EXT; XTN; INS element; FMOV, MOVI; REV; TBL 1-2 regs; TBX 1 reg; TRN; UZP, ZIP | 2 / 2 / V | default `N2Write_2c_1V`; MOVI #0 [C-4](#c-4) |
| DUP gen reg | 3 / 1 / M0 | `A78Write_3c_1M0`, M0 one cycle (`[a78_m0_pipelined]`; the N2 class held M0 3 cycles, a Facile C++ correction used to cap it) |
| SQXTN(2), SQXTUN(2), UQXTN(2) | 4 / 1 / V1 | `N2Write_4c_1V1` |
| URECPE, URSQRTE D; Q | 3 / 1; 4 / 1/2 / V0 | `N2Write_3c_1V0`; `N2Write_4c_2V0` |
| FRECPE, FRSQRTE: D F32 and scalar; D F16 and Q F32; Q F16 | 3 / 1; 4 / 1/2; 6 / 1/4 / V0 | `N2Write_3c_1V0`; `N2Write_4c_2V0`; `N2Write_6c_4V0` ([U-8](#u-8)) |
| FRECPX; FRECPS, FRSQRTS | 3 / 1 / V0; 4 / 2 / V | `N2Write_3c_1V0`; `N2Write_4c_1V` |
| TBL 3 regs; **4 regs** | 4 / 1; 4 / 2/3 / V | `N2Write_4c_2V`; `A78WriteV<4, 4, 3>` (*fixed*, `[a78_v_throughput]`; was 1/2) |
| TBX 2; **3; 4 regs** | 4 / 1; 6 / 2/3; 6 / 2/5 / V | `N2Write_4c_2V`; `A78WriteV<6, 4, 3>`; `A78WriteV<6, 8, 5>` (*fixed*; were 1/2 and 1/4) |
| **UMOV, SMOV** | 2 / 1 / V1 | `N2Write_2c_1V1` (*fixed*, `[a78_fp_cvt]`; were V) |
| INS gen reg | 5 / 1 / M0, V | `N2Write_5c_1M0_1V` |

The research binaries contain 1,207 DUP from a GPR, 2,501 UMOV/SMOV (including the MOV alias) and 62
TBL with four registers or TBX. Measured: two `umov` + branch 1.0 -> 2.0; `tbl` with four table
registers + branch 2.0 -> 1.5. Test: `Asimd`.

<a id="c-21"></a>
**C-21. Sec. 3.18, ASIMD loads (p.43-46, l.2060-2172).**
Checked row by row (`[a78_asimd_ld]`, *fixed 2026-10-31* where marked):

| SOG row | SOG | Model |
|---|---|---|
| **LD1 1 reg; 2 reg; 3 reg (D and Q)** | 5 / 3; 5 / 3/2; 5 / 1 / L | `A78Write_5c_1L`; `A78Write_5c_2L`; `A78Write_5c_3L` (were latency 6) |
| **LD1 4 reg, D-form** | 5 / 3/2 / L | `A78Write_5c_4L_2cyc`: 4 µOPs, L held 2 cycles ([D-6](#d-6); was 7, 3/4) |
| **LD1 4 reg, Q-form** | 6 / 3/4 / L | `A78Write_6c_4L` (was 7) |
| **LD1 one lane; LD1R** | 7 / 2 / L, V | `A78Write_7c_1L_1V` (was 8) |
| **LD2 multiple D; Q; one lane; LD2R** | 7 / 1 / L, V | `A78Write_7c_1L_2V`; `A78Write_7c_2L_2V`; `A78Write_7c_1L_2V` (were 8) |
| LD3 multiple, one lane, LD3R | 8 / 2/3 / L, V | `N2Write_8c_2L_3V` / `N2Write_8c_3L_3V` |
| LD4 multiple D; one lane; LD4R | 8 / 1/2 / L, V | `N2Write_8c_3L_4V` / `N2Write_8c_4L_4V` |
| **LD4 multiple, Q-form** | 10 / 1/4 / L, V | `A78WriteLDV<10, 8, 12, 4>` (L held 12 cycles, [D-6](#d-6); was 9, 1/2) |

Writeback forms add `WriteAdr` (1, I), as "(1) +I" (l.2172). The research binaries contain 869
LD1-LD4/ST1-ST4 in all (14 LD4/ST4). Sec. 4.18: [U-2](#u-2). Test: `AsimdLoadsAndStores`.

<a id="c-22"></a>
**C-22. Sec. 3.19, ASIMD stores (p.47-48, l.2244-2314).**
Checked row by row (`[a78_asimd_st]`). Unchanged: ST1 one to four registers (2 / 2, 2 / 1, 2 / 2/3,
2 / 1/2 as printed per form), ST4 one lane B/H/S (6 / 2/3). The following were up to twice as fast as
the SOG and are now `A78WriteSTV` (L01 held 2/T cycles, [D-6](#d-6); *fixed 2026-10-31*,
`[a78_ls_throughput]`): ST1 one lane (4 / 1), ST2 D (4 / 1), **ST2 Q (5 / 1/2, latency was 4)**, ST2
one lane (4 / 1), ST3 D (5 / 1/2), ST3 Q (6 / 1/3), **ST3 one lane (5 / 1/2, latency was 6)**, ST4 D
(6 / 1/3), ST4 Q B/H/S (7 / 1/6), ST4 Q D (5 / 1/4), ST4 one lane D (4 / 1/2). Writeback "(1) Add I"
(l.2314): `WriteAdr`. The data operands read `ReadST` ([B-1](#b-1)). Measured: two `st2 {.4s}` +
branch 2.0 -> 4.0. Test: `AsimdLoadsAndStores`.

<a id="c-23"></a>
**C-23. Sec. 3.20, cryptography (p.49, l.2367-2384).**
AESD, AESE, AESIMC, AESMC 2 / 2 / V: `N2Write_2c_1V`; PMULL 64x64, SHA1H, SHA1SU0, SHA1SU1,
SHA256SU0, SHA256SU1 2 / 1 / V0: `N2Write_2c_1V0`; SHA1C/M/P, SHA256H/H2 4 / 1 / V0:
`N2Write_4c_1V0`. Sec. 4.7 (p.53, l.2569-2570): "two AESE/AESMC/AESD/AESIMC instruction every
cycle (fully pipelined) with an execution latency of two cycles". AES pairs: [U-6](#u-6). None occurs
in the research binaries. Test: `CryptoAndCrc`.

<a id="c-24"></a>
**C-24. Sec. 3.21, CRC (p.50, l.2420-2436).**
CRC32, CRC32C 2 / 1 / M0, note 1: "late-forwarding of the result from a producer µOP to a consumer
µOP. This results in a 1 cycle reduction in latency": `N2Wr_CRC` 2, M0 + `N2Rd_CRC` (advance 1).
Measured (test): `crc32x w1, w1, x3` 1.0 per iteration. Test: `CryptoAndCrc`.

---

## 4. Not compatible with the SOG (N)

None open. The differences found by this audit where the SOG states a value unambiguously were
fixed; they are the C items marked *fixed 2026-10-31*.

---

## 5. Deliberate deviations and interpretations (D)

<a id="d-1"></a>
**D-1. Best case: shortest latency, best throughput, L1 hits, aligned accesses, zero-latency MOVs.**
SOG: integer divide "5 to 12" / "5 to 20", "Early termination is possible, depending upon the data
values" (l.699-700, l.787-788); FP divide and square root ranges (l.1213-1215, l.1239-1241,
l.1809-1813, l.1852-1861); load latencies assume an L1 hit (l.957, l.1367, l.2061); unaligned
penalties (sec. 4.5, l.2545-2551); "MOV Wd, Wn / MOV Xd, Xn ... may not be executed with zero latency
under certain conditions" (l.2858). Implementation (project policy: a data-dependent value takes its
best value; occupancy = best throughput rounded up): integer divide latency 5 and divider 5 cycles
for both forms (`[a78_div_best]`); FP divide / square root the shortest latency and `ceil(1 / best
throughput)` cycles ([docs.md](docs.md) section 4.I); L1 hits; aligned accesses; register MOVs always
zero latency. Effect: optimistic for divide-heavy code.

<a id="d-2"></a>
**D-2. Throughput as resource occupancy.**
SOG: throughput in instructions per cycle for the whole core (l.498-500). Implementation: a row with
throughput T on `n` pipelines holds them `n / T` cycles in total (ceil where needed); e.g. LDPSW 2
L-cycles (T = 3/2 on three L pipes), ADDV 16B 2 V1-cycles (T = 1/2), ST4 Q B/H/S 12 L01-cycles (T =
1/6). Facile divides the cycles of each resource by its units, so another instruction may use the
other unit meanwhile, which the SOG neither states nor excludes.

<a id="d-3"></a>
**D-3. The dividers are resources of their own.**
*Introduced 2026-10-31* (`[a78_dividers]`).
SOG: "Integer divides are performed using an iterative algorithm and block any subsequent divide
operations until complete" (sec. 3.6 note 1, l.787); "FP divide and square root operations are
performed using an iterative algorithm and block subsequent similar operations to the same pipeline
until complete" (sec. 3.11 note 1, l.1286-1287; sec. 3.16 note 3, l.1919-1920); "Each issue pipeline
can accept one µOP per cycle" (l.389-390). Implementation: a divide takes M0 (integer) or V0 (FP)
for one cycle and holds `A78UnitIDiv` / `A78UnitFDiv` for its occupancy; before, it held M0 or V0
itself, blocking every other M0 or V0 µOP (MADD, CRC, SCVTF, FMOV and DUP from a GPR, FP converts,
FRINT, FCMP, ...). Reason: the SOG names what a divide blocks (later divides / similar operations);
the reading that also blocks the pipeline has no SOG basis; the lower estimate is preferred
(project policy); the same decision was made for the Cortex-A76 ([a76-sog.md](a76-sog.md) D-3).
Measured: `udiv` + two `madd` + branch 7.0 -> 5.0 (with the latency of D-1); `fdiv; fcvtzs; frinta`
+ branch 5.0 -> 4.0. Test: `DivideAndMultiply`, `FpScalar`.

<a id="d-4"></a>
**D-4. FP conversions with operand and result in FP registers: 3 / 1 / V0.**
*Introduced 2026-10-31* (`[a78_fp_cvt_scalar]`).
SOG: no row names the scalar forms of FCVT[AMNPZ][SU], SCVTF, UCVTF whose operand and result are FP
registers (`scvtf d0, d1`, `fcvtzs s0, s1`, also with `#fbits`). The rows of these mnemonics give
3 / 1 (V0 from a vector register, M0 from a GPR, l.1302-1307), the ASIMD rows 3 / 1 / V0 for two
lanes, 4 / 1/2 for four and 6 / 1/4 for eight (l.1791-1808); "FP convert, from vec to vec reg" (2 /
1 / V0, l.1301) names only FCVT and FCVTXN. Implementation: `N2Write_3c_1V0`, i.e. one lane is taken
as the two-lane row, the lowest the SOG gives for these mnemonics, as for the Cortex-A76
([a76-sog.md](a76-sog.md) D-4). Before: the stock class, 2 on V, which no row supports. The research
binaries contain 10,434 such conversions. Measured: `scvtf d0, d0` + branch 2.0 -> 3.0. Test:
`FpScalar`.

<a id="d-5"></a>
**D-5. Scalar ASIMD forms take the row of their mnemonic.**
*Introduced 2026-10-31* (`[a78_asimd_scalar]`).
SOG: the ASIMD rows list mnemonics (SHL, SSHR, USHR, SLI, SRI, SRSHR, URSHR, SSRA, USRA, SRSRA,
URSRA, SQDMLAL, SQDMLSL, SQDMULL, SQSHRN, SQRSHRN, ...), without saying whether the scalar forms
(`shl d0, d1, #3`, `sqdmlal s0, h1, h2`, `sqshrn b0, h1, #3`) belong to them. Implementation: they
take the row of their mnemonic, as for the Cortex-A76 ([a76-sog.md](a76-sog.md) D-5). Before: the
stock default, 2 on V. The research binaries contain 56 scalar shifts by immediate. Test: `Asimd`.

<a id="d-6"></a>
**D-6. Load/store throughput that the µOPs do not give: expressed on the L / L01 pipelines.**
SOG: the rows of [C-21](#c-21) and [C-22](#c-22) marked there name "L, V" or "V, L01" (or "L") and a
throughput that their µOPs on three L (two L01) and two V pipelines do not give; which of the
pipelines limits is not stated. Sec. 4.18 (l.2883-2892) says that the bandwidth of some LD4/ST4
forms "is limited by decode constraints", without a number ([U-2](#u-2)). Implementation:
`A78WriteLDV<lat, uops, lcyc, vcyc>` / `A78WriteSTV<...>` keep the µOP count and the V cycles of the
write they replace and hold L for `3 / T` (L01 for `2 / T`) cycles in total; LD1 four registers
D-form (`5 3/2 L`) holds L for 2 cycles with its four µOPs. Reason: one rule for all rows, the rule
chosen for the Cortex-A76 ([a76-sog.md](a76-sog.md) D-7). Consequence: a mix of such an instruction
with other L users is limited together, with V users it is not.

<a id="d-7"></a>
**D-7. Store-to-load forwarding latency.**
SOG sec. 4.6 (l.2554-2564) gives the conditions of forwarding but no latency; the store tables give
store latencies (1 for integer stores, 2 or more for FP and ASIMD stores, "the minimum latency seen
by an operation dependent on" the store). Implementation (shared Facile code,
`addMemoryDependencies`, [docs.md](docs.md) section 4): a store -> load edge has the store's
latency, then the load's own latency applies; i.e. forwarding is taken to be no slower than an L1
hit.

<a id="d-8"></a>
**D-8. Not modelled: AArch32, dispatch stall, branch alignment, FPCR, Arm/Thumb, cache maintenance.**
The AArch32 tables, IT blocks (sec. 4.13) and mixed Arm/Thumb code (sec. 4.16) are not compared
(the research binaries are AArch64). The dispatch stall of sec. 4.2 (p.51, l.2470-2474: a V µOP with
more than one Q source partly written as S registers stalls three cycles, "only on the first such
instance") is not modelled: it does not repeat in a loop. Sec. 4.3, 4.4, 4.9 (no more than four
branches in an aligned 32-byte region, l.2673-2674), 4.10 and 4.17 give advice without a cost; not
modelled.

<a id="d-9"></a>
**D-9. Flag-setting throughput 3: a flags resource of three units.**
SOG: "ALU, basic, flagset ... 1 3 I", "Arithmetic, flagset, LSL shift, shift <= 4 ... 1 3 I",
"Conditional compare 1 3 I" (l.561-570), against "ALU, basic ... 1 4 I" on the same four I pipes.
The SOG does not say what limits them to three. Implementation (upstream N2): `N2UnitFlg`, a
resource of three units taken by every flag-setting integer operation (also the 2-cycle M forms),
which gives exactly the printed throughputs. Measured (test): six independent `adds` + branch 2.0.
Test: `DispatchAndPipelines`.

<a id="d-10"></a>
**D-10. Load pair: the second register has the row's latency and no µOP.**
*Introduced 2026-10-31* (`[a78_ldp_hi]`).
SOG: the load latencies "represent the maximum latency to load all the registers written by the
instruction" (l.957-958, l.1367-1368); the µOP count is not printed. Implementation: the second
destination of LDP W/X (`WriteLDHi`, 4), LDPSW and the FP pairs (`A78Write_5c_0Z`, 5) takes no µOP;
the row's pipelines are those of the first write. Before: `WriteLDHi` latency 4 with one µOP, also
for LDPSW and the FP pairs (5 in the SOG). As for the Cortex-A76 (C-11, C-15 there) and the X1
([docs.md](docs.md) 1.B). Effect: one µOP less per LDP in the issue bound and the MLP window.
Test: `Loads`, `FpLoadsAndStores`.

<a id="d-11"></a>
**D-11. Fused pairs: the absorbed instruction takes nothing; AES pairs are not masked.**
SOG sec. 4.14 (l.2805-2825) lists the pairs but not what a fused pair costs. Implementation
(`computeA78FusedMask`, [docs.md](docs.md) 4.C): the second instruction of items 1-9 and the NOP of
item 10 take no Mop, µOP or pipeline; the dependence graph keeps them. AESE + AESMC / AESD + AESIMC
are also listed as fused (l.2821-2823), but sec. 4.7 (l.2569-2570) gives two AES instructions per
cycle; masking the second would allow four per cycle, so the pair is left to [U-6](#u-6). Test:
`Fusion`.

---

## 6. Unclear in the SOG (U)

<a id="u-1"></a>
**U-1. ROB size and the MLP window.**
The SOG gives no reorder-buffer size or scheduler sizes. Current handling: `MicroOpBufferSize = 160`
(the upstream Neoverse N2 value), which is the MLP window (160 µOPs) and not used otherwise by the
A78 path; `IssueWidth = 5`, `MispredictPenalty = 10`, `LoopMicroOpBufferSize = 16` are not used by
the A78 path either. Impact: the MLP columns. How to settle: a documented source (whether the A78
Technical Reference Manual, 101430, gives it was not checked). Test: `DispatchAndPipelines`.

<a id="u-2"></a>
**U-2. Mops and µOPs per instruction.**
SOG l.386-389: instructions are decoded into MOPs, "A MOP can be split into two Micro-OPerations";
sec. 4.1: 6 MOPs and 12 µOPs per cycle; sec. 4.18 (p.59, l.2883-2892): the bandwidth of LD4R and LD4
one-lane post-index with 64-bit elements, LD4 multiple Q and D, ST4 multiple Q "is limited by decode
constraints", without a number. The SOG does not give µOP counts. Current handling: one Mop per
instruction; the model's µOP counts (e.g. scalar FDIV / FSQRT 7 to 16 µOPs, which the N2 model uses
to express the divide's iterations; LD4 Q 8, ST4 Q 12). With at most two µOPs per Mop the µOP bound
can bind only for instructions above two µOPs. Impact: the issue bound and the MLP window for such
instructions (rare in the research binaries apart from FDIV / FSQRT, 49,675). How to settle:
hardware measurement or an owner decision.

<a id="u-3"></a>
**U-3. Region-based fast forwarding.**
SOG sec. 4.8 (p.54-55, l.2597-2665): "The effective latency of FP and ASIMD instructions as
described in section 3 is increased by one cycle if the producer and consumer instructions are not
part of the same forwarding region", Table 4-1 with four regions, exclusions and special notes.
Unclear: the exact membership (e.g. the ASIMD miscellaneous instructions are in regions 1 and 2,
"can only be consumers" of region 2; FP converts and rounding, integer multiplies and reductions
belong to no region, so any edge into or out of them would cost one more cycle). Current handling:
not modelled (every FP/ASIMD edge has the table latency). Impact: optimistic by one cycle per
cross-region edge on FP/ASIMD dependence chains. How to settle: an owner decision on a
classification, or hardware measurement.

<a id="u-4"></a>
**U-4. ADD/SUB (extended register) with an identity extend.**
SOG "ALU, extend and shift ADD{S}, SUB{S} 2 2 M" (l.564). Unclear: whether the extended-register form
with a zero shift and a full-width extend (UXTX/SXTX for X), which assemblers write as `add x0, sp,
x1`, counts as "extend". Current handling: 2 on M (all extended forms). Impact: the research
binaries contain 25,199 such instructions (count of the Cortex-A76 audit, same binaries); at 1 / 4 /
I they would be cheaper. How to settle: hardware measurement or an owner decision.

<a id="u-5"></a>
**U-5. Multiply-accumulate forwarding between forms.**
SOG sec. 3.6 note 2 (l.789-791): "late-forwarding of accumulate operands from similar µOPs".
Unclear: whether a W/X-form MADD forwards to a long accumulator and vice versa, and whether a MUL
(zero accumulator) counts as similar. Current handling: `N2Rd_IMA` (advance 1) from `N2Wr_IMA`
(MADD/MSUB) and `A78Wr_IMAL` (SMADDL family) to the accumulator of all of them, i.e. any
multiply-accumulate to any (the lower estimate); a MUL / SMULL result is not forwarded. How to
settle: hardware measurement. Test: `DivideAndMultiply`.

<a id="u-6"></a>
**U-6. AESE/AESMC and AESD/AESIMC pairs.**
SOG sec. 4.14 (l.2821-2825) lists them as fused, sec. 4.7 (l.2584-2585): dependent pairs "exhibit
higher performance when they are adjacent in the program code and both instructions use the same
destination register"; no number. Current handling: no fusion and no forwarding; each AES
instruction 2 / 2 / V. No AES instruction occurs in the research binaries. How to settle: hardware
measurement.

<a id="u-7"></a>
**U-7. Exclusives, acquire/release, LSE atomics.**
The SOG has no rows for LDAR/LDAPR, STLR, LDXR/LDAXR/LDXP, STXR/STLXR/STXP/STLXP or the LSE atomics
(the A78 implements LSE and LDAPR, sec. 2, l.374-377; `llvm-mc -mcpu=cortex-a78` accepts LDADD).
Current handling: stock classes (LDAR, LDAPR, LDXR 4 on L; STLR as a store; STXR/STLXR `WriteSTX` =
store + load, latency 5); `WriteAtomic` is `Unsupported`, so InstrBuilder drops every LSE atomic
with a `[WARN]` ([docs.md](docs.md) section 2.F). The research binaries contain 139
LDAR/STLR/LDXR/STXR-family instructions and no LSE atomic; the only dropped opcode in the A78 runs is
UDF. How to settle: a project decision or hardware measurement (as for the Cortex-A76, U-7 there).

<a id="u-8"></a>
**U-8. FRECPE, FRSQRTE, Q-form F64.**
SOG sec. 3.17 (l.1952-1959): rows for D-form F32 and scalar, D-form F16 and Q-form F32, Q-form F16;
none for Q-form F64 (`frecpe v0.2d`). Current handling: the stock class (2, V). None in the research
binaries.

<a id="u-9"></a>
**U-9. Special-register access.**
SOG sec. 4.11 (p.55-57, l.2686-2759): reads and writes of most special registers are
non-speculative and/or in order, FPCR writes may act as a barrier, FPSR writes stall at dispatch
while another is pending; no cycle counts. Current handling: MRS, MSR, barriers and other system
instructions have no pipeline and latency 1 (`WriteSys`, `WriteBarrier`); no serialisation. NZCV and
SP "are fully renamed" (l.2723), which matches. The research binaries contain 10 MRS/MSR.

<a id="u-10"></a>
**U-10. Register forwarding hazards (S written, Q read).**
SOG sec. 4.12 (p.57, l.2763-2786): when an FP/ASIMD µOP reads a Q register recently written as S
registers, "Cortex-A78 performs slightly worse"; no number. Not modelled.

<a id="u-11"></a>
**U-11. Fusion: the "(register)" forms of CMP, CMN and TST.**
SOG sec. 4.14 (l.2809-2815) names "CMP/CMN (register)", "CMP (register)" and "TST (register)". The
A64 instruction set has CMP/CMN (shifted register), (extended register) and TST (shifted register).
Unclear: whether shifted (non-zero shift) and extended forms fuse. Current handling: every SUBS /
ADDS / ANDS with a zero destination fuses (also shifted and extended), i.e. the lower estimate.
Impact: the research binaries contain about 832,000 B.cond (a pair is counted only when adjacent to
a compare; the share with shifted or extended compares was not counted). How to settle: hardware
measurement.

<a id="u-12"></a>
**U-12. Model rows without an A78 SOG row or LLVM support.**
The N2 model has rows for FMLAL/FMLSL, BF16, FCMLA, SHA512/SHA3/SM3/SM4, pointer authentication,
MTE and SVE; the A78 SOG has no row for them, and LLVM 22 rejects them for `cortex-a78` (checked for
FMLAL, BFDOT, FCMLA, AUTIA, SHA512H, EOR3 with `llvm-mc`). They cannot occur in A78 code and were
not compared. PACIASP / AUTIASP in the research binaries decode as hints (`HINT #25`, `#29`), which
take no pipeline ([C-3](#c-3)).

---

## 7. Inconsistent within the SOG (I)

<a id="i-1"></a>
**I-1. AArch64 and AArch32 rows that disagree.**
"ASIMD absolute diff accum SABA, UABA 3(1)" and "... long SABAL(2), UABAL(2) 3(1)" (l.1559-1561)
against "VABA 4(1)", "VABAL 4(1)" (l.1689-1690); "ASIMD multiply long SMULL(2), UMULL(2), SQDMULL(2) 3
1 V0" (l.1638-1640) against "VMULL (.S, .I), VQDMULL 4 1 V0" (l.1733); "ASIMD reciprocal step
FRECPS, FRSQRTS 4 2 V" (l.1961) against "VRECPS, VRSQRTS 5 2 V" (l.2019). Implementation: the AArch64
rows.

<a id="i-2"></a>
**I-2. FP divide: "until complete" against the throughput column.**
Notes l.1286-1287 and l.1919-1920: divides "block subsequent similar operations to the same pipeline
until complete", which would make the throughput 1/latency; the tables print e.g. scalar H 7 / 4/7,
scalar D 7 to 15 / 1/7 to 2/7. Implementation: the throughput column ([D-1](#d-1)). For the integer
divide the note and the table agree (5 to 12 / 1/12 to 1/5).

<a id="i-3"></a>
**I-3. STRH "extend, scale by 1".**
Sec. 3.10 (l.1137-1138): "Store register, register offset, extend, scale by 1 STRH 2 2 I, L01, D";
the non-extended row says "scaled by 2" (l.1131-1132), and the load table "extend, scale by 2" for
LDRH (l.992-993). Implementation: scale by 2 (the S bit).

<a id="i-4"></a>
**I-4. FP register-offset loads: "basic" with I, "extend" without.**
Sec. 3.13 (l.1384-1386, l.1402-1403, checked in the PDF p.30-31): "Load vector reg, register offset,
basic LDR 5 3 L, I", but "..., extend LDR 5 3 L" and "..., scale, S/D-form 5 3 L"; the integer
"Load register, register offset, basic" is "L" (l.980-982). An I µOP for the unscaled X index alone
has no evident reason. Implementation: as printed (C-16), which adds an I µOP to 71,860 B/S/D loads
in the research binaries. How to settle: hardware measurement.

<a id="i-5"></a>
**I-5. FP register-offset stores, Q-form: "extend, scale" with I, "scale" without.**
Sec. 3.14 (l.1485-1486, l.1495-1496, checked in the PDF p.32): "Store vector reg, register offset,
scale, Q-form STR 2 2 L01, V" but "..., extend, scale, Q-form STR 2 2 I, L01, V"; for H both scaled
rows have I, for S/D neither. Implementation: as printed (C-17).

<a id="i-6"></a>
**I-6. STP Q immediate offset: 2 per cycle.**
Sec. 3.14 (l.1510-1511, checked in the PDF p.33): "Store vector pair, immed offset, Q-form STP, STNP 2
2 L01, V", while the post- and pre-index Q forms are "2 1" (l.1516-1517, l.1522-1523) and "ASIMD
store, 1 element, multiple, 2 reg, Q-form ST1 2 1 L01, V" (l.2257-2258) stores the same 32 bytes at
1 per cycle. Implementation: as printed (one L01 and one V µOP, C-17); this halves the store cost of
79,856 STP Q in the research binaries against the N2 row. How to settle: hardware measurement.

<a id="i-7"></a>
**I-7. Misprints and references.**
Table 3-26 and 3-27 are titled "ASIMD integer instructions" for the ASIMD floating-point tables
(l.1772, l.1873); "VCVTAU" in AArch64 rows (l.1797, l.1803; FCVTAU meant); "AAarch32" (l.540);
"Q2F" (l.2786); sec. 4.14 and Table 3-35 note 1 refer to "Section 4.6" for AES (l.2416, l.2822-2823;
4.7 meant); LDPSW pre/post-index is "I, L" like the offset form (l.998-999, l.1019-1020), while every
other pre/post-index integer load adds the I of the base update (the model gives the pre/post-index
LDPSW `WriteAdr` as well, i.e. two I µOPs). Implementation: the evident meaning.

---

## 8. Outside the A78 path: shared code (X)

<a id="x-1"></a>
**X-1. The test harness gives `cortex-a78` the stock model.**
`AArch64TestContext` (`tests/test_support.h`) installs the locally generated scheduling model only
for icestorm, firestorm, cortex-a720(ae), cortex-x1(c), cortex-a55 and cortex-a520. The A78 tests of
`tests/facile_test.cpp` (fusion, store writeback) therefore run on libLLVM's stock
`NeoverseN2Model` with the A78 fusion table and zero-latency switch (keyed by the CPU name). Not
changed; `tests/a78_sog_test.cpp` builds its own context (`A78Context`) the way `frontend.cpp`
does.

<a id="x-2"></a>
**X-2. MCA gives every call latency 0.**
LLVM MCA's `InstrBuilder` sets the latency of BL, BLR (and other calls) to 0, whatever the
scheduling model says; Facile then floors the LR edge to 1, which meets the SOG's BL / BLR latency 1
(C-8). Shared LLVM behaviour; not changed.

---

## 9. Taken over from other cores (B)

<a id="b-1"></a>
**B-1. Store data is read one cycle late.**
*Introduced 2026-10-31* (`[a78_from_a55_store_data]`). **Not SOG-based.**
A78 SOG: stores are split into address and data µOPs (l.1102-1103, l.1455, l.2245); no statement on
when the data operand is needed; store latencies 1 (integer), 2 or more (FP, ASIMD). The project
decided for the Cortex-A55 (SOG-based there) and the Cortex-A520 (borrowed) that store data is read
one cycle late, and applied it to the Cortex-A76 for consistency ([a76-sog.md](a76-sog.md) B-1); it
is applied to the A78 likewise. Implementation: `ReadAdvance<ReadST, 1>` (was 0), and `ReadST` on
the data operands of every store form (integer STR/STUR/STTR/STLR/STLLR, register offset, pre/post,
STP/STNP, STLXR/STLXP; FP STR/STUR/STP of every addressing mode; the `Vt` of ST1-ST4, with the lane
index of the one-lane forms reading `ReadDefault` in between); STXR/STXP have none. Effect: only
Facile's store -> load memory edges, and only for producers of latency 2 or more. Measured: `ldr x1,
[x0]; str x1, [x0]` + branch 5.0 -> 4.0. To be revisited when an A78 statement or measurement
appears. Test: `BorrowedStoreDataNeededLater`.

---

## 10. Reproducing the cycle counts quoted here

The cycles/iteration figures that are not from `tests/a78_sog_test.cpp` were measured by the author
on 2026-10-31 with private builds (CMake, Release; not `build/`): "before" is the working tree before
this audit, "after" the working tree with its changes. Each loop is assembled, linked and analysed
with the invocation of the research for the A78, plus `--fractional-cycles` (the unrounded Facile
estimate) and `--facile-reason`:

```sh
# body: the instructions of one iteration, e.g. "sdiv w1, w1, w2"
{ echo '.text'; echo '.globl f'; echo '.type f,%function'; echo 'f:'; echo '1:';
  printf '  %s\n' "${body[@]}"; echo '  b 1b'; echo '  ret'; } > t.s
aarch64-linux-gnu-gcc -march=armv8.2-a+crypto+lse+crc+dotprod+fp16+rcpc -nostdlib -static -Wl,-e,f t.s -o t
mca_tool --mcpu=cortex-a78 --mtriple=aarch64 --dependency ooo --facile --fractional-cycles --facile-reason t
# cycles/iteration = the cycles column of the first CSV row (the loop); the last column says
# which bound decided it (inst, exec, prec)
```

The loop branch `b 1b` is part of every body. The research invocations (without the last two flags)
are `mca_tool --merge-same-header --dependency ooo --mcpu cortex-a78 --facile <binary>` and the same
with `--disable-always-hit-loads-heuristic`. Instruction counts "in the research binaries" are
mnemonic/operand counts of `objdump -d` over the 46 binaries. Per-instruction values: `mca-insts-info
--mtriple=aarch64 --mcpu=cortex-a78`. Re-measure before quoting a number.

---

## 11. Maintaining this document

* Update an item when the SOG reading or the implementation changes. IDs run from 1 without gaps
  within each category, in the order of the items; an item whose difference has been fixed becomes a
  C item, and the code, tests and other documents that name an ID are updated with it.
* When a deviation is introduced on purpose, add a D item, and tag the code with a comment that names
  the ID.
* When a behaviour is taken over from another core's model by a project decision, without an A78
  SOG statement, add a B item and tag the code `[a78_from_<core>_<topic>]`.
* `NeoverseN2Model` in `ModifiedTarget/` is installed only for `cortex-a78` / `-a78ae` / `-a78c`. If
  another CPU is ever given this local model (e.g. `neoverse-n2`, `cortex-a710`), its own SOG must be
  compared first; the A78 changes above are not N2 statements.
* Evaluation results (accuracy metrics, totals over benchmark sets) are not recorded in this
  document; cycles/iteration only when reproduced with `mca_tool` as in section 10.
* [docs.md](docs.md) holds the rest of the technical documentation; [a55-sog.md](a55-sog.md),
  [a520-sog.md](a520-sog.md) and [a76-sog.md](a76-sog.md) are the counterparts of this document for
  the other cores. [README.md](README.md) lists the documents.
