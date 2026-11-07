# Cortex-A720: the model against the Software Optimization Guide

> **Who wrote this, and how far to trust it.** This document was written on **2026-11-01** by
> **Claude** (Anthropic's Claude Opus model, running as a sub-agent of Claude Code), after reading
> the Cortex-A720 Core Software Optimization Guide and the contents of this repository's working
> tree on that day. **It is not an Arm document**, and it has not been reviewed by Arm. Every claim
> about the guide is tied to a section, a page and (where useful) a line of the text extraction, so
> that it can be checked; it may still contain errors of reading or transcription. **Check against
> the original guide before relying on any statement here.** Statements that could not be checked
> are marked "not verified".

Contents

1. [About this document](#1-about-this-document)
2. [Implementation overview](#2-implementation-overview)
3. [Compatible with the SOG (C)](#3-compatible-with-the-sog-c)
4. [Not compatible with the SOG (N)](#4-not-compatible-with-the-sog-n)
5. [Deliberate deviations and interpretations (D)](#5-deliberate-deviations-and-interpretations-d)
6. [Unclear in the SOG (U)](#6-unclear-in-the-sog-u)
7. [Inconsistent within the SOG (I)](#7-inconsistent-within-the-sog-i)
8. [Outside the A720 path: shared code (X)](#8-outside-the-a720-path-shared-code-x)
9. [Taken over from other cores (B)](#9-taken-over-from-other-cores-b)
10. [Reproducing the cycle counts quoted here](#10-reproducing-the-cycle-counts-quoted-here)
11. [Maintaining this document](#11-maintaining-this-document)

| Category | Items |
|---|---|
| C, compatible | [C-1](#c-1) to [C-26](#c-26) |
| N, not compatible | none open |
| D, deliberate deviation or interpretation | [D-1](#d-1) to [D-12](#d-12) |
| U, unclear in the SOG | [U-1](#u-1) to [U-12](#u-12) |
| I, inconsistent within the SOG | [I-1](#i-1) to [I-5](#i-5) |
| X, outside the A720 path | [X-1](#x-1) to [X-3](#x-3) |
| B, taken over from the Cortex-A55 / A520 / A76 / A78 (not SOG-based) | [B-1](#b-1) |

---

## 1. About this document

### 1.1 Source

* **Guide**: "Arm® Cortex®-A720 Core Software Optimization Guide", document number **109720**,
  **Issue 7.0** of 30 November 2023 ("First release for r0p2 - Document confidentiality update"),
  revision r0p2, Copyright © 2022-2023 Arm Limited (or its affiliates), **Non-Confidential**, product
  status "Final" (title page and p.2-3, l.1-28, l.97). Called "the SOG" below. Page numbers are those
  of the guide's footers ("Page N of 74"); "l.N" is line N of the text extraction of the guide
  (not tracked). The row "Store vector reg, immed pre-index, B/H/S/D-form" was checked against the
  PDF (p.27, [I-3](#i-3)); all other rows were read from the text extraction only.

### 1.2 What "the implementation" is

The files concerned:

* `ModifiedTarget/AArch64/AArch64SchedA720.td`: the TableGen model `CortexA720Model`, compiled
  into the tool's own scheduling tables ([docs.md](docs.md) section 5). `AArch64Processors.td` maps
  `cortex-a720` and `cortex-a720ae` to it; no other CPU uses it. `src/cpu/sched_model.cpp`
  (`installLocalSchedModel`, `case CpuModel::A720`) installs it for these two CPUs
  (`src/cpu/cpu_traits.cpp`).
* `src/cpu/cpu_traits.cpp`: the A720 entry (`UopDispatchWidth = 10`, `MopDispatchWidth = 5`,
  `Fusion = FusionTable::A720`, `ZeroLatencyMov = true`, no MLP window resources).
* `src/facile/facile.cpp` and `src/facile/fusion.cpp`: the Facile predictor and the A720 fusion
  table (`computeA720FusedMask`), which produce the A720 cycle counts (`mca_tool --merge-same-header
  --dependency ooo --mcpu cortex-a720 --facile`); with `--facile` the MCA simulation is not run, the
  CSV `cycles` column is the rounded Facile estimate.
* `tests/a720_sog_test.cpp` (suite `A720SogTest`) pins the values below; the
  A720 fusion and zero-latency tests of `tests/facile_test.cpp` pin [C-4](#c-4) and [C-5](#c-5).
  `AArch64TestContext("cortex-a720")` installs the local model (`tests/test_support.h`).

Implementation facts are given by file and identifier (TableGen `def` names, C++ function names),
not by line number. The model's departures from a plain transcription of the SOG carry tags;
[section 2.3](#24-code-tags-and-the-items-of-this-document) maps them to the items.

### 1.3 Categories

| ID | Category | Meaning |
|---|---|---|
| **C-n** | compatible | The implementation does what the SOG states. |
| **N-n** | not compatible | The SOG states something and the implementation does not follow it. Each item says why it is still open. |
| **D-n** | deliberate deviation or interpretation | The implementation knowingly departs from the literal SOG, or fills a gap with an interpretation. Each item gives the SOG text, the implementation and the reason. |
| **U-n** | unclear | The SOG does not decide the question; the implementation makes an assumption or leaves it open. Each item gives the SOG text, what is unclear, the current handling, the expected impact, and how to settle it. |
| **I-n** | inconsistent | Two places of the SOG disagree, or the SOG contradicts itself or has a misprint. Each item says which reading the implementation follows. |
| **X-n** | outside the A720 path | Problems in shared code or in LLVM that affect the A720 results; not changed for the A720. |
| **B-n** | taken over from another core | The A720 SOG says nothing on the question, and the treatment the project decided for the Cortex-A55 / Cortex-A520 and applied to the Cortex-A76 / A78 is applied to the A720 for consistency. **Not SOG-based.** |

### 1.4 Reading the SOG instruction tables

Sec. 3.1 (p.16, l.568-579):

* **Exec latency** is "the minimum latency seen by an operation dependent on an instruction in the
  described group". A number in parentheses is the accumulate latency of a forwarding path (notes of
  sec. 3.5, 3.12, 3.16, 3.17, 3.18).
* **Execution throughput** is "the maximum throughput (in instructions per cycle) of the specified
  instruction group that can be achieved in the entirety of the Cortex-A720 core microarchitecture".
  The model holds a resource of `n` units for `n / T` cycles in total ([D-2](#d-2)).
* **Utilized pipelines**, Table 3-1 (p.16, l.584-596): B branch 0/1, S integer single-cycle 0/1, I
  "Integer single Cycle 0/1 and single/multicycle 0/1", M integer single/multicycle 0/1, M0 integer
  multicycle 0, L01 load/store 0/1, L "Load/Store 0/1 and Load 2", ID integer store data 0/1, V
  FP/ASIMD/vector store data 0/1, V0, V1.
* Ranges ("5 to 12", "1/12 to 1/5") are data dependent; the model takes the best value
  ([D-1](#d-1)).
* The A720 implements AArch64 only ("AArch64 Execution state at all Exception levels", l.416); the
  SOG has no AArch32 tables. The SVE, SME and MTE tables are outside the model ([D-8](#d-8)).

---

## 2. Implementation overview

### 2.1 Machine model (`CortexA720Model`)

`IssueWidth = 5` (not read by Facile, whose widths come from `cpu_traits.cpp`), `MicroOpBufferSize
= 192` (the MLP window width, [U-1](#u-1)), `LoadLatency = 4`, `MispredictPenalty = 11` (neither
read by the A720 path), `CompleteModel = 0`; `UnsupportedFeatures` = SVE, PA (an empty list in
LLVM 22), SME, MTE, CSSC.

| Resource | Units | Role |
|---|---|---|
| `A720UnitB0`, `A720UnitB1`; group `A720UnitB` | 1, 1; 2 | Branch 0/1 ("B") |
| `A720UnitI0`, `A720UnitI1`; group `A720UnitS` | 1, 1; 2 | Integer single-cycle 0/1 ("S") |
| `A720UnitM0`, `A720UnitM1`; group `A720UnitM` | 1, 1; 2 | Integer single/multicycle 0/1 ("M", "M0") |
| group `A720UnitI` | S0, S1, M0, M1 (4) | "I" |
| `A720UnitL0`, `A720UnitL1`, `A720UnitL2`; group `A720UnitL01`; group `A720UnitLS` | 1, 1, 1; 2; 3 | Load/store 0/1, load 2 ("L01", "L") |
| `A720UnitSD` | 2 | Integer store data 0/1 ("ID") |
| `A720UnitV0`, `A720UnitV1`; group `A720UnitV` | 1, 1; 2 | FP/ASIMD/vector store data 0/1 ("V0", "V1", "V") |
| `A720UnitIDiv` | 1 | not a pipeline: the integer divider inside M0 ([D-3](#d-3)) |

TableGen lists a use of a pipeline also under every group that contains it (a use of M0 appears as
M0, M and I), so `mca-insts-info` and the tests show, for example, `[A720UnitI:1 A720UnitM:1
A720UnitM0:1]` for `madd x1, x2, x3, x4`.

### 2.2 The Facile path (`src/facile/facile.cpp`)

For `--facile`, every region gets `EstimatedCycles = max(IssueBound, PortBound, PrecedenceBound)`:

* `calculateIssueBound`: `max(uops / 10, Mops / 5)` (uops = the model's `NumMicroOps`, at least one
  per instruction; one Mop per instruction; the absorbed half of a fused pair is not counted; [C-1](#c-1),
  [C-5](#c-5), [U-2](#u-2)).
* `calculatePortUsageBound`: for every resource (pipelines, groups, the divider), the sum of the
  resource cycles divided by its units; the absorbed half of a fused pair is not counted.
* `calculatePrecedenceBound`: the maximum cycle ratio of the dependence graph. Register edges have
  the latency of the producing def minus the consumer's `ReadAdvance`, floored at 1, and 0 for the
  zero-latency instructions (`getEdgeLatency`, [C-4](#c-4), [C-6](#c-6)); a store with a base update
  has a second node for the update ([C-7](#c-7)); store -> load memory edges, added only with `--facile-memory-deps` (off by default,
  [docs.md](docs.md) 4.G), have the store's latency ([D-7](#d-7)).
  Loop-carried edges (register and memory) are added only when the region is a loop, or with
  `--ignore-loop-carried disable` ([docs.md](docs.md) 4.J); for a region that is not a loop the bound is 0.

The MLP columns use a window of `MicroOpBufferSize` = 192 uops ([U-1](#u-1)); a change of an
instruction's uop count therefore also changes the `mlp` / `mlp_r` columns of the regions around it.

### 2.3 Unit tests

`tests/a720_sog_test.cpp` (suite `A720SogTest`) pins the values of the items below. The per-instruction
data (latency per def, uops, resource cycles, read advances) come from `sogDesc()`, the cycles per
iteration of small loops from `sogFacile()`, the Facile prediction as `mca_tool --facile` computes
it (both in `tests/sog_test_support.h`). `AArch64TestContext("cortex-a720")` (`tests/test_support.h`) installs the local scheduling model.
`A720SogTest.Dump` prints `sogDesc()` of the instructions listed in `$A720_DUMP` (`sogDump()`).

### 2.4 Code tags and the items of this document

| Tag in `AArch64SchedA720.td` (or file) | Items |
|---|---|
| `[a720_nop]` | [C-3](#c-3) |
| `[a720_store_uops]` | [C-7](#c-7), [C-14](#c-14), [C-18](#c-18), [C-24](#c-24) |
| `[a720_ext]` | [C-9](#c-9), [U-4](#u-4) |
| `[a720_div_best]` | [C-10](#c-10), [D-1](#d-1) |
| `[a720_dividers]` | [D-3](#d-3) |
| `[a720_pac]` | [C-11](#c-11) |
| `[a720_strq_ro]` | [C-18](#c-18) |
| `[a720_str_pre]` | [C-18](#c-18), [I-3](#i-3) |
| `[a720_frecpe_scalar]` | [C-22](#c-22) |
| `[a720_crc]` | [C-26](#c-26) |
| `[a720_fp_cvt_scalar]` | [D-4](#d-4) |
| `[a720_asimd_scalar]` | [D-5](#d-5) |
| `[a720_ldp_hi]` | [D-10](#d-10) |
| `[a720_sys]` | [D-12](#d-12) |
| `[a720_from_a55_store_data]` | [B-1](#b-1) |
| `[a720_fusion_simd]` (`src/facile/fusion.cpp`) | [C-5](#c-5), [D-11](#d-11) |

The rows without a tag are transcriptions of the SOG rows, with the SOG line numbers as comments.

---

## 3. Compatible with the SOG (C)

Values are "latency / throughput / pipelines" as printed. "Model" values were read with
`mca-insts-info --mtriple=aarch64 --mcpu=cortex-a720` and the unit tests (`tests/a720_sog_test.cpp`,
suite `A720SogTest`). Cycles per iteration are Facile estimates of small loops that include the loop
branch ([section 10](#10-reproducing-the-cycle-counts-quoted-here)). Values marked "(test)" are those of the unit tests, whose bodies have no loop branch.

### 3.1 Dispatch, pipelines and forwarding

<a id="c-1"></a>
**C-1. Five Mops and ten uOPs per cycle.**
SOG sec. 4.1 (p.58, l.2828): "The dispatch stage can process up to 5 MOPs per cycle and dispatch up
to 10 µOPs per cycle"; sec. 2.1 (p.14, l.469-474): instructions are decoded into MOPs, "A MOP can be
split into two Micro-OPerations (µOPs) further down the pipeline after the decode stage".
Implementation: `cpu_traits.cpp` `UopDispatchWidth = 10`, `MopDispatchWidth = 5`;
`calculateIssueBound` takes the larger of uops / 10 and instructions / 5. The per-type limits of
sec. 4.1 (l.2831-2836: four µOPs for S or B, four for M, two each for M0, V0, V1, five for L) are not
modelled; none is below the issue rate of the pipelines concerned (four S/B, two M, one M0, one V0,
one V1, three L), so none can bind in a steady state. Test: `DispatchAndPipelines`.

<a id="c-2"></a>
**C-2. Thirteen issue pipelines.**
SOG sec. 2.1 (p.14-15, l.474-477, Figure 2-1 l.479-525, Table 2-1 l.539-552): µOPs "issue
out-of-order to one of thirteen issue pipelines. Each issue pipeline can accept one µOP per cycle":
Branch 0, 1, Integer Single-Cycle 0, 1, Integer Single/Multi-Cycle 0, 1, FP/ASIMD 0/Vector Store data
0, FP/ASIMD 1/Vector Store data 1, Load/Store 0, 1, Load 2, Integer Store data 0, 1. Implementation:
the resources of [section 2.1](#21-machine-model-cortexa720model); the groups I, M, S, B, L01, L and V
are those of Table 3-1, as `ProcResGroup`s, so a group's bound counts the uses of its members.
"ALU, basic, flagset ... 1 4 I" (l.639-641): unlike the Cortex-A78 (3 per cycle), the flag-setting
ALU operations run at the full I rate. Measured (test, without the loop branch): eight independent
`adds` 2.0 per iteration; two `str` and four `add` 1.2 (the Mop bound, 6 / 5); with the loop branch
1.4 (7 / 5). Test: `DispatchAndPipelines`.

<a id="c-3"></a>
**C-3. NOP takes no execution resource.**
`[a720_nop]`.
The SOG has no row for NOP or the other hints (BTI); unlike the Cortex-A78 it has no "NOP + Any
instruction" fusion (sec. 4.11, l.3153-3167). Implementation: `WriteHint` has no resource; a NOP
counts as one Mop and one µOP. This is the decision made for the Cortex-A55, A520, A76 and A78
([a78-sog.md](a78-sog.md) C-3). Measured: five `nop` and an `add` + branch 1.4 (the Mop bound, 7 / 5). Test:
`DispatchAndPipelines`.

<a id="c-4"></a>
**C-4. Zero-latency instructions.**
SOG sec. 4.12 (p.65, l.3183-3227): "MOV Xd, #{12{1'b0},imm[3:0]} / MOV Xd, XZR / MOV Wd,
#{12{1'b0},imm[3:0]} / MOV Wd, WZR / MOV Hd, WZR / MOV Hd, XZR / MOV Sd, WZR / MOV Dd, XZR / MOVI Dd,
#0 / MOVI Vd.2D, #0 / MOV Wd, Wn / MOV Xd, Xn / FMOV Sd, Sn / FMOV Dd, Dn / MOV Vd, Vn (vector) / MOV
Zd.D, Zn.D / PTRUE / PFALSE / SETFFR ... are executed with zero latency. These instructions do not
utilize the scheduling and execution resources of the machine." Implementation ([docs.md](docs.md)
4.B): the predicate `A720ZeroMove` selects `A720Write_0c` (latency 0, no
resource) on MOVZ[WX]i, ORR[WX]rs, FMOV[WX][HSD]r from a zero register, MOVID / MOVIv2d_ns #0,
ORRv16i8 / ORRv8i8 with Rn == Rm, FMOVSr, FMOVDr; `CpuTraits::ZeroLatencyMov` lets Facile use the
latency 0. FMOV Hd, Hn is not listed and keeps "FP move, register 2 2 V" (l.1040); MOVI Vd.16B, #0
keeps "ASIMD move, integer immed 2 2 V" (l.1546). The SVE forms are outside the model ([D-8](#d-8)).
The immediate range and the caveat "may not be executed with zero latency under certain conditions"
(l.3226-3227): [D-9](#d-9), [D-1](#d-1). With `fadd` at its row's latency 2 ("FP arithmetic FADD,
FSUB 2 2 V", l.964), the tests `A720ZeroLatencyFmovFpToFp` and `A720FmovHalfPrecisionIsNotZeroLatency`
of `tests/facile_test.cpp` expect fadd + eliminated FMOV = 2.0 and fadd + FMOV H = 2 + 2 = 4.0. Test:
`ZeroLatencyInstructions`, `FacileTest.A720*`.

<a id="c-5"></a>
**C-5. Instruction fusion.**
SOG sec. 4.11 (p.64, l.3149-3171), compared row by row with `computeA720FusedMask`,
`isA720FusiblePair`, `isA720ShlSriPair`, `isA720FcmpAxflagPair` (`src/facile/fusion.cpp`; the
comparison with the A78 table is [docs.md](docs.md) 4.C):

| SOG row | Implementation |
|---|---|
| AESE + AESMC, AESD + AESIMC | not in the mask ([D-11](#d-11), [U-6](#u-6)) |
| CMP/CMN (immediate) + B.cond | SUBS/ADDS immediate with a ZR destination, then `Bcc` |
| CMP/CMN (register Rn != ZR) + B.cond | SUBS/ADDS register forms (shifted, extended) with a ZR destination and Rn not WZR/XZR, then `Bcc` |
| TST (immediate), (register) + B.cond | ANDS with a ZR destination, then `Bcc` |
| BICS ZR (register) + B.cond | BICS with a ZR destination, then `Bcc` |
| CMP (immediate), (register) + CSEL | SUBS with a ZR destination, then CSEL (not CMN) |
| CMP (immediate), (register) + CSET | SUBS with a ZR destination, then CSINC Rd, ZR, ZR (CSET; not CSETM) |
| BTI + Integer DP/BR/BLR/RET/B uncond/CBZ/TBZ | not in the mask ([U-11](#u-11)) |
| **SHL + SRI (both scalar or both vector)** | SHLd then SRId, or SHLv*_shift then SRIv*_shift (`[a720_fusion_simd]`) |
| **FCMP + AXFLAG** | any FCMP/FCMPE, then AXFLAG (`[a720_fusion_simd]`) |
| MOVPRFX + supported SVE instruction | outside the model ([D-8](#d-8)) |

"For CMP, CMN, TST fusion is allowed for shifted and/or extended register forms" (l.3169-3170):
these forms fuse. Pairs must be adjacent (l.3169); pairing is greedy and non-overlapping. The
restrictions on immediates (l.3170-3171): [U-10](#u-10); accounting of a fused pair: [D-11](#d-11).
Measured: two SHL + SRI pairs + branch 2.0. Test: `Fusion`,
`FacileTest.A720*`.

<a id="c-6"></a>
**C-6. Latency seen by dependent instructions; ALU -> address.**
SOG sec. 3.1 (l.574-575). Implementation: `getEdgeLatency` uses the latency of the def that the
consumer reads (so the base update of a post-index load has 1, not the load's 4) and subtracts the
consumer's `ReadAdvance` (the accumulator paths of [C-10](#c-10), [C-15](#c-15), [C-19](#c-19) to
[C-21](#c-21), [C-26](#c-26)), with a floor of 1 cycle except for the zero-latency instructions. The
address operands have `ReadAdrBase` advance 0: an ALU result is used as an address after its own
latency, no extra cycle (the decision taken for the Cortex-A55, A520, A76 and A78). Test:
`ZeroLatencyInstructions`, `DivideAndMultiply`.

<a id="c-7"></a>
**C-7. Stores are split into address and data µOPs; store-to-load forwarding.**
`[a720_store_uops]`.
SOG sec. 3.9 (p.23, l.867-868): "Stores µOPs are split into address and data µOPs"; sec. 3.15 and
3.21 (l.1110, l.1740): "Stores MOPs are split into store address and store data µOPs"; sec. 4.5
(p.60, l.2925-2934): forwarding needs the load to start at the start or the middle of the older
store, loads larger than 8 bytes from at most two stores, loads of up to 4 bytes from one.
Implementation: every store write has two µOPs (address on L01, data on ID or V), three where the row
adds an I (STR Q scaled, [C-18](#c-18)); a pre/post-index store adds the base update (`WriteAdr`).
The µOP count acts on the issue bound (only when the average exceeds two µOPs per
instruction) and on the MLP window ([U-1](#u-1)). Shared Facile code ([docs.md](docs.md) 4.F, 4.G):
the base update of a store is a node of its own that does not wait for the data, and, with `--facile-memory-deps`, a store -> load
edge is left out when the two register-offset accesses are provably disjoint. The size rules are not
checked; the latency of the forwarding is not stated ([D-7](#d-7)). Test: `Stores`, `FpLoadsAndStores`, `AsimdLoadsAndStores`.

### 3.2 Instruction tables, integer

<a id="c-8"></a>
**C-8. Sec. 3.3, branches (p.17, l.609-618).**

| SOG group | SOG | Model |
|---|---|---|
| B; BR, RET; CBZ, CBNZ, TBZ, TBNZ | 1 / 2 / B | `WriteBr`, `WriteBrReg` = 1, B |
| BL; BLR | 1 / 2 / B, S | `A720Wr_1c_B_S` |

B.cond is not listed; it has `WriteBr` like B. MCA gives every call latency 0 ([X-2](#x-2)); Facile
floors the edge to 1. Test: `Branches`.

<a id="c-9"></a>
**C-9. Sec. 3.4, arithmetic and logical (p.18, l.631-666).**

| SOG group | SOG | Model |
|---|---|---|
| ALU, basic; basic, flagset | 1 / 4 / I | `WriteI`, `WriteISReg` = 1, I |
| **ALU, extend and shift: ADD{S}, SUB{S}** | 2 / 2 / M | `A720Wr_2c_M` for every extended-register form, UXTB (extend operand 0) included (`[a720_ext]`, [U-4](#u-4)) |
| Arithmetic, LSL shift <= 4 (flagset or not) | 1 / 4 / I | `A720Wr_ArithShift`, `IsCheapLSL` -> `A720Wr_1c_I` |
| Arithmetic, LSR/ASR/ROR shift or LSL > 4 | 2 / 2 / M | the same variant otherwise -> `A720Wr_2c_M` |
| Conditional compare; select; AXFLAG, XAFLAG; SETF8, SETF16, RMIF, CFINV | 1 / 4 / I | `WriteI`; `A720Wr_1c_I` |
| Logical, shift, no flagset | 1 / 4 / I | `WriteISReg` (ORR: zero-move variant, [C-4](#c-4)) |
| Logical, shift, flagset: ANDS, BICS | 2 / 2 / M | `A720Wr_LogicSFlag`: `RegShiftedPred` -> 2 M, unshifted -> 1 I ("ALU, basic, flagset") |
| ADDG, SUBG | 1 / 4 / I | `WriteI` (MTE, [D-8](#d-8)) |
| GMI; IRG; SUBP, SUBPS | 1 / 4 / I; 2 / 1 / M0; 1 / 4, 1 / 3 / I | MTE: no scheduling data, dropped ([D-8](#d-8)) |

Measured: `add x1, x1, w2, uxtb` + branch 2.0. Test: `ArithmeticAndLogical`.

<a id="c-10"></a>
**C-10. Sec. 3.5, divide and multiply (p.19, l.679-699).**

| SOG group | SOG | Model |
|---|---|---|
| **Divide, W-form; X-form: SDIV, UDIV** | 5 to 12; 5 to 20 / 1/12 to 1/5; 1/20 to 1/5 / M0 (note 1) | latency 5, M0 1 cycle, divider 5 cycles (`[a720_div_best]`, [D-1](#d-1), [D-3](#d-3)) |
| Multiply accumulate, W-form; X-form; long: MADD, MSUB, SMADDL, SMSUBL, UMADDL, UMSUBL | 2(1) / 1 / M0 (notes 2, 3) | `A720Wr_MAC` -> `A720Wr_MAC_M0` 2, M0 + `A720Rd_MAcc` (advance 1) |
| MUL, MNEG, SMULL, SMNEGL, UMULL, UMNEGL (Ra = ZR, note 3) | 2 / 2 / M | `IsReg3ZeroPred` -> `A720Wr_MUL_M` 2, M |
| Multiply high: SMULH, UMULH | 3 / 2 / M | `A720Wr_3c_M` |

Note 2 (l.695-697), "Accumulator forwarding is not supported for consumers of 64 bit multiply high
operations": `A720Rd_MAcc` names only the multiply writes. Forwarding between forms and from a MUL:
[U-5](#u-5). Measured: `sdiv w1, w1, w2` + branch 5.0; `udiv x1, x2, x3` + two `madd` +
branch 5.0; `madd x1, x2, x3, x1` + branch 1.0. Test: `DivideAndMultiply`.

<a id="c-11"></a>
**C-11. Sec. 3.6, pointer authentication (p.20, l.712-750).**
`[a720_pac]`.

| SOG group | SOG | Model |
|---|---|---|
| Authenticate data address; instruction address: AUTDA, ..., AUTIASP, AUTIAZ, ... | 1 / 2 / M | `A720Wr_1c_M` |
| Branch and link, register; branch, register; return, with PA: BLRAA, ..., BRAA, ..., RETAA, RETAB (printed "RETA, RETB", [I-5](#i-5)) | 2 / 2 / M, B | `A720Wr_2c_M_B` |
| Compute PAC for data address; generic key; instruction address: PACDA, ..., PACGA, PACIA, ..., PACIASP, ... | 4 / 2 / M | `A720Wr_4c_M` |
| Load register, with PA: LDRAA, LDRAB | 5 / 2 / M, L, I (note 2: I only for the pre-index writeback) | `A720Wr_5c_M_L`; writeback form + `WriteAdr` |
| Strip PAC: XPACD, XPACI, XPACLRI | 1 / 2 / M | `A720Wr_1c_M` |

The hint-space forms (PACIASP, AUTIASP, XPACLRI, ...) take these rows, not the hint row. Note 1 (AUTH fail): [D-1](#d-1). Measured: `paciasp; autiasp` + branch 5.0 (the LR
recurrence 4 + 1). Test: `PointerAuthentication`.

<a id="c-12"></a>
**C-12. Sec. 3.7, miscellaneous data processing (p.21, l.763-777).**
ADR, ADRP 1 / 2 / S: `A720Wr_1c_S`; EXTR (one or two registers), SBFM, UBFM, BFM, CLS, CLZ, MOVN,
MOVK, MOVZ, RBIT, REV*, ASRV, LSLV, LSRV, RORV 1 / 4 / I: `WriteExtr`, `WriteIS`, `WriteImm`,
`WriteI` (MOVZ with a small immediate: [C-4](#c-4)). Test: `MiscDataProcessing`.

<a id="c-13"></a>
**C-13. Sec. 3.8, loads; sec. 4.14, L1 latency (p.21-22, l.782-853; p.66, l.3259).**

| SOG group | SOG | Model |
|---|---|---|
| Load register, literal: LDR, LDRSW, PRFM | 5 / 2 / L, S | `A720Wr_LD5_L_S` |
| Load register, unscaled, unprivileged, unsigned immed; register offset (all forms, note 2: [U-12](#u-12)) | 4 / 3 / L | `WriteLD`, `WriteLDIdx` = 4, L |
| Load register, immed post-index; pre-index | 4 / 3 / L, I | `WriteAdr` (1, I) + `WriteLD` ([I-1](#i-1)) |
| Load pair, W-form; X-form: LDP, LDNP | 4 / 3; 4 / 3/2 / L | `A720Wr_LD_W`; `A720Wr_LDP_X` (L 2 cycles), second register `A720Wr_LDHi_0` ([D-10](#d-10)) |
| Load pair, signed words: LDPSW | 4 / 3/2 / I, L | `A720Wr_LDPSW` (L 2 cycles, I 1) |
| Load pair, post-index or pre-index, W; X; LDPSW | 4 / 3; 3/2; 3/2 / L, I | `WriteAdr` + the offset-form writes ([I-4](#i-4)) |

"Level-1 Cache Hit 4 core cycles" (Table 4-4, l.3259): the model's `LoadLatency` is 4 (not read by
the A720 path). Acquire and exclusive loads: [U-7](#u-7). Test: `Loads`.

<a id="c-14"></a>
**C-14. Sec. 3.9, stores (p.23, l.866-895).**
Every row is 1 / 2 / L01, ID, plus I for the post- and pre-index forms of STR and STP; register
offset forms (also STRH scaled by 2) have no I. Model: `WriteST`, `WriteSTP`, `WriteSTIdx` = 1, L01 +
ID, two µOPs ([C-7](#c-7)), + `WriteAdr` for pre/post-index. A store's latency matters only on
Facile's store -> load edges ([D-7](#d-7)), which exist only with `--facile-memory-deps`. The data operand reads `ReadST` ([B-1](#b-1)). Release
and exclusive stores: [U-7](#u-7). Measured (test): four `str` 2.0 per iteration. Test: `Stores`.

### 3.3 Instruction tables, FP and ASIMD

<a id="c-15"></a>
**C-15. Sec. 3.12, FP data processing (p.24-25, l.959-1005).**

| SOG group | SOG | Model |
|---|---|---|
| FABS, FABD; FADD, FSUB; FCMP{E}, FCCMP{E}; FMIN/FMAX(NM); FNEG; FCSEL | 2 / 2 / V | `WriteF`, `WriteFCmp` = 2, V |
| FDIV, FSQRT H; S; D | 5; 7; 12 / 1 / V0 (note 1, "fully pipelined") | `A720Write_5c_1V0`, `_7c_1V0`, `_12c_1V0`: V0 one cycle |
| FMUL, FNMUL | 3 / 2 / V (note 2) | `WriteFMul` = 3, V |
| FMADD, FMSUB, FNMADD, FNMSUB | 4 (2) / 2 / V (note 3) | `A720Wr_FMA` 4 + `A720Rd_FMA` (advance 2 from `WriteFMul`, `A720Wr_FMA`, `A720Wr_VFMUL`) |
| FRINT* (incl. FRINT32X/Z, FRINT64X/Z) | 3 / 1 / V0 | `A720Wr_3c_V0` |

Note 2 (l.1002-1003): an FP multiply-accumulate "can potentially be issued 1 cycle after the FP
multiply µOP has been issued", i.e. FMUL -> accumulator 1 ([docs.md](docs.md) 4.H). Measured (test):
`fmadd d0, d1, d2, d0` 2.0; `fmul; fmadd (accumulator); fmov` 5.0 (1 + 4 + 0). Test: `FpScalar`.

<a id="c-16"></a>
**C-16. Sec. 3.13, FP miscellaneous (p.25-26, l.1009-1048).**

| SOG group | SOG | Model |
|---|---|---|
| FP convert, from gen to vec reg: SCVTF, UCVTF | 3 / 1 / M0 | `A720Wr_3c_M0` |
| FP convert, from vec to gen reg: FCVT[AMNPZ][SU]; FJCVTZS | 3 / 1 / V0 | `WriteFCvt` = 3, V0 |
| FP convert, from vec to vec reg: FCVT, FCVTXN | 3 / 1 / V0 | `WriteFCvt`; FCVTXN `A720Wr_3c_V0` |
| FP move, immed; register: FMOV | 2 / 2 / V (note 1) | `WriteFImm` 2, V; `A720Write_0or2c_1V` (FMOV S/D: [C-4](#c-4); FMOV H 2, V) |
| FP transfer, gen to low half: FMOV | 3 / 1 / M0 | `A720Write_0or3c_1M0` (from ZR: [C-4](#c-4)) |
| FP transfer, gen to high half | 5 / 1 / M0, V | `A720Wr_5c_M0V` |
| FP transfer, vec to gen reg: FMOV | 3 / 2 / V | `WriteFCopy` = 3, V |

Note 1 (l.1047-1048), "Particular FMOV #0 or Register to Register can be optimized in rename stage":
sec. 4.12 ([C-4](#c-4)). Conversions with the operand and the result in FP registers: [D-4](#d-4).
Measured: `fadd h0, h0, h1; fmov h0, h0` + branch 4.0. Test: `FpScalar`, `FacileTest.A720*`.

<a id="c-17"></a>
**C-17. Sec. 3.14, FP loads (p.26-27, l.1052-1104).**
All rows 6 / 3 / L, with I for the post- and pre-index LDR; register offset (all forms) without I;
LDP S/D 6 / 3, Q 6 / 3/2; LDP post- and pre-index (S/D and Q) 6 / 3/2 / L, I. Model:
`A720Wr_FLD` (6, L), `A720Wr_FLDP2` (L 2 cycles), `WriteAdr` for the writeback forms, second register
`A720Wr_FLDHi` ([D-10](#d-10)). "Compared to standard loads, an extra cycle is required to forward
results to FP/ASIMD pipelines" (l.1054-1055). Test: `FpLoadsAndStores`.

<a id="c-18"></a>
**C-18. Sec. 3.15, FP stores (p.27-28, l.1109-1180).**

| SOG group | SOG | Model |
|---|---|---|
| Store vector reg, unscaled; unsigned immed; register offset basic, extend; scale H, S/D; extend, scale H, S/D (B/H/S/D/Q) | 2 / 2 / L01, V | `A720Wr_FST` (two µOPs) |
| **Store vector reg, register offset, scale, Q; extend, scale, Q** | 2 / 2 / I, L01, V | `A720Wr_STRQro`: `A720IsMemScaled` -> `A720Wr_FST_I`, else `A720Wr_FST` (`[a720_strq_ro]`: `ScaledIdxPred` is true for every register-offset form and would give the unscaled Q stores an I µOP) |
| Store vector reg, post-index (B/H/S/D/Q); pre-index Q | 2 / 2 / L01, V, I | `WriteAdr` + `A720Wr_FST` |
| **Store vector reg, pre-index, B/H/S/D** | **3** / 2 / L01, V, I | `WriteAdr` + `A720Wr_FST3` (`[a720_str_pre]`, [I-3](#i-3)) |
| Store vector pair, immed offset, S; D; Q: STP, STNP | 2 / 2 / L01, V | `A720Wr_FST` |
| Store vector pair, post/pre-index, S; D; Q | 2 / 2 / I, L01, V | `WriteAdr` + `A720Wr_FST` |

"Scaled" is the S bit of the addressing mode (`AArch64_AM::getMemDoShift`), as for the Cortex-A76
and A78. The data operands read `ReadST` ([B-1](#b-1)). Measured: three `str q, [xN, x2]` and
an `add` + branch 1.5 (the L01 bound). Test: `FpLoadsAndStores`.

<a id="c-19"></a>
**C-19. Sec. 3.16, ASIMD integer (p.28-31, l.1185-1337).**

| SOG row | SOG | Model |
|---|---|---|
| Absolute diff, arith basic, complex, pair-wise, compare, logical, max/min basic and pair-wise, absolute diff long | 2 / 2 / V | `WriteVd` / `WriteVq` = 2, V |
| SABA, UABA; SABAL(2), UABAL(2) | 4(1) / 1 / V1 (note 2) | `A720Wr_ACC4_V1` + `A720Rd_ACC4_V1` (advance 3) |
| Reduce 4H/4S; 8B/8H; 16B: ADDV, SADDLV, UADDLV and SMAXV, ... | 3 / 1 / V1; 5 / 1 / V1, V; 6 / 1/2 / V1 | `A720Wr_3c_V1`; `A720Wr_5c_V1V`; `A720Wr_6c_V1r2` |
| Dot product SDOT, UDOT, SUDOT, USDOT; matrix multiply-accumulate SMMLA, UMMLA, USMMLA | 3(1) / 2 / V (note 2) | `A720Wr_DOT` 3 + `A720Rd_DOT` (advance 2) |
| Multiply MUL, SQDMULH, SQRDMULH; multiply long SMULL(2), UMULL(2), SQDMULL(2) | 4 / 1 / V0 | `A720Wr_VMUL` |
| MLA, MLS; [SU]MLAL(2), [SU]MLSL(2) | 4(1) / 1 / V0 (note 1) | `A720Wr_VMUL` + `A720Rd_VAcc1` (advance 3) |
| SQRDMLAH, SQRDMLSH; SQDMLAL(2), SQDMLSL(2) | 4(2) / 1 / V0 (note 1) | `A720Wr_VMUL` + `A720Rd_VAcc2` (advance 2) |
| PMUL, PMULL(2) 8x8, D; Q | 2 / 1 / V0 (note 3) | `A720Wr_2c_V0` |
| SADALP, UADALP; SSRA, SRSRA, USRA, URSRA | 4(1) / 1 / V1 (note 2) | `A720Wr_VAccV1` + `A720Rd_VAccV1` (advance 3) |
| Shift by immed basic (incl. SXTL, UXTL), and insert; by register basic | 2 / 1 / V1 | `A720Wr_2c_V1` |
| Shift by immed complex; by register complex | 4 / 1 / V1 | `A720Wr_4c_V1` |

Scalar forms: [D-5](#d-5). Forwarding from a MUL to an MLA accumulator: [U-5](#u-5). Measured
(test): `saba`, `mla`, `sdot` chains 1.0, `sqrdmlah` 2.0 per iteration. Test: `Asimd`.

<a id="c-20"></a>
**C-20. Sec. 3.17, ASIMD FP (p.31-34, l.1341-1495).**

| SOG row | SOG | Model |
|---|---|---|
| FABS, FABD, FADD, FSUB, compare, max/min normal, FNEG | 2 / 2 / V | `WriteVd` / `WriteVq` |
| FCADD; FCMLA | 3 / 2 / V; 4(2) / 2 / V | `A720Wr_3c_V`; `A720Wr_FCMLA` + `A720Rd_FCMLA` |
| FCVTL, FCVTN: F16<->F32; F32<->F64, FCVTXN | 4 / 1/2; 3 / 1 / V0 | `A720Wr_4c_V0r2`; `A720Wr_3c_V0` |
| Convert other, D F32 and Q F64; D F16 and Q F32; Q F16 (also #fbits) | 3 / 1; 4 / 1/2; 6 / 1/4 / V0 | `A720Wr_3c_V0`; `A720Wr_4c_V0r2`; `A720Wr_6c_V0r4` |
| FDIV, FSQRT D F16; D F32; Q F16; Q F32; Q F64 | 8 / 1/4; 8 / 1/2; 12 / 1/8; 10 / 1/4; 13 / 1/2 / V0 (note 3) | latency as printed, V0 held 4, 2, 8, 4, 2 cycles ([I-2](#i-2)) |
| FADDP, FMAXP, FMAXNMP, FMINP, FMINNMP | 3 / 2 / V | `A720Wr_3c_V` |
| FMAXV etc. reduce, F32 and D-form F16; Q-form F16 | 4 / 1; 6 / 2/3 / V | `A720Wr_4c_Vt1`; `A720Wr_6c_Vr3` |
| FMUL, FMULX; FMLA, FMLS; FMLAL(2), FMLSL(2) | 3 / 2; 4(2) / 2; 4(2) / 2 / V (notes 1, 2) | `A720Wr_VFMUL`; `A720Wr_FMA` + `A720Rd_FMA` |
| FRINT*: D F32 and Q F64; D F16 and Q F32; Q F16 | 3 / 1; 4 / 1/2; 6 / 1/4 / V0 | `A720Wr_3c_V0`; `A720Wr_4c_V0r2`; `A720Wr_6c_V0r4` |

Scalar FMULX: [D-5](#d-5). The region notes of sec. 4.7: [U-3](#u-3). Measured (test): `fmla`
chain 2.0. Test: `Asimd`.

<a id="c-21"></a>
**C-21. Sec. 3.18, ASIMD BF16 (p.34, l.1499-1516).**
BFCVTN 4 / 1/2 / V0: `A720Wr_4c_V0r2`; BFDOT 4(2), BFMLALB/T 4(2), BFMMLA 5(3) / 2 / V: `A720Wr_BF42`,
`A720Wr_BF53` + `A720Rd_BF` (advance 2); BFCVT 3 / 1 / V0: `A720Wr_3c_V0`. Measured (test): `bfmmla` chain 3.0. Test: `Asimd`.

<a id="c-22"></a>
**C-22. Sec. 3.19, ASIMD miscellaneous (p.35-36, l.1529-1599).**

| SOG row | SOG | Model |
|---|---|---|
| RBIT; BIF, BIT, BSL; CLS, CLZ, CNT; DUP element; EXT; XTN; INS element; FMOV, MOVI, MVNI; REV; TBL 1-2 regs; TBX 1 reg; TRN; UZP, ZIP | 2 / 2 / V | default 2, V (zero idioms: [C-4](#c-4)) |
| DUP gen reg | 3 / 1 / M0 | `A720Wr_3c_M0` |
| SQXTN(2), SQXTUN(2), UQXTN(2) | 4 / 1 / V1 | `A720Wr_4c_V1` |
| URECPE, URSQRTE D; Q | 3 / 1; 4 / 1/2 / V0 | `A720Wr_3c_V0`; `A720Wr_4c_V0r2` |
| **FRECPE, FRSQRTE: D F32 and scalar forms**; D F16 and Q F32; Q F16 | 3 / 1; 4 / 1/2; 6 / 1/4 / V0 | `A720Wr_3c_V0` (also the scalar H forms, `[a720_frecpe_scalar]`); `A720Wr_4c_V0r2`; `A720Wr_6c_V0r4` ([U-8](#u-8)) |
| FRECPX; FRECPS, FRSQRTS | 3 / 1 / V0; 4 / 2 / V | `A720Wr_3c_V0`; `A720Wr_4c_V` |
| TBL 3; 4 regs; TBX 2; 3; 4 regs | 4 / 1; 4 / 2/3; 4 / 1; 6 / 2/3; 6 / 1/2 / V | V held 2, 3, 2, 3, 4 cycles |
| UMOV, SMOV | 2 / 1 / V | `A720Wr_2c_Vt1` (V 2 cycles) |
| INS gen reg | 5 / 1 / M0, V | `A720Wr_5c_M0V_INS` |

Measured: `frecpe h0, h0` + branch 3.0. Test: `Asimd`.

<a id="c-23"></a>
**C-23. Sec. 3.20, ASIMD loads (p.37-39, l.1612-1735).**
One `InstRW` per SOG row, one µOP whose L (and, where the row names V, V) cycles give the row's
throughput; the writeback forms add `WriteAdr` ("(ASIMD load, writeback form) I", l.1729-1735).
Checked by script: all 202 LD1-LD4 opcodes (with their post-index forms) have the latency and the
throughput of their row in `mca-insts-info`. Examples: LD1 one register 6 / 3 / L; LD1 four registers 7 / 3/4 (L 4 cycles);
LD1 one lane 8 / 2 / L, V; LD1R 6 / 3 / L; LD3 Q 10 / 2/3 (V 3 cycles); LD4 multiple 8 / 1/2 (L 6
cycles).
[D-6](#d-6), [U-2](#u-2). Test: `AsimdLoadsAndStores`.

<a id="c-24"></a>
**C-24. Sec. 3.21, ASIMD stores (p.39-40, l.1739-1816).**
One `InstRW` per SOG row, L01 held `2 / T` cycles and V one cycle; all 138 ST1-ST4 opcodes checked
by script as for [C-23](#c-23). Examples: ST1 one or two registers 2 / 2; ST1 three or four registers 2
/ 1 (L01 2 cycles); ST3 Q B/H/S 4 / 2/3 (3 cycles); ST4 Q B/H/S 4 / 1/2 (4 cycles); ST4 Q D 2 / 1.
Writeback: `WriteAdr`. Every store write has two µOPs ([C-7](#c-7)); the data operands read `ReadST`
([B-1](#b-1)). Test:
`AsimdLoadsAndStores`.

<a id="c-25"></a>
**C-25. Sec. 3.22, cryptography (p.40-41, l.1820-1867).**
AESD, AESE, AESIMC, AESMC 2 / 2 / V: default 2, V; PMULL 64x64 2 / 1 / V0: `A720Wr_2c_V0`; SHA1H,
SHA1SU0/1, SHA256SU0/1, SHA512H/H2/SU0/SU1, SM3* 2 / 1 / V0: `A720Wr_2c_V0r1`; SHA1C/M/P,
SHA256H/H2, SM4E, SM4EKEY 4 / 1 / V0: `A720Wr_4c_V0r1`; BCAX, EOR3, RAX1, XAR 2 / 2 / V ("executed
from the ALU pipeline", note 2): default 2, V. Sec. 4.6 (p.60, l.2939-2940): "two
AESE/AESMC/AESD/AESIMC instruction every cycle (fully pipelined) with an execution latency of two
cycles". AES pairs: [U-6](#u-6). Decoding: [X-1](#x-1). Test:
`CryptoAndCrc`.

<a id="c-26"></a>
**C-26. Sec. 3.23, CRC (p.41, l.1871-1878).**
`[a720_crc]`.
CRC32, CRC32C 2 / 1 / M0, note 1: "late forwarding of the result from a producer µOP to a consumer µOP.
This results in a 1 cycle reduction in latency as seen by the consumer": `A720Wr_CRC` 2, M0 +
`A720Rd_CRC` (advance 1 on the accumulated CRC, Rn). Measured (test):
`crc32x w1, w1, x3` 1.0 per iteration (2 - 1), two independent chains 2.0 (M0). Test: `CryptoAndCrc`.

---

## 4. Not compatible with the SOG (N)

None open. Where the SOG states a value unambiguously, the model follows it (section 3).

---

## 5. Deliberate deviations and interpretations (D)

<a id="d-1"></a>
**D-1. Best case: shortest latency, best throughput, L1 hits, aligned accesses, zero-latency forms.**
SOG: integer divide "5 to 12" / "5 to 20", "Early termination is possible, depending upon the data
values" (l.683-684, l.693-694); load latencies assume an L1 hit (l.783, l.1053, l.1613); unaligned
accesses (sec. 4.4, l.2914-2921); "AUTH FAIL" degrades PAC instructions (l.749); "MOV Wd, Wn, MOV Xd,
Xn and FMOV Sd, Sn, FMOV Dd, Dn, MOV Vd, Vn (vector) ... may not be executed with zero latency under
certain conditions" (l.3226-3227); TLB misses (sec. 4.13, l.3240-3248). Implementation (project
policy: a data-dependent value takes its best value; occupancy = best throughput rounded up):
integer divide latency 5 and divider 5 cycles for both forms (`[a720_div_best]`); L1 hits; aligned
accesses; authentication succeeds; the listed moves always zero latency; TLB hits. Effect:
optimistic for divide-heavy code.

<a id="d-2"></a>
**D-2. Throughput as resource occupancy.**
SOG: throughput in instructions per cycle for the whole core (l.577-579). Implementation: a row with
throughput T on `n` pipelines holds them `n / T` cycles in total; e.g. LDP X 2 L-cycles (T = 3/2 on
three L pipes), ADDV 16B 2 V1-cycles (T = 1/2), UMOV 2 V-cycles (T = 1 on two V pipes), FMAXV Q F16 3
V-cycles (T = 2/3). Facile divides the cycles of each resource by its units, so another instruction
may use the other unit meanwhile, which the SOG neither states nor excludes.

<a id="d-3"></a>
**D-3. The integer divider is a resource of its own.**
`[a720_dividers]`.
SOG: "Integer divides are performed using an iterative algorithm and block any subsequent divide
operations until complete" (sec. 3.5 note 1, l.693); "Each issue pipeline can accept one µOP per
cycle" (l.477). Implementation: a divide takes M0 for one cycle and holds `A720UnitIDiv` for its
occupancy, so that it does not block the other M0 µOPs (MADD, SCVTF, FMOV and DUP from a GPR, CRC).
Reason: the SOG names what a divide blocks (later divides); the lower estimate is preferred (project
policy); the decision taken for the Cortex-A76 and A78 ([a78-sog.md](a78-sog.md) D-3). FP divide and square root are "fully pipelined" (l.1001, l.1495) and
need no such resource. Measured: `udiv` + two `madd` + branch 5.0 (with the latency of D-1).
Test: `DivideAndMultiply`.

<a id="d-4"></a>
**D-4. FP conversions with operand and result in FP registers: 3 / 1 / V0.**
`[a720_fp_cvt_scalar]` (H forms).
SOG: no row names the scalar forms of FCVT[AMNPZ][SU], SCVTF, UCVTF whose operand and result are FP
registers (`scvtf d0, d1`, `fcvtzs h0, h1`, also with `#fbits`). The rows of these mnemonics give 3 /
1 (V0 from a vector register, M0 from a GPR, l.1013-1023), the ASIMD rows 3 / 1 / V0 for the D-form
F32 and Q-form F64, 4 / 1/2 for D-form F16 and Q-form F32, 6 / 1/4 for Q-form F16 (l.1373-1405).
Implementation: every scalar form takes the 3 / 1 / V0 row, the lowest the SOG gives for these
mnemonics, as for the Cortex-A76 and A78 (D-4 there); the tag marks the H forms (FCVT*, `#fbits`
forms, SCVTF/UCVTF H), which would otherwise take an ASIMD F16 row or no row. Measured: `scvtf h0, h0` + branch 3.0. Test:
`FpScalar`.

<a id="d-5"></a>
**D-5. Scalar ASIMD forms take the row of their mnemonic.**
`[a720_asimd_scalar]`.
SOG: the ASIMD rows list mnemonics (SHL, SSHR, USHR, SLI, SRI, SRSHR, URSHR, SSRA, USRA, SRSRA,
URSRA, SQSHL, UQSHL, SQSHLU, SQSHRN, ..., SQDMLAL, SQDMLSL, SQDMULL, FMULX), without saying whether
the scalar forms (`shl d0, d1, #3`, `sqdmlal s0, h1, h2`, `fmulx d0, d1, d2`) belong to them.
Implementation: they take the row of their mnemonic, as for the Cortex-A76 and A78 (D-5 there). Measured (test): `ssra d0, d1, #3` chain
1.0. Test: `Asimd`.

<a id="d-6"></a>
**D-6. ASIMD load/store throughput expressed by pipeline occupancy.**
SOG: the rows of [C-23](#c-23) and [C-24](#c-24) give a throughput that one µOP per pipeline
does not give; which pipeline limits is not stated. Sec. 4.18 (p.68, l.3366-3383) says that the
bandwidth of some LD4/ST4 forms "is limited by decode constraints", without a number ([U-2](#u-2)).
Implementation: a load holds L, or V where the row names V, for more than one cycle so that the
row's throughput results (e.g. LD1 four registers L 4 cycles, LD3 multiple D-form V 3 cycles, LD4
multiple L 6 cycles); a store holds L01 for `2 / T` cycles. A load has one µOP, a store two
([C-7](#c-7)). Reason: one rule for all rows,
the rule chosen for the Cortex-A76 and A78 ([a78-sog.md](a78-sog.md) D-6). Consequence: a mix with
other L users is limited together.

<a id="d-7"></a>
**D-7. Store-to-load forwarding latency.**
SOG sec. 4.5 (l.2925-2934) gives the conditions of forwarding but no latency; the store tables give
store latencies (1 for integer stores, 2 or 3 for FP stores, 2 or 4 for ASIMD stores). Implementation
(shared Facile code, `addMemoryDependencies`, [docs.md](docs.md) section 4.G): by default Facile adds no store -> load edge; with `--facile-memory-deps` a store -> load edge has
the store's latency, then the load's own latency applies; i.e. forwarding is taken to be no slower than
an L1 hit.

<a id="d-8"></a>
**D-8. Not modelled: SVE, SME, MTE, branch alignment, FPCR, memory routines, caches beyond L1.**
The SVE tables (sec. 3.24-3.31, p.42-57, l.1891-2815), the MOVPRFX fusion (sec. 4.19, p.69-72,
l.3414-3593), the SVE forms of sec. 4.12 and 4.18, and SME: `UnsupportedFeatures` lists SVE and SME,
so these instructions have no scheduling data. MTE (Table 3-3 ADDG/SUBG/IRG/GMI/SUBP(S), l.648-666;
sec. 3.10, 3.11, l.900-954; sec. 4.16, 4.17, l.3302-3362): `HasMTE` is unsupported; ADDG/SUBG take
`WriteI` (1, I, as printed), STGPi, STGPpre and STGPpost have scheduling data (the store-pair class;
they match the rows of l.942-948: 1 / 2 / L01, ID, plus I for the write-back forms, two or three
µOPs), the others (IRG, GMI, SUBP(S), LDG, LDGM, STG, ST2G, STZG, STZ2G, STGM, STZGM) are dropped. Sec. 4.2, 4.3, 4.8 (aligned 32-byte
regions, l.3045-3060), 4.9, 4.15 give advice without a cost; sec. 4.13, 4.14 give TLB and L2/L3
latencies, which the load rows (L1 hit, [D-1](#d-1)) do not use. Not modelled.

<a id="d-9"></a>
**D-9. MOV Xd, #{12{1'b0},imm[3:0]}: immediates 0 to 15, unshifted.**
SOG l.3188, l.3192: the immediate form is written as a 16-bit field whose upper 12 bits are zero.
Implementation: `A720ZeroMove` accepts MOVZ[WX]i with an immediate 0 to 15
and shift 0; a shifted MOVZ (`movz x0, #1, lsl #16`) and MOVN are not zero latency. Every
zero-latency form still takes one Mop and one µOP (dispatch), no pipeline. Test: `ZeroLatencyInstructions`, `FacileTest.A720MovzZeroLatencyImmediateRange`.

<a id="d-10"></a>
**D-10. Load pair: the second register has the row's latency and no µOP.**
`[a720_ldp_hi]`.
SOG: the load latencies "represent the maximum latency to load all the registers written by the
instruction" (l.783-784, l.1053-1054); the µOP count is not printed. Implementation: the second
destination of LDP W/X, LDPSW and the FP pairs (`A720Wr_LDHi_0`, 4; `A720Wr_FLDHi`, 6) and the
generic `WriteLDHi` (LDXP, LDAXP; 4, no pipeline) take no µOP; the row's pipelines are those of the
first write. The Cortex-A78 does the
same ([a78-sog.md](a78-sog.md) D-10); the Cortex-A76 also gives the second register the row's
latency, with no µOP for LDP W and LDPSW but an L µOP for LDP X and the FP pairs ([a76-sog.md](a76-sog.md)
C-11, C-15); the Cortex-X1 keeps the L µOP of LDP X ([x1-sog.md](x1-sog.md) D-10). Effect: one µOP less per LDP in the issue bound and the MLP window. Test: `Loads`, `FpLoadsAndStores`.

<a id="d-11"></a>
**D-11. Fused pairs: the second instruction takes nothing; AES pairs are not masked.**
SOG sec. 4.11 (l.3149-3171) lists the pairs but not what a fused pair costs. Implementation
(`computeA720FusedMask`, [docs.md](docs.md) 4.C): the second instruction of a pair takes no Mop, µOP
or pipeline; the dependence graph keeps it. SHL + SRI and FCMP + AXFLAG need not be dependent (the SOG
states no such condition). AESE + AESMC / AESD + AESIMC are also listed (l.3153-3154), but sec. 4.6
(l.2939-2944) gives two AES instructions per cycle and "virtually 4 instructions issue rate" for
interleaved pairs; masking the second would allow four AES µOPs per cycle on two V pipes, so the pair
is left to [U-6](#u-6). Test: `Fusion`.

<a id="d-12"></a>
**D-12. System instructions: latency 1, no pipeline.**
`[a720_sys]`.
SOG: no row for MRS, MSR, DMB, DSB, ISB or other system instructions; sec. 4.10 (p.63-64,
l.3080-3145) lists non-speculative, in-order and flush constraints without cycle counts
([U-9](#u-9)). Implementation: `WriteSys` and `WriteBarrier` latency 1, no pipeline, the lower
estimate (project policy), as the Cortex-A78 model. Test: `SystemInstructions`.

---

## 6. Unclear in the SOG (U)

<a id="u-1"></a>
**U-1. ROB size and the MLP window.**
The SOG gives no reorder-buffer size, no scheduler size and no mispredict penalty. Current handling:
`MicroOpBufferSize = 192`, which is the MLP window (192 µOPs) and not used otherwise by the A720
path. The value is from a secondary source (Cardyak's microarchitecture cheat sheet, unofficial; the
model's comment notes that another secondary source gives 160). `IssueWidth = 5` and
`MispredictPenalty = 11` are not read by the A720 path. Impact: the MLP columns, which also move
with the µOP counts ([C-7](#c-7), [D-10](#d-10)). Whether the ROB counts MOPs or µOPs is not stated.
How to settle: a documented source (whether the A720 Technical Reference Manual, 102530, gives it was
not checked). Test: `DispatchAndPipelines`.

<a id="u-2"></a>
**U-2. Mops and µOPs per instruction.**
SOG l.469-474: instructions are decoded into MOPs, "A MOP can be split into two Micro-OPerations";
sec. 4.1: 5 MOPs and 10 µOPs per cycle; sec. 4.18 (l.3366-3383): the bandwidth of LD4R and LD4 one
lane post-index with 64-bit elements, LD4 multiple Q with elements below 64 bits, ST4 multiple Q "is
limited by decode constraints", without a number. Apart from stores ([C-7](#c-7)) the SOG gives no
µOP counts. Current handling: one Mop per instruction; one µOP per write of the model, plus
`WriteAdr` for writeback forms (rows naming two pipelines, e.g. LD2 "L, V", BL "B, S", LDRAA "M, L",
have one µOP). With at most two µOPs per Mop the µOP bound can bind only above two µOPs per
instruction on average. How to settle: hardware measurement or an owner decision.

<a id="u-3"></a>
**U-3. Region-based fast forwarding.**
SOG sec. 4.7 (p.61-62, l.2968-3040): the forwarding logic of the V pipelines is optimized within the
regions of Tables 4-1 (INT1 ALU/shift/insert/AES/reduction/PERM, INT2 multiply/MAC, INT3
SHA1/SHA256) and 4-2 (FP1 multiply/MAC/compare/add and PERM, FP2 BFDOT/BFMMLA); "interleaving
producers and consumers from regions INT1 and INT2 ... will result in an additional latency of 1
cycle"; "Intercluster communication requires one cycle penalty"; exclusions (FP convert and rounding,
FP divide/sqrt belong to no region; precision must match in FP1; pairwise instructions are neither
producers nor consumers in FP1). Unclear: the exact membership of many instructions, and whether the
table latencies are the in-region or the cross-region values. Current handling: not modelled (every
FP/ASIMD edge has the table latency). Impact: optimistic by one cycle per cross-region edge on
FP/ASIMD dependence chains. How to settle: an owner decision on a classification, or hardware
measurement.

<a id="u-4"></a>
**U-4. ADD/SUB (extended register) with an identity extend.**
SOG "ALU, extend and shift ADD{S}, SUB{S} 2 2 M" (l.642). Unclear: whether the extended-register form
with a zero shift and a full-width extend (UXTX for X, UXTW for W), which assemblers write as `add
x0, sp, x1`, counts as "extend". Current handling: 2 on M (all extended forms, [C-9](#c-9)). Impact: at 1 / 4 / I such instructions would be cheaper. How to
settle: hardware measurement or an owner decision (the same question is U-4 of the A78 document).

<a id="u-5"></a>
**U-5. Multiply-accumulate forwarding between forms and from multiplies.**
SOG sec. 3.5 note 2 (l.695-696) and sec. 3.16 note 1 (l.1332-1334): "late-forwarding of accumulate
operands from similar µOPs"; sec. 4.7 Table 4-1 note 2 (l.2994): "ASIM/SVE INT multiply accumulate
only fast forward to accumulation source". Unclear: whether a MUL (Ra = ZR, on M) or SMULL counts as a
similar µOP for a MADD accumulator, and whether an ASIMD MUL forwards to an MLA accumulator. Current
handling: `A720Rd_MAcc` (advance 1) from `A720Wr_MAC_M0` and `A720Wr_MUL_M`,
`A720Rd_VAcc1` / `A720Rd_VAcc2` from `A720Wr_VMUL` (ASIMD MUL, SQDMULH, SMULL, UMULL, SQDMULL and the
multiply-accumulates alike), i.e. any to any, the lower estimate; SMULH/UMULH are not forwarded
(note 2). Measured (test): `mul x1, x1, x2; madd x1, x3, x4, x1` 3.0 (1 + 2), `smulh` instead of
`mul` 5.0; `mul v0.4s, v0.4s, v1.4s; mla v0.4s, v2.4s, v3.4s` 5.0 (4 + 1); `mla; smlal` on one
accumulator 2.0. The Cortex-X1 takes the same reading ([x1-sog.md](x1-sog.md) U-5). How to settle:
hardware measurement. Test: `DivideAndMultiply`, `Asimd`.

<a id="u-6"></a>
**U-6. AESE/AESMC and AESD/AESIMC pairs.**
SOG sec. 4.11 (l.3153-3154) lists them as fused; sec. 4.6 (l.2939-2955): two AES instructions per
cycle with latency 2, dependent pairs "are higher performance when they are adjacent ... since they
are fused", "virtually 4 instructions issue rate" with four interleaved chunks; no latency for the
pair. Current handling: no fusion and no forwarding; each AES instruction 2 / 2 / V. How to settle: hardware measurement.

<a id="u-7"></a>
**U-7. Exclusives, acquire/release, LSE atomics.**
The SOG has no rows for LDAR/LDAPR, STLR, LDXR/LDAXR/LDXP, STXR/STLXR/STXP or the LSE atomics.
Current handling: stock classes (LDAR, LDAPR, LDXR 4 on L; STLR as a store with `ReadST`; STXR/STLXR
latency 5); `WriteAtomic` is `Unsupported`, so InstrBuilder drops every LSE atomic with a `[WARN]`
([docs.md](docs.md) section 2.F). How to settle: a project decision or
hardware measurement (as for the Cortex-A76 and A78, U-7 there).

<a id="u-8"></a>
**U-8. FRECPE, FRSQRTE, Q-form F64.**
SOG sec. 3.19 (l.1551-1558): rows for D-form F32 and scalar forms, D-form F16 and Q-form F32, Q-form
F16; none for Q-form F64 (`frecpe v0.2d`). Current handling: 3 / 1 / V0 (the lowest row).

<a id="u-9"></a>
**U-9. Special-register access.**
SOG sec. 4.10 (p.63-64, l.3080-3145): reads and writes of most special registers are non-speculative
and/or in order, an FPCR write may act as a barrier or flush, FPSR writes stall at dispatch while
another is pending, FPSR reads wait for prior flag updates to retire; no cycle counts. NZCV and SP
"are fully renamed" (l.3110, l.3141), which matches. Current handling: [D-12](#d-12); no serialisation.

<a id="u-10"></a>
**U-10. Fusion: immediate restrictions and other restrictions.**
SOG sec. 4.11 (l.3170-3171): "For CMP, CMN, TST and BICS, there are restrictions on immediate values
for both instructions of the pair for which fusion is supported. Other particular restrictions apply
on instruction fusion." The restrictions are not given. Also unclear: whether "Rn != ZR" (printed on
the B.cond row only, l.3156) applies to the CSEL/CSET rows. Current handling: no immediate
restriction; Rn != ZR only on the B.cond row ([docs.md](docs.md) 4.C), i.e. the lower estimate. Impact: every compare directly followed by a B.cond. How to settle: hardware
measurement.

<a id="u-11"></a>
**U-11. Fusion: BTI + "Integer DP/BR/BLR/RET/B uncond/CBZ/TBZ".**
SOG sec. 4.11 (l.3164). Unclear: which instructions "Integer DP" covers, and whether the BTI or the
second instruction is absorbed. Current handling: not fused; a BTI takes a Mop and a µOP but no
pipeline ([C-3](#c-3)). Impact: BTI occurs at function entries and branch targets. How to settle: an owner decision.

<a id="u-12"></a>
**U-12. Integer loads, note 2: "Execution Latency is 5 ... when scale with aligned offset of 128 bits".**
SOG sec. 3.8 note 2 (l.853), attached to every register-offset row: "Execution Latency is 5 and
Utilized Pipelines are L, I when scale with aligned offset of 128 bits". Integer loads cannot scale
the index by 16 bytes, and the FP load table (where LDR Q can) has no such note. Current handling: not
modelled (4, L). How to settle: an owner decision or a corrected SOG.

---

## 7. Inconsistent within the SOG (I)

<a id="i-1"></a>
**I-1. "Only Immed pre-index with write back use I pipes".**
Sec. 3.8 note 1 (l.852) on the "Load register, immed pre-index" row (l.800), while the post-index row
(l.797) also names "L, I" without a note, and the load pair post/pre-index rows name I too (l.844-850);
sec. 3.6 note 2 (l.750) repeats the sentence for LDRAA. Implementation: the rows (both forms take
`WriteAdr` on I).

<a id="i-2"></a>
**I-2. ASIMD FP divide: "fully pipelined" against the throughput column.**
Sec. 3.17 note 3 (l.1495): "ASIMD FP divide and square root operations are now performed using a
fully pipelined data path", while the rows print 1/4, 1/2, 1/8, 1/4, 1/2 (l.1406-1410,
l.1468-1477); the scalar rows print 1 with the same note (l.967-999, l.1001). Implementation: the
throughput column (V0 held 4, 2, 8, 4, 2 cycles) ([docs.md](docs.md) 4.I).

<a id="i-3"></a>
**I-3. FP store, pre-index, B/H/S/D: latency 3.**
Sec. 3.15 (p.27, l.1124-1125, checked in the PDF): "Store vector reg, immed pre-index, B/H/S/D-form
STR 3 2 L01, V, I", against 2 for the post-index forms and the Q-form pre-index (l.1120-1127).
Implementation: as printed (C-18). Effect: only on store -> load edges (`--facile-memory-deps`).

<a id="i-4"></a>
**I-4. LDPSW post/pre-index: "I, L" like the offset form.**
Sec. 3.8 (l.842, l.849-850): LDPSW with an immediate offset and with post/pre-index are both "4 3/2
I, L", while every other pre/post-index load adds the I of the base update. Implementation: the
writeback LDPSW has `WriteAdr` and the I of the offset form, i.e. two I µOPs (as the A78 model, I-7
there).

<a id="i-5"></a>
**I-5. Misprints.**
"VCVTAU" in AArch64 rows (l.1385, l.1396; FCVTAU meant); "SM3PARTW2SM 3SS1" (l.1858-1859;
SM3PARTW2, SM3SS1 meant); "Table 4-3" for both the read and the write table (l.3094, l.3125);
"Table A-8" twice (l.3678, l.3687); "ASIM/SVE" (l.2994); the second example of sec. 4.7 (l.3038-3040)
gives the code but not its effect; Table 2-1 "AES µOPs." (l.552); "RETA, RETB" for RETAA, RETAB
(l.730, [C-11](#c-11)). Implementation: the evident meaning.

---

## 8. Outside the A720 path: shared code (X)

<a id="x-1"></a>
**X-1. Cryptographic instructions do not assemble for `cortex-a720` by default.**
LLVM's `cortex-a720` processor has no AES/SHA2/SHA3/SM4 feature (the Cryptographic Extension is
optional, l.426): `aese v0.16b, v1.16b` fails to assemble in the unit tests ("instruction requires:
aes") unless `.arch_extension aes` is given. For a binary, `mca_tool` does not decode them
for `cortex-a720` either: a loop of `aese`, `pmull v.1q`, `sha256h` or `eor3` is analysed as the loop
branch alone, and `sm4e` is decoded as the SVE gather `GLD1B_S_IMM`, which has no scheduling data and
is dropped with a `[WARN]`. The crypto rows of the instruction tables are therefore checked through the assembler path only. Outside the A720 model.

<a id="x-2"></a>
**X-2. MCA gives every call latency 0.**
LLVM MCA's `InstrBuilder` sets the latency of BL, BLR (and other calls) to 0, whatever the scheduling
model says; Facile then floors the LR edge to 1, which meets the SOG's BL / BLR latency 1 (C-8).
Shared LLVM behaviour; not changed.

<a id="x-3"></a>
**X-3. No store -> load memory edge for ASIMD structure stores.**
By default Facile adds no store -> load edge at all ([docs.md](docs.md) 4.G). With
`--facile-memory-deps`, `addMemoryDependencies` still adds no edge from ST1-ST4 to a later load of the same
address (measured with that option: `ldr q1, [x0]; st1 {v1.16b}, [x0]` + branch 0.67 per iteration on the A720 and the A78,
against 7.0 and 6.0 with `str q1, [x0]`). Shared code; not changed. The `ReadST` of their data
operands ([B-1](#b-1)) therefore shows only in the model data.

---

## 9. Taken over from other cores (B)

<a id="b-1"></a>
**B-1. Store data is read one cycle late.**
`[a720_from_a55_store_data]`. **Not SOG-based.**
A720 SOG: stores are split into address and data µOPs (l.867-868, l.1110, l.1740); no statement on
when the data operand is needed. The project decided for the Cortex-A55 (SOG-based there) and the
Cortex-A520 (borrowed) that store data is read one cycle late, and applied it to the Cortex-A76 and
A78 for consistency ([a78-sog.md](a78-sog.md) B-1); it is applied to the A720 likewise.
Implementation: `ReadAdvance<ReadST, 1>`, and `ReadST` on the data operands of every store
form (integer STR/STUR/STTR/STLR/STLLR, register offset, pre/post, STP/STNP; FP STR/STUR/STP of every
addressing mode; the `Vt` of ST1-ST4, with the lane index of the one-lane forms reading `ReadDefault`
in between); STXR/STXP have none; the release forms STLXR/STLXP read `ReadST` on the data (with their stock
writes), as the other cores do. Effect: only Facile's store -> load memory
edges (`--facile-memory-deps`; none by default), and only for producers of latency 2 or more. Measured with `--facile-memory-deps`: `ldr x1, [x0]; str x1, [x0]` + branch
4.0; with Q registers 7.0. To be revisited when an A720 statement or measurement
appears. Tests: `BorrowedStoreDataNeededLater`, `MemoryEdgesOffByDefault`.

---

## 10. Reproducing the cycle counts quoted here

The cycles/iteration figures that are not from `tests/a720_sog_test.cpp` are Facile estimates of
`mca_tool`. Each loop is assembled, linked and analysed with the A720 invocation, plus
`--fractional-cycles` (the unrounded Facile estimate) and `--facile-reason`:

```sh
# body: the instructions of one iteration, e.g. "sdiv w1, w1, w2"
{ echo '.text'; echo '.globl f'; echo '.type f,%function'; echo 'f:'; echo '1:';
  printf '  %s\n' "${body[@]}"; echo '  b 1b'; echo '  ret'; } > t.s
aarch64-linux-gnu-gcc -march=armv8.5-a+crypto+crc+fp16+sha3 -nostdlib -static -Wl,-e,f t.s -o t
mca_tool --mcpu=cortex-a720 --mtriple=aarch64 --dependency ooo --facile --fractional-cycles --facile-reason t
# cycles/iteration = the cycles column of the first CSV row (the loop); the last column says
# which bound decided it (inst, exec, prec)
```

The loop branch `b 1b` is part of every body. The invocation for a binary (without the last two
flags) is `mca_tool --merge-same-header --dependency ooo --mcpu cortex-a720 --facile <binary>`. Per-instruction values: `mca-insts-info --mtriple=aarch64 --mcpu=cortex-a720`, or the
`A720SogTest.Dump` test, which prints opcode, latencies, µOPs, resource cycles and ReadAdvance entries
for the instructions in `$A720_DUMP` (separated by `;`). Re-measure before quoting a number.

---

## 11. Maintaining this document

* Update an item when the SOG reading or the implementation changes. IDs run from 1 without gaps
  within each category, in the order of the items; an item whose difference is resolved becomes a
  C item, and the code, tests and other documents that name an ID are updated with it.
* When a deviation is introduced on purpose, add a D item, and tag the code with a comment that names
  the ID.
* When a behaviour is taken over from another core's model by a project decision, without an A720
  SOG statement, add a B item and tag the code `[a720_from_<core>_<topic>]`.
* `CortexA720Model` is used only by `cortex-a720` and `cortex-a720ae`. If another CPU is ever given
  this model, its own SOG must be compared first.
* Evaluation results (accuracy metrics, totals over benchmark sets) are not recorded in this
  document; cycles/iteration only when reproduced with `mca_tool` as in section 10.
* [docs.md](docs.md) holds the rest of the technical documentation; [a55-sog.md](a55-sog.md),
  [a520-sog.md](a520-sog.md), [a76-sog.md](a76-sog.md), [a78-sog.md](a78-sog.md) and
  [x1-sog.md](x1-sog.md) are the counterparts of this document for the other cores. [README.md](README.md) lists the documents.
