# Cortex-X1: the model against the Software Optimization Guide

> **Who wrote this, and how far to trust it.** This document was written on **2026-11-02** by
> **Claude** (Anthropic's Claude Opus model, running as a sub-agent of Claude Code), after reading
> the Cortex-X1 Core Software Optimization Guide and the contents of this repository's working
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
8. [Outside the X1 path: shared code (X)](#8-outside-the-x1-path-shared-code-x)
9. [Taken over from other cores (B)](#9-taken-over-from-other-cores-b)
10. [Reproducing the cycle counts quoted here](#10-reproducing-the-cycle-counts-quoted-here)
11. [Maintaining this document](#11-maintaining-this-document)

| Category | Items |
|---|---|
| C, compatible | [C-1](#c-1) to [C-24](#c-24) |
| N, not compatible | none open |
| D, deliberate deviation or interpretation | [D-1](#d-1) to [D-13](#d-13) |
| U, unclear in the SOG | [U-1](#u-1) to [U-12](#u-12) |
| I, inconsistent within the SOG | [I-1](#i-1) to [I-8](#i-8) |
| X, outside the X1 path | [X-1](#x-1) to [X-3](#x-3) |
| B, taken over from the Cortex-A55 / A520 / A76 / A78 / A720 (not SOG-based) | [B-1](#b-1) |

---

## 1. About this document

### 1.1 Source

* **Guide**: "Arm® Cortex®-X1 Core Software Optimization Guide", document number
  **PJDOC-466751330-12804**, **Issue 4.0** of 28 April 2021 ("First release for r1p2"), revision
  r1p2, Copyright © 2019-2021 Arm Limited (or its affiliates), **Non-Confidential**, product status
  "Final" (title page and p.2-3, l.1-23, l.89-90). Called "the SOG" below. Page numbers are those of
  the guide's footers ("Page N of 59"); "l.N" is line N of the text extraction of the guide (not
  tracked). The section numbers are those of the PDF (the text extraction drops most of them); they
  are the same as in the Cortex-A78 SOG (3.6 divide and multiply, 4.14 instruction fusion, ...).
  Table 3-7 (p.18, the SMADDL row of [I-5](#i-5)) was checked against the PDF; all other rows were
  read from the text extraction only.
* **Sibling guide**: the Cortex-X1 SOG is the sibling of the Cortex-A78 SOG (same tables, same
  revision). Where the X1 SOG is unclear or inconsistent, the decision taken for the A78
  ([a78-sog.md](a78-sog.md)) is followed and named in the item ("same as A78 X-n"). Where the X1
  SOG is clear and differs from the A78's, the X1 SOG is followed.

### 1.2 What "the implementation" is

The files concerned:

* `ModifiedTarget/AArch64/AArch64SchedNeoverseV1.td`: the TableGen model `NeoverseV1Model` (LLVM's
  Neoverse V1 model with local changes), compiled into the tool's own scheduling tables
  ([docs.md](docs.md) section 5). `src/cpu/sched_model.cpp` (`installLocalSchedModel`, `case
  CpuModel::X1`) installs it **only** for `cortex-x1` and `cortex-x1c` (`src/cpu/cpu_traits.cpp`).
  `neoverse-v1` and `neoverse-512tvb`, which `AArch64Processors.td` also maps to `NeoverseV1Model`,
  have `CpuModel::None` and keep libLLVM's stock model (checked: their `mca-insts-info` output is
  unchanged by the local model). The file is therefore the X1's, as the N2 copy is the A78's; the X1 additions are named `X1...` (`X1UnitIDiv`, `X1Write_...`), the rows
  kept from upstream keep their `V1...` names.
* `src/cpu/cpu_traits.cpp`: the X1 entry (`UopDispatchWidth = 16`, `MopDispatchWidth = 8`,
  `Fusion = FusionTable::X1`, `ZeroLatencyMov = true`, no MLP window resources).
* `src/facile/facile.cpp` and `src/facile/fusion.cpp`: the Facile predictor and the X1 fusion
  table (`computeA78FusedMask` with `HasSelectFusion = false`). With `--facile` the MCA simulation
  is not run, the CSV `cycles` column is the rounded Facile estimate.
* `tests/x1_sog_test.cpp` (suite `X1SogTest`) pins the values below.
  `AArch64TestContext("cortex-x1")` installs the local model (`tests/test_support.h`).

Implementation facts are given by file and identifier (TableGen `def` names, C++ function names),
not by line number. The X1-specific rows of the model carry tags; [section 2.3](#24-code-tags-and-the-items-of-this-document)
maps them to the items.

### 1.3 Categories

| ID | Category | Meaning |
|---|---|---|
| **C-n** | compatible | The implementation does what the SOG states. |
| **N-n** | not compatible | The SOG states something and the implementation does not follow it. Each item says why it is still open. |
| **D-n** | deliberate deviation or interpretation | The implementation knowingly departs from the literal SOG, or fills a gap with an interpretation. Each item gives the SOG text, the implementation and the reason. |
| **U-n** | unclear | The SOG does not decide the question; the implementation makes an assumption or leaves it open. Each item gives the SOG text, what is unclear, the current handling, the expected impact, how to settle it, and the A78 item it follows. |
| **I-n** | inconsistent | Two places of the SOG disagree, or the SOG contradicts itself or has a misprint. Each item says which reading the implementation follows and the A78 item it follows. |
| **X-n** | outside the X1 path | Problems in shared code, in LLVM or in another core's path, found while comparing the X1 model with the SOG. Not part of the X1 model. |
| **B-n** | taken over from another core | The X1 SOG says nothing on the question, and the treatment the project decided for the Cortex-A55 / A520 and applied to the Cortex-A76 / A78 / A720 is applied to the X1 for consistency. **Not SOG-based.** |

### 1.4 Reading the SOG instruction tables

Sec. 3.1 (p.14, l.488-500):

* **Execution latency** is "the minimum latency seen by an operation dependent on an instruction in
  the described group". A number in parentheses is the accumulate latency of a forwarding path
  (notes of sec. 3.6, 3.11, 3.15, 3.16).
* **Execution throughput** is "the maximum throughput (in instructions per cycle) of the specified
  instruction group that can be achieved in the entirety of the Cortex-X1 microarchitecture". The
  model holds a resource of `n` units for `n / T` cycles in total ([D-2](#d-2)).
* **Utilized pipelines**, Table 3-1 (p.14, l.505-520): B branch 0/1, S integer single cycle 0/1, I
  "Integer single Cycle 0/1 and single/multicycle 0/1", M integer single/multicycle 0/1, M0 integer
  multicycle 0, L01 load/store 0/1, L "Load/Store 0/1 and Load 2", D store data 0/1, V FP/ASIMD
  0/1/2/3, V01 FP/ASIMD 0/1, V02 FP/ASIMD 0/2, V13 FP/ASIMD 1/3, V0, V1.
* Ranges ("5 to 12", "1/12 to 1/5") are data dependent; the model takes the best value
  ([D-1](#d-1)).
* The AArch32 tables are not compared (the tool analyses AArch64 code; [D-8](#d-8)); they are
  quoted only where they disagree with an AArch64 row ([I-1](#i-1)).

---

## 2. Implementation overview

### 2.1 Machine model (`NeoverseV1Model`)

`IssueWidth = 16` (not read by Facile, whose widths come from `cpu_traits.cpp`), `MicroOpBufferSize
= 224` (the MLP window width, [U-1](#u-1)), `LoadLatency = 4`, `MispredictPenalty = 11`,
`LoopMicroOpBufferSize = 16` (none of the three read by the X1 path), `CompleteModel = 1`;
`UnsupportedFeatures` = SVE (and SVE2), SME, MTE, CPA, CSSC, MOPS ([D-8](#d-8)).

| Resource | Units | Role |
|---|---|---|
| `V1UnitB` | 2 | Branch 0/1 |
| `V1UnitS` | 2 | Integer single cycle 0/1 |
| `V1UnitM0`, `V1UnitM1` | 1, 1 | Integer single/multicycle 0, 1 |
| `V1UnitL01` | 2 | Load/Store 0/1 |
| `V1UnitL2` | 1 | Load 2 |
| `V1UnitD` | 2 | Store data 0/1 |
| `V1UnitV0` to `V1UnitV3` | 1 each | FP/ASIMD 0 to 3 |
| `V1UnitI` | group S, M0, M1 (4) | "I" |
| `V1UnitM` | group M0, M1 (2) | "M" |
| `V1UnitL` | group L01, L2 (3) | "L" |
| `V1UnitV` | group V0-V3 (4) | "V" |
| `V1UnitV01`, `V1UnitV02`, `V1UnitV13` | groups (2 each) | "V01", "V02", "V13" |
| `V1UnitFlg` | 3 | not a pipeline: flag-setting ALU operations ([D-9](#d-9)) |
| `X1UnitIDiv` | 1 | not a pipeline: the integer divider inside M0 ([D-3](#d-3)) |
| `X1UnitFDiv` | 2 | not a pipeline: the FP divide / square root units inside V0 and V2 ([D-3](#d-3)) |

TableGen lists a use of a pipeline also under every group that contains it (a use of M0 appears as
M0, M and I), so `mca-insts-info` and the tests show, for example, `[V1UnitI:1 V1UnitM:1
V1UnitM0:1]` for `madd x1, x2, x3, x4`.

### 2.2 The Facile path (`src/facile/facile.cpp`)

For `--facile`, every region gets `EstimatedCycles = max(IssueBound, PortBound, PrecedenceBound)`:

* `calculateIssueBound`: `max(uops / 16, Mops / 8)` (uops = the model's `NumMicroOps`, at least one
  per instruction; one Mop per instruction; the absorbed half of a fused pair is not counted;
  [C-1](#c-1), [C-5](#c-5), [U-2](#u-2)).
* `calculatePortUsageBound`: for every resource (pipelines, groups, `V1UnitFlg`, the dividers), the
  sum of the resource cycles divided by its units; the absorbed half of a fused pair is not counted.
* `calculatePrecedenceBound`: the maximum cycle ratio of the dependence graph. Register edges have
  the latency of the producing def minus the consumer's `ReadAdvance`, floored at 1, and 0 for the
  zero-latency MOVs (`getEdgeLatency`, [C-4](#c-4), [C-6](#c-6)); a store with a base update has a
  second node for the update ([C-7](#c-7)); store -> load memory edges, added only with `--facile-memory-deps` (off by default,
  [docs.md](docs.md) 4.G), have the store's latency ([D-7](#d-7)). Loop-carried edges (register and memory) are added only when the region is a loop,
  or with `--ignore-loop-carried disable` ([docs.md](docs.md) 4.J); for a region that is not a loop
  the bound is 0.

The MLP columns use a window of `MicroOpBufferSize` = 224 uops ([U-1](#u-1)); a change of an
instruction's uop count therefore also changes the `mlp` / `mlp_r` columns of the regions around it.

### 2.3 Unit tests

`tests/x1_sog_test.cpp` (suite `X1SogTest`) pins the values of the items below. The per-instruction
data (latency per def, uops, resource cycles, read advances) come from `sogDesc()`, the cycles per
iteration of small loops from `sogFacile()`, the Facile prediction as `mca_tool --facile` computes
it (both in `tests/sog_test_support.h`). `AArch64TestContext("cortex-x1")` (`tests/test_support.h`) installs the local scheduling model.
`X1SogTest.Dump` prints `sogDesc()` of the instructions listed in `$X1_DUMP` (`sogDump()`).

### 2.4 Code tags and the items of this document

| Tag in `AArch64SchedNeoverseV1.td` | Items |
|---|---|
| `[x1_dividers]` | [D-3](#d-3) |
| `[x1_div_best]` | [C-10](#c-10), [D-1](#d-1) |
| `[x1_maddl]` | [C-10](#c-10), [I-5](#i-5), [U-5](#u-5) |
| `[x1_mul_acc_fwd]` | [C-10](#c-10), [C-18](#c-18), [U-5](#u-5) |
| `[x1_logical_flagset]` | [C-9](#c-9) |
| `[x1_ext_flagset]` | [C-9](#c-9), [U-4](#u-4) |
| `[x1_extr]` | [C-11](#c-11) |
| `[x1_ldpsw]` | [C-12](#c-12) |
| `[x1_reg_offset_h]` | [C-12](#c-12), [C-16](#c-16), [I-4](#i-4) |
| `[x1_ldp_hi]` | [C-16](#c-16), [D-10](#d-10) |
| `[x1_store_rows]` | [C-13](#c-13), [C-17](#c-17), [I-3](#i-3) |
| `[x1_fdiv_best]` | [C-14](#c-14), [C-19](#c-19), [D-1](#d-1) |
| `[x1_fp_cvt]` | [C-15](#c-15) |
| `[x1_fp_cvt_scalar]` | [D-4](#d-4) |
| `[x1_asimd_scalar]` | [D-5](#d-5) |
| `[x1_v_throughput]` | [C-19](#c-19), [C-20](#c-20), [D-2](#d-2) |
| `[x1_ls_throughput]` | [C-21](#c-21), [C-22](#c-22), [D-6](#d-6) |
| `[x1_aes]` | [C-23](#c-23), [U-6](#u-6) |
| `[x1_pac_hint]` | [D-13](#d-13), [U-12](#u-12) |
| `[x1_sve]` | [D-8](#d-8) |
| `[x1_from_a55_store_data]` | [B-1](#b-1) |

The rows without a tag carry the SOG group names as comments.

---

## 3. Compatible with the SOG (C)

Values are "latency / throughput / pipelines" as printed. "Model" values were read with
`mca-insts-info --mtriple=aarch64 --mcpu=cortex-x1` and, for variant classes, with the unit tests
(`tests/x1_sog_test.cpp`, suite `X1SogTest`; `X1SogTest.Dump` prints any instruction's data).
Cycles per iteration are Facile estimates of small loops that include the loop branch
([section 10](#10-reproducing-the-cycle-counts-quoted-here)).

### 3.1 Dispatch, pipelines and forwarding

<a id="c-1"></a>
**C-1. Eight Mops and sixteen µOPs per cycle.**
SOG sec. 4.1 (p.51, l.2474): "The dispatch stage can process up to 8 MOPs per cycle and dispatch up
to 16 µOPs per cycle"; sec. 2.1 (p.11, l.384-388): instructions are decoded into MOPs, "A MOP can be
split into two Micro-OPerations (µOPs) further down the pipeline after the decode stage".
Implementation: `cpu_traits.cpp` `UopDispatchWidth = 16`, `MopDispatchWidth = 8`;
`calculateIssueBound` takes the larger of uops / 16 and instructions / 8. The per-type limits of
sec. 4.1 (l.2476-2481: four µOPs for S or B, four for M, two each for M0, V0, V1, six for L) are not
modelled; none is below the issue rate of the pipelines concerned (four S/B, two M, one M0, one V0,
one V1, three L), so none can bind in a steady state. Test: `DispatchAndPipelines`.

<a id="c-2"></a>
**C-2. Fifteen issue pipelines.**
SOG sec. 2.1 (p.11-13, l.388-389, Figure 2-1 l.401-444, Table 2-1 l.461-475): µOPs "issue
out-of-order to one of fifteen issue pipelines. Each issue pipeline can accept one µOP per cycle":
Branch 0, 1, Integer Single-Cycle 0, 1, Integer Single/Multi-Cycle 0, 1, FP/ASIMD 0 to 3, Load/Store
0, 1, Load 2, Store data 0, 1. Implementation: the resources of
[section 2.1](#21-machine-model-neoversev1model); the groups I, M, L, V, V01, V02 and V13 are those
of Table 3-1. Unlike the Cortex-A78, "ALU, basic, flagset" is 3 per cycle but "Conditional compare
CCMN, CCMP 1 4 I" (l.574) is not flag-limited: CCMP takes `WriteI` (no `V1UnitFlg`). Measured
(test): eight independent `add` 2.0; two `str` and four `add` 1.0. Test: `DispatchAndPipelines`.

<a id="c-3"></a>
**C-3. NOP takes no execution resource.**
The SOG has no row for NOP or the other hints; it lists "NOP + Any instruction" as a fused pair
(sec. 4.14, l.2839, [C-5](#c-5)). Implementation (upstream): `WriteHint` has no resource; a NOP that
is not fused counts as one Mop and one µOP. The decision of the Cortex-A55, A520, A76 and A78
([a78-sog.md](a78-sog.md) C-3). Hint-space pointer
authentication: [D-13](#d-13). Test: `DispatchAndPipelines`.

<a id="c-4"></a>
**C-4. Zero-latency MOVs.**
SOG sec. 4.15 (p.58-59, l.2849-2879): "MOV Xd, #0 / MOV Xd, XZR / MOV Wd, #0 / MOV Wd, WZR / MOV
Wd, Wn / MOV Xd, Xn" (and two AArch32 forms) "are executed with zero latency. These instructions do
not utilize the scheduling and execution resources of the machine." The list is word for word the
A78's. Implementation: the stock `NeoverseZeroMove` variant on `MOVZ[WX]i` and `ORR[WX]rs` gives
`V1Write_0c` (latency 0, no resource), and `CpuTraits::ZeroLatencyMov` lets Facile use the latency 0
([docs.md](docs.md) 4.B). FMOV from WZR/XZR and MOVI #0 are not bound to the variant (they take "FP
transfer, from gen to low half" 3 1 M0, l.1319, and "ASIMD move, integer immed" 2 4 V, l.1961), as
for the A78 ([a78-sog.md](a78-sog.md) C-4). `V1Write_0c` is the only latency-0 write of the model:
the stock V1's latency-0, zero-µOP write `V1Write_0c_0Z` is not defined (the second destination of a
load pair has the load latency, [D-10](#d-10)).
"The last 3 instructions may not be executed with zero latency under certain conditions" (l.2879):
[D-1](#d-1). Test: `ZeroLatencyMovs`.

<a id="c-5"></a>
**C-5. Instruction fusion.**
SOG sec. 4.14 (p.58, l.2831-2845), compared row by row with `computeA78FusedMask` /
`isA78FusiblePair` with `HasSelectFusion = false` (`FusionTable::X1`, `src/facile/fusion.cpp`; the
comparison with the A78 table is [docs.md](docs.md) 4.C):

| SOG row | Implementation |
|---|---|
| 1, 2. CMP/CMN (immediate), (register) + B.cond | SUBS/ADDS (any form) with a ZR destination, then `Bcc` |
| 3, 4. TST (immediate), (register) + B.cond | ANDS with a ZR destination, then `Bcc` |
| 5. BICS (register) + B.cond | any BICS (also with a real destination), then `Bcc` |
| 6. NOP + Any instruction | `NOP` or `HINT #0`, then any instruction |
| AESE + AESMC, AESD + AESIMC | not in the mask ([D-11](#d-11), [U-6](#u-6)) |

The A78's CMP + CSEL and CMP + CSET rows do not exist on the X1 and are not fused. Pairs must be
adjacent (l.2845); pairing is greedy and non-overlapping. The "(register)" forms: [U-11](#u-11);
accounting: [D-11](#d-11). Test: `Fusion`.

<a id="c-6"></a>
**C-6. Latency seen by dependent instructions; ALU -> address.**
SOG sec. 3.1 (l.495-496). Implementation: `getEdgeLatency` uses the latency of the def that the
consumer reads (so the base update of a post-index load has 1, not the load's 4) and subtracts the
consumer's `ReadAdvance` (the accumulator paths of [C-10](#c-10), [C-14](#c-14), [C-18](#c-18),
[C-19](#c-19), [C-24](#c-24)), with a floor of 1 cycle except for the zero-latency MOVs. The address
operands have `ReadAdrBase` advance 0: an ALU result is used as an address after its own latency,
no extra cycle (the decision for the Cortex-A55, A520, A76, A78 and A720). Test:
`ZeroLatencyMovs`, `DivideAndMultiply`.

<a id="c-7"></a>
**C-7. Stores are split into address and data µOPs; store-to-load forwarding.**
SOG sec. 3.10 (p.25, l.1106-1108): "Stores µOPs are split into address and data µOPs"; sec. 3.14
and 3.19 (l.1460, l.2260): "split into store address and store data µOPs at dispatch time"; sec. 4.6
(p.53, l.2575-2585): forwarding needs the load to start at the start or the middle of the older
store (not for LDP of two 32-bit registers), loads larger than 8 bytes from at most two stores,
smaller ones from one. Implementation: every store write has two µOPs (address on L01, data on D or
V01). Shared Facile code ([docs.md](docs.md) 4.F, 4.G): the base update of a store is a node of its
own that does not wait for the data, and, with `--facile-memory-deps`, a store -> load edge is left out when the two
register-offset accesses are provably disjoint. The size rules are not checked; the latency of the
forwarding is not stated ([D-7](#d-7)). ASIMD structure stores: [X-2](#x-2). Test: `Stores`.

### 3.2 Instruction tables, integer

<a id="c-8"></a>
**C-8. Sec. 3.3, branches (p.15, l.532-541).**

| SOG group | SOG | Model |
|---|---|---|
| B; BR, RET; CBZ, CBNZ, TBZ, TBNZ | 1 / 2 / B | `V1Write_1c_1B` (two branch pipes) |
| BL; BLR | 1 / 2 / B, S | `V1Write_1c_1B_1S` (two µOPs) |

B.cond is not listed; it has `WriteBr` like B. MCA gives every call latency 0 ([X-1](#x-1));
Facile floors the edge to 1. Measured (test): four `cbz` 2.0. Test: `Branches`.

<a id="c-9"></a>
**C-9. Sec. 3.4, arithmetic and logical (p.15-16, l.557-590).**

| SOG group | SOG | Model |
|---|---|---|
| ALU, basic: ADD, ADC, AND, BIC, EON, EOR, ORN, ORR, SUB, SBC | 1 / 4 / I | `WriteI` = `V1Write_1c_1I` |
| ALU, basic, flagset: ADDS, ADCS, ANDS, BICS, SUBS, SBCS | 1 / 3 / I | `V1Write_1c_1I_1Flg` ([D-9](#d-9)) |
| **ANDS, BICS (TST) register, no shift** | 1 / 3 / I (basic, flagset) | `X1Write_LogicalS`: `NeoverseNoLSL` -> `V1Write_1c_1I_1Flg` (`[x1_logical_flagset]`) |
| ALU, extend and shift: ADD{S}, SUB{S} | 2 / 2 / M | `WriteIEReg` = `V1Write_2c_1M`; **flag-setting extended forms** `V1Write_2c_1M_1Flg` (`[x1_ext_flagset]`: not the `IsCheapLSL` variant, which would read the extend operand as a shift and give UXTB 1 / I; [U-4](#u-4)) |
| Arithmetic, LSL shift <= 4 (flagset or not) | 1 / 4; 1 / 3 / I | `V1WriteISReg` / `V1WriteISRegS`, `IsCheapLSL` |
| Arithmetic, LSR/ASR/ROR shift or LSL > 4 | 2 / 2 / M | the same variants otherwise |
| Conditional compare; conditional select | 1 / 4 / I | `WriteI` (no flags resource, [C-2](#c-2)) |
| Logical, shift, no flagset | 1 / 4 / I | `V1Write_1c_1I` (ORR: zero-move variant) |
| Logical, shift, flagset: ANDS, BICS | 2 / 2 / M | `X1Write_LogicalS` shifted -> `V1Write_2c_1M_1Flg` |

Measured: `ands x1, x1, x2` + branch 1.0. Test: `ArithmeticAndLogical`.

<a id="c-10"></a>
**C-10. Sec. 3.6, divide and multiply (p.18-19, l.699-795; Table 3-7 checked in the PDF).**

| SOG group | SOG | Model |
|---|---|---|
| **Divide, W-form; X-form: SDIV, UDIV** | 5 to 12; 5 to 20 / 1/12 to 1/5; 1/20 to 1/5 / M0 (note 1) | `X1Write_5c_1M0_IDiv`: 5, M0 1 cycle, divider 5 cycles (`[x1_div_best]`, [D-1](#d-1), [D-3](#d-3)) |
| Multiply: MUL, MNEG | 2 / 2 / M | `V1WriteIM` with a zero accumulator -> `V1Wr_IM` 2, M; forwards to an accumulator ([U-5](#u-5), `[x1_mul_acc_fwd]`) |
| Multiply accumulate, W-form; X-form: MADD, MSUB | 2(1) / 1 / M0 (note 2) | `V1Wr_IMA` 2, M0 + `V1Rd_IMA` (advance 1) |
| **Multiply accumulate long: SMADDL, SMSUBL, UMADDL, UMSUBL** | 2(1) / 2 / M0 (sic) | `X1WriteIML` -> `X1Wr_IMAL` 2, M + `V1Rd_IMA` (`[x1_maddl]`, [I-5](#i-5)) |
| Multiply high: SMULH, UMULH | 3 / 2 / M | `V1Write_3c_1M` |
| Multiply long: SMNEGL, SMULL, UMNEGL, UMULL | 2 / 2 / M | zero accumulator -> `V1Wr_IM` |

Note 2 (l.793-795), "Accumulator forwarding is not supported for consumers of 64 bit multiply high
operations": `V1Rd_IMA` names the multiply-accumulate writes and the multiplies without an
accumulator (`V1Wr_IM`, [U-5](#u-5)), not SMULH/UMULH. Measured: `mul x1, x1, x2; madd x1, x3, x4,
x1` + branch 3.0 (1 + 2); `sdiv w1, w1, w2` +
branch 5.0; `udiv x1, x2, x3` + two `madd` + branch 5.0; four independent `smaddl` +
branch 2.0. Forwarding between forms: [U-5](#u-5). Test: `DivideAndMultiply`.

<a id="c-11"></a>
**C-11. Sec. 3.8, miscellaneous data processing (p.21, l.885-903).**

| SOG group | SOG | Model |
|---|---|---|
| Address generation: ADR, ADRP | 1 / 4 / I | `WriteI` |
| **Bitfield extract, one reg; two regs: EXTR** | 1 / 4 / I; 3 / 2 / I, M (note 1) | `V1WriteExtr`: `X1IsExtrOneRegPred` (Rn == Rm **or imm == 0**) -> `V1Write_1c_1I`, else `V1Write_3c_1I_1M` (`[x1_extr]`; the stock predicate checks only Rn == Rm) |
| Bitfield move, basic: SBFM, UBFM; insert: BFM | 1 / 4 / I; 2 / 2 / M | `WriteIS`; `V1Write_2c_1M` |
| Count leading; move immed; reverse; variable shift | 1 / 4 / I | `WriteI` / `WriteImm` / `WriteIS` 1, I |

`mca-insts-info` shows EXTR with 1 cycle only because it
resolves a variant with synthesized operands (Rn == Rm, printed as `ror`) and so shows only the
one-register arm ([X-3](#x-3)). Measured: `extr
x1, x1, x2, #0` + branch 1.0. Test:
`MiscDataProcessing`.

<a id="c-12"></a>
**C-12. Sec. 3.9, loads (p.23-24, l.960-1024).**

| SOG group | SOG | Model |
|---|---|---|
| Load register, literal, unscaled, unprivileged, unsigned immed | 4 / 3 / L | `WriteLD` = `V1Write_4c_1L` |
| Load register, immed post-index / pre-index | 4 / 3 / L, I | `WriteAdr` (1, I) + `WriteLD` |
| Load register, register offset: basic; scale by 4/8; extend; extend, scale by 4/8 | 4 / 3 / L | `WriteLDIdx` = `V1Write_4c_1L` |
| **Load register, register offset, scale by 2; extend, scale by 2: LDRH, LDRSH** | 5 / 3 / I, L | `X1WriteLDRH`: `X1IsMemScaled` -> `V1Write_5c_1I_1L`, else `V1Write_4c_1L` (`[x1_reg_offset_h]`) |
| Load pair, W-form; X-form: LDP, LDNP | 4 / 3; 4 / 1.5 / L | `V1Write_4c_1L` + `V1Write_4c_0Z`; `WriteLD` + `WriteLDHi` (each 4, one L µOP; [D-10](#d-10)) |
| Load pair, signed words: LDPSW (offset, pre, post) | 5 / 1.5 / I, L | `V1Write_5c_1I_2L` (L held 2 cycles) + `V1Write_5c_0Z` (`[x1_ldpsw]`) |
| Load pair, post-index / pre-index, W; X | 4 / 3; 4 / 1.5 / L, I | `WriteAdr` + the offset-form writes |

"Scaled" is the S bit of the addressing mode (`AArch64_AM::getMemDoShift`), so `ldrh w1, [x0, x1]`
and `ldrh w1, [x0, w1, uxtw]` stay 4. The pre/post-index LDPSW also has `WriteAdr`, i.e. two I µOPs
for "I, L" ([I-8](#i-8)). Measured: `ldrh w1, [x0, x1, lsl #1]` + branch 5.0. Acquire and exclusive loads: [U-7](#u-7). Test:
`Loads`.

<a id="c-13"></a>
**C-13. Sec. 3.10, stores (p.25-26, l.1105-1145).**

| SOG group | SOG | Model |
|---|---|---|
| Store register, unscaled; unprivileged; unsigned immed | 1 / 2 / L01, D | `WriteST` = `V1Write_1c_1L01_1D` |
| Store register, immed post-index / pre-index | 1 / 2 / L01, D, I | `WriteAdr` + `WriteST` |
| Store register, register offset: basic; scaled by 4/8; extend; extend, scale by 4/8 | 1 / 2 / L01, D | `WriteSTIdx` = `V1Write_1c_1L01_1D` |
| **Store register, register offset, scaled by 2; extend, scale by 1 (sic, [I-3](#i-3)): STRH** | 2 / 2 / I, L01, D | `X1WriteSTRH`: scaled -> `X1Write_2c_1I_1L01_1D`, else `V1Write_1c_1L01_1D` (`[x1_store_rows]`) |
| Store pair, immed offset; post-index; pre-index: STP, STNP | 1 / 2 / L01, D (+ I) | `WriteSTP` = `V1Write_1c_1L01_1D` (+ `WriteAdr`) |

A store's latency matters only on Facile's store -> load edges ([D-7](#d-7)), which exist only with `--facile-memory-deps`. The data operand reads
`ReadST` ([B-1](#b-1)). Release and exclusive
stores: [U-7](#u-7). Measured (test): four `str` 2.0. Test: `Stores`.

### 3.3 Instruction tables, FP and ASIMD

<a id="c-14"></a>
**C-14. Sec. 3.11, FP data processing (p.27-28, l.1209-1245, notes l.1294-1302).**

| SOG group | SOG | Model |
|---|---|---|
| FABS; FADD, FSUB; FMIN/FMAX(NM); FNEG | 2 / 4 / V | `WriteF` = `V1Write_2c_1V` |
| FCMP{E}, FCCMP{E} | 2 / 1 / V0 | `WriteFCmp` = `V1Write_2c_1V0` |
| **FDIV H; S; D** | 7 / 8/7; 7 to 10 / 8/9 to 8/7; 7 to 15 / 2/7 to 4/7 / V02 (note 1) | `V1WriteFDS_7c2`, `_7c2`, `_7c4`: 7, V02 1 cycle, an FP divide unit 2, 2, 4 cycles (`[x1_dividers]`, [D-3](#d-3); `[x1_fdiv_best]`, [D-1](#d-1)) |
| **FSQRT H; S; D** | 7 / 8/7; 7 to 9 / 1 to 8/7; 7 to 16 / 4/15 to 4/7 / V02 | `V1WriteFDS_7c2`, `_7c2`, `_7c4` (likewise) |
| FMUL, FNMUL | 3 / 4 / V (note 2) | `WriteFMul` 3, V |
| FMADD, FMSUB, FNMADD, FNMSUB | 4 (2) / 4 / V (note 3) | `V1Wr_FMA` 4 + `V1Rd_FMA` (advance 2 from `WriteFMul` and `V1Wr_FMA`) |
| FRINT* | 3 / 2 / V02 | `V1Write_3c_1V02` |
| FCSEL | 2 / 2 / V02 | `V1Write_2c_1V02` |

The SOG throughput of the divide rows is for the V0/V2 pair: one unit is held `ceil(2 / T)` cycles
([D-2](#d-2)); the best latency is the start of each range. The occupancy is held on the divide unit, not on V02,
so that FRINT, FCSEL, converts, ASIMD multiplies etc. are not blocked. Note 2 (l.1297-1299): FMUL ->
accumulator 1 (`V1Rd_FMA` names `WriteFMul`, a `WriteRes` here). Measured: `fdiv d0, d2, d1; fcvtzs
w1, d3; frinta d4, d5` + branch 2.0; (test) `fmul; fmadd (accumulator); fmov` 7.0 (1 + 4 + 2).
Test: `FpScalar`.

<a id="c-15"></a>
**C-15. Sec. 3.12, FP miscellaneous (p.29, l.1306-1323).**

| SOG group | SOG | Model |
|---|---|---|
| FP convert, from vec to vec reg: FCVT, FCVTXN | 3 / 2 / V02 | `WriteFCvt` = `V1Write_3c_1V02`; FCVTXN `V1Write_3c_1V02` |
| FP convert, from gen to vec reg: SCVTF, UCVTF | 3 / 1 / M0 | `V1Write_3c_1M0` (M0 one cycle) |
| FP convert, from vec to gen reg: FCVT[AMNPZ][SU] | 3 / 1 / V02 | `V1Write_3c2_1V02` (V02 2 cycles, [D-2](#d-2)); **also the fixed-point forms** (`#fbits`, `[x1_fp_cvt]`; not `WriteFCvt`, 2 per cycle) |
| FP move, immed; register | 2 / 4 / V | `WriteFImm`; `V1Write_2c_1V` |
| FP transfer, gen to low half: FMOV | 3 / 1 / M0 | `V1Write_3c_1M0` (also from ZR, [C-4](#c-4)) |
| FP transfer, gen to high half: FMOV | 5 / 1 / M0, V | `V1Write_5c_1M0_1V` |
| FP transfer, vec to gen reg: FMOV | 2 / 1 / V1 | `WriteFCopy` = `V1Write_2c_1V1` |

Measured: two `fcvtzs x, d, #3` + branch 2.0. Conversions with the operand and the result in FP registers: [D-4](#d-4). Test:
`FpScalar`.

<a id="c-16"></a>
**C-16. Sec. 3.13, FP loads (p.30-31, l.1373-1426).**

| SOG group | SOG | Model |
|---|---|---|
| Load vector reg, literal; unscaled immed; unsigned immed | 6 / 3 / L | `V1Write_6c_1L` |
| Load vector reg, immed post-index / pre-index | 6 / 3 / L, I | `WriteAdr` + `V1Write_6c_1L` |
| Load vector reg, register offset, basic | 6 / 3 / L, **I** | X index without the scale: `V1Write_6c_1L`, the printed I is not modelled ([I-4](#i-4)) |
| Load vector reg, register offset, scale, S/D; extend; extend, scale, S/D | 6 / 3 / L | `V1Write_6c_1L` |
| **Load vector reg, register offset, H/Q without the scale** ("basic"; "extend") | 6 / 3 / L, I ([I-4](#i-4)); 6 / 3 / L | `X1WriteLDRHQro` unscaled -> `V1Write_6c_1L` (`[x1_reg_offset_h]`) |
| Load vector reg, register offset, scale, H/Q; extend, scale, H/Q | 7 / 3 / I, L | `X1WriteLDRHQro` scaled -> `V1Write_7c_1I_1L` |
| Load vector pair, immed offset, S/D | 6 / 3 / L | `V1Write_6c_1L` + `V1Write_6c_0Z` |
| **Load vector pair, immed offset, Q: LDP, LDNP** | 6 / 3/2 / L | `V1Write_6c_2L` + `V1Write_6c_0Z` (`[x1_ldp_hi]`, [D-10](#d-10): the second register has the FP load latency 6, not `WriteLDHi`'s 4) |
| Load vector pair, post/pre-index, S/D; **Q** | 6 / 3; 6 / 3/2 / L, I | `WriteAdr` + the offset-form writes (Q likewise `[x1_ldp_hi]`) |

"Compared to standard loads, an extra cycle is required to forward results to FP/ASIMD pipelines"
(l.1375-1376). Measured: `ldp q0, q1, [x0]; fadd v1.4s, v1.4s, v1.4s; fmov x0, d1` + branch
10.0; `ldr q0, [x0, x1]; fmov x1, d0` + branch 8.0. Test:
`FpLoadsAndStores`.

<a id="c-17"></a>
**C-17. Sec. 3.14, FP stores (p.32-33, l.1459-1528).**

| SOG group | SOG | Model |
|---|---|---|
| Store vector reg, unscaled immed; unsigned immed, B/H/S/D/Q | 2 / 2 / L01, V01 | `V1Write_2c_1L01_1V01` |
| Store vector reg, immed post-index / pre-index | 2 / 2 / L01, V01, I | `WriteAdr` + `V1Write_2c_1L01_1V01` |
| Store vector reg, register offset: basic (B/H/S/D, Q); scale S/D; extend (B/H/S/D, Q); extend, scale S/D | 2 / 2 / L01, V01 | `V1Write_2c_1L01_1V01`; **H and Q without the scale** (`[x1_store_rows]`: no I µOP) |
| Store vector reg, register offset, scale, H; scale, Q; extend, scale, H; extend, scale, Q | 2 / 2 / I, L01, V01 | `X1WriteSTRHQro` scaled -> `V1Write_2c_1I_1L01_1V01` |
| Store vector pair, immed offset, S; D; Q | 2 / 2 / L01, V01 | `V1Write_2c_1L01_1V01` (Q: [I-7](#i-7)) |
| Store vector pair, post/pre-index, S; D | 2 / 2 / I, L01, V01 | `WriteAdr` + `V1Write_2c_1L01_1V01` |
| Store vector pair, post/pre-index, Q | 2 / 1 / I, L01, V01 | `WriteAdr` + `V1Write_2c_2L01_1V01` (L01 2 cycles) |

Unlike the A78 SOG (its I-5), the X1 SOG gives the I to both scaled Q rows. The data operands read
`ReadST` ([B-1](#b-1)). Test: `FpLoadsAndStores`.

<a id="c-18"></a>
**C-18. Sec. 3.15, ASIMD integer (p.34-36, l.1559-1684, notes l.1757-1764).**

| SOG row | SOG | Model |
|---|---|---|
| Absolute diff; arith basic, complex, pair-wise; compare; logical; max/min basic and pair-wise; absolute diff long | 2 / 4 / V | `WriteVd` / `WriteVq` = `V1Write_2c_1V` |
| SABA, UABA; SABAL(2), UABAL(2); SADALP, UADALP | 4(1) / 2 / V13 (note 2) | `V1Wr_ADA` 4 + `V1Rd_ADA` (advance 3) |
| Reduce 4H/4S; 8B/8H; 16B: ADDV, SADDLV, UADDLV and SMAXV, ... | 2 / 2 / V13; 4 / 2 / V13, V; 4 / 1 / V13 | `V1Write_2c_1V13`; `V1Write_4c_1V13_1V`; `V1Write_4c_2V13` |
| Dot product: SDOT, UDOT | 2 (1) / 4 / V (note 2) | `V1Wr_VDOT` 2 + `V1Rd_VDOT` (advance 1) |
| Multiply: MUL; SQDMULH, SQRDMULH | 4 / 2 / V02 | `X1Wr_VMUL`; `V1Write_4c_1V02` |
| Multiply accumulate: MLA, MLS; long: [SU]MLAL(2), [SU]MLSL(2) | 4(1) / 2 / V02 (note 1) | `V1Wr_VMA` / `V1Wr_VMAL` + `V1Rd_VMA` / `V1Rd_VMAL` (advance 3 from either and from MUL, SMULL, UMULL: [U-5](#u-5), `[x1_mul_acc_fwd]`) |
| Multiply accumulate high: SQRDMLAH, SQRDMLSH; saturating long: SQDMLAL(2), SQDMLSL(2) | 4 / 2 / V02 | `V1Write_4c_1V02` (no forwarding) |
| PMUL, PMULL(2) 8x8, D; Q | 3 / 2 / V01 (note 3) | `V1Write_3c_1V01` |
| Multiply long: SMULL(2), UMULL(2); SQDMULL(2) | 3 / 2 / V02 | `X1Wr_VMULL`; `V1Write_3c_1V02` |
| SSRA, SRSRA, USRA, URSRA | 4(1) / 2 / V13 (note 2) | `V1Wr_VSA` + advance 3 |
| Shift by immed basic; and insert; by register basic | 2 / 2 / V13 | `V1Write_2c_1V13` |
| Shift by immed, complex; by register, complex | 4 / 2 / V13 | `V1Write_4c_1V13` |

The X1 rows are the A78's with twice the throughput and other pipelines; SABA is 4(1) here (A78:
3(1), its I-1). SQSHRUN takes the complex row through an extra regular expression (the stock
`^SQSHU?RNv` cannot match it). Scalar forms: [D-5](#d-5). Measured (test): `saba`,
`sdot`, `mla` chains 1.0 per iteration; `mul v0.4s, v0.4s, v1.4s; mla v0.4s, v2.4s, v3.4s`
5.0 ([U-5](#u-5)). Test: `Asimd`.

<a id="c-19"></a>
**C-19. Sec. 3.16, ASIMD FP (p.38-40, l.1776-1866, notes l.1924-1932).**

| SOG row | SOG | Model |
|---|---|---|
| FABS, FABD, FADD, FSUB, FADDP, compare, max/min normal and pairwise, FNEG | 2 / 4 / V | `WriteVd` / `WriteVq` |
| FCVTL, FCVTN: F16<->F32; F32<->F64, FCVTXN | 4 / 1; 3 / 2 / V02 | `V1Write_4c_2V02`; `V1Write_3c_1V02` |
| Convert other, D F32 and Q F64; D F16 and Q F32 | 3 / 2; 4 / 1 / V02 | `V1Write_3c_1V02`; `V1Write_4c_2V02` |
| **Convert other, Q F16** | 6 / 1/2 / V02 | `X1WriteV02<6, 4, 4>` (V02 4 cycles; `[x1_v_throughput]`) |
| FDIV, FSQRT (all forms) | ranges / V02 (note 3) | `V1WriteFDS_*`, best values ([D-1](#d-1), [D-3](#d-3)): D F16 7/7, D F32 7/4, Q F16 10/10 (FSQRT 11/11), Q F32 7/7, Q F64 7/7 (latency / divide unit cycles) |
| FMAXV etc. reduce, F32 and D-form F16; Q-form F16 | 4 / 2; 6 / 4/3 / V | `V1Write_4c_2V`; `V1Write_6c_3V` |
| FMUL, FMULX; FMLA, FMLS | 3 / 4; 4 (2) / 4 / V (notes 1, 2) | `V1Wr_FPM`; `V1Wr_FPMA` + `V1Rd_FPMA` (from `V1Wr_FPM` and `V1Wr_FPMA`) |
| FRINT*: D F32 and Q F64; D F16 and Q F32; **Q F16** | 3 / 2; 4 / 1; 6 / 1/2 / V02 | `V1Write_3c_1V02`; `V1Write_4c_2V02`; `X1WriteV02<6, 4, 4>` (`[x1_v_throughput]`) |

FMLAL: [U-12](#u-12). Test:
`Asimd`.

<a id="c-20"></a>
**C-20. Sec. 3.17, ASIMD miscellaneous (p.41-42, l.1944-2008).**

| SOG row | SOG | Model |
|---|---|---|
| RBIT; BIF, BIT, BSL; CLS, CLZ, CNT; DUP element; EXT; XTN; INS element; FMOV, MOVI; REV; TRN; UZP, ZIP | 2 / 4 / V | default `V1Write_2c_1V` |
| DUP gen reg | 3 / 1 / M0 | `V1Write_3c_1M0` |
| SQXTN(2), SQXTUN(2), UQXTN(2) | 4 / 2 / V13 | `V1Write_4c_1V13` |
| URECPE, URSQRTE D; **Q** | 3 / 2; 4 / 1 / V02 | `V1Write_3c_1V02`; `X1WriteV02<4, 1, 2>` (`[x1_v_throughput]`) |
| FRECPE, FRSQRTE: D F32 and **scalar forms**; **D F16 and Q F32**; **Q F16** | 3 / 2; 4 / 1; 6 / 1/2 / V02 | `V1Write_3c_1V02` (also the scalar H forms); `X1WriteV02<4, 1, 2>`; `X1WriteV02<6, 2, 4>` (`[x1_v_throughput]`, [U-8](#u-8)) |
| FRECPX; FRECPS, FRSQRTS | 3 / 2 / V02; 4 / 4 / V | `V1Write_3c_1V02`; `V1Write_4c_1V` |
| TBL 1-2 regs; 3; 4 regs | 2 / 2; 4 / 1; 4 / 2/3 / V01 | `V1Write_2c_1V01`; `V1Write_4c_2V01`; `V1Write_4c_3V01` |
| TBX 1; 2; 3; 4 regs | 2 / 2; 4 / 1; 6 / 2/3; 6 / 2/5 / V01 | `V1Write_2c_1V01`; `V1Write_4c_2V01`; `V1Write_6c_3V01`; `V1Write_6c_5V01` |
| **UMOV, SMOV** | 2 / 1 / V | `X1WriteV<2, 1, 4>`: V held 4 cycles (`[x1_v_throughput]`, [D-2](#d-2)) |
| INS gen reg | 5 / 1 / M0, V | `V1Write_5c_1M0_1V` |

The X1's UMOV/SMOV row names V with throughput 1 (the A78's: V1, 1); all four V pipes are taken as
eligible, each instruction holding the group for 4 / 1 = 4 cycles in total. Measured: two
`umov` + branch 2.0; two `frecpe v.4s` + branch 2.0. Test: `Asimd`.

<a id="c-21"></a>
**C-21. Sec. 3.18, ASIMD loads (p.43-45, l.2075-2187).**
Checked row by row; every row but one has the printed latency and throughput with the stock rows (one L µOP
per L slot, V µOPs as printed). Examples: LD1 one register 6 / 3 / L; LD1 four registers D 6 / 3/2
(two L µOPs), Q 7 / 3/4; LD1 one lane, LD1R 8 / 3 / L, V; LD2 D 8 / 2, Q 8 / 3/2; LD3 D, one lane,
LD3R 8 / 4/3, LD3 Q 8 / 1; LD4 D, one lane, LD4R 8 / 1.

| SOG row | SOG | Model |
|---|---|---|
| **LD4 multiple, Q-form, B/H/S; D** | 9 / 1/2 / L, V | `X1WriteLDV<9, 8, 6, 4>`: 8 µOPs, L held 6 cycles, V 4 (`[x1_ls_throughput]`, [D-6](#d-6)) |

Writeback forms add `WriteAdr` (1, I), "(ASIMD load, writeback form) I" (l.2187). Measured: `ld4
{v0.16b-v3.16b}, [x0]` + branch 2.0. Sec. 4.18:
[U-2](#u-2). Test: `AsimdLoadsAndStores`.

<a id="c-22"></a>
**C-22. Sec. 3.19, ASIMD stores (p.47-48, l.2259-2329).**
Checked row by row, with the data side on V01 ([I-6](#i-6)): ST1 one to four registers (2 / 2, 2 /
1, 2 / 2/3, 2 / 1/2 as printed per form), ST1 one lane 4 / 2, ST2 D 4 / 2, Q 4 / 1, one lane 4 / 2,
ST3 D 4 / 1, Q 5 / 2/3, one lane 4 / 1, ST4 D 6 / 2/3, Q B/H/S 7 / 1/3, Q D 4 / 1/2, one lane D 4 / 1:
each L01 µOP holds one L01 cycle and the throughput is reached.

| SOG row | SOG | Model |
|---|---|---|
| **ST4 one lane, B/H; S** | 6 / 4/3 / V, L01 | `X1WriteSTV<6, 6, 2, 2>`: L01 and V01 2 cycles each, 1 per cycle (`[x1_ls_throughput]`, [D-6](#d-6)) |

Writeback "(1) +I" (l.2329, l.2375): `WriteAdr`. The data operands read `ReadST` ([B-1](#b-1)). Test: `AsimdLoadsAndStores`.

<a id="c-23"></a>
**C-23. Sec. 3.20, cryptography (p.49, l.2382-2399).**
AESD, AESE, AESIMC, AESMC 2 / 2 / V01: `V1Write_2c_1V01`; PMULL 64x64 2 / 2 / V01: `V1Write_2c_1V01`;
SHA1H, SHA1SU0, SHA1SU1, SHA256SU0, SHA256SU1 2 / 1 / V0: `V1Write_2c_1V0`; SHA1C/M/P, SHA256H/H2 4
/ 1 / V0: `V1Write_4c_1V0`. Sec. 4.7 (p.53, l.2590-2591): "two
AESE/AESMC/AESD/AESIMC instruction every cycle (fully pipelined) with an execution latency of two
cycles". The stock read advance of 2 from AESE/AESD to AESMC/AESIMC is not in the model (`[x1_aes]`,
[U-6](#u-6)); Facile would not apply it (the write is a `WriteSequence`): `aese; aesmc` + branch
4.0. Unlike the A720, `cortex-x1` has the AES and SHA2 features, so these
instructions are decoded in binaries. Test: `CryptoAndCrc`.

<a id="c-24"></a>
**C-24. Sec. 3.21, CRC (p.50, l.2439-2455).**
CRC32, CRC32C 2 / 1 / M0, note 1: "late-forwarding of the result from a producer µOP to a consumer
µOP. This results in a 1 cycle reduction in latency": `V1Wr_CRC` 2, M0 + `V1Rd_CRC` (advance 1 on
the accumulated CRC). Measured (test): `crc32x w1, w1, x3` 1.0 per iteration. Test: `CryptoAndCrc`.

---

## 4. Not compatible with the SOG (N)

None open. Where the SOG states a value unambiguously, the model follows it (section 3).

---

## 5. Deliberate deviations and interpretations (D)

<a id="d-1"></a>
**D-1. Best case: shortest latency, best throughput, L1 hits, aligned accesses, zero-latency MOVs.**
SOG: integer divide "5 to 12" / "5 to 20", "Early termination is possible, depending upon the data
values" (l.703-704, l.791-792); FP divide and square root ranges (l.1217-1245, l.1814-1818,
l.1857-1866); load latencies assume an L1 hit (l.961, l.1374, l.2076); unaligned accesses (sec. 4.5,
l.2566-2572); "The last 3 instructions may not be executed with zero latency under certain
conditions" (l.2879). Implementation (project policy, as for the A78, [a78-sog.md](a78-sog.md) D-1):
integer divide latency 5 and divider 5 cycles for both forms (`[x1_div_best]`); FP divide / square
root the shortest latency and `ceil(2 / best throughput)` cycles of one unit ([docs.md](docs.md)
4.I); L1 hits; aligned accesses; register MOVs always zero latency. Effect: optimistic for
divide-heavy code.

<a id="d-2"></a>
**D-2. Throughput as resource occupancy.**
SOG: throughput in instructions per cycle for the whole core (l.498-500). Implementation: a row with
throughput T on `n` pipelines holds them `n / T` cycles in total (rounded up where needed), as for
the A78 ([a78-sog.md](a78-sog.md) D-2); e.g. FP convert to GPR 2 V02-cycles (T = 1 on two V02
pipes), UMOV 4 V-cycles (T = 1 on four V pipes), Q-form F16 converts 4 V02-cycles (T = 1/2), LD4 Q 6
L-cycles (T = 1/2), ST4 one lane B/H/S 2 L01-cycles (2 / (4/3) = 1.5, rounded up, so 1 per cycle
instead of 4/3), FDIV D 4 divider cycles (2 / (4/7) = 3.5, so 1/2 instead of 4/7 per cycle). Facile
divides the cycles of each resource by its units, so another instruction may use the other units
meanwhile, which the SOG neither states nor excludes.

<a id="d-3"></a>
**D-3. The dividers are resources of their own.**
(`[x1_dividers]`)
SOG: "Integer divides are performed using an iterative algorithm and block any subsequent divide
operations until complete" (sec. 3.6 note 1, l.791-792); "FP divide and square root operations are
performed using an iterative algorithm and block subsequent similar operations to the same pipeline
until complete" (sec. 3.11 note 1, l.1295-1296; sec. 3.16 note 3, l.1931-1932); "Each issue pipeline
can accept one µOP per cycle" (l.388-389); FP divide and square root issue to V0 and V2 (Table 2-1,
l.469-474, "V02" in the rows). Implementation: a divide takes M0 (integer) or V02 (FP) for one cycle
and holds `X1UnitIDiv` (1 unit) / `X1UnitFDiv` (2 units, one per pipeline) for its occupancy, so
that it does not block every other M0 or V02 µOP (MADD, CRC, SCVTF, FMOV and DUP from a GPR; FRINT,
FCSEL, converts, ASIMD multiplies, ...). Reason: the SOG names what a divide
blocks; the lower estimate is preferred; the decision of the Cortex-A78 ([a78-sog.md](a78-sog.md)
D-3). Measured: `udiv` + two `madd` + branch 5.0 (with the latency of D-1); `fdiv; fcvtzs;
frinta` + branch 2.0. Test: `DivideAndMultiply`, `FpScalar`.

<a id="d-4"></a>
**D-4. FP conversions with operand and result in FP registers: 3 / 2 / V02.**
(`[x1_fp_cvt_scalar]`)
SOG: no row names the scalar forms of FCVT[AMNPZ][SU], SCVTF, UCVTF whose operand and result are FP
registers (`scvtf s0, s1`, `fcvtzs h0, h1`, also with `#fbits`). The rows of these mnemonics give 3
/ 1 (V02 from a vector register, M0 from a GPR, l.1311-1316) and the ASIMD rows 3 / 2 / V02 for
D-form F32 and Q-form F64, 4 / 1 for D-form F16 and Q-form F32, 6 / 1/2 for Q-form F16
(l.1796-1813). Implementation: every scalar form takes the 3 / 2 / V02 row, the lowest the SOG gives
for these mnemonics. The Cortex-A78 follows the same rule (lowest SOG row for the mnemonic) and gets
3 / 1 / V0 ([a78-sog.md](a78-sog.md) D-4, `N2Write_3c_1V0`); the values differ because the two SOGs
print different rows (X1 here: `V1Write_3c_1V02`, 3 / 2 / V02). The 4 / 1 and 6 / 1 rows that the
stock regular expressions would give the S and H forms by element count are not used. Measured: `scvtf s0, s0` + branch 3.0. Test: `FpScalar`.

<a id="d-5"></a>
**D-5. Scalar ASIMD forms take the row of their mnemonic.**
(`[x1_asimd_scalar]`)
SOG: the ASIMD rows list mnemonics (SHL, SSHR, USHR, SLI, SRI, SRSHR, URSHR, SSRA, USRA, SRSRA,
URSRA, SQSHRN, SQRSHRN, UQSHRN, UQRSHRN, SQSHRUN, SQRSHRUN, SQDMULL, FMULX), without saying whether
the scalar forms (`shl d0, d1, #3`, `sqshrn b0, h1, #3`, `sqdmull s0, h1, h2`, `fmulx d0, d1, d2`)
belong to them. Implementation: they take the row of their mnemonic (not the stock default, 2 on
V), same as A78 D-5. Measured: `ssra d0,
d1, #3` + branch 1.0 (4(1)). Test: `Asimd`.

<a id="d-6"></a>
**D-6. Load/store throughput that the µOPs do not give: expressed on the L / L01 pipelines.**
SOG: LD4 multiple Q "9 1/2 L, V" (l.2169-2172) and ST4 one lane B/H/S "6 4/3 V, L01" (l.2323-2326)
give a throughput that their µOPs do not give; which pipeline limits is not stated. Sec. 4.18
(l.2902-2911) says that the bandwidth of some LD4/ST4 forms "is limited by decode constraints",
without a number ([U-2](#u-2)). Implementation: `X1WriteLDV<lat, uops, lcyc, vcyc>` /
`X1WriteSTV<...>` keep the µOP count of the write they replace and hold L for `3 / T` (L01 for `2 /
T`, rounded up) cycles in total; the V side keeps its cycles (LD4) or takes the L01 count on V01
(ST4, [I-6](#i-6)). Reason: one rule for all rows, the rule chosen for the Cortex-A76 and A78
([a78-sog.md](a78-sog.md) D-6). Consequence: a mix of such an instruction with other L users is
limited together, with V users it is not.

<a id="d-7"></a>
**D-7. Store-to-load forwarding latency.**
SOG sec. 4.6 (l.2575-2585) gives the conditions of forwarding but no latency; the store tables give
store latencies (1 for integer stores, 2 or more for FP and ASIMD stores). Implementation (shared
Facile code, `addMemoryDependencies`, [docs.md](docs.md) section 4.G): by default Facile adds no store -> load edge; with `--facile-memory-deps` a store -> load edge has the store's latency, then the
load's own latency applies; i.e. forwarding is taken to be no slower than an L1 hit (same as A78
D-7).

<a id="d-8"></a>
**D-8. Not modelled: AArch32, dispatch stall, branch alignment, FPCR, Arm/Thumb, cache maintenance;
no SVE.**
(SVE part: `[x1_sve]`)
The AArch32 tables, IT blocks (sec. 4.13) and mixed Arm/Thumb code (sec. 4.16) are not compared (the
tool analyses AArch64 code). The dispatch stall of sec. 4.2 (p.51, l.2489-2493: a V µOP with more
than one Q source partly written as S registers stalls three cycles, "only on the first such
instance") is not modelled: it does not repeat in a loop. Sec. 4.3, 4.4, 4.9 (no more than four
branches in an aligned 32-byte region, l.2692-2693), 4.10 and 4.17 give advice without a cost; not
modelled (same as A78 D-8). The Cortex-X1 implements Armv8.2-A with RAS, LDAPR and the dot product
(sec. 2, l.373-376), no SVE; LLVM's `cortex-x1` has no SVE feature, and `mca_tool --mcpu cortex-x1`
does not decode SVE encodings (checked: for a loop of `add z0.d, z1.d, z2.d; b 1b` the SVE
instruction is not decoded and no loop is formed; the only region reported is the branch and the
`ret` after it, not a loop). The model has no SVE rows and SVE is in
`UnsupportedFeatures`.

<a id="d-9"></a>
**D-9. Flag-setting throughput 3: a flags resource of three units.**
SOG: "ALU, basic, flagset ... 1 3 I", "Arithmetic, flagset, LSL shift, shift <= 4 ... 1 3 I"
(l.565-571), against "ALU, basic ... 1 4 I" on the same four I pipes; the SOG does not say what
limits them to three. Implementation (upstream V1): `V1UnitFlg`, a resource of three units taken by
every flag-setting integer operation (also the 2-cycle M forms), which gives exactly the printed
throughputs (same as A78 D-9). Unlike the A78, conditional compare is "1 4 I" (l.574) and does not
take it. Measured (test): six independent `adds` + branch 2.0. Test: `DispatchAndPipelines`.

<a id="d-10"></a>
**D-10. Load pair: the second register has the row's latency.**
SOG: the load latencies "represent the maximum latency to load all the registers written by the
instruction" (l.961-962, l.1374-1376); the µOP count is not printed. Implementation: the second
destination of LDP W, LDPSW and the FP pairs (`V1Write_4c_0Z`, `V1Write_5c_0Z`, `V1Write_6c_0Z`,
`[x1_ldp_hi]`) takes no µOP and no pipeline; the row's pipelines are those of
the first write. LDP X keeps `WriteLD` + `WriteLDHi` (one L µOP each, latency 4 each: the two L
slots of "1.5"). LDP Q has `V1Write_6c_2L` + `V1Write_6c_0Z` ([C-16](#c-16)).
Same as A78 D-10.

<a id="d-11"></a>
**D-11. Fused pairs: the absorbed instruction takes nothing; AES pairs are not masked.**
SOG sec. 4.14 (l.2831-2845) lists the pairs but not what a fused pair costs. Implementation
(`computeA78FusedMask`, [docs.md](docs.md) 4.C): the second instruction of items 1-5 and the NOP of
item 6 take no Mop, µOP or pipeline; the dependence graph keeps them. AESE + AESMC / AESD + AESIMC
are also listed as fused (l.2841-2843), but sec. 4.7 (l.2590-2591) gives two AES instructions per
cycle; masking the second would allow four per cycle, so the pair is left to [U-6](#u-6) (same as
A78 D-11). Test: `Fusion`.

<a id="d-12"></a>
**D-12. Decode width and scheduler entries: not modelled.**
The SOG gives neither; the project owner supplied a decode width of 5 and 150 scheduler entries
([docs.md](docs.md) 1.B). Implementation: neither is modelled (MCA and Facile have no decode
stage; the hot-loop front end is bounded by the 8-Mop dispatch of [C-1](#c-1); a split of 150
entries over fifteen pipelines would be invented). The A78's decode width (4) is likewise not
modelled.

<a id="d-13"></a>
**D-13. Pointer-authentication instructions of the HINT space are NOPs on the X1.**
(`[x1_pac_hint]`)
SOG: the X1 implements Armv8.2-A with RAS, LDAPR (Armv8.3-A) and the dot product (Armv8.4-A) (sec.
2, l.373-376); pointer authentication is not among them, and the SOG has no PAC row. LLVM gives
`cortex-x1` no PAuth feature (`cortex-x1c` has it). PACIASP, AUTIASP, PACIAZ, ..., PACIA1716, ...,
XPACLRI are HINT encodings (LLVM decodes them by name for every CPU, "software compatible with both
CPUs that do or don't implement PA"), so on the X1 they execute as NOPs. Implementation: a
`SchedWriteVariant` on `FeatureSchedPredicate<FeaturePAuth>`: without the feature `X1Write_Hint`
(latency 1, no pipeline, as `WriteHint`, [C-3](#c-3)); with it the stock rows (`V1Write_5c_1M0`,
XPACLRI `V1Write_2c_1M0`; [U-12](#u-12)). The stock rows would cost 5 cycles on M0 in every function
prologue and epilogue. Not fused as a NOP ([C-5](#c-5) matches `NOP` and `HINT #0` only). The X1 TRM was not
consulted (not verified beyond the SOG and LLVM's processor table). The A55, A76 and A78 take the same decision ([a55-sog.md](a55-sog.md)
U-13, [a76-sog.md](a76-sog.md) C-3, [a78-sog.md](a78-sog.md) D-12). Measured: `paciasp; autiasp` +
branch 2.0 (they keep LLVM's implicit read and write of LR with latency 1, a chain of one
cycle each). Test: `PointerAuthenticationHints`.

---

## 6. Unclear in the SOG (U)

<a id="u-1"></a>
**U-1. ROB size and the MLP window.**
The SOG gives no reorder-buffer size, no scheduler size and no mispredict penalty. Current handling:
`MicroOpBufferSize = 224`, the value the project owner supplied ([docs.md](docs.md) 1.B), which is the
MLP window (224 µOPs) and not used otherwise by the X1 path; `IssueWidth = 16`, `MispredictPenalty =
11`, `LoopMicroOpBufferSize = 16` are not read by the X1 path. Same handling as A78 U-1 (a value
without an SOG source, used only for the MLP window). Impact: the MLP columns, which also move with
the µOP counts ([C-12](#c-12), [C-16](#c-16), [D-13](#d-13)). Whether the ROB counts MOPs or µOPs is
not stated. How to settle: a documented source (whether the X1 Technical Reference Manual, 101433,
gives it was not checked). Test: `DispatchAndPipelines`.

<a id="u-2"></a>
**U-2. Mops and µOPs per instruction.**
SOG l.384-388: instructions are decoded into MOPs, "A MOP can be split into two Micro-OPerations";
sec. 4.1: 8 MOPs and 16 µOPs per cycle; sec. 4.18 (p.59, l.2902-2911): the bandwidth of LD4R and LD4
one-lane post-index with 64-bit elements, LD4 multiple Q and D, ST4 multiple Q "is limited by decode
constraints", without a number. The SOG does not give µOP counts. Current handling: one Mop per
instruction; the model's µOP counts (e.g. LD4 Q 8, ST4 Q 12, FP divide 1). With at most two µOPs per
Mop the µOP bound can bind only above two µOPs per instruction on average. Same as A78 U-2. How to
settle: hardware measurement or an owner decision.

<a id="u-3"></a>
**U-3. Region-based fast forwarding.**
SOG sec. 4.8 (p.54-55, l.2617-2684): "The effective latency of FP and ASIMD instructions as described
in section 3 is increased by one cycle if the producer and consumer instructions are not part of the
same forwarding region", Table 4-1 with four regions (1 ASIMD integer ALU/shift/insert/move/abs/cmp
and the miscellaneous tables; 2 FP multiply/MAC/compare/add; 3 crypto and SHA; 4 AES, polynomial
multiply and region 1), exclusions (div/sqrt, converts and rounding without a GPR result, integer
mul/mac, reductions belong to no region) and special notes. Unclear: the exact membership. Current
handling: not modelled (every FP/ASIMD edge has the table latency), same as A78 U-3. Impact:
optimistic by one cycle per cross-region edge on FP/ASIMD dependence chains. How to settle: an owner
decision on a classification, or hardware measurement.

<a id="u-4"></a>
**U-4. ADD/SUB (extended register) with an identity extend.**
SOG "ALU, extend and shift ADD{S}, SUB{S} 2 2 M" (l.568). Unclear: whether the extended-register form
with a zero shift and a full-width extend (UXTX for X), which assemblers write as `add x0, sp, x1`,
counts as "extend". Current handling: 2 on M (all extended forms, also flag-setting, [C-9](#c-9));
same as A78 U-4. Impact: at 1 / 4 / I these instructions would be cheaper. How to settle: hardware
measurement or an owner decision.

<a id="u-5"></a>
**U-5. Multiply-accumulate forwarding between forms and from multiplies.**
(`[x1_mul_acc_fwd]`)
SOG sec. 3.6 note 2 (l.793-795): "Multiply-accumulate pipelines support late-forwarding of
accumulate operands from similar µOPs ... Accumulator forwarding is not supported for consumers of
64 bit multiply high operations"; sec. 3.15 note 1 (l.1758-1760), the same sentence for the ASIMD
MLA/MLS and [SU]MLAL/[SU]MLSL rows "4(1)". Unclear: whether a W/X-form MADD forwards to a long
accumulator and vice versa (ASIMD: MLA and MLAL), and whether a multiply without an accumulator (MUL,
SMULL; ASIMD MUL, SMULL, UMULL) counts as similar. Current handling: any to any, the lower estimate;
the reading of the Cortex-A55, A520, A76 and A720 models for the integer forms and of the A720 model
([a720-sog.md](a720-sog.md) U-5) for the ASIMD forms, whose SOGs have the same sentence.
Integer: `V1Rd_IMA` (advance 1) from `V1Wr_IMA` (MADD/MSUB), `X1Wr_IMAL` (SMADDL family) and `V1Wr_IM`
(MUL, MNEG, SMULL, UMULL, SMNEGL, UMNEGL) to the accumulator of MADD/MSUB and the SMADDL family;
SMULH/UMULH are not forwarded (the note). ASIMD: `V1Rd_VMA` / `V1Rd_VMAL` (advance 3) from MLA/MLS,
the long forms, `X1Wr_VMUL` (MUL, 4) and `X1Wr_VMULL` (SMULL, UMULL, 3; the edge is floored at 1);
SQDMULH, SQRDMULH and SQDMULL are not forwarded, as their accumulating rows (SQRDMLAH, SQDMLAL, "4",
l.1631-1632, l.1637-1638) have no forwarding. Measured: `mul x1, x1, x2; madd x1, x3, x4,
x1` + branch 3.0; `mul v0.4s, v0.4s, v1.4s; mla v0.4s, v2.4s, v3.4s` + branch 5.0. How to settle: hardware measurement. Test:
`DivideAndMultiply`, `Asimd`.

<a id="u-6"></a>
**U-6. AESE/AESMC and AESD/AESIMC pairs.**
(`[x1_aes]`)
SOG sec. 4.14 (l.2841-2845) lists them as fused, sec. 4.7 (l.2604-2605): dependent pairs "exhibit
higher performance when they are adjacent in the program code and both instructions use the same
destination register"; no number. Current handling: no fusion and no forwarding; each AES
instruction 2 / 2 / V01 — same as A78 U-6. The stock V1 read advance of 2 (which Facile would not
apply, [C-23](#c-23)) is not in the model. How to
settle: hardware measurement.

<a id="u-7"></a>
**U-7. Exclusives, acquire/release, LSE atomics.**
The SOG has no rows for LDAR/LDAPR, STLR, LDXR/LDAXR/LDXP, STXR/STLXR/STXP/STLXP or the LSE atomics
(the X1 implements LSE and LDAPR, sec. 2, l.373-376). Current handling: stock classes (LDAR, LDAPR,
LDXR 4 on L; STLR as a store with `ReadST`; STXR/STLXR `WriteSTX` = store + load, latency 5);
`WriteAtomic` is `Unsupported`, so InstrBuilder drops every LSE atomic with a `[WARN]`
([docs.md](docs.md) section 2.F). Same as A78 U-7. How to settle: a project decision or hardware measurement.

<a id="u-8"></a>
**U-8. FRECPE, FRSQRTE, Q-form F64.**
SOG sec. 3.17 (l.1966-1973): rows for D-form F32 and scalar forms, D-form F16 and Q-form F32, Q-form
F16; none for Q-form F64 (`frecpe v0.2d`). Current handling: the model's row (4, V02 one cycle), as
the A78 kept its model's row (A78 U-8).

<a id="u-9"></a>
**U-9. Special-register access.**
SOG sec. 4.11 (p.55-57, l.2704-2781): reads and writes of most special registers are non-speculative
and/or in order, FPCR writes may act as a barrier, FPSR writes stall at dispatch while another is
pending; no cycle counts. Current handling: MRS, MSR, barriers and other system instructions have
no pipeline and latency 1 (`WriteSys`, `WriteBarrier`); no serialisation. NZCV and SP "are fully
renamed" (l.2743, l.2775), which matches. Same as A78 U-9.

<a id="u-10"></a>
**U-10. Register forwarding hazards (S written, Q read).**
SOG sec. 4.12 (p.57, l.2785-2812): when an FP/ASIMD µOP reads a Q register recently written as S
registers, "Cortex-X1 performs slightly worse"; no number. Not modelled, same as A78 U-10.

<a id="u-11"></a>
**U-11. Fusion: the "(register)" forms of CMP, CMN and TST.**
SOG sec. 4.14 (l.2835-2838) names "CMP/CMN (register)" and "TST (register)". The A64 instruction set
has CMP/CMN (shifted register), (extended register) and TST (shifted register). Unclear: whether
shifted and extended forms fuse. Current handling: every SUBS / ADDS / ANDS with a zero destination
fuses (also shifted and extended), i.e. the lower estimate; same as A78 U-11. Impact: every compare + B.cond pair of these forms is affected. How to settle: hardware measurement.

<a id="u-12"></a>
**U-12. Model rows without an X1 SOG row; cortex-x1c.**
The V1 model has rows for FMLAL/FMLSL, BF16, FCMLA, FCADD, I8MM (SMMLA), SHA512/SHA3/SM3/SM4,
FJCVTZS, the flag-manipulation instructions (SETF8, CFINV, ...) and pointer authentication; the X1
SOG has no row for them, and `cortex-x1` lacks the features (the assembler rejects `fmlal`, `setf8`,
`cfinv` for it; the HINT-space PAC forms: [D-13](#d-13)). They cannot occur in X1 code and were not
compared (same as A78 U-12). `cortex-x1c` shares the model and the X1 traits; it has PAuth, FlagM and
LSE2, and gets the stock PAC rows. No Cortex-X1C guide was consulted (not verified).

---

## 7. Inconsistent within the SOG (I)

<a id="i-1"></a>
**I-1. AArch64 and AArch32 rows that disagree.**
The AArch32 tables print other pipelines and half the throughput for many FP/ASIMD rows: "VFP
divide ... V0 4/7" (l.1262) against "FP divide ... V02 8/7" (l.1217), "VFP arith VADD 2 2 V01"
(l.1255) against "FP arithmetic FADD 2 4 V" (l.1214), "ASIMD absolute diff accum VABA 4(1) 1 V1"
(l.1694) against "SABA 4(1) 2 V13" (l.1564), "ASIMD reciprocal step VRECPS 5 2 V01" (l.2034) against
"FRECPS 4 4 V" (l.1975), "Crypto polynomial VMULL.P64 2 1 V0" (l.2410) against "PMULL 2 2 V01"
(l.2388), "Crypto AES ops ... V" (l.2408) against "V01" (l.2386). Implementation: the AArch64 rows
(same as A78 I-1).

<a id="i-2"></a>
**I-2. FP divide: "until complete" against the throughput column.**
Notes l.1295-1296 and l.1931-1932: divides "block subsequent similar operations to the same pipeline
until complete", which would make the throughput 1/latency per pipeline; the tables print e.g.
scalar H 7 / 8/7 on V02. Implementation: the throughput column ([D-1](#d-1), [D-2](#d-2)), same as
A78 I-2. For the integer divide the note and the table agree (5 to 12 / 1/12 to 1/5).

<a id="i-3"></a>
**I-3. STRH "extend, scale by 1".**
Sec. 3.10 (l.1141-1142): "Store register, register offset, extend, scale by 1 STRH 2 2 I, L01, D";
the non-extended row says "scaled by 2" (l.1135-1136), and the load table "extend, scale by 2" for
LDRH (l.996-997). Implementation: scale by 2 (the S bit), same as A78 I-3.

<a id="i-4"></a>
**I-4. FP register-offset loads: "basic" with I, "extend" without.**
Sec. 3.13 (l.1391-1392, l.1409-1410): "Load vector reg, register offset, basic LDR 6 3 L, I", but
"..., extend LDR 6 3 L" and "..., scale, S/D-form 6 3 L"; the integer "Load register, register
offset, basic" is "L" (l.984-986). Implementation: the I of the "basic" row is taken as a misprint
and not modelled: LDR B, H, S, D and Q with an unscaled X index are `V1Write_6c_1L` (the scaled H/Q
rows, "7 I, L", keep their I) — same as A78 I-4. How to settle: hardware measurement; if the I is
real, the unscaled X-index forms take an I µOP and latency 6.

<a id="i-5"></a>
**I-5. SMADDL family: throughput 2 on the single M0 pipe.**
Sec. 3.6, Table 3-7 (p.18, l.708-709, checked in the PDF): "Multiply accumulate long SMADDL, SMSUBL,
UMADDL, UMSUBL 2(1) 2 M0". M0 is one pipeline that "can accept one µOP per cycle" (l.388-389), so
the throughput cannot be 2; the MADD rows above it are "2(1) 1 M0" (l.706-707). The sibling
Cortex-A78 SOG prints the row as "2(1) 2 M" (its Table 3-7), which the A78 model follows
([a78-sog.md](a78-sog.md) C-10, `[a78_maddl]`). Implementation: the A78 reading, M (both pipes), 2
per cycle (`[x1_maddl]`, [C-10](#c-10)), which is also the lower estimate. How to settle: hardware measurement.

<a id="i-6"></a>
**I-6. ASIMD stores: "V" in Table 3-32, V0/V1 in Table 2-1 and Table 3-22.**
Table 3-32 (l.2266-2328) gives every ASIMD store "L01, V" or "V, L01"; Table 2-1 (l.469-475) lists
"store data µOPs" for FP/ASIMD-0 and -1 only, and the FP store table (Table 3-22, l.1466-1528) gives
"L01, V01". The A78 has only two V pipes, so it has no corresponding decision. Implementation: V01
for the data side of every ASIMD store (the stock V1 model's choice), the reading of two
places of the SOG against one. It is not the lower estimate: with V the data µOPs
could use four pipes. Effect: none on the printed throughputs, which the L01 side already limits
([C-22](#c-22)).

<a id="i-7"></a>
**I-7. STP Q immediate offset: 2 per cycle.**
Sec. 3.14 (l.1515-1516): "Store vector pair, immed offset, Q-form STP, STNP 2 2 L01, V01", while
the post- and pre-index Q forms are "2 1" (l.1521-1522, l.1527-1528) and "ASIMD store, 1 element,
multiple, 2 reg, Q-form ST1 2 1 L01, V" (l.2272-2273) stores the same 32 bytes at 1 per cycle.
Implementation: as printed (one L01 and one V01 µOP, [C-17](#c-17)), same as A78 I-6. How to settle: hardware measurement.

<a id="i-8"></a>
**I-8. Misprints and references.**
Tables 3-26 and 3-27 are titled "ASIMD integer instructions" for the ASIMD floating-point tables
(l.1777, l.1871); "VCVTAU" in AArch64 rows (l.1802, l.1808; FCVTAU meant); "AAarch32" (l.544);
"L0,1 D" (l.1133; L01, D meant); "Q2F" (l.2812); sec. 4.14 and Table 3-35 note 1 refer to "Section
4.6" for AES (l.2435, l.2842-2843; 4.7 meant); LDPSW pre/post-index is "I, L" like the offset form
(l.1002-1003, l.1023-1024), while every other pre/post-index integer load adds the I of the base
update (the model gives the pre/post-index LDPSW `WriteAdr` as well, i.e. two I µOPs).
Implementation: the evident meaning, same as A78 I-7.

---

## 8. Outside the X1 path: shared code (X)

<a id="x-1"></a>
**X-1. MCA gives every call latency 0.**
LLVM MCA's `InstrBuilder` sets the latency of BL, BLR (and other calls) to 0, whatever the scheduling
model says; Facile then floors the LR edge to 1, which meets the SOG's BL / BLR latency 1
([C-8](#c-8)). Shared LLVM behaviour, outside the X1 model (as A78 X-2).

<a id="x-2"></a>
**X-2. No store -> load memory edge for ASIMD structure stores.**
By default Facile adds no store -> load edge at all ([docs.md](docs.md) 4.G). With
`--facile-memory-deps`, `addMemoryDependencies` still adds no edge from ST1-ST4 to a later load of the same
address (measured with that option for the X1: `ldr q1, [x0]; st1 {v1.16b}, [x0]` + branch 0.67 per iteration, against 7.0
with `str q1, [x0]`). Shared code, outside the X1 model (as A720 X-3). The `ReadST` of their data operands
([B-1](#b-1)) therefore shows only in the model data.

<a id="x-3"></a>
**X-3. `mca-insts-info` and `tools/sog_compare.py` show one arm of each variant.**
`mca-insts-info` resolves a `SchedWriteVariant` with synthesized operands, so EXTR shows only the
one-register arm (printed as `ror`) and the scaled/unscaled register-offset variants show one arm;
`tools/sog_compare.py`, which reads that output, cannot see the others (EXTR: [C-11](#c-11)). The unit
tests and `X1SogTest.Dump` build each instruction from assembly and show the arm it takes.

---

## 9. Taken over from other cores (B)

<a id="b-1"></a>
**B-1. Store data is read one cycle late.**
(`[x1_from_a55_store_data]`) **Not SOG-based.**
X1 SOG: stores are split into address and data µOPs (l.1106-1107, l.1460, l.2260); no statement on
when the data operand is needed. The project decided for the Cortex-A55 (SOG-based there) and the
Cortex-A520 (borrowed) that store data is read one cycle late, and applied it to the Cortex-A76,
A78 and A720 for consistency ([a78-sog.md](a78-sog.md) B-1); it is applied to the X1 likewise.
Implementation: `ReadAdvance<ReadST, 1>`, and `ReadST` on the data operands of every store
form (integer STR/STUR/STTR/STLR/STLLR, register offset, pre/post, STP/STNP, STLXR/STLXP; FP
STR/STUR/STP of every addressing mode; the `Vt` of ST1-ST4, with the lane index of the one-lane
forms reading `ReadDefault` in between); STXR/STXP have none. Effect: only Facile's store -> load
memory edges (`--facile-memory-deps`; none by default), and only for producers of latency 2 or more. Measured with `--facile-memory-deps`: `ldr x1, [x0]; str x1, [x0]` +
branch 4.0; with Q registers 7.0. To be revisited when an X1 statement or measurement
appears. Tests: `BorrowedStoreDataNeededLater`, `MemoryEdgesOffByDefault`.

---

## 10. Reproducing the cycle counts quoted here

The cycles/iteration figures that are not from `tests/x1_sog_test.cpp` come from `mca_tool` (CMake,
Release build). Each loop is assembled, linked and analysed with the X1 Facile invocation below,
including `--fractional-cycles` (the unrounded Facile estimate) and `--facile-reason`:

```sh
# body: the instructions of one iteration, e.g. "sdiv w1, w1, w2"
{ echo '.text'; echo '.globl f'; echo '.type f,%function'; echo 'f:'; echo '1:';
  printf '  %s\n' "${body[@]}"; echo '  b 1b'; echo '  ret'; } > t.s
aarch64-linux-gnu-gcc -march=armv8.2-a+crypto+lse+crc+dotprod+fp16+rcpc -nostdlib -static -Wl,-e,f t.s -o t
mca_tool --mcpu=cortex-x1 --mtriple=aarch64 --dependency ooo --facile --disable-always-hit-loads-heuristic \
         --fractional-cycles --facile-reason t
# cycles/iteration = the cycles column of the first CSV row (the loop); the last column says
# which bound decided it (inst, exec, prec)
```

The loop branch `b 1b` is part of every body. For a whole binary the invocation (without the last
two flags) is `mca_tool --merge-same-header --dependency ooo --mcpu cortex-x1 --facile
--disable-always-hit-loads-heuristic <binary>`. Per-instruction
values: `mca-insts-info --mtriple=aarch64 --mcpu=cortex-x1` ([X-3](#x-3)), or the `X1SogTest.Dump`
test, which prints opcode, latencies, µOPs, resource cycles and ReadAdvance entries for the
instructions in `$X1_DUMP` (separated by `;`). Re-measure before quoting a number.

---

## 11. Maintaining this document

* Update an item when the SOG reading or the implementation changes. IDs run from 1 without gaps
  within each category, in the order of the items; an item whose difference is resolved in the model
  becomes a C item, and the code, tests and other documents that name an ID are updated with it.
* When a deviation is introduced on purpose, add a D item, and tag the code with a comment that names
  the ID.
* When a behaviour is taken over from another core's model by a project decision, without an X1 SOG
  statement, add a B item and tag the code `[x1_from_<core>_<topic>]`.
* Where the X1 SOG is unclear or inconsistent, follow the decision of the Cortex-A78 document and name
  the A78 item.
* `NeoverseV1Model` in `ModifiedTarget/` is installed only for `cortex-x1` / `cortex-x1c`. If another
  CPU is ever given this local model (e.g. `neoverse-v1`), its own SOG must be compared first; the X1
  rows above are not Neoverse V1 statements (the V1 has SVE, for one).
* Evaluation results (accuracy metrics, totals over benchmark sets) are not recorded in this
  document; cycles/iteration only when reproduced with `mca_tool` as in section 10.
* [docs.md](docs.md) holds the rest of the technical documentation; [a55-sog.md](a55-sog.md),
  [a520-sog.md](a520-sog.md), [a76-sog.md](a76-sog.md), [a78-sog.md](a78-sog.md) and
  [a720-sog.md](a720-sog.md) are the counterparts of this document for the other cores.
  [README.md](README.md) lists the documents.
