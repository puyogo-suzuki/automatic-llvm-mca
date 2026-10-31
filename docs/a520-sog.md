# Cortex-A520: the model against the Software Optimization Guide

> **Who wrote this, and how far to trust it.** This document was written on **2026-10-31** by
> **Claude** (Anthropic's Claude Opus model, running as a sub-agent of Claude Code), after reading
> the Cortex-A520 Software Optimization Guide and the contents of this repository's working tree on
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
8. [Outside the A520 path: shared code (X)](#8-outside-the-a520-path-shared-code-x)
9. [Borrowed from the Cortex-A55 (B)](#9-borrowed-from-the-cortex-a55-b)
10. [Reproducing the cycle counts quoted here](#10-reproducing-the-cycle-counts-quoted-here)
11. [Maintaining this document](#11-maintaining-this-document)

| Category | Items |
|---|---|
| C, compatible | [C-1](#c-1) to [C-28](#c-28) |
| N, not compatible | [N-1](#n-1) |
| D, deliberate deviation or interpretation | [D-1](#d-1) to [D-14](#d-14) |
| U, unclear in the SOG | [U-1](#u-1) to [U-9](#u-9) |
| I, inconsistent within the SOG | [I-1](#i-1) to [I-7](#i-7) |
| X, outside the A520 path | [X-1](#x-1) |
| B, borrowed from the Cortex-A55 (not SOG-based) | [B-1](#b-1) to [B-4](#b-4) |

---

## 1. About this document

### 1.1 Source

* **Guide**: "Arm® Cortex®-A520 Core Software Optimization Guide", Revision r0p1, document number
  **PJDOC-1505342170-671342**, **Issue 1.2** (29 May 2023, "Second release for r0p1"), Copyright ©
  2023 Arm Limited (or its affiliates), **Non-Confidential** (title page and "Release information",
  p.1-2). Called "the SOG" below.

### 1.2 What "the implementation" is

The state of the working tree on 2026-10-31 (last commit `7e98c91`, plus uncommitted changes):

* `ModifiedTarget/AArch64/AArch64SchedA520.td`: the TableGen model `CortexA520Model`, compiled into
  the tool's own scheduling tables (see [docs.md](docs.md) section 5). The file had been rewritten
  by another session on 2026-10-30 (uncommitted); this audit edited it in place.
* `src/a520/a520_issue_stage.{h,cpp}`: `A520InOrderIssueStage`, a copy of LLVM 22.1.8's
  `llvm::mca::InOrderIssueStage` with three changes ([section 2.2](#22-issue-stage-srca520a520_issue_stagecpp)),
  added by this audit and by the borrowing of [chapter 9](#9-borrowed-from-the-cortex-a55-b).
* `src/mca/mca_simulation.cpp` (`analyzeMcaRegion`): selects that stage when the CPU's `CpuModel` is
  `A520` (`src/cpu/cpu_traits.cpp`: `cortex-a520`, `cortex-a520ae`).
* `src/cpu/sched_model.cpp` (`installLocalSchedModel`): installs `CortexA520Model`.
* `tests/a520_sog_test.cpp`: unit tests that pin the values below (added by this audit, not yet
  committed). `tests/test_support.h` gives `AArch64TestContext("cortex-a520")` the local model.

The Cortex-A520 uses no A55 code (`src/a55/*`) except the flag-bypass helper `a55IsFlagBypassEligible`
of `src/a55/a55_bypass.h` ([B-4](#b-4)), and no Facile (`src/facile/*`: the research
script runs it with `--dependency dependency` and without `--facile`, like the A55;
`/home/go/Workspace/research/analyze.bash`). `cpu_traits.cpp` gives it no dispatch width, fusion
table or MLP window of its own (all defaults).

Implementation facts are given by file and identifier (TableGen `def` names, C++ function names),
not by line number. The code marks the changes of this audit with tags; [section 2.3](#23-code-tags-and-the-items-of-this-document)
maps them to the items.

### 1.3 Categories

| ID | Category | Meaning |
|---|---|---|
| **C-n** | compatible | The implementation does what the SOG states. |
| **N-n** | not compatible | The SOG states something and the implementation does not follow it. Each item says why it is still open. |
| **D-n** | deliberate deviation or interpretation | The implementation knowingly departs from the literal SOG, or fills a gap with an interpretation. Each item gives the SOG text, the implementation and the reason. |
| **U-n** | unclear | The SOG does not decide the question; the implementation makes an assumption or leaves it open. Each item gives the SOG text, what is unclear, the current handling, the expected impact, and how to settle it. |
| **I-n** | inconsistent | Two places of the SOG disagree, or the SOG contradicts itself or has a misprint. Each item says which reading the implementation follows. |
| **X-n** | outside the A520 path | Problems in shared code, found during this audit. |
| **B-n** | borrowed from the Cortex-A55 | The A520 SOG says nothing (or nothing usable) on the question, and the project owner decided to give the A520 the Cortex-A55 model's treatment, whose basis is the A55 SOG ([a55-sog.md](a55-sog.md)). **Not SOG-based.** Each item gives what the A520 SOG says, what the A55 does and why, what is implemented, the measured effect and the tests; it is to be revisited when an A520 measurement or SOG statement appears. |

### 1.4 Reading the SOG instruction tables

Sec. 3.1 (p.12, l.529-544):

* **Exec Latency** is "the minimum latency seen by an operation dependent on an instruction in the
  described group"; **Load Latency** likewise for loads, assuming an L1 data cache hit. A "-" means
  no latency is given (branches without a link, stores).
* **Execution Throughput** is "maximum throughput (in instructions per cycle) of the specified
  instruction group that can be achieved in the entirety of Cortex-A520 Core". The model holds a
  resource for `units / T` cycles, rounded up ([D-3](#d-3)).
* **"A,B"**: "The Vector datapath size may affect the operation of ASIMD, FP, Neon, and SVE
  instructions. In such cases the Exec Latency and Execution Throughput will be defined with two
  value, 'A,B'. A is for a 2x128-bit configuration or a non-Q or scalar form of a 2x64-bit
  configuration. B is for a 2x64-bit configuration." The model uses A ([D-1](#d-1)).
* **Utilized Pipeline** names the pipelines of fig. 1 (p.10, l.403-454) and of the table after it
  (p.10-11, l.457-507): ALU0, ALU1, Branch, DIV, Load/Store, Load, MAC, PAC; VPU: VALU0, VALU1,
  VMAC0, VMAC1, VMC, Crypto0, Crypto1 (VALU1, VMAC1 and Crypto1 only in the 2x128-bit
  configuration). The A520 SOG has **no** dual-issue column; the issue rules are in sec. 4.1
  ([C-1](#c-1), [C-2](#c-2)).

---

## 2. Implementation overview

### 2.1 Machine model and resources (`AArch64SchedA520.td`)

`CortexA520Model`: `MicroOpBufferSize = 0` (in-order), `IssueWidth = 3`, `LoadLatency = 2`,
`MispredictPenalty = 10` (not in the SOG; not used by the simulation), `CompleteModel = 0`;
`UnsupportedFeatures` = SVE, PAuth (as a feature predicate), SME, MTE, CSSC. All resources have the
default buffer (in-order, so no reservation station).

| Resource | Units | Role |
|---|---|---|
| `A520UnitALU` | 2 | ALU0, ALU1 |
| `A520UnitB` | 1 | Branch |
| `A520UnitMAC` | 1 | MAC (integer multiply, CRC) |
| `A520UnitDIV` | 1 | DIV (iterative) |
| `A520UnitLdSt` | 1 | Load/Store pipe (the only store pipe); an LSE atomic holds it with `A520UnitLd` ([B-1](#b-1)) |
| `A520UnitLd` | 1 | Load pipe |
| `A520UnitLoad` | group of LdSt + Ld | loads ("Load/Store, Load") |
| `A520UnitPAC` | 1 | PAC (added by this audit, [C-13](#c-13)) |
| `A520UnitVALU0`, `A520UnitVALU1`, group `A520UnitVALU` | 1, 1 | VALU0, VALU1 |
| `A520UnitVMAC` | 2 | VMAC0, VMAC1 |
| `A520UnitVMC` | 2 | VMC, as two units ([D-5](#d-5)) |
| `A520UnitCrypto` | 2 | Crypto0, Crypto1 |
| `A520UnitVPU` | 2 | not a pipe: "Two VPU data processing instructions" per cycle (sec. 4.1); every VPU write takes one unit for one cycle |

### 2.2 Issue stage (`src/a520/a520_issue_stage.cpp`)

`analyzeMcaRegion` builds the pipeline with `createA520InOrderPipeline` when the CPU's `CpuModel`
is `A520`. The stage `A520InOrderIssueStage` is LLVM 22.1.8's `InOrderIssueStage`
(`llvm/lib/MCA/Stages/InOrderIssueStage.cpp`, fetched from the `llvmorg-22.1.8` tag; the installed
libLLVM is 22.1.8 and its header is identical) function by function, with three changes:

1. **No general write-back ordering** (`[a520_no_wb_order]`, [D-9](#d-9)). The stock rule
   "an instruction may not issue if it would write back before the last issued one"
   (`LastWriteBackCycle`) is applied only when the constructor argument `EnforceWriteBackOrder` is
   true; the pipeline is built with `false`.
2. **A hook before the register hazard check** (`[a520_lc_hook]`). With `--ignore-loop-carried`,
   `LoopCarriedIgnoreSteadyStateTracker` forgot the previous iteration's producers on the
   Dispatched event, which the stock stage raises after its hazard check, so the first instruction
   of every basic block still waited for the previous iteration (issue #12, [docs.md](docs.md)
   section 2.C). `analyzeMcaRegion` now wires
   `LoopCarriedIgnoreSteadyStateTracker::beforeRegisterHazardCheck()` into the A520 stage
   (`setBeforeRegisterHazardCheckHook`), as for the A55. This is a tool fix, not an SOG item.
   Measured (section 10, basic block `fadd d0, d0, d1; fadd d2, d3, d1; ret`): 4.0 cycles/iteration
   before, 1.0 after (3.0 -> 1.0 with the two FADDs swapped); with
   `--ignore-loop-carried=disable` 4.0 (unchanged). Test:
   `A520SogTest.IgnoreLoopCarriedAppliesToFirstInstruction`.
3. **The Cortex-A55's flag bypass** (`[a520_from_a55_flags]`, [B-4](#b-4)). `checkRegisterHazard`
   skips an NZCV read when `a55IsFlagBypassEligible` (`src/a55/a55_bypass.h`) allows it, as the A55
   stage does. Not SOG-based. It is active only when `createA520InOrderPipeline` is given the
   `MCInstrInfo` and the static instruction sequence, which `analyzeMcaRegion` passes.

Everything else is the stock behaviour: the issue width (3) is counted in `NumMicroOps`
([C-2](#c-2)); an instruction that does not fit in the rest of the cycle waits for the next one;
a stalled instruction stops all younger ones ([C-3](#c-3)); the resources, the LSU and
`CustomBehaviour` are checked as before.

**Faithfulness check.** A build with `EnforceWriteBackOrder = true` and the hook disconnected gave
byte-identical CSV output to the stock `InOrderIssueStage` (A520, 2026-10-31). The flag bypass came later; with no `MCInstrInfo` (the default argument) the hazard
check is the stock one again, so the check still applies to that configuration (not re-run). Other cores are not affected: the A55 keeps its own stage, the out-of-order cores the
stock default pipeline, and every other in-order CPU the stock `InOrderIssueStage`.

### 2.3 Code tags and the items of this document

| Tag in the code | Items |
|---|---|
| `[a520_sog_fix]` (`.td`) | [C-14](#c-14) variable shifts, [C-21](#c-21) ADDV and SMMLA/UMMLA/USMMLA, [C-23](#c-23) BFMMLA, [C-27](#c-27) XAR, [C-28](#c-28) CRC |
| `[a520_pac]` (`.td`) | [C-13](#c-13), [C-7](#c-7), [D-11](#d-11) |
| `[a520_no_wb_order]` (`src/a520`) | [D-9](#d-9) |
| `[a520_nop]`, `[a520_mul_zr]` (`.td`) | [D-14](#d-14), [D-15](#d-15) |
| `[a520_lc_hook]` (`src/a520`, `mca_simulation.cpp`) | [section 2.2](#22-issue-stage-srca520a520_issue_stagecpp) |
| `[a520_from_a55_atomics]` (`.td`) | [B-1](#b-1) |
| `[a520_from_a55_store_data]` (`.td`) | [B-2](#b-2) |
| `[a520_from_a55_base_update]` (`.td`) | [B-3](#b-3) |
| `[a520_from_a55_flags]` (`src/a520`, `mca_simulation.cpp`) | [B-4](#b-4) |
| "docs/a520-sog.md ..." in the `.td` header | [D-1](#d-1), [D-9](#d-9), [D-13](#d-13), [D-14](#d-14), [D-15](#d-15), [U-2](#u-2), [U-3](#u-3), [U-4](#u-4), [C-2](#c-2), [C-25](#c-25), [C-26](#c-26) |

The rest of the `.td` cites the SOG by line ("l.N" = line N of `tmp/a520sog.txt`); those comments
were written by the session that rewrote the file on 2026-10-30.

---

## 3. Compatible with the SOG (C)

Values are "latency / throughput / pipeline" as printed (A column). "Model" values were read with
`mca-insts-info --mcpu=cortex-a520` (latency, reciprocal throughput, resource cycles) and, where
the class is a `SchedWriteVariant`, with the unit tests.

### 3.1 Issue and forwarding

<a id="c-1"></a>
**C-1. Three issue-queue entries per cycle; per-class limits.**
SOG sec. 4.1 (p.72, l.3585-3594): "A maximum of three issue queue entries can be co-issued per cycle
(ignoring hazards) consisting of at most: Two ALU instructions / Two load instructions / One store
instruction / Two VPU data processing instructions." Implementation: `IssueWidth = 3` (counted in
`NumMicroOps` by the issue stage); two `A520UnitALU`, the two-unit load group `A520UnitLoad`, one
`A520UnitLdSt` for stores (a store also takes one of the two load pipes, since `A520UnitLdSt` is in
the group, which matches the pipeline column "Load/Store" of stores and "Load/Store, Load" of
loads), two `A520UnitVPU`. Measured (section 10): four independent ADDs + branch 2.0
cycles/iteration. Tests: `IssueWidthAndEntries`. What counts as an "ALU instruction": [U-1](#u-1).

<a id="c-2"></a>
**C-2. Instructions that take two issue-queue entries.**
SOG sec. 4.1 (p.72, l.3574-3578): "An instruction will occupy two entries when it has either: Three
or more general purpose destination registers / Three or more general purpose source registers"
(FP/SIMD/SVE/predicate registers excluded, l.3567-3568). Implementation: `NumMicroOps = 2` on
`A520Wr_IMW`, `A520Wr_IMX`, `A520Wr_IML` (MADD, MSUB, SMADDL, SMSUBL, UMADDL, UMSUBL with a real
accumulator; with the zero register as accumulator, i.e. MUL, MNEG, SMULL, UMULL, ..., one entry,
`A520Wr_IMW1`, `A520Wr_IMX1`, `A520Wr_IML1`, [D-15](#d-15)), `A520Wr_ST_2e` (STR register offset of a GPR, STP/STNP of
GPRs, incl. pre/post-index) and `A520Wr_LD_t1_2e` (LDP/LDPSW pre/post-index: Rt, Rt2 and the base);
the writeback and second-register writes have `NumMicroOps = 0`. Measured: `madd w1, w2, w3, w4;
add; add` + branch 3.0 cycles/iteration before the audit, 2.0 after (the change comes from
[D-9](#d-9)); without the branch 1.5 (test `IssueWidthAndEntries`, which also checks that an FP
`str d1, [x2, x3]` takes one entry). The zero register does not count as a source for MUL, MNEG, SMULL, UMULL, SMNEGL, UMNEGL, which take one entry ([D-15](#d-15)).
Missing for instructions without an SOG row: [N-1](#n-1). The queue capacity ("Four general purpose
destination registers / Six general purpose source registers", l.3570-3572) is not modelled beyond
this (not stated in the SOG how it limits issue).

<a id="c-3"></a>
**C-3. In-order issue.**
SOG sec. 2 (p.9, l.363): "In-order pipeline with direct and indirect branch prediction"; sec. 4.1
(l.3580): "An instruction will stall if insufficient space is available in the issue queue."
Implementation: the stock in-order stage ([section 2.2](#22-issue-stage-srca520a520_issue_stagecpp)): an instruction stalled on a
register or resource hazard stops every younger one. Test: `NoWriteBackOrderStall` (third case,
`fadd d0, d0, d1; fadd d2, d0, d1; add; add; add`: 5.0).

<a id="c-4"></a>
**C-4. Latency as seen by dependent instructions.**
SOG sec. 3.1 (p.12, l.531-535): Exec/Load Latency is "the minimum latency seen by an operation
dependent on" the instruction. The A520 SOG describes no other forwarding between the integer pipes
(no stage-level paths like the A55 SOG's sec. 3.1.1) apart from the notes of [C-5](#c-5) to
[C-8](#c-8) and the Rm note ([D-6](#d-6)); it gives no operand timing for store data and no flag
forwarding. Implementation: every read sees the producer's latency (`ReadAdvance 0` on `ReadI`,
`ReadISReg`, `ReadIEReg`, `ReadIM`, `ReadIMA`, `ReadID`, `ReadAdrBase`, `ReadVLD`, ...). A load is
seen at 2 by every consumer, including the base of the next load (measured `ldr x1, [x1]` + branch:
2.0). Two exceptions are borrowed from the Cortex-A55 and are not SOG-based: the data of a store is
read one cycle late (`ReadAdvance<ReadST, 1>`, [B-2](#b-2)), and a flag-setting instruction and a
conditional consumer (B.cond, CSEL, CCMP, ...) issue in the same cycle ([B-4](#b-4)).

<a id="c-5"></a>
**C-5. Integer multiply-accumulate: accumulator forwarding.**
SOG Table 3-3 note 1 (p.15, l.677-681): "a dedicated forwarding path in the accumulate portion of
the unit that allows the result of one MAC operation to be used as the accumulate operand of a
following MAC operation with no interlock. Thanks to this, a typical sequence of multiply-accumulate
instructions can issue one every 2 cycles). Accumulator forwarding is not supported for consumers of
64 bit multiply high operations." Implementation: `A520Rd_MAccW` (`SchedReadAdvance<1,
[A520Wr_IMW]>`) and `A520Rd_MAccX` (`SchedReadAdvance<2, [A520Wr_IMX]>`) on the accumulator of
MADD/MSUB W and X: a chain through the accumulator costs 2 (3 - 1, 4 - 2); SMADDL & co. have
latency 2 and no advance; SMULH is not a producer. Measured: `madd w1, w2, w3, w1` + branch 2.0;
`madd x1, x2, x3, x1` + branch 2.0 (3.0 before the audit, [D-9](#d-9)). Test: `DivideAndMultiply`.
Forwarding between different forms (MADD X -> SMADDL accumulator): [U-7](#u-7).

<a id="c-6"></a>
**C-6. ADRP -> load pointer forwarding.**
SOG sec. 4.5 (p.73-74, l.3664-3681): for "adrp x0, <const>" followed by a load from "[x0, #lo12
<const>]" (printed "ldrp", [I-7](#i-7)) "there are dedicated forwarding paths that always allow this
sequence to be executed without incurring a dependency-based stall". Implementation: `A520Wr_ADRP`
(ADRP only; ADR has the same row but is not named in sec. 4.5) and `A520Rd_PtrFwd`
(`SchedReadAdvance<1, [A520Wr_ADRP]>`) on the base of `LDR*ui` (GPR and FP/SIMD) and of `STR*ui`
([D-7](#d-7)). Measured (section 10): `adrp x1, 0; ldr x2, [x1, #8]` + branch 1.0 cycle/iteration;
`ldr x2, [x5, x1]` 1.0 (2.0 before the audit; the register-offset form waits 1 cycle for the ADRP,
which the branch hides). Test: `AdrpPointerForwarding` (4/3 and 2.0 without a branch). The
`#lo12` relation between the two instructions is not checked.

<a id="c-7"></a>
**C-7. AUT* -> RET forwarding.**
SOG sec. 4.6 (p.74, l.3686-3689): "any variant of the AUT instruction will be dual issued with the
directly following RET instruction. The latency of the AUT instruction for the dependency of the LR
does not apply for these cases." Implementation (`[a520_pac]`, added by this audit): `A520Wr_AUT` on
the AUTI* instructions and `A520Rd_AutRet` (`SchedReadAdvance<5, [A520Wr_AUT]>`) on the register
operand of `RET`, so RET sees the AUT result at 0. Test: `PointerAuthentication`
(`paciasp; autiasp; ret`: 10 = 5 + 5 + 0). "directly following": [D-11](#d-11).

<a id="c-8"></a>
**C-8. ASIMD multiply-accumulate forwarding.**
SOG sec. 4.7 (p.74, l.3694-3706): for MUL, MLA, MLS, [SU]MULL(2), [SU]MLAL(2), [SU]MLSL(2), UDOT,
SDOT, "the latency between the above instructions will be 1 cycle" when two consecutive instructions
use the same destination/accumulator register, element size and register size. Implementation:
`A520Wr_VMUL` (latency 4, VMAC) for all of them and `A520Rd_VMACFwd` (`SchedReadAdvance<3,
[A520Wr_VMUL]>`) on the accumulator of MLA/MLS (incl. by element), [SU]MLAL/[SU]MLSL, SDOT/UDOT (incl.
by element). Measured: `mla v0.4s, v1.4s, v2.4s` + branch 1.0 (3.0 before the audit, [D-9](#d-9)).
Test: `SimdMacForwarding`. Conditions not checked: [D-8](#d-8). FMLA has no such path (4 per step).

<a id="c-9"></a>
**C-9. Integer divides block later divides.**
SOG sec. 3.4 (p.15, l.653-654) and note 2 (l.683-689): divides "block any subsequent divide operations
until complete". Implementation: the one-unit `A520UnitDIV` held for the whole latency (UDIV 3, SDIV 4
cycles, [D-13](#d-13)). Unlike the A55 SOG, the A520 SOG does not call divides "serializing" for other
instructions; the model has no such rule. Measured: `udiv w1, w2, w3; add; add; add` + branch 3.0, `sdiv` 4.0, `udiv x1, x2, x3; udiv x4, x5, x6` + branch 6.0, `sdiv` pair 8.0.

### 3.2 Instruction tables, integer

<a id="c-10"></a>
**C-10. Sec. 3.2, branches (Table 3-1, p.12, l.547-559).**

| SOG group | SOG | Model |
|---|---|---|
| B; BR, RET; CBZ, CBNZ, TBZ, TBNZ | - / 1 / Branch | `WriteBr` / `WriteBrReg` latency 1 on `A520UnitB` (no register result; RET: [C-7](#c-7)) |
| BL; BLR | 1 / 1 / Branch | latency 1 on `A520UnitB` |

B.cond is not listed; it takes `WriteBr` like B. The second table on p.13 has AArch32 mnemonics
([I-6](#i-6)).

<a id="c-11"></a>
**C-11. Sec. 3.3, arithmetic and logical (Table 3-2, p.13-14, l.589-639).**

| SOG group | SOG | Model |
|---|---|---|
| Arithmetic, basic: ADD, ADC, SUB, SBC | 1 / 2 / ALU | `WriteI` 1, one ALU |
| Arithmetic, basic, flagset: ADDS, SUBS; ADCS, SBCS | 1 / 2; 1 / 1 / ALU | `WriteI`; ADCS/SBCS `A520Wr_1c_ALU_t1` (one ALU 2 cycles = throughput 1) |
| Arithmetic, extend and shift: ADD{S}, SUB{S} | 1 (note 1: 2 on Rm) / 2 / ALU | `A520Wr_1c_ALU` + `A520Rd_ShiftedRm` / `A520Rd_ExtendedRm` ([D-6](#d-6)); shift amount 0: basic |
| Conditional compare: CCMN, CCMP | 1 / 1 / ALU | `A520Wr_1c_ALU_t1` |
| Conditional select: CSEL, CSINC, CSINV, CSNEG | 1 / 2 / ALU | stock class, latency 1 |
| Logical, basic; logical, shift | 1 / 2 / ALU | latency 1 (no Rm note) |

Test: `ArithmeticAndLogical`. The note-1 marker on "flagset": [I-1](#i-1).

<a id="c-12"></a>
**C-12. Sec. 3.4, divide and multiply (Table 3-3, p.15, l.656-672).**

| SOG group | SOG | Model |
|---|---|---|
| Divide, W-form: SDIV, UDIV | 12 / 1/12 / DIV (worst case) | `A520Wr_UDIV` 3, DIV 3 cycles; `A520Wr_SDIV` 4, DIV 4 cycles ([D-13](#d-13)) |
| Divide, X-form | 20 / 1/20 / DIV (worst case) | same as the W-form ([D-13](#d-13)) |
| Multiply accumulate, W-form: MADD, MSUB; MUL | 3 / 1 / MAC | `A520Wr_IMW` 3, MAC 1 cycle, 2 entries |
| Multiply accumulate, X-form: MADD, MSUB, MUL | 4 / 1/2 / MAC | `A520Wr_IMX` 4, MAC 2 cycles, 2 entries |
| Multiply accumulate long: SMADDL, SMSUBL, UMADDL, UMSUBL | 2 / 1 / MAC | `A520Wr_IML` 2, MAC 1 cycle, 2 entries |
| Multiply high: SMULH, UMULH | 6 / 1/4 / MAC | `A520Wr_IMH` 6, MAC 4 cycles |

Accumulator: [C-5](#c-5). Test: `DivideAndMultiply`.

<a id="c-13"></a>
**C-13. Sec. 3.5, pointer authentication (Table 3-4, p.16-17, l.702-773).**
*Fixed 2026-10-31* (`[a520_pac]`); before, the stock classes gave these instructions latency 1 on an
ALU (PACIASP, AUTIASP, PACIA, ...) or no data at all (BRAA, BLRAA, RETAA, PACDZA, AUTIZA, ...).

| SOG group | SOG | Model |
|---|---|---|
| Authenticate instruction address: AUTIA, AUTIB, AUTIA1716, AUTIB1716, AUTIASP, AUTIBSP, AUTIAZ, AUTIBZ, AUTIZA, AUTIZB | 5 / 1 / PAC | `A520Wr_AUT` 5 on the new one-unit `A520UnitPAC` |
| Compute PAC, data / generic / instruction: PACDA, PACDB, PACDZA, PACDZB, PACGA, PACIA, PACIB, PACIZA, PACIZB, PACIA1716, PACIB1716, PACIASP, PACIBSP, PACIAZ, PACIBZ | 5 / 1 / PAC | `A520Wr_PAC` 5 |
| Strip PAC: XPACD, XPACI, XPACLRI | 5 / 1/5 / PAC | `A520Wr_XPAC` 5, PAC 5 cycles |
| Branch, register, with PAC: BRAA, BRAAZ, BRAB, BRABZ | - / 1 / Branch, PAC | `A520Wr_BPAC` 1 on Branch + PAC |
| Branch and link, register, with PAC: BLRAA, BLRAAZ, BLRAB, BLRABZ | 1 / 1 / Branch, PAC | `A520Wr_BPAC` |
| Branch, return, with PAC: RETAA, RETAB | - / 1 / Branch | `A520Wr_1c_B` |

Not changed: AUTDA/AUTDB/AUTDZA/AUTDZB ([U-8](#u-8)) and LDRAA/LDRAB ([U-9](#u-9)). Measured
(section 10): `paciasp; autiasp` + branch 2.0 -> 10.0 cycles/iteration (a chain through LR, 5 + 5).
The 46 research binaries contain 143 PACIASP and 147 AUTIASP (in total; mnemonic count of
`objdump -d`; no BRAA/BLRAA/RETAA/LDRAA). Test: `PointerAuthentication`.

<a id="c-14"></a>
**C-14. Sec. 3.6, miscellaneous data processing (Table 3-5, p.17-18, l.777-833).**

| SOG group | SOG | Model |
|---|---|---|
| Address generation: ADR, ADRP | 1 / 2 / ALU | `A520Wr_1c_ALU` / `A520Wr_ADRP` ([C-6](#c-6)) |
| Bitfield extract: EXTR | 2 (note 1: ROR alias 1) / 2 / ALU | `A520Wr_EXTR`: `A520IsRorImm` (Rn == Rm) -> 1, else 2 |
| Bitfield move, basic: SBFM, SBFIZ, SBFX, SXTH, SXTW, UBFM, UBFIZ, UBFX, UXTH | 2 (note 2: LSL, LSR, UXTB, SXTB, ASR aliases 1) / 2 / ALU | `A520Wr_BFM`: `A520BfmAlias1c` -> 1 (LSR/ASR: imms = 31/63; UXTB: UBFMWri 0, 7; SXTB: SBFM 0, 7; LSL: enumerated), else 2 |
| Bitfield move, insert: BFM | 2 / 2 / ALU | `A520Wr_2c_ALU` |
| AXFLAG, XAFLAG; SETF8, SETF16; RMIF; CFINV | - / 1/2; 2 / 1/2; 1 / 1; 1 / 1/2 / ALU | `A520Wr_ALU_t1q` (1, ALU 4 cycles); `A520Wr_ALU2_t1q`; `A520Wr_1c_ALU_t1`; `A520Wr_ALU_t1q` |
| Count leading: CLS, CLZ | 1 / 2 / ALU | stock, latency 1 |
| Move immed: MOVN, MOVK, MOVZ | 1 / 2 / ALU | `WriteImm` / `WriteI` 1 |
| Reverse: REV, REV16, REV32; RBIT | 1 / 2; 2 / 2 / ALU | stock 1; `A520Wr_2c_ALU` |
| **Variable shift: ASRV, LSLV, LSRV, RORV** | 1 / 2 / ALU | `A520Wr_1c_ALU` (*fixed 2026-10-31*, `[a520_sog_fix]`) |

Variable shifts: they used to take the stock `WriteIS`, which the model maps to Bitfield move (2).
The research binaries contain 12,367 register-form LSL/LSR/ASR/ROR. Measured (section 10): `lsl x1,
x1, x2` twice + branch 4.0 -> 2.0 cycles/iteration. Test: `MiscDataProcessing`.

<a id="c-15"></a>
**C-15. Sec. 3.7, loads (Table 3-6, p.18-20, l.837-947).**

| SOG group | SOG | Model |
|---|---|---|
| Load register: literal, unscaled, pre/post-index, unprivileged, unsigned immediate, register offset (all scale/extend variants); PRFM, PRFUM | 2 / 2 / Load/Store, Load | `WriteLD` / `WriteLDIdx` / `A520Wr_LD` 2 on `A520UnitLoad`; writeback `WriteAdr` 1 ([B-3](#b-3): T = 2) |
| Load pair, signed offset: LDP, LDNP W/X; LDPSW | 2 / 2 | `A520Wr_LD` + `A520Wr_LDHi` (no resource): one load pipe 1 cycle |
| Load pair, pre/post-index: LDP W/X, LDPSW | 2 / 1 | `A520Wr_WB` + `A520Wr_LD_t1_2e` (one load pipe 2 cycles, 2 entries) + `A520Wr_LDHi` |

Writeback of LDP pre/post-index: `A520Wr_WB` 1 ([B-3](#b-3): T = 1). Test: `LoadsAndStores` (`ldp`
signed offset four per two cycles, post-index one per cycle). Acquire and exclusive loads have no row
([U-6](#u-6)).

<a id="c-16"></a>
**C-16. Sec. 3.8, stores (Table 3-7, p.21, l.965-1003).** Every row (STUR, STR pre/post,
STTR, STR unsigned immediate, STR register offset, STP/STNP all addressing modes) "- / 1 /
Load/Store": `WriteST`/`WriteSTP`/`WriteSTIdx` 1 on `A520UnitLdSt`; STR register offset and STP of
GPRs `A520Wr_ST_2e` (2 entries, [C-2](#c-2)); writeback 1 ([B-3](#b-3): T = 1). The data operands
read `ReadST`, one cycle late ([B-2](#b-2), borrowed from the A55; the SOG gives no operand timing).
Release and exclusive stores have no row ([U-6](#u-6)). Test: `LoadsAndStores` (one store per
cycle).

### 3.3 Instruction tables, FP and ASIMD

<a id="c-17"></a>
**C-17. Sec. 3.12, FP data processing (Table 3-11, p.24-25, l.1117-1175).**

| SOG group | SOG | Model |
|---|---|---|
| FABS, FABD; FADD, FSUB; FMIN/FMINNM/FMAX/FMAXNM; FNEG; FRINT* | 4 / 2 / VALU | `WriteF` (and `FABD16/32/64` InstRW) 4 on `A520UnitVALU` |
| FCMP{E} | 1 / 1 / VALU | `WriteFCmp` 1 on `A520UnitVALU0` ([D-4](#d-4)) |
| FCCMP{E} | 5 / 1/5 / VALU | `A520Wr_VALU0_5r5`: 5, VALU0 5 cycles ([D-4](#d-4)) |
| FDIV H / S / D | 8 / 2/5; 13 / 2/10; 22 / 2/19 / VMC | `A520Wr_VMC<8,5>`, `<13,10>`, `<22,19>` ([D-5](#d-5)) |
| FMUL, FNMUL; FMADD, FMSUB, FNMADD, FNMSUB | 4 / 2 / VMAC | `WriteFMul` 4 on `A520UnitVMAC` (no accumulator forwarding is stated) |
| FCSEL | 3 / 1 / VALU | `A520Wr_VALU0_3` |
| FSQRT H / S / D | 11 / 2/5; 14 / 2/9; 25 / 2/19 / VMC | `A520Wr_VMC<11,5>`, `<14,9>`, `<25,19>` |

Note 1 (FDIV may finish early for a power-of-two divisor, l.1172-1175) is not modelled
([D-2](#d-2)). Test: `FpScalar`.

<a id="c-18"></a>
**C-18. Sec. 3.13, FP miscellaneous (Table 3-12, p.25, l.1180-1207).**

| SOG group | SOG | Model |
|---|---|---|
| FP convert, gen to vec: SCVTF, UCVTF | 4 / 2 / VALU | `A520Wr_VALU_4` (fixed-point scalar-SIMD forms likewise) |
| FP convert, vec to gen: FCVT[AMNPZ][SU]; FJCVTZS | 4 / 1 / VALU | `A520Wr_VALU0_4` |
| FP convert, vec to vec: FCVT, FCVTXN | 4 / 2 / VALU | `A520Wr_VALU_4` / `WriteFCvt` |
| FP move, immed | 3 / 2 / VALU | `WriteFImm` 3 |
| FP move, register | 3 / 1 / VALU | `A520Wr_VALU0_3` |
| FP transfer, gen to vec; vec to gen | 3 / 2; 3 / 1 / VALU | `A520Wr_VALU_3`; `A520Wr_VALU0_3` |

<a id="c-19"></a>
**C-19. Sec. 3.14, FP loads (Table 3-13, p.26-27, l.1219-1286).** Load vector register, every
addressing mode and size: "3 / 2 / Load/Store, Load" -> `A520Wr_FLD` 3 on `A520UnitLoad` (writeback
`A520Wr_WB` 1, [B-3](#b-3): T = 2). Load vector pair LDP/LDNP S/D/Q, every addressing mode: "3 / 1" -> `A520Wr_FLD_t1`
(one load pipe 2 cycles) + `A520Wr_FLDHi`. Test: `LoadsAndStores`.

<a id="c-20"></a>
**C-20. Sec. 3.15, FP stores (Table 3-14, p.27-29, l.1290-1406).** STUR/STR of every size and
addressing mode, STP/STNP S and D: "- / 1 / Load/Store" -> 1 on `A520UnitLdSt` (one entry: the data
is not a GPR). STP/STNP Q, every addressing mode: "2 / 1/2" -> `A520Wr_STPQ` (2, LdSt 2 cycles);
its writeback has latency 2 (1/T, [B-3](#b-3)). The data operands read `ReadST` ([B-2](#b-2)).
Multicycle co-issue rule for stores of more than 128 bits: [U-2](#u-2).

<a id="c-21"></a>
**C-21. Sec. 3.16, ASIMD integer (Table 3-15, p.29-33, l.1410-1618).** "Default" means
`WriteVd`/`WriteVq`: 3 on `A520UnitVALU`.

| SOG row | SOG | Model |
|---|---|---|
| SABD, UABD; SABDL, UABDL | 3 / 2 / VALU | default |
| SABA, UABA | 6 / 2/7 | `A520Wr_VALU_6r7` (VALU 7 cycles) |
| SABAL(2), UABAL(2) | 6 / 1/2 | `A520Wr_VALU_6r4` |
| ABS, ADD, NEG, SHADD, SHSUB, SUB, UHADD, UHSUB; [SU]ADDL(2), [SU]ADDW(2), [SU]SUBL(2), [SU]SUBW(2); SRHADD, URHADD; ADDP, SADDLP, UADDLP | 3 / 2 | default |
| ADDHN(2), RSUBHN(2), SQABS, SQADD, SQNEG, SQSUB, SUBHN(2), SUQADD, UQADD, UQSUB, USQADD | 4 / 2 | `A520Wr_VALU_4` |
| RADDHN(2) | 8 / 2/5 | `A520Wr_VALU_8r5` |
| ADDV, SADDLV, UADDLV, 4H/4S | 4 / 1 | `A520Wr_VALU0_4` |
| **ADDV (8B, 16B, 8H)** | 3 / 1 | `A520Wr_VALU0_3` (*fixed 2026-10-31*, was 4) |
| SADDLV, UADDLV (other) | 4 / 1 | `A520Wr_VALU0_4` |
| CMEQ, CMGE, CMGT, CMHI, CMHS, CMLE, CMLT; CMTST | 3; 4 / 2 | default; `A520Wr_VALU_4` |
| SDOT, UDOT; SUDOT, USDOT | 4 / 2 / VMAC | `A520Wr_VMUL` ([C-8](#c-8)); `A520Wr_VMAC_4` |
| AND, BIC, EOR, MOV, MVN, NOT, ORN, ORR | 3 / 2 | default |
| **SMMLA, UMMLA, USMMLA** | 4 / 2 / VALU | `A520Wr_VALU_4` (*fixed 2026-10-31*, was the default 3) |
| [SU]MAX(P), [SU]MIN(P); [SU]MAXV, [SU]MINV (4H/4S, 8B/8H, 16B) | 3 / 2; 4 / 1 | default; `A520Wr_VALU0_4` |
| MUL, SQDMULH, SQRDMULH; MLA, MLS | 4 / 2 / VMAC | `A520Wr_VMUL` (MUL, MLA, MLS), `A520Wr_VMAC_4` |
| SQRDMLAH, SQRDMLSH (D, Q) | 4 / 1 / VMAC | `A520Wr_VMAC_4t1` (VMAC 2 cycles) |
| [SU]MLAL(2), [SU]MLSL(2); SQDMLAL(2), SQDMLSL(2) | 4 / 2 / VMAC | `A520Wr_VMUL`; `A520Wr_VMAC_4` |
| PMUL, PMULL(2) 8x8 (D, Q) | 4 / 2 / VALU | `A520Wr_VALU_4` |
| SMULL(2), UMULL(2), SQDMULL(2) | 4 / 2 / VMAC | `A520Wr_VMUL`, `A520Wr_VMAC_4` |
| SADALP, UADALP | 6 / 2/5 | `A520Wr_VALU_6r5` |
| SRSRA, URSRA; SSRA, USRA | 8 / 2/5; 3 / 2 | `A520Wr_VALU_8r5`; default |
| SHL, SHLL(2), SSHLL(2), SSHR, SXTL(2), USHLL(2), USHR, UXTL(2); SLI, SRI | 3 / 2 | default |
| SHRN(2); RSHRN(2), SQRSHRN(2), SQRSHRUN(2), SQSHL{U} (imm), SQSHRN(2), SQSHRUN(2), UQRSHRN(2), UQSHL (imm), UQSHRN(2) | 4 / 2 | `A520Wr_VALU_4` |
| SSHL, USHL, SRSHL, SRSHR, URSHL, URSHR | 3 / 2 | default |
| SQRSHL, SQSHL, UQRSHL, UQSHL (register) | 4 / 2 | `A520Wr_VALU_4` (the SOG prints "VALUE", [I-7](#i-7)) |

ADDV: the SOG has three rows, "ASIMD arith, reduce, 4H/4S ADDV, SADDLV, UADDLV 4 1", "ASIMD arith,
reduce ADDV 3 1" and "ASIMD arith, reduce SADDLV, UADDLV 4 1" (p.31, checked in the PDF); every ADDV
used to have 4. Measured: `addv b0, v0.16b` twice + branch 8.0 -> 6.0 cycles/iteration. Test:
`AsimdFixedRows`, `AsimdSpotChecks`.

<a id="c-22"></a>
**C-22. Sec. 3.17, ASIMD FP (Table 3-16, p.34-37, l.1631-1814).**

| SOG row | SOG | Model |
|---|---|---|
| FABS, FABD; FADD, FSUB, FADDP; FMAX/FMIN(NM)(P); FNEG; FRINT*; FCVTL(2), FCVTN(2), FCVTXN(2); FCVT[AMNPZ][SU], SCVTF, UCVTF (all forms) | 4 / 2 / VALU | `A520Wr_VALU_4` |
| FACGE, FACGT, FCMEQ, FCMGE, FCMGT, FCMLE, FCMLT | 3 / 2 / VALU | default |
| FCADD; FCMLA | 4 / 2 / VMAC | `A520Wr_VMAC_4` |
| FDIV D F16; D F32; Q F16; Q F32; Q F64 | 8 / 2/5; 13 / 2/10; 8 / 1/5; 13 / 1/10; 22 / 1/19 | `A520Wr_VMC<8,5>`, `<13,10>`, `<8,10>`, `<13,20>`, `<22,38>` ([I-5](#i-5)) |
| FMAXV, FMAXNMV, FMINV, FMINNMV | 4 / 1 | `A520Wr_VALU0_4` |
| FMUL, FMULX; FMLA, FMLS; FMLAL(2), FMLSL(2) (incl. by element) | 4 / 2 / VMAC | `A520Wr_VMAC_4` |
| FSQRT D F16; D F32; Q F16; Q F32; Q F64 | 8 / 2/5; 12 / 2/9; 8 / 1/5; 12 / 1/9; 22 / 1/19 | `A520Wr_VMC<8,5>`, `<12,9>`, `<8,10>`, `<12,18>`, `<22,38>` |

Scalar and vector square roots differ ([I-3](#i-3)).

<a id="c-23"></a>
**C-23. Sec. 3.18, BF16 (Table 3-17, p.37-38, l.1817-1842).** BFCVTN(2), BFCVT "4 / 2 / VALU":
`A520Wr_VALU_4`. BFDOT "10 / 2 / VMAC, VALU": `A520Wr_BFDOT` (10, one VMAC + one VALU unit).
BFMLALB/T "4 / 2 / VMAC": `A520Wr_VMAC_4`. **BFMMLA "14, 15 / 1, 1/2 / VMAC, VALU"**:
`A520Wr_BFMMLA` (14, VMAC 2 cycles, VALU 2 cycles; *fixed 2026-10-31*, was the default 3). Test:
`AsimdFixedRows`.

<a id="c-24"></a>
**C-24. Sec. 3.19, ASIMD miscellaneous (Table 3-18, p.38-40, l.1846-1946).**

| SOG row | SOG | Model |
|---|---|---|
| RBIT; BIF, BIT, BSL; CLS, CLZ, CNT; DUP element; EXT; FMOV immed; MOVI, MVNI; REV16/32/64; TRN1/2; UZP1/2, ZIP1/2 | 3 / 2 / VALU | default |
| DUP gen reg; UMOV, SMOV; INS gen reg | 3 / 1 | `A520Wr_VALU0_3` |
| XTN; SQXTN(2), SQXTUN(2), UQXTN(2); INS element | 4 / 2 | `A520Wr_VALU_4` |
| FRECPE, FRECPX, FRSQRTE, URECPE, URSQRTE; FRECPS, FRSQRTS | 4 / 2 / VMAC | `A520Wr_VMAC_4` |
| TBL 1, 2, 3, 4 regs | 4 / 2; 8 / 2/5; 12 / 1/5; 16 / 1/9 | `A520Wr_VALU_4`, `A520Wr_V_8r5`, `A520Wr_V_12r10`, `A520Wr_V_16r18` |
| TBX 1, 2, 3, 4 regs | 8 / 2/5; 12 / 1/5; 16 / 1/9; 20 / 1/13 | `A520Wr_V_8r5`, `A520Wr_V_12r10`, `A520Wr_V_16r18`, `A520Wr_V_20r26` |

<a id="c-25"></a>
**C-25. Sec. 3.20, ASIMD loads (Table 3-19, p.40-43, l.1950-2111).** Every row has its own
`InstRW` (generated from the text by the 2026-10-30 session, `A520Wr_VLS_<lat>_<pipe><cycles>`;
writeback forms add `A520Wr_WB`, latency 1, or `A520Wr_WBn<1/T>` where the row's throughput T is
below 1, [B-3](#b-3)). Checked row by row against l.1959-2111 for this document: LD1 1-4
registers D/Q (3 / 2, 3 / 2, 3 / 1, 3 / 1, 4 / 1/2 x 4), LD1 one lane and LD1R (3 / 2), LD2 multiple
(4 / 1, 1/2, 1), LD2 one lane (4 / 1/2), LD2R (3 / 1), LD3 multiple and one lane (5 / 1/3), LD3R
(4 / 1/2), LD4 multiple (5 / 1/3, 1/3, 1/4), LD4 one lane (6 / 1/4), LD4R (4 / 1/2). Rows with
"Load/Store, Load" use the load group, rows with "Load/Store" only `A520UnitLdSt`, except LD1 one
lane B/H/S ([D-12](#d-12), [I-4](#i-4)). The multicycle co-issue rule for loads of more than 256
bits: [U-2](#u-2).

<a id="c-26"></a>
**C-26. Sec. 3.21, ASIMD stores (Table 3-20, p.43-45, l.2114-2214).** Same scheme, latency 1 (the
SOG prints "-"), `A520UnitLdSt` held 1/T cycles; checked row by row against l.2121-2214: ST1 1-4
registers (1, 1, 1, 1/2, 1/3, 1/3, 1/2, 1/4), ST1 one lane (1), ST2 (1, 1/2, 1/2, one lane 1),
ST3 (1/4, 1/6, 1/3, one lane 1/2), ST4 (1/4, 1/8, 1/4, one lane 1/2). Writeback as for the loads
([B-3](#b-3)); the data `Vt` reads `ReadST` ([B-2](#b-2)). Test: `AsimdSpotChecks`.

<a id="c-27"></a>
**C-27. Sec. 3.22, cryptography (Table 3-21, p.45-46, l.2218-2274).**

| SOG row | SOG | Model |
|---|---|---|
| AESD, AESE, AESIMC, AESMC | 3 / 2 / Crypto | `A520Wr_AES` 3 on `A520UnitCrypto` (2 units) |
| PMULL(2) 64x64 | 4 / 2 / VMC | `A520Wr_VMC4` |
| SHA1H | 3 / 1 / VALU | `A520Wr_V3t1` (VALU0) |
| SHA1C, SHA1M, SHA1P; SHA1SU0, SHA1SU1 | 4 / 2; 3 / 2 / VMC | `A520Wr_VMC4`; `A520Wr_VMC3` |
| SHA256H, SHA256H2, SHA256SU0, SHA256SU1 | 4 / 2 / VMC | `A520Wr_VMC4` |
| SHA512*, RAX1, SM3*, SM4E, SM4EKEY | 9 / 1/9 / VMC | `A520Wr_VMC9` (VMC 18 cycles on two units) |
| BCAX, EOR3; **XAR** | 3; **4** / 2 / VALU | default; `A520Wr_VALU_4` (*fixed 2026-10-31*, was 3; checked in the PDF, p.46) |

AES fusion: [U-4](#u-4). None of these instructions can be decoded by `mca_tool --mcpu cortex-a520`
at present ([X-1](#x-1)); the values were read with `mca-insts-info`.

<a id="c-28"></a>
**C-28. Sec. 3.23, CRC (Table 3-22, p.46, l.2278-2284).** "CRC checksum ops CRC32, CRC32C 2 / 1 /
MAC" (checked in the PDF): `A520Wr_CRC` 2 on `A520UnitMAC` (*fixed 2026-10-31*, `[a520_sog_fix]`;
was the stock class, latency 1 on an ALU). Measured: `crc32x w1, w1, x2` twice + branch 2.0 -> 4.0
cycles/iteration. Test: `CrcOnMac`.

Tables 3-8 to 3-10 (MTE) and 3-23 to the end of chapter 3 (SVE, p.47-71) are not compared
([D-10](#d-10)).

---

## 4. Not compatible with the SOG (N)

<a id="n-1"></a>
**N-1. Instructions with three or more GPR sources that have no SOG row take one entry.**
SOG sec. 4.1 (p.72, l.3574-3578): the two-entry rule is stated for every instruction. Implementation:
`NumMicroOps = 2` only on the classes of [C-2](#c-2). Other A64 instructions with three GPR sources
or destinations keep one entry, e.g. STXP/STLXP (Rt, Rt2, Rn; status Ws). Why open: these
instructions have no row in the A520 SOG, so giving them a write of their own would also mean
choosing a latency and pipe the SOG does not give; they are rare in the research binaries (not
counted). The LSE atomics CAS and CASP, which have rows borrowed from the A55 ([B-1](#b-1)), take
two entries by this rule.

---

## 5. Deliberate deviations and interpretations (D)

<a id="d-1"></a>
**D-1. The "A" column (2x128-bit vector datapath).**
SOG sec. 2 (p.9, l.385-387) and sec. 3.1 (p.12, l.541-544): the vector datapath is 2x64 or 2x128
bit; "A" is for 2x128-bit (or non-Q / scalar forms on 2x64-bit), "B" for 2x64-bit. Implementation:
every value is the A value, and VALU1, VMAC1 and Crypto1 exist (two units each). Reason: the
configuration of the evaluated board is not known (`.td` header). Effect: Q-form throughput is
optimistic by up to 2x if the board has the 2x64-bit configuration.

<a id="d-2"></a>
**D-2. Best case: L1 hits, aligned accesses, no early termination, no shared-VPU contention.**
SOG: load latencies assume an L1 hit (sec. 3.7, l.838); "L1 cache hit 2-4 cycles (2 is best case, 4
is normal case)" (sec. 4.11, p.77, l.3845); unaligned penalties (sec. 4.4, p.73, l.3651-3661); FDIV
may finish early (Table 3-11 note 1, l.1172-1175; Table 3-16 note 1, l.1814); in a dual-core complex
the VPU is shared and "may reduce performance" (sec. 4.12, p.77, l.3851-3855). Implementation: load
latency 2 (the best case of the range, project policy), aligned accesses, the printed FDIV values,
no contention from a second core. Reason: none of this is known statically.

<a id="d-3"></a>
**D-3. Throughput as resource occupancy.**
SOG: throughput in instructions per cycle for the whole core (sec. 3.1, l.537-539).
Implementation: a resource of `n` units is held `n / T` cycles (rounded up): ADCS on two ALUs 2
cycles (T = 1), SABA on two VALUs 7 cycles (T = 2/7), TBL 4 regs 18 cycles (T = 1/9), SHA512 on two
VMC units 18 cycles (T = 1/9), STP Q on the one LdSt pipe 2 cycles (T = 1/2). This allows another
instruction to use the other unit meanwhile, which the SOG neither states nor excludes.

<a id="d-4"></a>
**D-4. Throughput-1 rows on the VALU: VALU0.**
SOG: FCMP "1 / 1", FCCMP "5 / 1/5", FCSEL "3 / 1", FMOV register and vec->gen "3 / 1", FCVT to gen
"4 / 1", the reductions "x / 1", DUP/INS from gen and UMOV/SMOV "3 / 1", SHA1H "3 / 1", all "VALU"
(two pipes in the A configuration). Implementation: these hold `A520UnitVALU0` (FCCMP: 5 cycles),
the other rows the VALU group. Reason: the SOG gives the throughput but not which VALU; pinning one
unit caps the throughput and leaves the other VALU free. Consequence: an FCCMP blocks every other
VALU0-only instruction for 5 cycles (not stated by the SOG).

<a id="d-5"></a>
**D-5. VMC as two units.**
SOG: one VMC pipeline that "Supports 2x64-bit or 1x128-bit operations" (p.11, l.504-506); scalar
and D-form FDIV/FSQRT throughputs are printed as "2/N", Q-forms as "1/N" (l.1127-1168,
l.1709-1809); PMULL 64x64, SHA1/SHA256 throughput 2. Implementation: `A520UnitVMC` has two units;
scalar/D-form divides hold one for N cycles (two in flight), Q-forms 2N cycles. Reason: the only
reading under which "2/N" is reachable with one VMC pipeline is two 64-bit operations at a time.

<a id="d-6"></a>
**D-6. "Latency=2 when the dependency is on Rm" (Table 3-2 note 1).**
SOG p.14 (l.616, l.639, checked in the PDF): "Arithmetic, extend and shift ADD{S}, SUB{S} 1[1]";
note 1 "Latency=2 when the dependency is on Rm". Implementation: the Rm read of `(ADD|SUB)S?[WX]rs`
with a non-zero shift (`RegShiftedPred`) and of `(ADD|SUB)S?[WX]rx(64)?` with an extend
(`RegExtendedPred`) is `A520Rd_RmLate` (`SchedReadAdvance<-1>`): a producer is seen one cycle later
through Rm; Rn and the result keep latency 1. Reading: the note qualifies this instruction's latency
from its Rm input (the operand that goes through the shifter). Not applied to the logical shifted
forms (no note in that row) or to "Arithmetic, basic, flagset" ([I-1](#i-1)). Test:
`ArithmeticAndLogical` (chain through Rm 2 per step, through Rn 1).

<a id="d-7"></a>
**D-7. Scope of the ADRP forwarding.**
SOG sec. 4.5 (l.3664-3681) shows only "adrp x0; ldr x0, [x0, #lo12]" and speaks of the sequence
used "to generate load-store addresses". Implementation ([C-6](#c-6)): the base of the
unsigned-immediate LDR (all GPR and FP/SIMD sizes) and STR (GPR and FP/SIMD) forms; producer ADRP
only. Not covered: LDUR/STUR, pre/post-index, register offset, LDP/STP, ADR.

<a id="d-8"></a>
**D-8. The conditions of the SIMD MAC forwarding are not checked.**
SOG sec. 4.7 (l.3699-3704): only between two *consecutive* instructions of the list that use the
same destination/accumulator register, the same element size and the same register size.
Implementation ([C-8](#c-8)): every listed producer -> accumulator of every listed consumer, at any
distance and for any sizes. Reason: a `ReadAdvance` cannot express adjacency or size equality.
Effect: optimistic (by up to 3 cycles per step) for mixed sequences.

<a id="d-9"></a>
**D-9. No general write-back ordering.**
*Introduced 2026-10-31* (`[a520_no_wb_order]`).
SOG: the A520 SOG states no completion-order constraint on issue. The only issue constraints are
those of sec. 4.1 (co-issue limits "ignoring hazards", the two-entry rule, multicycle entries) and
the hazards; the instruction tables give pipelined throughputs (e.g. FADD 2 per cycle, loads 2 per
cycle) that the pipelines of fig. 1 (p.10: integer EX1-EX3 then WR; VPU V0-V5 then RC) deliver
without an ordering stall. Implementation: `A520InOrderIssueStage` does not apply the stock
`InOrderIssueStage` rule that a younger instruction may not write back before an older one
([section 2.2](#22-issue-stage-srca520a520_issue_stagecpp)). Reason: that rule is an LLVM default
for in-order models, not a statement of the SOG, and it turned e.g. an FADD (4) followed by an
independent ADD (1) into a 3-cycle stall. Measured (section 10, + branch, before -> after): `fadd d0;
add; fadd d2; add` 6.0 -> 1.68; `ldr; add; ldr; add` 2.0 -> 1.67; `mul x3; add; mul x4; add` 7.0 ->
4.01; `madd w1, w2, w3, w4; add; add` 3.0 -> 2.0. Whether the
core limits the number of instructions in flight (a retire buffer) is not stated; none is modelled.
Test: `NoWriteBackOrderStall`.
Decision of the project owner: follow the SOG, which has no such rule, and keep the change.

<a id="d-10"></a>
**D-10. SVE, SME, MTE and the PAuth feature predicate are outside the model.**
`UnsupportedFeatures` lists SVE, PAuth, SME, MTE and CSSC; the SVE tables (sec. 3.24-3.31,
p.47-71), the MOVPRFX fusion (sec. 4.2), the tag tables (sec. 3.9-3.11) and RDFFRS (sec. 4.1) are
not compared. The research binaries were checked to contain no MOVPRFX (mnemonic count); SVE in
general was not checked (not verified). The PAuth instructions that do occur are modelled anyway
([C-13](#c-13)).

<a id="d-11"></a>
**D-11. AUT* -> RET: "directly following" is not checked.**
SOG sec. 4.6 (l.3687-3689). Implementation ([C-7](#c-7)): any RET reading an AUTI* result gets it
at 0, also when other instructions lie between them (where the result would be ready anyway after
5 cycles in most cases). Reason: a `ReadAdvance` cannot express adjacency.

<a id="d-12"></a>
**D-12. LD1 (one lane, B/H/S) on both load pipes.**
SOG l.1994-1996: "3 / 2 / Load/Store" (one pipe, throughput 2; [I-4](#i-4)). Implementation:
`A520Wr_VLS_3_Load1` on the load group (both pipes), i.e. the throughput column. The LD1 one lane D
row (l.1997-1998) names both pipes.

<a id="d-13"></a>
**D-13. Integer divide: the best case of the formula.**
SOG Table 3-3 (p.15, l.660-661): W "12, 1/12", X "20, 1/20"; note 2 (l.683-689): these are "the
worst-case values. Early termination is possible, depending upon the data values"; the cycles "can be
calculated using the formula [N + bits/4] (N=3 for UDIV, N=4 for SDIV ...)". Implementation (project
policy: a latency that depends on the data takes the best value): bits = 0, i.e. UDIV latency 3 and
SDIV latency 4, for the W and the X form alike; the divider is held for the latency (divides "block any
subsequent divide operations until complete", [C-9](#c-9)), so the occupancy is 3 and 4 cycles
(`A520Wr_UDIV`, `A520Wr_SDIV`). The SOG does not say what "bits" counts, so this is the lower
bound of the formula, not a value the SOG states for any operand; with the operand width the formula
gives UDIV 11 / 19 and SDIV 12 / 20 ([I-2](#i-2)). Effect: optimistic for divide-heavy code (a divide can
take up to 12 or 20 cycles). The research binaries contain 5,499 UDIV and 3,304 SDIV. Test:
`DivideAndMultiply`.

<a id="d-14"></a>
**D-14. NOP takes no execution resource.**
The SOG has no row for NOP (nor for the other hints, MRS/MSR, barriers). Implementation
(`[a520_nop]`): the `NOP` opcode (`nop`, 0xd503201f) has a write with latency 1 and no resource
(`A520Wr_Nop`); it still takes one issue entry. The other hints (`HINT`: YIELD, BTI, ...) keep
`WriteHint` on an ALU, as do the system instructions and barriers (their rows are not in the SOG). Reason: a NOP
does no work, and the SOG gives no pipe for it; modelling it on an ALU (as before) limited four NOPs
plus the loop branch to 2.0 cycles/iteration, with no basis in the guide (same choice as for the A55,
[a55-sog.md](a55-sog.md) C-3 and U-13). Measured (section 10): six `nop` 2.0 cycles per iteration, three entries per
cycle. The research binaries contain 348,224 NOPs (mostly alignment padding; not verified). Test:
`NopHasNoResourceAndMulTakesOneEntry`.

<a id="d-15"></a>
**D-15. The zero register is not a GPR source: MUL, MNEG, SMULL, UMULL take one issue entry.**
SOG sec. 4.1 (p.72, l.3574-3578): an instruction takes two issue entries with three or more general
purpose source registers; the SOG does not say whether the zero register WZR/XZR is a "general purpose
source register". Implementation (`[a520_mul_zr]`): MADD, MSUB, SMADDL, UMADDL, SMSUBL, UMSUBL whose
accumulator operand (operand 3) is WZR or XZR, i.e. the aliases MUL, MNEG, SMULL, UMULL, SMNEGL,
UMNEGL, take one entry (`A520Wr_IMW1`, `A520Wr_IMX1`, `A520Wr_IML1`: the latency and the MAC
occupancy of the MADD rows, [C-12](#c-12)); with a real accumulator they take two ([C-2](#c-2)). The
accumulator-forwarding advance of [C-5](#c-5) applies to the results of both. Reason: a register that
holds no value takes no read port, and underestimation is preferred over an unexplained
overestimation (project policy). MUL is common: 97,837 in the research binaries. Measured (section 10): `mul w1, w2, w3; add;
add` + branch 1.34 cycles/iteration (`madd w1, w2, w3, w4`: 2.0). The queue capacity of four GPR
destinations / six sources is still not modelled. Test: `NopHasNoResourceAndMulTakesOneEntry`.

---

## 6. Unclear in the SOG (U)

<a id="u-1"></a>
**U-1. Which instructions count as "ALU instructions" and "load instructions" for co-issue.**
SOG sec. 4.1 (l.3588-3594). Unclear: whether MAC, DIV, PAC, CRC (MAC pipe) and branch instructions
count towards "Two ALU instructions", and whether PRFM counts as a load. Current handling: only the
instructions on `A520UnitALU` are limited to two; MAC, DIV, PAC and branch have their own one-unit
resources and are limited only by the three entries. Expected impact: small. How to settle:
hardware measurement of `madd; add; add` and `b; add; add` groups.

<a id="u-2"></a>
**U-2. Multicycle issue-queue entries.**
SOG sec. 4.1 (p.72, l.3596-3608): "Multicycle entries disable co-issuing for all cycles of the
operation but the last", listing atomics with acquire/release, loads of more than 256 bits, stores
of more than 128 bits, stores with release, RDFFRS. Unclear: how many cycles "the operation" takes
(the issue occupancy, i.e. 1/T, is the natural guess, but STLR/STLLR and the atomics have no row at
all), and whether co-issue with older instructions of the same cycle is also blocked. Current
handling: not modelled; e.g. STP Q (256 bits, 1/2) holds the store pipe 2 cycles but ALU and load
instructions issue with it. The LSE atomics carry the A55's group bits instead ([B-1](#b-1): first
instruction of its cycle; CASP alone), which is not this rule. Expected impact: small (memcpy-like Q-register loops). How to settle:
hardware measurement of `stp q0, q1; add; add` loops.

<a id="u-3"></a>
**U-3. Decoupling of the VPU.**
SOG fig. 1 (p.10) draws the VPU pipeline (V0-V5, RC) after ISS, shared between the cores of a
complex. Unclear: whether VPU instructions wait in a queue of their own, so that a stalled VPU
instruction does not stop younger integer instructions. Current handling: none; a stalled FP/ASIMD
instruction stops every younger instruction ([C-3](#c-3)). How to settle: hardware measurement of
`fadd d0, d0, d1` chains interleaved with independent integer instructions.

<a id="u-4"></a>
**U-4. AES fusion and the AES issue rule.**
SOG sec. 4.2 (p.73, l.3626) and 4.13 (p.77, l.3858-3870): AESE + AESMC and AESD + AESIMC pairs are
fused; sec. 4.1 (l.3582-3583): "AES instructions will stall until there is at least one other
instruction available to be issued". Unclear: the latency and throughput of a fused pair. Current
handling: not modelled (each instruction 3, Crypto). Expected impact: none on the research binaries
(no AES instruction found, and AES cannot be decoded at present, [X-1](#x-1)).

<a id="u-5"></a>
**U-5. Branches: alignment, taken branches, prediction.**
SOG sec. 4.3 (p.73, l.3642-3647): avoid more than one conditional branch in an aligned 16-byte
region. The SOG gives no fetch or redirect cost for a taken branch. Current handling: none; a taken
loop branch issues together with the first instructions of the next iteration (measured: `fadd;
add; fadd; add` + branch 1.68 cycles/iteration, i.e. five instructions in 5/3 cycles). Expected
impact: possibly optimistic for short loops. How to settle: hardware measurement of short loops.

<a id="u-6"></a>
**U-6. Exclusives and acquire/release.**
The A520 SOG has no rows for LDAR/LDAPR, STLR, LDXR/LDAXR/LDXP/LDAXP, STXR/STLXR/STXP/STLXP or the
LSE atomics (sec. 4.1 only mentions atomics and release stores as multicycle, [U-2](#u-2)). The LSE
atomics have the A55's rows by a project decision ([B-1](#b-1)); the newer CAST/CASPT (LSUI) and
LDCLRP/LDSETP/SWPP (LSE128), which the A55 model does not cover either, are still `WriteAtomic`
`Unsupported` and dropped by InstrBuilder (`[WARN]`). Current handling of the others (as found):
stock classes, e.g. LDAR 2 on the load group, STLR 1 on the store pipe, STLXR latency 3 (as reported
by `mca-insts-info`); the data of STLR/STLLR/STLXR/STLXP reads `ReadST` ([B-2](#b-2)). The A55's rows
for these opcodes (A55 SOG sec. 4.8/4.9: LDAR/LDAXR "3 (2)", STLR 2 at 1/2, STLXR 4 at 1/2) were not
taken over: the A55's acquire load costs what its ordinary load costs, which the A520 model already
does (2, its own load latency), and STLR/STLXR were not part of the decision of [chapter 9](#9-borrowed-from-the-cortex-a55-b).
None of these occur in the research binaries except 139 LDAR (mnemonic count; no LSE atomic). How to
settle: hardware measurement, or adopt the A55's rows by a further project decision.

<a id="u-7"></a>
**U-7. MAC accumulator forwarding between different forms.**
SOG Table 3-3 note 1: "the result of one MAC operation ... as the accumulate operand of a following
MAC operation". Unclear: whether this includes a W producer into an X consumer, an X MADD into an
SMADDL accumulator, or SMADDL into MADD X. Current handling: only within the W forms and within
the X MADD/MSUB forms ([C-5](#c-5)); `madd x1, x2, x3, x4; smaddl x4, w2, w3, x1` + branch 6.0
cycles/iteration (4 + 2, no forwarding). How to settle: hardware measurement.

<a id="u-8"></a>
**U-8. Latency of AUTDA, AUTDB, AUTDZA, AUTDZB.**
SOG Table 3-4 (p.16, l.707-710): "Authenticate data address ... - 1 PAC": no latency, although the
instructions write a register. Current handling: unchanged (stock class: latency 1 on an ALU for
AUTDA/AUTDB; AUTDZA/AUTDZB have no scheduling data and are dropped by InstrBuilder, with the
`[WARN]` report of [docs.md](docs.md) section 2.F; checked 2026-10-31). How to settle: adopt the 5 of the other PAC rows (a
project decision) or hardware measurement.

<a id="u-9"></a>
**U-9. LDRAA, LDRAB.**
SOG Table 3-4 (p.17, l.763-770): offset "2 2 PAC", pre-indexed "2 1 PAC": a load whose only
pipeline is PAC (one pipe, yet throughput 2). Unclear: whether it also uses a load pipe, and how
throughput 2 is reached on one PAC pipe. Current handling: unchanged (no scheduling data: LDRAA is
dropped by InstrBuilder and reported with `[WARN]`, checked 2026-10-31). None occur in the research
binaries (mnemonic count).

---

## 7. Inconsistent within the SOG (I)

<a id="i-1"></a>
**I-1. The note-1 marker on "Arithmetic, basic, flagset".**
Table 3-2 (p.13, checked in the PDF): the group name "Arithmetic, basic, flagset [1]" carries the
marker of note 1 ("Latency=2 when the dependency is on Rm", p.14), and so does "Arithmetic, extend
and shift 1[1]". ADDS/SUBS/ADCS/SBCS (basic) have no shift or extend. Implementation: the note is
applied only to the shifted/extended forms ([D-6](#d-6)).

<a id="i-2"></a>
**I-2. Divide: table values against the formula.**
Table 3-3 (l.660-661): W 12, X 20 for SDIV and UDIV. Note 2 (l.687-689): "[N + bits/4]" with N = 3
for UDIV and 4 for SDIV, "signed division takes one more cycle than unsigned". With bits = 32 / 64
the formula gives SDIV 12 / 20 but UDIV 11 / 19. Implementation: the best case of the formula ([D-13](#d-13)).

<a id="i-3"></a>
**I-3. Scalar and vector FSQRT differ.**
FSQRT H 11, S 14, D 25 (Table 3-11, l.1153-1168) against ASIMD FSQRT D-form F16 8, D-form F32 12,
Q-form F64 22 (Table 3-16, l.1800-1809); FDIV is the same in both tables (8, 13, 22). The
implementation follows each table.

<a id="i-4"></a>
**I-4. ASIMD load rows: pipeline column.**
LD1 one lane B/H/S (l.1994-1996): throughput 2 on "Load/Store" only, one pipe. LD4 one lane D
(l.2098-2099): no pipeline printed. Implementation: LD1 one lane B/H/S on both load pipes
([D-12](#d-12)); LD4 one lane D on `A520UnitLdSt` like the other LD4 rows.

<a id="i-5"></a>
**I-5. ASIMD FDIV Q-form F64 on "VALU".**
Table 3-16 (p.35, l.1717-1718): "ASIMD FP divide, Q-form, F64 FDIV 22 1/19 VALU"; every other
divide and square root row says VMC. Implementation: VMC (`A520Wr_VMC<22,38>`).

<a id="i-6"></a>
**I-6. AArch32 mnemonics in an AArch64-only guide.**
Sec. 3.2 has a second, untitled branch table (p.13, l.574-583) with BX, BLX and "CBZ, CBNZ" only,
although the Cortex-A520 implements AArch64 at all exception levels (sec. 2, l.358). Implementation:
Table 3-1.

<a id="i-7"></a>
**I-7. Misprints.**
"ldrp x0, [x0, #lo12 <const>]" (sec. 4.5, l.3678; LDR meant). Pipeline "VALUE" (l.1615, l.1661;
VALU meant). "VCVTAU" (l.1698; FCVTAU). "PACIA171" (l.757; PACIA1716). Sec. 4.7 lists "SMLAL2"
twice and no SMLSL2 (l.3697). "Store register, register offset, extend, scale by 1 STRH" (l.995-997;
scale by 2 meant, as in the row for the non-extended form, l.986-988). Implementation: the evident
meaning.

---

## 8. Outside the A520 path: shared code (X)

<a id="x-1"></a>
**X-1. Cryptographic instructions are not decoded for `cortex-a520`.**
`initTarget` (`src/frontend/frontend.cpp`) creates the subtarget with an empty feature string.
LLVM 22's `cortex-a520` (and `cortex-a720`) has no AES/SHA2/SHA3 feature (the Cryptographic
Extension is optional on these cores, SOG sec. 2, l.376), so `llvm-mc --disassemble
-mcpu=cortex-a520` rejects AESE, XAR and SHA256H while it accepts them with `-mcpu=cortex-a55` or
`cortex-a78` (checked 2026-10-31). The rows of [C-27](#c-27) therefore never apply to binaries; the
undecodable words are skipped by the frontend (behaviour not examined further). The research
binaries contain no such instruction (mnemonic count), so the research results are not affected.
Fixing it means passing a feature string (e.g. `+aes,+sha2,+sha3`) for these CPUs in shared frontend
code; not done.

---

## 9. Borrowed from the Cortex-A55 (B)

These items have no basis in the A520 SOG. The project owner decided on 2026-10-31 to give the
Cortex-A520 the treatment of the Cortex-A55 model ([a55-sog.md](a55-sog.md)) in four places where the
A520 SOG says nothing usable. The A55's values come from the A55 SOG; for the A520 they are an
assumption. **None of them is SOG-based**; each is to be revisited as soon as an A520 measurement or
an A520 SOG statement on the question appears. The code marks them with `[a520_from_a55_*]` tags
([section 2.3](#23-code-tags-and-the-items-of-this-document)). Tests: `A520SogTest.Borrowed*`.

Decision of the project owner: all four (B-1 to B-4) are kept.

<a id="b-1"></a>
**B-1. LSE atomics: the A55's rows.**
A520 SOG: no instruction table has a row for an atomic (loads sec. 3.7, p.18-20, l.837-947; stores
sec. 3.8, p.20-21, l.952-1003). Sec. 4.1 (p.72, l.3596-3600) names "Atomic instructions with Acquire
or Release semantics" among the multicycle entries ([U-2](#u-2)), and its two-entry rule (l.3574-3578)
applies to every instruction. Before: `WriteAtomic` `Unsupported`, so InstrBuilder dropped every LSE
atomic (reported with `[WARN]`). A55: [a55-sog.md](a55-sog.md) C-19 (A55 SOG sec. 4.10 gives latency,
throughput and dual issue per group) and U-5 (no pipeline is given; the A55 model holds its load and
its store pipe for `1/T` cycles each, since an atomic is a load and a store); dual issue "01"
(`BeginGroup`), CASP "00" (`SingleIssue`); ST&lt;OP&gt; is LD&lt;OP&gt; with Rt = WZR/XZR.
Implementation (`[a520_from_a55_atomics]`, `A520Wr_Atomic<lat, occ, entries>`,
`A520Wr_AtomicSI`, `A520Wr_LdOp`, `A520Wr_LdOpL` with the predicate `A520AtomicToZR`):

| Group (A55 SOG 4.10) | Latency / occupancy of `A520UnitLd` and `A520UnitLdSt` | Entries | Group bits |
|---|---|---|---|
| LD&lt;OP&gt;{A} (ADD, CLR, EOR, SET, SMAX, SMIN, UMAX, UMIN) / ST&lt;OP&gt; | 2 / 2; 1 / 1 | 1 | `BeginGroup` |
| LD&lt;OP&gt;{A}L / ST&lt;OP&gt;L | 3 / 3; 2 / 2 | 1 | `BeginGroup` |
| SWP{A}; SWPL, SWPAL | 2 / 2; 3 / 3 | 1 | `BeginGroup` |
| CAS{A}; CASL, CASAL | 4 / 4; 3 / 3 | 2 | `BeginGroup` |
| CASP{A}; CASPL, CASPAL | 5 / 5; 4 / 4 | 2 | `SingleIssue` |

168 opcodes (B, H, W and X forms). The A520's store pipe is its Load/Store pipe, so the load part of
an atomic takes the Load pipe: no other load or store issues while an atomic holds them. In the
A520 issue stage `BeginGroup` means "first instruction of its cycle" and `SingleIssue` "alone in its
cycle". The number of issue entries is not borrowed: it follows the A520 SOG's rule ([C-2](#c-2)):
CAS and CASP read three or more GPRs (Rs, Rt, Rn), LD&lt;OP&gt; and SWP two. The latencies are the
A55's absolute values, not scaled to the A520 (whose ordinary load is 2 where the A55's is 3); there is
no basis for a scaling. Not covered, still dropped (as on the A55): CAST/CASPT (FEAT_LSUI),
LDCLRP/LDSETP/SWPP (FEAT_LSE128), RCW*. LDAR/STLR and the exclusives keep the stock classes
([U-6](#u-6)). Measured (section 10, + branch; before: dropped): `ldadd x1, x2, [x0]` 2.0
cycles/iteration, `ldaddal` 3.0, `stadd x1, [x0]` 1.0, `cas x1, x2, [x0]` 4.0, `casp x2, x3, x4, x5,
[x0]` 5.0, `ldadd x1, x2, [x0]; ldr x5, [x6]` 3.0. The research binaries contain no LSE atomic
(mnemonic count). Test: `BorrowedAtomicsFromA55`.

<a id="b-2"></a>
**B-2. Store data is read one cycle late.**
A520 SOG: the store tables print no latency ("-", sec. 3.8 Table 3-7, p.20-21, l.965-1003; sec. 3.15;
sec. 3.21) and say nothing about when the data operand of a store is needed; the latency is "the
minimum latency seen by an operation dependent on" the producer (sec. 3.1, p.12, l.531-535), and the
only forwarding paths described are those of sec. 4.5-4.7 ([C-4](#c-4)). Before: `ReadAdvance<ReadST,
0>`, and `ReadST` only on the FP register-offset stores (stock `Sched` lists; the GPR register-offset
rows of the model named no read). A55: [a55-sog.md](a55-sog.md) C-18 and U-9 (A55 SOG sec. 4.9
"do not need to wait for data operands (which are required at wr)", sec. 3.1.1 the ALU0 ex2 -> store
ex2 path, sec. 4.14 FP data "required at f2"): `ReadAdvance<ReadST, 1>` from every producer, on the
data of every store form with an A55 SOG row. Implementation (`[a520_from_a55_store_data]`): the same
advance, and `ReadST` on the data operands of STR/STUR/STTR of every size and addressing mode (GPR and
FP; the unsigned-offset forms keep the ADRP forwarding on the base, [C-6](#c-6)), STP/STNP W/X/S/D/Q of
every addressing mode, STLR/STLLR/STLXR/STLXP (with their stock writes) and the `Vt` of every ASIMD
ST1-ST4 row; reads in operand order (data, then the lane index of the one-lane forms, then the base).
As on the A55, STXR/STXP and the atomics have no `ReadST`. The writes (latency, resources, entries)
are unchanged (`mca-insts-info` identical). Measured (section 10, + branch, before -> after): `mul x1,
x2, x3; str x1, [x0]` 4.0 -> 3.0 (register offset `str x1, [x0, x4]` 5.0 -> 4.0); `ldr x1, [x0]; str
x1, [x5]` 2.0 -> 1.0 (the same for `stur`, `str ..., #8` post-index and `stlr`); `ldr x1, [x0]; stp
x1, x2, [x5]` 3.0 -> 2.0; two load -> store pairs 4.0 -> 2.0; `fadd d0, d2, d2; str d0, [x5]` 4.0 ->
3.0 (also `stp d1, d0` and `stp q1, q0`); `add v0.8b, ...; st1 {v0.8b}, [x5]` 3.0 -> 2.0 (also the
one-lane post-index form); `ld1 {v0.8b}, [x0]; st1 {v0.8b}, [x5]` 3.0 -> 2.0. Test: `BorrowedStoreDataNeededLater`.

<a id="b-3"></a>
**B-3. Base-register update latency 1/T.**
A520 SOG: "Base register updates are done in parallel to the operation" (sec. 3.7, p.18, l.840; sec.
3.8, p.20, l.953; sec. 3.14, p.26, l.1222; sec. 3.15, p.27, l.1291; sec. 3.20, p.40, l.1953; sec.
3.21, p.43, l.2115); no latency is given. Before: latency 1 for every writeback form (`WriteAdr`,
`A520Wr_WB`: the update is an address addition, taken to be available like an ALU result). A55:
[a55-sog.md](a55-sog.md) U-11: the A55 SOG prints "(1/T)" as the update latency of writeback stores
and of the ASIMD loads and stores (A55 SOG sec. 4.9 note 3, 4.14, 4.18, 4.19), read as the reciprocal
of the row's throughput; the A55 model has `WriteAdr` 1, 2 for STP Q (T = 1/2) and
`CortexA55WriteVAdr<1/T>` on the ASIMD rows. Implementation (`[a520_from_a55_base_update]`): the
update latency is `1/T` of the A520 row's own throughput, rounded up, and 1 when T >= 1
(`A520Wr_WBn<N>`, no resource, no entry). Rows with T >= 1 keep 1: LDR/STR pre/post (all sizes),
LDP/STP W/X/S/D pre/post, LDP Q pre/post, and the ASIMD rows with T = 1 or 2. Changed (1 -> N):

| Rows (writeback forms) | A520 T | N |
|---|---|---|
| STP Q pre/post | 1/2 | 2 |
| LD1 3 and 4 registers (D, Q); LD2 multiple Q B/H/S; LD2 one lane (B/H, S, D); LD3R; LD4R | 1/2 | 2 |
| LD3 multiple (all), LD3 one lane (all); LD4 multiple B/H/S (D, Q) | 1/3 | 3 |
| LD4 multiple Q D; LD4 one lane (all) | 1/4 | 4 |
| ST1 2 registers Q, ST1 4 registers D; ST2 multiple Q (B/H/S, D); ST3 one lane (B/H/S, D); ST4 one lane (B/H/S, D) | 1/2 | 2 |
| ST1 3 registers (D, Q); ST3 multiple Q D | 1/3 | 3 |
| ST1 4 registers Q; ST3 multiple D B/H/S; ST4 multiple D B/H/S, Q D | 1/4 | 4 |
| ST3 multiple Q B/H/S | 1/6 | 6 |
| ST4 multiple Q B/H/S | 1/8 | 8 |

For the loads N does not exceed the data latency, so only a consumer of the base that is not
waiting for the data sees the change. A chain of the same instruction through its base is bounded by
its occupancy (N cycles) anyway; the change shows when another instruction uses the updated base.
Measured (section 10, + branch, before -> after): `stp q0, q1, [x0], #32` + three dependent ADDs from
x0 3.0 -> 4.0; `st4 {v0.16b-v3.16b}, [x0], #64; ldr x2, [x0]; add x3, x2, #1` 8.0 -> 10.0; `ld4
{v0.16b-v3.16b}, [x0], #64; ldr x2, [x0]` + two dependent ADDs 4.0 -> 6.0; `st1 {v0.16b-v2.16b},
[x0], #48; ldr x2, [x0]; ldr x2, [x2]` 3.5 -> 5.5; `stp q0, q1, [x0], #32` twice unchanged (4.0).
This is the one borrowed item
that makes the model slower, against the A520 SOG's "in parallel" read literally; it is kept by the
owner's decision. Test: `BorrowedBaseUpdateLatency`.

<a id="b-4"></a>
**B-4. Same-cycle flag bypass to a conditional instruction.**
A520 SOG: no statement on flag forwarding. Flag-setting instructions have latency 1 (ADDS/SUBS,
ADCS/SBCS, CCMN/CCMP, Table 3-2, p.13-14, l.597-624; SETF8/SETF16 2, Table 3-5), the latency is
the one "seen by an operation dependent on" the instruction (sec. 3.1, l.531-535), and the forwarding
sections (4.5-4.7) do not mention the flags; there is no dual-issue table like the A55 SOG's Table 2.
Before: a conditional instruction (B.cond, CSEL, CCMP, ...) issued one cycle after its flag setter.
A55: [a55-sog.md](a55-sog.md) C-4 (A55 SOG Table 2 "Conditional": a conditional instruction-1 can be
dual-issued with a flag-setting instruction-0, except arithmetic-with-carry instructions and other
AArch32-only cases) and D-8 (FCMP/FCCMP are FP flag producers, A55 SOG sec. 5.11: the transfer from
the FP flags takes a cycle); open points U-2 (FCCMP/FCSEL as consumers) and U-10 (2-cycle flag
setters). The A55 stage skips such an NZCV read in `A55DecoupledIssueStage::checkRegisterHazard`
through `a55IsFlagBypassEligible` (`src/a55/a55_bypass.h`). Implementation
(`[a520_from_a55_flags]`, [section 2.2](#22-issue-stage-srca520a520_issue_stagecpp)):
`A520InOrderIssueStage::checkRegisterHazard` calls the same `a55IsFlagBypassEligible` (the A55 code
is unchanged) before the RAW check of each NZCV read, so the eligibility is exactly the A55's: every
consumer except ADC/ADCS/SBC/SBCS (and NGC/NGCS), every producer except FCMP{E}/FCCMP{E} (and the
AArch32 VMRS/VMSR); the producer's latency is ignored (A55 U-10, e.g. SETF8). Adapted to the A520
pipeline only in what follows from it: the A55 pairs the two instructions of a dual-issue pair, the
A520 issues up to three entries per cycle, so a setter and up to two consumers (or a chain `cmp;
csel; ...` limited by the two ALUs) can issue in one cycle; the bypass does not depend on the slot.
Measured (section 10, + branch, before -> after): `cmp x1, x2; csel; csel` 2.0 -> 1.5; `cmp x0, x1;
csel x0, x4, x0, eq` 2.0 -> 1.0; `cmp; ccmp; csel` 3.0 -> 2.0; `cmp x1, x2; csel x3, ...; cmp x3, x6;
csel` 3.0 -> 2.0; `ldr x1, [x0]; cmp x1, #0; b.eq` 4.0 -> 2.67; unchanged: `cmp; adc; adc` 2.0,
`fcmp s0, s1; csel w3, ...; fmov s0, w3` 5.0. Test: `BorrowedFlagBypass`.

---

## 10. Reproducing the cycle counts quoted here

The cycles/iteration figures that are not from `tests/a520_sog_test.cpp` were measured by the author
on 2026-10-31 with private builds (CMake, Release; not `build/`): "before" is a snapshot of the
working tree before this audit, "after" the working tree with the changes of this audit. For the
B items ([chapter 9](#9-borrowed-from-the-cortex-a55-b)) the builds were made one item after the
other, in the order B-1 to B-4, and "before" is the build with the preceding items only:

```sh
# body: the instructions of one iteration, e.g. "lsl x1, x1, x2" "lsl x1, x1, x2"
{ echo '.text'; echo '.globl f'; echo '.type f,%function'; echo 'f:'; echo '1:';
  printf '  %s\n' "${body[@]}"; echo '  b 1b'; echo '  ret'; } > t.s
aarch64-linux-gnu-gcc -march=armv8.5-a+crypto+lse+crc+i8mm+bf16+sha3 -nostdlib -static -Wl,-e,f t.s -o t
mca_tool --mcpu=cortex-a520 --mtriple=aarch64 --iterations=200 t
# cycles/iteration = cycles / (retired_instructions / length), first CSV row (the loop)
```

The loop branch `b 1b` is part of every body. For the basic-block figures of
[section 2.2](#22-issue-stage-srca520a520_issue_stagecpp) the body ends in `ret` instead of the loop
(`f: <body>; ret`), which `mca_tool` analyses as a basic block (`--ignore-loop-carried=default`
ignores loop-carried dependencies there). Re-measure before quoting a number.

---

## 11. Maintaining this document

* Update an item when the SOG reading or the implementation changes. IDs run from 1 without gaps
  within each category, in the order of the items; an item whose difference has been fixed becomes a
  C item, and the code, tests and other documents that name an ID are updated with it.
* When a deviation is introduced on purpose, add a D item, and tag the code with a comment that names
  the ID.
* When a behaviour is taken over from another core's model by a project decision, without an A520
  SOG statement, add a B item (chapter 9: what the A520 SOG says, what the other core does and why,
  what is implemented, the measured effect, the tests) and tag the code `[a520_from_<core>_<topic>]`.
  When an A520 measurement or SOG statement appears, the item is revisited and becomes a C, D or N
  item (or is removed with its code).
* Evaluation results (accuracy metrics, totals over benchmark sets) are not recorded in this document;
  cycles/iteration only when reproduced with `mca_tool` as in section 10.
* [docs.md](docs.md) holds the rest of the technical documentation; [a55-sog.md](a55-sog.md) is the
  Cortex-A55 counterpart of this document. [README.md](README.md) lists the documents.
