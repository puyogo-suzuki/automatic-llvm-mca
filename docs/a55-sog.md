# Cortex-A55: the model against the Software Optimization Guide

> **Who wrote this, and how far to trust it.** This document was written on **2026-10-31** by
> **Claude** (Anthropic's Claude Opus model, running as a sub-agent of Claude Code), after reading
> the Cortex-A55 Software Optimization Guide and the contents of this repository's working tree on
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
8. [Reproducing the cycle counts quoted here](#8-reproducing-the-cycle-counts-quoted-here)
9. [Maintaining this document](#9-maintaining-this-document)

| Category | Items |
|---|---|
| C, compatible | [C-1](#c-1) to [C-30](#c-30) |
| N, not compatible | [N-1](#n-1) to [N-3](#n-3) |
| D, deliberate deviation or interpretation | [D-1](#d-1) to [D-11](#d-11) |
| U, unclear in the SOG | [U-1](#u-1) to [U-16](#u-16) |
| I, inconsistent within the SOG | [I-1](#i-1) to [I-10](#i-10) |

---

## 1. About this document

### 1.1 Source

* **Guide**: "Arm® Cortex®-A55 Software Optimization Guide", document number **ARM-EPM-128372**,
  **Version 3.0** (30 November 2018), Copyright © 2017, 2018 Arm Limited (or its affiliates),
  **Non-Confidential**. Called "the SOG" below.

### 1.2 What "the implementation" is

The state of the working tree on 2026-10-31 (last commit `023c744`, plus uncommitted changes):

* `ModifiedTarget/AArch64/AArch64SchedA55.td`: the TableGen model `CortexA55Model`, compiled into
  the tool's own scheduling tables (see [docs.md](docs.md) section 5).
* `src/a55/a55_issue_stage.{h,cpp}`: `A55DecoupledIssueStage`, the issue stage used for
  `cortex-a55` instead of `llvm::mca::InOrderIssueStage`.
* `src/a55/a55_bypass.h`: the flag bypass and the ADRP pointer forwarding.
* `src/a55/a55_adaptations.{h,cpp}`: corrections applied after `InstrBuilder` and to the cycle count.
* `src/mca/mca_simulation.cpp` (`analyzeMcaRegion`): selects the A55 pipeline.
* `tests/a55_sog_test.cpp`: unit tests that pin the values below to the SOG (the file was not yet
  committed on 2026-10-31).

Implementation facts are given by file and identifier (TableGen `def` names, C++ function names),
not by line number, because line numbers change. The code marks its SOG-related changes with tags
such as `[a55_alu_fwd]`; [section 2.4](#24-code-tags-and-the-items-of-this-document) maps the tags
to the items of this document.

### 1.3 Categories

Every item has an ID that other documents and code comments can refer to.

| ID | Category | Meaning |
|---|---|---|
| **C-n** | compatible | The implementation does what the SOG states. |
| **N-n** | not compatible | The SOG states something and the implementation does not follow it (wrong or missing). Each item says why it is still open. |
| **D-n** | deliberate deviation or interpretation | The implementation knowingly departs from the literal SOG, or fills a gap with an interpretation the project chose. Each item gives the SOG text, the implementation, the reason, and an evaluation result where one exists. |
| **U-n** | unclear | The SOG does not decide the question; the implementation makes an assumption or leaves it open. Each item gives the SOG text, what is unclear, the current handling, the expected impact, and how to settle it. |
| **I-n** | inconsistent | Two places of the SOG disagree. Each item gives both places and which one the implementation follows. |

### 1.4 Reading the SOG instruction tables

Sec. 4.1 (p.14, l.636-648):

* **Exec latency** is "the minimum latency seen by an operation dependent on an instruction in the
  described group" (but see [I-9](#i-9)). A number in parentheses is a shorter latency on a
  special forwarding path, explained by the table notes (pointer chasing for loads, sec. 4.8
  note 1; the accumulator for multiply-accumulates, sec. 4.5 note 2, sec. 4.15 notes 2-3).
  "a, b" in the load tables is "data latency, base-register update latency" (sec. 4.8 note 2).
* **Execution throughput** is instructions per cycle for the whole core. The model represents a
  throughput `T < 1` by holding the resource for `1/T` cycles; on the two-unit FP/ASIMD resources a
  throughput `T` is one unit held for `2/T` cycles ([D-1](#d-1)).
* **Dual-issue**: `00` not dual-issuable, `01` dual-issuable from slot 0, `10` dual-issuable from
  slot 1, `11` from both slots. Slot 0 is "instruction-0" (the older instruction of a pair) and
  slot 1 is "instruction-1" (sec. 3.2, p.9, l.371-374). The model encodes `01` as `BeginGroup`,
  `10` as `EndGroup` and `00` as `SingleIssue` (= both) ([C-2](#c-2)).
* **`2*` / `11*`** (only in the ASIMD tables of sec. 4.15-4.17; the load/store tables of
  sec. 4.18-4.19 have no asterisks): note 1 of those sections, "If the instruction has Q-form, the
  Q-form of the instruction can only be dual issued as instruction 0 and execution throughput is 1"
  (e.g. p.32, l.1785-1786). The D-form (and scalar) keeps throughput 2 and `11`. In the model the
  D-form is one of the two `CortexA55UnitFPALU` units for one cycle (`CortexA55WriteVd`,
  `CortexA55WriteAluVd_n`) and the Q-form holds both units for one cycle with `BeginGroup`
  (`CortexA55WriteVq`, `CortexA55WriteAluVq_n`).

---

## 2. Implementation overview

### 2.1 Machine model and resources (`AArch64SchedA55.td`)

`CortexA55Model`: `MicroOpBufferSize = 0` (in-order), `IssueWidth = 2`, `LoadLatency = 3`,
`MispredictPenalty = 8`. All resources have `BufferSize = 0`.

| Resource | Units | Role |
|---|---|---|
| `CortexA55UnitALU` | 2 | integer ALU0 / ALU1 (SOG fig. 1, p.8) |
| `CortexA55UnitMAC` | 1 | integer multiply-accumulate |
| `CortexA55UnitDiv` | 1 | integer divide (iterative) |
| `CortexA55UnitLd` | 1 | load pipe |
| `CortexA55UnitSt` | 1 | store pipe |
| `CortexA55UnitB` | 1 | branch (also used by `WriteSys`, `WriteBarrier`, `WriteHint`; not by NOP, [C-3](#c-3), [U-13](#u-13)) |
| `CortexA55UnitVarShift` | 1 | not a pipe: caps the variable shifts at one per cycle ([D-7](#d-7), [U-1](#u-1)) |
| `CortexA55UnitFPALU` | 2 | FP/ASIMD ALU |
| `CortexA55UnitFPCmp` | 1 | not a pipe: caps the FP compares at one per cycle ([C-21](#c-21)) |
| `CortexA55UnitFPMAC` | 2 | FP multiply / multiply-accumulate |
| `CortexA55UnitFPDIV` | 1 | FP divide / square root |

### 2.2 Issue stage (`src/a55/a55_issue_stage.cpp`)

`analyzeMcaRegion` (`src/mca/mca_simulation.cpp`) builds the pipeline with
`createA55DecoupledPipeline` when the CPU's `CpuModel` is `A55` (`src/cpu/cpu_traits.cpp`) and the
model is in-order. `A55DecoupledIssueStage` keeps a two-instruction window and issues from it in
program order (`drainDecodeWindow`). For each candidate, `canExecute` checks, in this order:

1. the group bits (`BeginGroup` refused as instruction-1, nothing after an `EndGroup`) ([C-2](#c-2));
2. the `--ignore-loop-carried` hook (below);
3. `checkRegisterHazard`: the flag bypass and the ADRP forwarding (`a55_bypass.h`,
   [C-4](#c-4), [C-5](#c-5), [D-8](#d-8)), then the register file's RAW hazard, which is waived by
   `isSameCycleALUForward` for the ALU0 -> ALU1 path ([D-6](#d-6));
4. the resource hazard, the LSU, `CustomBehaviour`;
5. the integer-divide rule ([U-4](#u-4)).

The issue width is counted in instructions, not micro-ops ([C-1](#c-1)).

**`--ignore-loop-carried`.** The tracker that forgets the previous iteration's producers
(`LoopCarriedIgnoreSteadyStateTracker`) used to do so on the Dispatched event, which the issue stage
raises after its hazard check, so the first instruction of every iteration still waited for the
previous iteration (issue #12, see [docs.md](docs.md) section 2.C). The A55 issue stage calls
`LoopCarriedIgnoreSteadyStateTracker::beforeRegisterHazardCheck()` before each hazard check
(`setBeforeRegisterHazardCheckHook`, wired in `analyzeMcaRegion` only when the tracker is the
loop-carried-ignoring one). Test: `A55SogTest.IgnoreLoopCarriedAppliesToFirstInstruction`. The
Cortex-A520 has its own issue stage with the same hook ([a520-sog.md](a520-sog.md)).

### 2.3 Corrections after `InstrBuilder` (`src/a55/a55_adaptations.cpp`)

* `applyA55InstructionAdaptations`: for `ADD/SUB/ADDS/SUBS (W|X)rs` with shift amount 0 the latency
  is set to 1. The `.td` already resolves this through `CortexA55WriteISRegV`, so the override is
  redundant but harmless ([D-5](#d-5)). Its former second part (branches with latency 0 and no
  resources) is disabled ([C-12](#c-12)).
* `applyA55BranchOverlapCorrection`: formerly subtracted 0.5 cycle per steady-state iteration;
  now returns the cycle count unchanged ([C-12](#c-12)).

Instructions that `InstrBuilder` rejects are counted and reported for every core, not only the
A55 ([docs.md](docs.md) section 2.F).

### 2.4 Code tags and the items of this document

| Tag in the code | Items |
|---|---|
| `[a55_alu_fwd]` | [D-6](#d-6) |
| `[a55_shift_imm]` | [D-3](#d-3), [I-4](#i-4) |
| `[a55_ror_imm]` | [D-4](#d-4), [I-4](#i-4) |
| `[a55_varshift]` | [D-7](#d-7), [U-1](#u-1) |
| `[a55_flag_bypass]` | [C-4](#c-4), [D-8](#d-8), [U-2](#u-2) |
| `[a55_planB]` | [C-6](#c-6), [D-5](#d-5), [D-10](#d-10), [U-3](#u-3) |
| `[a55_td_fix]` | [C-15](#c-15), [C-16](#c-16), [C-17](#c-17), [C-22](#c-22); the `U` sub-tag: [C-12](#c-12) |
| `[a55_sog_impl #3]` atomics | [C-19](#c-19), [U-5](#u-5) |
| `[a55_sog_impl #4]` dual-issue column | [C-2](#c-2), [C-12](#c-12) |
| `[a55_sog_impl #5]` FSQRT | [C-11](#c-11), [D-11](#d-11) |
| `[a55_sog_impl #6a]` / `#6b` accumulators | [C-9](#c-9) / [C-10](#c-10) |
| `[a55_sog_impl #7]` RBIT, CRC | [C-16](#c-16), [C-20](#c-20) |
| `[a55_sog_impl #9*]` loads and stores | [C-7](#c-7), [C-17](#c-17), [C-18](#c-18), [C-23](#c-23), [C-24](#c-24), [C-28](#c-28), [C-29](#c-29) |
| `[a55_sog_impl #10]` ASIMD rows | [C-25](#c-25) to [C-30](#c-30) |
| `[a55_sog_impl #11]` integer divide | [C-15](#c-15), [D-1](#d-1) |
| `[a55_sog_n5]` FP compare | [C-21](#c-21) |
| `[a55_sog_n6]` ASIMD MOV | [C-25](#c-25), [I-8](#i-8) |
| `[a55_sog_n7]` SQXTN / SQXTUN / UQXTN | [C-27](#c-27) |
| `[a55_sog_n8]` NOP | [C-3](#c-3), [U-13](#u-13) |
| `[a55_pac_hint]` pointer-authentication hints | [C-3](#c-3), [U-13](#u-13) |
| `[a55_sog_n2]` pointer chasing from load pairs, LDPSW | [C-7](#c-7), [C-17](#c-17), [U-15](#u-15) |
| `[a55_sog_n3]` pointer chasing into store and ASIMD load/store bases | [C-7](#c-7), [C-18](#c-18), [U-16](#u-16) |
| `[a55_sog_u9]` store data read advance | [C-18](#c-18), [U-9](#u-9) |
| `#12` (tests) loop-carried | [section 2.2](#22-issue-stage-srca55a55_issue_stagecpp) |

---

## 3. Compatible with the SOG (C)

### 3.1 Issue and forwarding

<a id="c-1"></a>
**C-1. In-order dual issue, counted in instructions.**
SOG: at most an instruction-0 / instruction-1 pair per cycle (sec. 3.2, p.9, l.371-374); "only
issues instructions in-order" (sec. 3.5.1, p.12, l.551); "does not support any instruction
reordering at issue" (sec. 5.1, p.42, l.2386). Implementation: `A55DecoupledIssueStage::drainDecodeWindow`
issues the younger instruction of the window only in the same cycle as, and after, the older one,
and caps the cycle at two (`NumIssued`). `issue()` counts instructions, not `NumMicroOps` (TableGen
gives every writeback load/store and every LDP two micro-ops, which would otherwise fill the issue
width alone). Tests: `StrictlyInOrderIssue`, `DualIssueColumn` (LDR post-index + ADD at one cycle).
History: an earlier "decoupled" issue let an integer instruction-1 pass a stalled FP instruction-0;
it had no SOG basis and was removed.

<a id="c-2"></a>
**C-2. The dual-issue column.**
SOG sec. 4.1 (p.14, l.644-648). Implementation: `BeginGroup` (01), `EndGroup` (10), `SingleIssue`
(00) on the scheduling classes; `A55DecoupledIssueStage::canExecute` refuses a `BeginGroup`
instruction as instruction-1 and issues nothing after an `EndGroup` one (bits read with
`mca::Instruction::getBeginGroup()/getEndGroup()`). Test: `DualIssueColumn`.

<a id="c-3"></a>
**C-3. The structural rules of Tables 1 and 2.**
SOG sec. 3.2 Table 1 (p.9-10, l.376-414) and Table 2 (p.10-11, l.417-460).

| SOG rule | Implementation |
|---|---|
| loads not dual-issued with loads, stores not with stores (Table 2, l.423-428) | one-unit `CortexA55UnitLd`, `CortexA55UnitSt` |
| branches not dual-issued with branches (Table 2, l.436-438) | one-unit `CortexA55UnitB` |
| divides not as instruction-1 (Table 2, l.421) | `WriteID32`/`WriteID64` have `BeginGroup` (also the "01" of sec. 4.5) |
| load double/pair not as instruction-1 (Table 2, l.428) | the first (data) write of every LDP/LDNP/LDPSW/LDAXP class has `BeginGroup` or `SingleIssue` (the instruction takes the bits of any of its writes) |
| indirect branches not as instruction-0 (Table 1, l.408-409) | `WriteBrReg` has `EndGroup` (BR, BLR, RET: "10", sec. 4.2) |
| control instructions (MRS/MSR, WFI, WFE, barriers) not dual-issued (Table 1, l.410-412) | `WriteSys`, `WriteBarrier` have `SingleIssue`; **not WFE and WFI**, which are `HINT` in LLVM and use `WriteHint`: [N-3](#n-3) |
| atomic compare-and-swap pair not as instruction-0 (Table 1, l.386) | CASP: `CortexA55WriteAtomicSI` (00) |
| "Architectural NOP" can be instruction-1 (Table 2, l.458); not restricted as instruction-0 (Table 1) | `NOP`: `CortexA55WriteNop`, no resource, no group bits (`[a55_sog_n8]`) |

AArch32-only rules (load/store multiple, LDRD, VMRS/VMSR, IT) cannot occur in A64 code.
NOP: resolved 2026-10-31. The NOP opcode (`nop`) used to be `WriteHint` on
`CortexA55UnitB`, so it could pair neither with a branch nor with another NOP; it now uses no
resource, so a NOP pairs with anything the issue rules allow. The HINT-space pointer-authentication
instructions (PACIASP, AUTIASP, PACIAZ, ..., XPACLRI) are NOPs as well (the project owner's decision
of 2026-11-02, `[a55_pac_hint]`, [U-13](#u-13); they keep LLVM's implicit read and write of LR with
latency 1; measured: `paciasp; autiasp` + branch 3.0 -> 2.0, `paciasp; add x1, x1, #1` + branch 2.0
-> 1.5). The SOG names no pipe for a NOP, and which other
hint instructions count as "Architectural NOP" is open: [U-13](#u-13). Measured
(section 8): four `nop` + branch 2.5 cycles/iteration (was 5.0), `add; nop` + branch 1.5 (was
2.0). Test: `NopDualIssues`. The compare-and-branch rule of Table 2 contradicts sec. 4.2:
[U-14](#u-14).

<a id="c-4"></a>
**C-4. Flag bypass to a conditional instruction-1.**
SOG Table 2 "Conditional" (p.11, l.452-457): a conditional (flag-dependent) instruction can be
dual-issued with a flag-setting instruction-0, except instructions that execute an RRX operation,
arithmetic-with-carry instructions, instruction-0 being MULS/MLAS, and instruction-0 being a NEON
instruction. Implementation: `a55IsFlagBypassEligible` (`a55_bypass.h`), applied in
`A55DecoupledIssueStage::checkRegisterHazard` before the RAW check: an NZCV read does not stall
unless the consumer is ADC/ADCS/SBC/SBCS (W and X; NGC/NGCS are aliases), `a55IsArithmeticWithCarry`,
or the producer is an FP compare ([D-8](#d-8)). RRX and MULS/MLAS exist only in AArch32, and no A64
NEON instruction writes NZCV, so those exceptions cannot arise. The interlock after a flag-setting
MUL (sec. 3.4, p.12, l.530-532) likewise has no A64 counterpart. Test:
`FlagBypassExceptCarryAndFpCompare`. Open questions: [U-2](#u-2), [U-10](#u-10).

<a id="c-5"></a>
**C-5. ADRP -> LDR pointer forwarding.**
SOG sec. 5.10 (p.44, l.2528-2541): "adrp x0, <const>" followed by a load from "[x0, #lo12 <const>]"
executes "without incurring a dependency based stall" (the SOG prints the load as "ldrp", see
[I-10](#i-10)). Implementation: `a55IsAdrpForwardConsumer` and `a55IsPointerForwardEligible`
(`a55_bypass.h`): a register whose current producer is `ADRP` causes no stall when it is the **base
register (operand 1) of an unsigned-immediate-offset load**, i.e. an opcode `LDR*ui` (LDR, LDRB, LDRH,
LDRSB, LDRSH, LDRSW, and the FP/SIMD LDR of B/H/S/D/Q), which is the form the SOG shows. Not covered
(the ADRP result is waited for, latency 1): LDUR, pre/post-index and register-offset loads, the index
register of a register-offset load, and every other consumer. Decision of the project owner
2026-10-31 (the check used to be only "opcode name starts with LDR or LDUR, any operand"; that also
waived the index register, LDUR and the pre/post-index forms, which the SOG does not show). The SOG's
`#lo12 <const>` (the offset is the low 12 bits of the same symbol as the ADRP) cannot be known from the
instructions and is **not checked**; compiler output pairs them by construction. Whether the SOG's
rule extends to the forms not covered is not stated (not verified on hardware). Test:
`AdrpPointerForwardingBaseOfUnsignedImmediateLoads` (with the loop branch: `adrp; ldr x2, [x1, #8]` 1.5
cycles per iteration, `adrp; ldur` / `ldr [x1, #8]!` / `ldr [x3, x1]` 2.0).

<a id="c-6"></a>
**C-6. Load-use latency 2 to the ALU.**
SOG sec. 3.3 (p.11, l.484-485): "The load-use latency from the data of a load instruction to the
ALU of a dependent datapath instruction is two cycles." Implementation: `ReadAdvance<ReadI, 1,
[loads]>` and `CortexA55ReadNotShifted` (advance 1 from the load writes `WriteLD`, `WriteLDIdx`,
`WriteLDHi`, `CortexA55WriteLDP1`, `CortexA55WriteLDPW3SI`, `CortexA55WriteLDPX4`,
`CortexA55WriteLDPX4b`, `CortexA55WriteLDPX4SI`, `CortexA55WriteLdAcq`, `CortexA55WriteLdAXP`,
`CortexA55WriteLdAXPb`, `CortexA55WriteLDPSW3SI`, `CortexA55WriteLDPSW3b`: every register of every
load, including both registers of every pair); a load of latency 3 is seen at 2. Since 2026-10-31 the
pointer-chasing path has a producer list of its own ([C-7](#c-7)). Operands that go through the shifter do not get it ([D-10](#d-10)).
Test: `StrictlyInOrderIssue`, `SameCycleAluForwarding`. See also [I-2](#i-2).

<a id="c-7"></a>
**C-7. Pointer chasing into load and store bases.**
SOG sec. 3.3 (p.11, l.486-495) and sec. 4.8 note 1 (p.21, l.1088-1089): "The first stage (ex1) of
both the load and store pipeline contains an AGU. To lower the latency on pointer chasing operations
to 2-cycle, load data from a limited set of load instructions can be forwarded from the beginning of
the wr pipeline stage to either the load or store AGU base operand", limited by Table 3; the table
latency in parentheses ("3 (2)", "4 (3)"). Implementation: `ReadAdvance<ReadAdrBase, 1, [...]>`.

*Producers* (`[a55_sog_n2]`, resolved 2026-10-31): a list of their own, no
longer the ALU load-use list of [C-6](#c-6). Table 3 (l.492-495): "LDR & LDRT/LDTR (all
variants)" -> `WriteLD`, `WriteLDIdx` (every single-register load, see [I-7](#i-7)); "LDRD/LDP (all
variants): Only the first register of the pair if 32-bit load or the second register of the pair if
64-bit load" -> LDP/LDNP W-form: the first register only (`CortexA55WriteLDPW3SI`, not
`CortexA55WriteLDP1`), X-form: the second register only (`CortexA55WriteLDPX4b`, not
`CortexA55WriteLDPX4`/`CortexA55WriteLDPX4SI`); LDPSW "3" with no parentheses and no note 1 (sec.
4.8, p.21, l.1078): none (LDPSW has its own writes `CortexA55WriteLDPSW3SI`/`CortexA55WriteLDPSW3b`,
which are only in the ALU list). Load acquire (`CortexA55WriteLdAcq`) and LDAXP (both registers,
`CortexA55WriteLdAXP`, `CortexA55WriteLdAXPb`) keep the path as before: "3 (2)" with note 1 in sec.
4.8, but Table 3 does not list them ([U-15](#u-15)). `WriteLDHi` (second register of the stock
LDXP class, no SOG row, [U-12](#u-12)) unchanged. Little-endian and aligned addresses are assumed
([D-2](#d-2)). Measured (section 8, one instruction + loop branch, before -> after): `ldp x1, x2,
[x1]` 3.0 -> 4.0; `ldp x1, x2, [x2]` 3.0 (unchanged); `ldp w1, w2, [x2]` 2.0 -> 3.0; `ldpsw x1, x2,
[x1]` 2.0 -> 3.0. Test: `LoadPairPointerChasingPerRegister`.

*Consumers* (`[a55_sog_n3]`, resolved 2026-10-31): `ReadAdrBase` is on the
base operand of every LDR/LDUR/LDTR/LDP/LDNP/LDAR/LDAXR/LDAXP form (`InstRW` rows, and the stock
`Sched` lists of the register-offset loads) and now also of every store form with an SOG row: STR/STUR
/STTR (immediate, unscaled, unprivileged, pre/post-index, all register sizes incl. FP B/H/S/D/Q),
STP/STNP (all sizes and addressing modes, incl. Q), STLR/STLLR, STLXR, STLXP (the register-offset
stores had it already through their stock `Sched` list), and of the ASIMD LD1-LD4 / ST1-ST4 rows
(multiple, one lane, replicate, post-index). The SOG names the ASIMD structure loads/stores neither in
sec. 3.3 nor in sec. 4.18/4.19; the statement is about the AGU base operand of the load/store
pipelines, which they use like every other load/store, so it is applied to them
([U-16](#u-16)). Not covered (no SOG row): atomics, PRFM, LDXR/LDXP/STXR/STXP ([U-16](#u-16)).
Measured (section 8, before -> after): `ldr x1, [x0]; str x2, [x1]` 4.0 -> 3.0 (= the register-offset
`str x2, [x1, x4]`, 3.0); `ldr x1, [x1]; ld1 {v0.8b}, [x1]` 4.0 -> 3.0. Tests:
`StoreAndAsimdBasePointerChasing`, `ScalarAndPairLoadsStores` (9g).

<a id="c-8"></a>
**C-8. Integer multiply-accumulate: accumulator forwarding.**
SOG sec. 3.1.1 (p.9, l.354-356), sec. 3.4 (p.12, l.526-528), sec. 4.5 note 2 (p.17, l.827-828):
the MAC result can be forwarded to the accumulate operand of a following MAC operation. Values:
MADD/MSUB 32-bit "3 (1)", 64-bit "4-5 (2-3)", SMADDL/SMSUBL/UMADDL/UMSUBL "3 (1)" (p.16-17,
l.800-817). Implementation: `ReadAdvance<ReadIMA, 2, [WriteIM32, WriteIM64]>`: 3 - 2 = 1 for
`WriteIM32` and 4 - 2 = 2 for `WriteIM64` (the best of "2-3", [D-1](#d-1)). Multiplicands
(`ReadIM`) and divide operands (`ReadID`) have advance 0 ([U-3](#u-3)).

<a id="c-9"></a>
**C-9. ASIMD integer multiply-accumulate and dot product: accumulator forwarding.**
SOG sec. 4.15 notes 2-3 (p.32, l.1787-1790): "4 (1)" for MLA/MLS (incl. by element), [SU]MLAL(2)
/ [SU]MLSL(2), SDOT/UDOT (incl. by scalar), forwarding "from similar instructions" / "between Dot
Product instructions". Implementation: `CortexA55ReadMlaAcc` (advance 3 from
`CortexA55WriteMlaVd_4`, `CortexA55WriteMlaVq_4`, `CortexA55WriteMlaIxVq_4`,
`CortexA55WriteMlaLVq_4`) and `CortexA55ReadDotAcc` (advance 3 from the three `CortexA55WriteDot*`
writes), on the accumulator operand only. Test: `AccumulatorForwarding`.

<a id="c-10"></a>
**C-10. FP multiply-accumulate: no accumulator forwarding.**
SOG Table 4 (sec. 3.5.1, p.13, l.583-590): note b (accumulate forwarding) appears only on the
non-fused AArch32 VMLA/VNMLA/VMLS/VNMLS/VRECPS/VRSQRTS row ("8 (4b)"); the fused row
"VFMA/FMADD, ..." gives a plain 4. Sec. 4.11 FMADD "4, 2, 11" (p.26, l.1367) and sec. 4.16 FMLA/FMLS
"4, 2*, 11*" (p.34, l.1889) have no number in parentheses. Implementation: `CortexA55WriteFMAC`
/ `CortexA55WriteFMACq` with no read advance; a chain through the accumulator costs 4 per step.
History: the former `CortexA55ReadFMAC` (advance 2, citing "SOG Table 4 Note b") was removed
(`[a55_sog_impl #6b]`). Test: `AccumulatorForwarding`.

<a id="c-11"></a>
**C-11. FP divide and square root.**
SOG Table 4 (p.13, l.567-590): hazard / latency for half, single, double precision: FDIV 5 / 8,
10 / 13, 19 / 22; FSQRT 5 / 8, 9 / 12, 19 / 22. Sec. 4.11 (p.26, l.1357-1380): FDIV throughput
1/5, 1/10, 1/19 and FSQRT 1/5, 1/9, 1/19, dual-issue 01. Implementation: `CortexA55WriteFDivHP/SP/DP`
and `CortexA55WriteFSqrtHP/SP/DP` on `CortexA55UnitFPDIV`, latency = SOG latency,
`ReleaseAtCycles` = hazard, `BeginGroup`. Scalar and vector forms are bound per element size
(ASIMD FSQRT: [D-11](#d-11)). Denormal inputs (note c, l.609-610) are not modelled ([D-2](#d-2)).
Test: `SquareRootAndReciprocalSqrt`.

<a id="c-12"></a>
**C-12. Branches.**
SOG sec. 4.2 (p.14, l.666-676): B, BL, CBZ/CBNZ/TBZ/TBNZ "1, 1, 11"; BR, RET, BLR "1, 1, 10".
Implementation: `WriteBr` (latency 1, `CortexA55UnitB`), `WriteBrReg` (latency 1, `EndGroup`).
History: two corrections without an SOG basis were disabled (`[a55_td_fix U]`): the zero-latency,
resource-free override of B/B.cond in `applyA55InstructionAdaptations`, and the -0.5
cycle/iteration "branch overlap" correction (`applyA55BranchOverlapCorrection` now returns its
input). See [U-14](#u-14) for compare-and-branch as instruction-1.

### 3.2 Instruction tables, integer

Values are "Exec latency / Execution throughput / Dual-issue" as printed. "ALU" is
`CortexA55UnitALU`.

<a id="c-13"></a>
**C-13. Sec. 4.3, arithmetic and logical (A64 table, p.15, l.711-721).**

| SOG group | SOG | Implementation |
|---|---|---|
| ALU, basic, incl. flag setting (ADD{S}, ADC{S}, AND{S}, BIC{S}, EON, EOR, ORN, ORR, SUB{S}, SBC{S}) | 1 / 2 / 11 | `WriteI` 1 on ALU; shifted-register forms with shift 0: `CortexA55WriteISReg1` ([D-5](#d-5)) |
| ALU, extend and/or shift | 2 / 2 / 11 | `CortexA55WriteISReg2` (shift != 0, `RegShiftedPred`), `WriteIEReg` 2 |
| ALU, conditional compare (CCMN, CCMP) | 1 / 2 / 11 | latency 1 on ALU (stock class) |
| ALU, conditional select (CSEL, CSINC, CSINV, CSNEG) | 1 / 2 / 11 | latency 1 on ALU (stock class) |

<a id="c-14"></a>
**C-14. Sec. 4.4, move and shift (A64 table, p.16, l.752-758).**

| SOG group | SOG | Implementation |
|---|---|---|
| Address generation (ADR, ADRP) | 1 / 2 / 11 | latency 1 on ALU |
| Move immed (MOVN, MOVK, MOVZ) | 1 / 2 / 11 | latency 1 on ALU (`WriteImm` for MOVZ/MOVN) |
| Variable shift (ASRV, LSLV, LSRV, RORV) | 1 / 1 / 11 | `CortexA55WriteVarShift`: latency 1, one ALU + `CortexA55UnitVarShift` for one cycle ([D-7](#d-7), [U-1](#u-1)); test `VariableShiftThroughputOne` |

<a id="c-15"></a>
**C-15. Sec. 4.5, divide and multiply (A64 table, p.16-17, l.793-819).**

| SOG group | SOG | Implementation |
|---|---|---|
| Divide, W-form (SDIV, UDIV) | 3 - 12 (11) / 1/12 (11) - 1/3 / 01 | `WriteID32`: latency 3, `CortexA55UnitDiv` 3 cycles, `BeginGroup` ([D-1](#d-1)) |
| Divide, X-form | 3 - 20 (19) / 1/20 (19) - 1/3 / 01 | `WriteID64`: same as W ([D-1](#d-1)) |
| Multiply accumulate 32-bit (MADD, MSUB) | 3 (1) / 1 / 11 | `WriteIM32` 3, MAC 1 cycle; accumulator [C-8](#c-8) |
| Multiply accumulate 64-bit | 4-5 (2-3) / 1/3 - 1/2 / 11 | `WriteIM64` 4, MAC 2 cycles ([D-1](#d-1)) |
| Multiply accumulate long (SMADDL, SMSUBL, UMADDL, UMSUBL) | 3 (1) / 1 / 11 | `WriteIM32` (stock class of these opcodes) |
| Multiply high (SMULH, UMULH) | 6 / 1/3 / 11 | `CortexA55WriteIMH`: 6, MAC 3 cycles |

MUL/MNEG and SMULL/UMULL are aliases of MADD/MSUB and SMADDL/UMADDL and share their rows. The
"serializing" rule of sec. 3.4: [U-4](#u-4). Tests: `IntegerDivideBestCase`,
`NoWriteBackOrderStallAfterLongLatencyOp` (64-bit MUL).

<a id="c-16"></a>
**C-16. Sec. 4.7, miscellaneous data processing (A64 table, p.18, l.903-917).**

| SOG group | SOG | Implementation |
|---|---|---|
| Bitfield extract (EXTR) | 2 / 2 / 11 | `CortexA55WriteExtr2` for Rn != Rm; ROR alias 1 ([D-4](#d-4)) |
| Sign/zero extend (SXTB, UXTB) | 1 / 2 / 11 | `CortexA55WriteBfm1` via `CortexA55IsExtendAlias` (scope: [U-8](#u-8)) |
| Bitfield move, basic (SBFM, UBFM) | 2 / 2 / 11 | `CortexA55WriteBfm2`; LSL/LSR/ASR aliases 1 ([D-3](#d-3)) |
| Bitfield move, insert (BFM) | 2 / 2 / 11 | `CortexA55WriteBfm2` |
| Count leading (CLS, CLZ) | 1 / 2 / 11 | latency 1 on ALU |
| Reverse bits (RBIT) | 2 / 2 / 11 | `CortexA55WriteALU2`; test `RbitAndCrcLatency` |
| Reverse bytes (REV, REV16, REVSH) | 1 / 2 / 11 | latency 1 on ALU (on REVSH see [I-10](#i-10)) |

<a id="c-17"></a>
**C-17. Sec. 4.8, loads (A64 table, p.21, l.1047-1083).**

| SOG group | SOG | Implementation |
|---|---|---|
| Load register, literal (LDR, LDRSW) | 3 (2) / 1 / 11 | `WriteLD` 3 (stock class; no base register) |
| Load register, unscaled immed (LDUR*) | 3 (2) / 1 / 11 | `WriteLD` + `ReadAdrBase` |
| Load register, immed, pre/post-indexed | 3 (2), 1 / 1 / 11 | `WriteAdr` 1 + `WriteLD` + `ReadAdrBase` |
| Load register, immed unprivileged (LDTR*) | 3 (2) / 1 / 11 | `WriteLD` + `ReadAdrBase` |
| Load register, unsigned immed | 3 (2) / 1 / 11 | `WriteLD` + `ReadAdrBase` |
| Load register, register offset | 3 (2) / 1 / 11 | `WriteLDIdx` 3 + `ReadAdrBase` (stock) |
| Preload (PRFM) | 1 / 1 / 11 | `CortexA55WritePRFM` |
| Load acquire (LDAR*, LDLAR*), load acquire exclusive (LDAXR*) | 3 (2) / 1 / 01 | `CortexA55WriteLdAcq` |
| Load acquire exclusive, pair (LDAXP) | 3(2) / 1 / 00 | `CortexA55WriteLdAXP` + `CortexA55WriteLdAXPb` |
| Load pair, W-form (LDP, LDNP) | 3 (2) / 1 / 00 | `CortexA55WriteLDPW3SI` + `CortexA55WriteLDP1` |
| Load pair, X-form (LDP, LDNP) | 4 (3) / 1/2 / 01 | `CortexA55WriteLDPX4` (Ld 2 cycles) + `CortexA55WriteLDPX4b` |
| Load pair, signed words (LDPSW) | 3 / 1 / 00 | `CortexA55WriteLDPSW3SI` + `CortexA55WriteLDPSW3b` (no pointer chasing, [C-7](#c-7)); pre/post with `WriteAdr` |
| Load pair, W-form, pre/post | 3 (2), 1 / 1 / 00 | `WriteAdr` + `CortexA55WriteLDPW3SI` + `CortexA55WriteLDP1` |
| Load pair, X-form, pre/post | 4 (3), 2 / 1/2 / 00 | `CortexA55WriteAdr2` + `CortexA55WriteLDPX4SI` + `CortexA55WriteLDPX4b` |

Test: `ScalarAndPairLoadsStores`, `DualIssueColumn`.

<a id="c-18"></a>
**C-18. Sec. 4.9, stores (A64 table, p.23, l.1162-1190).**

| SOG group | SOG | Implementation |
|---|---|---|
| STUR*, STR immed pre/post, STTR*, STR unsigned immed, STR register offset | 1 / 1 / 11 | `WriteST` / `WriteSTIdx` 1 on `CortexA55UnitSt`; pre/post `WriteAdr` 1 |
| Store release (STLR*, STLLR*) | 2 / 1/2 / 01 | `CortexA55WriteSTLR` |
| Store release exclusive (STLXR*, STLXP) | 4 / 1/2 / 01 | `CortexA55WriteSTLXR` |
| Store pair, immed, all addressing modes (STP, STNP) | 1 / 1 / 11 | `WriteSTP` |
| (Store, writeback form) | (1/T) / same as before | `WriteAdr` 1 (T = 1) ([U-11](#u-11)) |

Operand timing of stores. Base: pointer chasing from loads ([C-7](#c-7)).
Data (`[a55_sog_u9]`, resolved 2026-10-31): `ReadAdvance<ReadST, 1>` (advance 1
from every producer) is now on the data operands of every store with an SOG row, not only of the
register-offset stores (whose stock `Sched` lists had it): STR/STUR/STTR (all addressing modes and
sizes, incl. FP), STP/STNP (both data registers, incl. Q), STLR/STLLR, STLXR/STLXP, and the data
`Vt` of every ASIMD ST1-ST4 row ([C-29](#c-29)). SOG: sec. 4.9 (p.22, l.1108-1109) stores "do not need
to wait for data operands (which are required at wr)"; sec. 3.1.1 (p.9, l.351-352) "a dedicated
forwarding path ... from the ex2 stage of the ALU0 to the ex2 stage of the store pipeline"; sec. 4.14
(p.28, l.1521-1522) FP store data "required at f2" (FP operands are read in f1, p.8); sec. 4.19
(p.38, l.2193) ASIMD stores "do not need to wait for data operands" (no stage given); sec. 5.7
(p.43, l.2453-2456) similar instructions behave alike. The amount (1) is the project's reading
([U-9](#u-9)). Measured (section 8, before -> after): `mul x1, x2, x3; str x1, [x0]` 5.0 -> 4.0 (=
register-offset, 4.0); `ldr x1, [x0]; str x1, [x5]; ldr x2, [x0, #8]; str x2, [x5, #8]` 7.0 -> 5.0;
`add x1, x1, #1; str x1, [x0]` twice 3.0 -> 2.5. Not covered: atomics, STXR/STXP (no SOG row).
Test: `StoreDataNeededLater`.

<a id="c-19"></a>
**C-19. Sec. 4.10, atomics (p.23-25, l.1193-1290): latency, throughput, dual issue.**

| SOG group | SOG | Implementation (`CortexA55WriteAtomic<lat, occ>`, `BeginGroup`) |
|---|---|---|
| LD&lt;OP&gt;, without release | 2 / 1/2 / 01 | `CortexA55WriteLdOp` -> `<2, 2>` |
| LD&lt;OP&gt;, with release (…L, …AL) | 3 / 1/3 / 01 | `CortexA55WriteLdOpL` -> `<3, 3>` |
| ST&lt;OP&gt;, without release | 1 / 1 / 01 | `CortexA55WriteLdOp`, `CortexA55AtomicToZR` arm -> `<1, 1>` |
| ST&lt;OP&gt;, with release | 2 / 1/2 / 01 | `CortexA55WriteLdOpL`, ZR arm -> `<2, 2>` |
| CAS{A} / CASL, CASAL | 4 / 1/4 / 01, 3 / 1/3 / 01 | `<4, 4>`, `<3, 3>` |
| CASP{A} / CASPL, CASPAL | 5 / 1/5 / 00, 4 / 1/4 / 00 | `CortexA55WriteAtomicSI<5, 5>`, `<4, 4>` |
| SWP{A} / SWPL, SWPAL | 2 / 1/2 / 01, 3 / 1/3 / 01 | `<2, 2>`, `<3, 3>` |

ST&lt;OP&gt; is LD&lt;OP&gt; with Rt = WZR/XZR (same opcode), recognised by `CortexA55AtomicToZR`.
Atomics used to be `Unsupported` and silently dropped. Which resources they occupy: [U-5](#u-5).
Test: `AtomicsAreModelled`.

<a id="c-20"></a>
**C-20. Sec. 4.21, CRC (A64 table, p.41, l.2362-2366).** CRC32B/H/CB/CH/X/CX "2, 2, 11" ->
`CortexA55WriteALU2`; CRC32W/CW "1, 2, 11" -> `CortexA55WriteCRC1`. Test: `RbitAndCrcLatency`.
(CRC32W/CW are not producers of the ALU0 -> ALU1 path, [D-6](#d-6).)

### 3.3 Instruction tables, FP and ASIMD

<a id="c-21"></a>
**C-21. Sec. 4.11, FP data processing (A64 table, p.25-26, l.1338-1380).**

| SOG group | SOG | Implementation |
|---|---|---|
| FABS, FNEG, FADD/FSUB, FMIN/FMINNM/FMAX/FMAXNM, FRINT* | 4 / 2 / 11 | latency 4 on one `CortexA55UnitFPALU` unit (`WriteF`) |
| FP compare (FCCMP{E}, FCMP{E}) | 1 / 1 / 11 | `CortexA55WriteFCmp`: latency 1, one `CortexA55UnitFPALU` unit + the one-unit `CortexA55UnitFPCmp` for one cycle (below) |
| FDIV, FSQRT | see [C-11](#c-11) | see [C-11](#c-11) |
| FP multiply (FMUL, FNMUL) | 4 / 2 / 11 | `WriteFMul` 4 on `CortexA55UnitFPMAC` |
| FP multiply accumulate (FMADD, FMSUB, FNMADD, FNMSUB) | 4 / 2 / 11 | `CortexA55WriteFMAC` ([C-10](#c-10)) |
| FP select (FCSEL) | 2 / 1 / 01 | `CortexA55WriteFCSEL` (both FP ALU units, `BeginGroup`) |

FP compare: resolved 2026-10-31 (`[a55_sog_n5]`). It used to be
`CortexA55WriteFPALU_F1` (one of the two FP ALU units), i.e. throughput 2. The SOG gives throughput
1 (AArch32 VCMP likewise "1, 1, 11", p.25, l.1302) but not which FP ALU unit compares; as for the
variable shifts ([D-7](#d-7)), a one-unit resource caps the FP compares at one per cycle and leaves
the other FP ALU unit free, so an FP compare still pairs with another FP ALU instruction (11). (The
other "throughput 1, dual-issue 11" row, AESMC/AESIMC, holds both FP ALU units instead.) Measured
(section 8): four independent `fcmp` + branch 4.0 cycles/iteration (was 2.5); `fcmp; fadd; fcmp;
fadd` + branch 2.5 (unchanged). Test: `FpCompareThroughputOne`.

<a id="c-22"></a>
**C-22. Sec. 4.12, FP miscellaneous (A64 table, p.27, l.1417-1443).**

| SOG group | SOG | Implementation |
|---|---|---|
| FP convert, vec to vec (FCVT) | 4 / 2 / 11 | `WriteFCvt` 4 |
| FP convert, vec to vec (FCVTXN) | 4 / 1 / 01 | `CortexA55WriteAluVq_4` (regex `^FCVT(L|N|XN)v`) |
| FP convert, gen to vec (SCVTF, UCVTF) | 5 / 2 / 11 | `CortexA55WriteFPALU_F5` |
| FP convert, vec to gen (FCVT[AMNPZ][SU]) | 3 / 2 / 11 | `CortexA55WriteFPALU_F3` |
| FP move, immed / register (FMOV) | 1 / 2 / 11 | `CortexA55WriteFPALU_F1` |
| FP transfer, gen to half/single/double (FMOV) | 2 / 2 / 11 | `CortexA55WriteFPALU_F2` |
| FP transfer, single/double/half to gen (FMOV) | 3 / 2 / 11 | `WriteFCopy` 3 |

<a id="c-23"></a>
**C-23. Sec. 4.13, FP loads (A64 table, p.27-28, l.1464-1508).**

| SOG group | SOG | Implementation |
|---|---|---|
| LDR literal, S/D | 3 / 1 / 11 | `WriteLD` 3 |
| LDR literal, Q | 4 / 1/2 / 01 | `CortexA55WriteLdQ4` (Ld 2 cycles, `BeginGroup`) |
| LDUR, B/H | 3 / 1 / 01 | `CortexA55WriteLdBH3` |
| LDUR, S/D | 3 / 1 / 11 | `WriteLD` |
| LDUR, Q | 4 / 1/2 / 01 | `CortexA55WriteLdQ4` |
| LDR pre/post, B/H; S/D; Q | 3, 1 / 1 / 01; 3, 1 / 1 / 11; 4, 2 / 1/2 / 01 | `WriteAdr` + `CortexA55WriteLdBH3`; `WriteAdr` + `WriteLD`; `CortexA55WriteAdr2` + `CortexA55WriteLdQ4` |
| LDR unsigned immed / register offset, B/H; S/D; Q | 3 / 1 / 01; 3 / 1 / 11; 4 / 1/2 / 01 | `CortexA55WriteLdBH3`; `WriteLD` / `WriteLDIdx`; `CortexA55WriteLdQ4` |
| LDP/LDNP, S; D; Q | 3 / 1 / 01; 4 / 1/2 / 01; 6 / 1/4 / 01 | `CortexA55WriteLDPS3`; `CortexA55WriteLDPX4`; `CortexA55WriteLDPQ6` (Ld 4 cycles) |
| LDP pre/post, S; D; Q | 3, 1; 4, 2; 6, 4 (T and DI as above) | `WriteAdr`, `CortexA55WriteAdr2`, `CortexA55WriteAdr4` + the same data writes |

<a id="c-24"></a>
**C-24. Sec. 4.14, FP stores (A64 table, p.29, l.1544-1557).** STUR, STR (immed, register
offset), STP S/D-form "1, 1, 11": `WriteST` / `WriteSTIdx` / `WriteSTP`. STP Q-form "2, 1/2, 01":
`CortexA55WriteSTPQ` (also STNP Q); its writeback "(1/T)" = 2: `CortexA55WriteAdr2`. Data "required
at f2" (l.1521-1522): `ReadST` on every FP store ([C-18](#c-18)); base: [C-7](#c-7).

<a id="c-25"></a>
**C-25. Sec. 4.15, ASIMD integer (A64 table, p.30-32, l.1655-1780).** D-forms (and scalar forms)
on one FP ALU unit for one cycle; Q-forms of `2*, 11*` rows on both units with `BeginGroup`
([1.4](#14-reading-the-sog-instruction-tables)). "Default" means the stock class falls on the
model's `WriteVd`/`WriteVq` aliases (latency 4).

| SOG row | SOG | Implementation |
|---|---|---|
| SABD, UABD | 3 / 2* / 11* | `CortexA55WriteAluVd_3` / `CortexA55WriteAluVq_3` |
| SABA, UABA, SABAL(2), UABAL(2) | 4 / 1/2 / 01 | `CortexA55WriteAluT2_4` (one unit 4 cycles, `BeginGroup`) |
| SABDL, UABDL | 3 / 1 / 01 | `CortexA55WriteAluVq_3` |
| ADD, SUB, NEG, [SU]HADD, [SU]HSUB, [SU]RHADD | 2 / 2* / 11* | `AluVd_2` / `AluVq_2` |
| ABS, ADDP, [SU]ADDLP, SQADD, UQADD, SQNEG, SQSUB, UQSUB, SUQADD, USQADD | 3 / 2* / 11* | `AluVd_3` / `AluVq_3` |
| [SU]ADDL(2), [SU]ADDW(2), [SU]SUBL(2), [SU]SUBW(2), ADDHN(2), SUBHN(2) | 3 / 1 / 01 | `AluVq_3` |
| SQABS | 4 / 2* / 11* | default |
| RADDHN(2), RSUBHN(2) | 4 / 1/2 / 01 | `AluT2_4` |
| ADDV, SADDLV, UADDLV | 3 / 1 / 01 | `AluVq_3` |
| CMEQ, CMGE, CMGT, CMHI, CMHS, CMLE, CMLT | 2 / 2* / 11* | `AluVd_2` / `AluVq_2` |
| CMTST | 3 / 2* / 11* | `AluVd_3` / `AluVq_3` |
| AND, BIC, EOR, MVN, ORN, ORR | 1 / 2* / 11* | `AluVd_1` / `AluVq_1` (ORR register form: `CortexA55WriteOrrVd` / `OrrVq`, latency 1 unless Rn == Rm) |
| MOV (ASIMD logical) | 2 / 2* / 11* | ORR with Rn == Rm (`CortexA55IsVecMovAlias`): `AluVd_2` / `AluVq_2` (below) |
| [SU]MAX(P), [SU]MIN(P) | 2 / 2* / 11* | `AluVd_2` / `AluVq_2` |
| [SU]MAXV, [SU]MINV | 4 / 1 / 01 | `AluVq_4` |
| MUL, SQDMULH, SQRDMULH | 4 / 2* / 11* | default |
| the same, by element | 4 / 1 / 01 | `AluVq_4` |
| PMUL | 3 / 2* / 11* | `AluVd_3` / `AluVq_3` |
| MLA, MLS; by element | 4 (1) / 2* / 11*; 4 (1) / 1 / 01 | `CortexA55WriteMlaVd_4` / `MlaVq_4`; `MlaIxVq_4` ([C-9](#c-9)) |
| SQRDMLAH, SQRDMLSH | 4 / 1 / 01 | `AluVq_4` |
| [SU]MLAL(2), [SU]MLSL(2) | 4 (1) / 1 / 01 | `CortexA55WriteMlaLVq_4` ([C-9](#c-9)) |
| SQDMLAL(2), SQDMLSL(2) | 4 / 1 / 01 | `AluVq_4` |
| UDOT, SDOT; by scalar | 4 (1) / 2* / 11*; 4 (1) / 1 / 01 | `CortexA55WriteDotVd_4` / `DotVq_4`; `DotScVq_4` ([C-9](#c-9)) |
| SMULL(2), UMULL(2), SQDMULL(2) | 4 / 1 / 01 | `AluVq_4` |
| PMULL.8B, PMULL2.16B | 3 / 1 / 01 | `AluVq_3` |
| SADALP, UADALP | 4 / 1/2 / 01 | `AluT2_4` |
| SRA, USRA | 3 / 2* / 11* | `AluVd_3` / `AluVq_3` |
| SRSRA, URSRA | 4 / 1/2 / 01 | `AluT2_4` |
| SHL, SHRN(2), SLI, SRI, SSHR, USHR | 2 / 2 / 11 | `AluVd_2` for every form ([U-7](#u-7); SLI/SRI: [I-3](#i-3)) |
| SLI, SRI (shift and insert) | 2 / 2* / 11* | vector `AluVd_2` / `AluVq_2`; scalar `AluVd_2` ([I-3](#i-3)) |
| SHLL(2), SSHLL(2), USHLL(2), SXTL(2), UXTL(2) | 2 / 1 / 01 | `AluVq_2` |
| RSHRN(2), SRSHR, URSHR, SQSHRN(2), UQSHRN(2) | 3 / 2* / 11* | `AluVd_3` / `AluVq_3` |
| SQSHL{U}, UQSHL, SQRSHRN(2), UQRSHRN(2), SQRSHRUN(2), SQSHRUN(2) | 4 / 2* / 11* | default |
| SSHL, USHL; SRSHL, URSHL; SQSHL, UQSHL, SQRSHL, UQRSHL (register) | 2; 3; 4 / 2* / 11* | `AluVd_2`/`AluVq_2`; `AluVd_3`/`AluVq_3`; default |

ASIMD MOV: resolved 2026-10-31 (`[a55_sog_n6]`). `MOV Vd.T, Vn.T` is
`ORR Vd.T, Vn.T, Vn.T` (`ORRv8i8` / `ORRv16i8`) and used to take the ORR row (latency 1). The
register-form ORR now has a `SchedWriteVariant` on `CheckSameRegOperand<1, 2>`: with Rn == Rm it
takes the A64 MOV row (2, 2*, 11*), otherwise the ORR row (1, 2*, 11*). ORR (immediate) is
unchanged. The A64 MOV row is the SOG's statement for this A64 instruction; the AArch32 VMOV
(register) row says 1, so the SOG remains inconsistent ([I-8](#i-8)). Measured (section 8): a chain
of four `mov v0.16b, v0.16b` + branch 8.0 cycles/iteration (was 4.0). Test:
`AsimdMovRegisterLatencyTwo`.

<a id="c-26"></a>
**C-26. Sec. 4.16, ASIMD FP (A64 table, p.33-34, l.1843-1898).**

| SOG row | SOG | Implementation |
|---|---|---|
| FABS, FABD, FADD, FSUB, FADDP | 4 / 2* / 11* | default |
| FACGE, FACGT, FCMEQ, FCMGE, FCMGT, FCMLE, FCMLT | 2 / 2* / 11* | `AluVd_2` / `AluVq_2` |
| FCVTL(2); FCVTN(2), FCVTXN(2) | 4 / 1 / 01 | `AluVq_4` |
| FCVT[AMNPZ][SU], SCVTF, UCVTF (vector) | 4 / 2* / 11* | `CortexA55WriteFPALU_F4` / `AluVq_4` |
| FDIV, H / S / D | 8, 13, 22 / 1/5, 1/10, 1/19 / 01 | `CortexA55WriteFDivHP/SP/DP` ([C-11](#c-11)) |
| FMAX, FMAXNM, FMIN, FMINNM; pairwise | 4 / 2* / 11* | default |
| FMAXV, FMAXNMV, FMINV, FMINNMV | 4 / 1 / 01 | default (`WriteVq`) |
| FMUL, FMULX; by element | 4 / 2* / 11*; 4 / 1 / 01 | default; `AluVq_4` |
| FMLA, FMLS; by element | 4 / 2* / 11*; 4 / 1 / 01 | `CortexA55WriteFMAC` / `CortexA55WriteFMACq`; `FMACq` ([C-10](#c-10)) |
| FNEG, FRINT* | 4 / 2* / 11* | default |

<a id="c-27"></a>
**C-27. Sec. 4.17, ASIMD miscellaneous (A64 table, p.35-36, l.1973-2030).**

| SOG row | SOG | Implementation |
|---|---|---|
| RBIT; BIF, BIT, BSL; CLZ, CNT | 2 / 2* / 11* | `AluVd_2` / `AluVq_2` |
| CLS | 3 / 2* / 11* | `AluVd_3` / `AluVq_3` |
| DUP, gen reg | 3 / 2 / 11 | `CortexA55WriteFPALU_F3` (every form) |
| DUP, element | 2 / 2* / 11* | `AluVd_2` (incl. scalar DUPi*) / `AluVq_2` |
| EXT | 2 / 2* / 11* | `AluVd_2` / `AluVq_2` |
| XTN | 2 / 2 / 11 | `CortexA55WriteFPALU_F2` |
| SQXTN(2), SQXTUN(2), UQXTN(2) | 4 / 2 / 11 | `CortexA55WriteFPALU_F4` (every form; below) |
| INS, element to element; INS, gen reg to element | 2 / 2 / 11 | `CortexA55WriteFPALU_F2` |
| MOVI | 1 / 2* / 11* | `AluVd_1` / `AluVq_1` (MOVID: `FPALU_F1`) |
| FMOV, FP immed | 1 / 2 / 11 | `AluVd_1`; Q-form `CortexA55WriteFPALU_F1` |
| FRECPE, FRECPX, FRSQRTE, URECPE, URSQRTE; FRECPS, FRSQRTS | 4 / 2* / 11* | `AluVd_4` / `AluVq_4` |
| REV16, REV32, REV64 | 2 / 2* / 11* | `AluVd_2` / `AluVq_2` |
| TBL | 2+N-1 / 1/N / 01 | `CortexA55WriteTBL1..4` (latency 2..5, one unit 2N cycles, `BeginGroup`) |
| TBX | 2+N / 1/(N+1) / 01 | `CortexA55WriteTBX1..4` (latency 3..6, one unit 2(N+1) cycles) |
| SMOV, UMOV | 2 / 2 / 11 | `CortexA55WriteFPALU_F2` |
| TRN1, TRN2 .2D; other | 2 / 1 / 01; 2 / 2* / 11* | `AluVq_2`; `AluVd_2` / `AluVq_2` |
| UZP1, UZP2, ZIP1, ZIP2 | 2 / 2* / 11* | `AluVd_2` / `AluVq_2` |

SQXTN / SQXTUN / UQXTN: resolved 2026-10-31 (`[a55_sog_n7]`). The vector
forms (D and Q) used to fall on the default `WriteVq` (both FP ALU units, `BeginGroup`): throughput
1, slot 0 only. The row has no asterisk (no Q-form note), and the AArch32 row "ASIMD move,
saturating VQMOVN, VQMOVUN 4, 2, 11" (p.34, l.1931) agrees, so every form (D, Q = the "2" forms,
scalar) is now one FP ALU unit for one cycle, latency 4, both slots (`InstRW` on
`^(SQXTN|SQXTUN|UQXTN)v`). Measured (section 8): `add; sqxtn v0.8b, v1.8h; add; sqxtn` + branch
2.5 cycles/iteration (was 3.0), the same as with XTN. Test: `SaturatingExtractNarrowDualIssues`.

<a id="c-28"></a>
**C-28. Sec. 4.18, ASIMD loads (A64 table, p.37-38, l.2102-2183).** Every row has its own
`InstRW` with `CortexA55WriteVLDx<lat, 1/T, BeginGroup>`: latency = SOG latency, load pipe held
`1/T` cycles, `BeginGroup` for `01` (all rows except LD1 one register D-form, `11`). Writeback
forms: base update "(1/T)" as `CortexA55WriteVAdr<1/T>` ([U-11](#u-11)). Checked row by row against
l.2105-2181 (LD1 1-4 registers D/Q, one lane, LD1R; LD2 D/Q, one lane B/H/S and D, LD2R; LD3
D/Q by element size, one lane, LD3R; LD4 D/Q by element size, one lane, LD4R). Exception: the
dual-issue of the writeback form of LD1 one register D-form ([N-2](#n-2)). The base reads
`ReadAdrBase` ([C-7](#c-7)). Test: `AsimdLoadsStores`.

<a id="c-29"></a>
**C-29. Sec. 4.19, ASIMD stores (A64 table, p.39-40, l.2239-2291).** Same scheme with
`CortexA55WriteVSTx<lat, 1/T, BeginGroup>` (`RetireOOO`): ST1 one register "1, 1, 11", every
other row `01`; checked row by row against l.2242-2289. Exception: ST4 one lane S/D has no row
([U-6](#u-6)). Every row reads the data `Vt` with `ReadST` ([C-18](#c-18)) and the base with
`ReadAdrBase` ([C-7](#c-7)). Test: `AsimdLoadsStores`.

<a id="c-30"></a>
**C-30. Sec. 4.20, Cryptographic Extension (A64 table, p.40-41, l.2316-2346).**

| SOG row | SOG | Implementation |
|---|---|---|
| AESD, AESE | 2 / 1 / 01 | `AluVq_2` |
| AESIMC, AESMC | 2 / 1 / 11 | `CortexA55WriteAluT1_2_11` (both FP ALU units, no `BeginGroup`) |
| PMULL(2) 64x64 | 2 / 1 / 01 | `AluVq_2` |
| SHA1SU0; SHA1H, SHA1SU1 | 2 / 1 / 01 | `AluVq_2` |
| SHA1C, SHA1M, SHA1P | 4 / 1 / 01 | default `WriteVq` |
| SHA256SU0 | 3 / 1 / 01 | `AluVq_3` |
| SHA256SU1; SHA256H, SHA256H2 | 4 / 1 / 01 | default `WriteVq` |

The AESE/AESMC pairing of sec. 3.5.2 is not modelled: [N-1](#n-1).

---

## 4. Not compatible with the SOG (N)

Cycle counts in this section were reproduced as described in [section 8](#8-reproducing-the-cycle-counts-quoted-here).

<a id="n-1"></a>
**N-1. AESE + AESMC (AESD + AESIMC) pairs are not dual-issued.**
SOG: sec. 3.5.2 (p.13, l.613-618): dependent pairs "can be dual-issued, if they meet the following
template: AESE Vn, _ / AESMC Vn, Vn"; sec. 4.20 note 1 (p.41, l.2350): "Adjacent AESE/AESMC
instruction pairs ... can be dual-issued together." Implementation: not modelled (comment in
`AArch64SchedA55.td` next to the crypto rows). The dependent AESMC waits for AESE's latency 2, and
even an independent AESE + AESMC pair cannot issue together because `CortexA55WriteAluVq_2` (AESE)
and `CortexA55WriteAluT1_2_11` (AESMC) each hold both `CortexA55UnitFPALU` units. Measured: body
`aese v0; aesmc v0, v0; aese v2; aesmc v2, v2` 6.0 cycles/iteration, independent registers 4.0.
Why open: needs an issue-stage rule for the template (same register Vn), similar to
[D-6](#d-6), and a resource representation that lets the two halves share a cycle.

<a id="n-2"></a>
**N-2. Dual issue of LD1 (one register, D-form) with writeback.**
SOG sec. 4.18 A64 (p.38, l.2183): "(ASIMD load, writeback form) - (1/T) Same as before 01": the
dual-issue column of the writeback row is `01`, while the non-writeback LD1 one register D-form is
`11` (l.2105). Implementation: `LD1Onev(8b|4h|2s|1d)_POST` uses `CortexA55WriteVLDx<3, 1, 0>`
(no `BeginGroup`), i.e. `11`. All other ASIMD load rows are `01` anyway. Why open: small; whether
the writeback row's `01` is meant to override the base row is not stated (not verified).

<a id="n-3"></a>
**N-3. WFE and WFI can be dual-issued.**
*Status: recorded 2026-10-31; deliberately not fixed (decision of the project owner).*
SOG sec. 3.2 Table 1 (p.10-11, l.410-412): "Control instructions cannot be dual issued. These include
MRC/MCR, MRS/MSR, WFI, WFE, CPS, and barriers." Implementation: `WriteSys` and `WriteBarrier` have
`SingleIssue` ([C-3](#c-3)), but in LLVM WFE and WFI are not separate opcodes: they are `HINT #2`
and `HINT #3`, the same opcode as YIELD, SEV, SEVL, BTI, CSDB, ..., whose class is `WriteHint`
(`CortexA55UnitB`, no `SingleIssue`; see [U-13](#u-13)). So WFE and WFI pair with other
instructions as far as the other rules allow. Why open, and why left: the two instructions have to be
told apart from the other hints by the immediate operand of `HINT` (2 and 3), which takes a new
scheduling predicate (like `CortexA55IsRorImmAlias`) and a `SingleIssue` write; the fix is small, but
WFE and WFI do not occur in user-space programs (not verified), so the project owner decided to
record the difference only.

---

## 5. Deliberate deviations and interpretations (D)

<a id="d-1"></a>
**D-1. Ranges: the best value; throughput as occupancy.**
SOG: several entries are ranges: integer divide "3 - 12 (11)" / "3 - 20 (19)" with throughput
"1/12 (11) - 1/3" (sec. 4.5, p.16, l.794-797; note 1 on early termination, p.17, l.824-826); MADD 64-bit
"4-5 (2-3)", "1/3 - 1/2" (l.802); and pointer chasing "can be activated in some cases" (sec. 4.8
note 1, l.1088). Implementation (project policy): the best (lowest-latency) value of a range, and
the resource held for `1/T` of the best throughput, rounded up: divide latency 3, `CortexA55UnitDiv`
3 cycles (`WriteID32`/`WriteID64`); MADD X latency 4, MAC 2 cycles, accumulator 2; the
parenthesised pointer-chasing latency on every LDR form ([C-7](#c-7)). On the two-unit FP/ASIMD
resources a throughput `T` holds one unit `2/T` cycles (`CortexA55WriteVT`). Reason: the static
model has one value per write; a value that depends on data is not known. Effect: optimistic for
divide-heavy code (a divide can take up to 12 or 20 cycles).

<a id="d-2"></a>
**D-2. L1 hits, aligned accesses, normal FP inputs, near atomics.**
SOG: load latencies "assume the memory access hits in the Level 1 (L1) data cache" and correctly
aligned accesses (sec. 4.8, p.18, l.922-923); alignment penalties (sec. 3.3, p.11, l.469-483);
pointer chasing only for aligned addresses (Table 3, l.493); denormal FP inputs add a cycle
(Table 4 note c, p.13, l.609-610); atomics may be performed far (sec. 5.6, p.43, l.2441-2450).
Implementation: none of these conditions is modelled; every access is taken to be an aligned L1
hit, FP inputs normal, atomics near. Reason: addresses and data are not known statically; misses
are handled outside this model (MLP, see [docs.md](docs.md) section 3).

<a id="d-3"></a>
**D-3. LSL/LSR/ASR by immediate: 1 cycle.**
SOG: the A64 table has no shift-by-immediate row; these are aliases of UBFM/SBFM, "Bitfield move,
basic SBFM, UBFM 2, 2, 11" (sec. 4.7, p.18, l.909). The AArch32 table gives "Move, shift by immed
ASR{S}, LSL{S}, LSR{S}, ROR{S}, RRX{S} 1, 2, 11" (sec. 4.4, p.15, l.729-730). Implementation
(`[a55_shift_imm]`): `CortexA55WriteUBFMW/X` and `CortexA55WriteSBFMW/X` select
`CortexA55WriteShiftImm1` (latency 1) for LSR #n (UBFM immr = n, imms = size-1), ASR #n (SBFM, same
condition; `CortexA55IsShiftRightAlias`) and LSL #n (UBFM imms != size-1, immr = imms+1;
`CortexA55IsLslAlias`, enumerated with `!foreach`). UBFX/SBFX/UBFIZ/SBFIZ keep 2. The source still
goes through the shifter (`CortexA55ReadShifted`): an LSL is a producer, not a consumer, of the
same-cycle ALU path ([D-6](#d-6)). Reason: AArch32 and AArch64 shifts are taken to use the same
shifter (the shift box of both ALU pipes, fig. 1, p.8); the SOG itself is inconsistent here
([I-4](#i-4)). Test:
`ShiftImmediateAliasesTakeOneCycle`.

<a id="d-4"></a>
**D-4. ROR by immediate: 1 cycle.**
SOG: "Bitfield extract EXTR 2, 2, 11" (sec. 4.7, p.18, l.905) covers every EXTR, including
`ROR Rd, Rn, #n` = `EXTR Rd, Rn, Rn, #n`; the AArch32 row of [D-3](#d-3) includes ROR.
Implementation (`[a55_ror_imm]`): `CortexA55WriteEXTR` selects `CortexA55WriteShiftImm1` when
`CortexA55IsRorImmAlias` (`CheckSameRegOperand<1, 2>`, Rn == Rm) holds, else
`CortexA55WriteExtr2`. Both sources are `CortexA55ReadShifted`. Reason: as D-3. Test: `RorImmediateAliasTakesOneCycle`.

<a id="d-5"></a>
**D-5. Shifted-register ALU forms with shift amount 0 are "ALU, basic".**
SOG: "ALU, basic" 1 and "ALU, extend and/or shift" 2 (sec. 4.3, p.15, l.713-718); the SOG does not
say how a zero shift is classified. Implementation: `CortexA55WriteISRegV` (`RegShiftedPred` ->
`CortexA55WriteISReg2`, else `CortexA55WriteISReg1`), and for the reads `CortexA55ReadISReg` /
`CortexA55ReadIEReg` (a zero shift reads like a plain ALU operand). `applyA55InstructionAdaptations`
sets the same latency 1 for `ADD/SUB/ADDS/SUBS (W|X)rs` with shift 0 (redundant). Reason: a zero
shift does no shifting; `MOV Xd, Xn` (ORR with XZR) is the common case.

<a id="d-6"></a>
**D-6. The ALU0 -> ALU1 dedicated path as an issue-stage rule.**
SOG sec. 3.1.1 (p.9, l.349-352): the ALU pipeline forwards "from the end of ex1 and ex2 to earlier
stages of both ALU pipelines", and "There is also a dedicated forwarding path from the ex1 stage of
the ALU0 to the ex1 stage of the ALU1". Implementation (`[a55_alu_fwd]`):
`A55DecoupledIssueStage::isSameCycleALUForward` waives a RAW hazard of instruction-1 when (a) the
hazard comes only from this cycle's instruction-0, (b) instruction-0 runs only on
`CortexA55UnitALU` (`isALUPipeOnly`; `CortexA55UnitVarShift` counts as part of the ALU pipes) and
the write has latency 1 (`isALUForwardProducer`; CRC32W/CW excluded because sec. 4.21 does not place
CRC in the ALU pipes), (c) instruction-1 runs only on `CortexA55UnitALU`, and (d) the operand is
read by the ALU, not the shifter, which the stage recognises as "the read has a `ReadAdvance` >= 1
from `WriteLD`" (`ReadI`, `CortexA55ReadNotShifted`). Across cycles an ALU result is seen at its
exec latency (1), the plain register-file hazard; there is no ALU -> ALU `ReadAdvance` in the `.td`.
A chain of dependent ADDs therefore issues two per cycle. Assumptions: instruction-0 executes in
ALU0 and instruction-1 in ALU1 (the SOG does not give the slot -> pipe mapping; it is the only one
under which an ALU0 -> ALU1 path within ex1 is usable by an in-order core); the path exists only
within a pair. Reason for the representation: a `ReadAdvance` from the 1-cycle ALU writes (the
earlier form) expressed the same-cycle pair only implicitly and for every reader of `ReadI`; the
issue-stage rule states which pipes it connects. Test: `SameCycleAluForwarding`. See also
[I-5](#i-5).

<a id="d-7"></a>
**D-7. Variable shifts: a one-unit resource.**
SOG: "Variable shift ASRV, LSLV, LSRV, RORV 1, 1, 11" (sec. 4.4, p.16, l.758), against throughput 2
for ALU basic. Implementation (`[a55_varshift]`): `CortexA55WriteVarShift` holds one
`CortexA55UnitALU` and the one-unit `CortexA55UnitVarShift` for one cycle: at most one variable
shift per cycle, and the other ALU stays free, so a variable shift still pairs with another ALU
instruction. It is a producer of the ALU0 -> ALU1 path ([D-6](#d-6)); its operands are shifter
operands (`CortexA55ReadShifted`), so it is not a consumer. Reason: the SOG gives the throughput but
not the pipe ([U-1](#u-1)); this representation adds no constraint beyond the stated throughput.
Test: `VariableShiftThroughputOne`.

<a id="d-8"></a>
**D-8. FCMP and FCCMP are FP flag producers (no same-cycle flag transfer).**
SOG sec. 5.11 (p.44, l.2544-2546): "Flag transfer from the floating-point flags to the integer flags
takes a single cycle. This prevents dual-issue of a VMRS instruction-0 with an integer flag testing
instruction-1". In A64, FCMP{E}/FCCMP{E} write NZCV directly; there is no VMRS. Implementation
(`[a55_flag_bypass]`): `a55IsFlagBypassEligible` refuses the bypass when the NZCV producer's name
starts with FCMP, FCCMP, VMRS or VMSR; the consumer then sees the producer's latency (1, [C-21](#c-21))
and issues in the next cycle. Reason: the A64 FP compares are taken to need the same transfer from
the FP side. Test: `FlagBypassExceptCarryAndFpCompare` (`fcmp`/`fccmp` + two `csel`: 2.0).

<a id="d-9"></a>
**D-9. No general write-back ordering.**
SOG: integer instructions "complete in the wr (writeback) pipeline stage" and pipeline stages
"cannot be skipped" so that branch and store instructions complete in order (sec. 3.1, p.8-9,
l.328, l.340-341); FP/NEON instructions of Table 4 may complete out of order (sec. 3.5.1). The
throughput and dual-issue tables allow, for example, FADD + ADD or LDR + ADD pairs every cycle.
Implementation: `llvm::mca::InOrderIssueStage`'s rule "a younger instruction may not write back
before an older one" is not used; the only ordering rule kept is the divide rule ([U-4](#u-4)).
Reason: with fixed completion stages in-order completion needs no stall, and the general rule
turned such pairs into up to three stall cycles each. Test: `NoWriteBackOrderStallAfterLongLatencyOp`.

<a id="d-10"></a>
**D-10. Operands that pass through the shifter get no load-use advance.**
SOG: load-use latency 2 "to the ALU" (sec. 3.3, l.484); fig. 1 (p.8) draws, in each ALU pipe, a
"shift" box and an "ALU" box (in the text extraction they fall under ex1 and ex2, l.271-278; check
the PDF). The SOG gives no number for load -> shifter. Implementation (`[a55_planB]`):
`CortexA55ReadShifted` (`SchedReadAdvance<0>`) for shifted/extended Rm (`RegShiftedPred`,
`RegExtendedPred`), the source of UBFM/SBFM/BFM, both sources of EXTR and of the variable shifts;
these operands see a load at 3, other ALU operands at 2. Reason: an operand that is shifted is
needed one stage earlier than one consumed by the ALU box. Not verified on hardware.

<a id="d-11"></a>
**D-11. ASIMD FSQRT takes the scalar rows.**
SOG: sec. 4.16 (A64) has FDIV rows but no FSQRT row; sec. 3.5.1 (p.12, l.552-554): "NEON and
floating-point instructions of the same type have the same timing characteristics". Implementation:
`FSQRTv(4f16|8f16)`, `FSQRTv(2f32|4f32)`, `FSQRTv2f64` use `CortexA55WriteFSqrtHP/SP/DP` (sec. 4.11
values). Reason: sec. 3.5.1.

---

## 6. Unclear in the SOG (U)

<a id="u-1"></a>
**U-1. Which ALU pipe executes a variable shift.**
SOG: "Variable shift ASRV, LSLV, LSRV, RORV 1, 1, 11" (sec. 4.4, p.16, l.758); fig. 1 (p.8) draws a
shift box in both ALU0 and ALU1. Unclear: whether only one of the pipes can do a variable shift
(which the throughput 1 suggests) and which one, and whether a variable shift can be the producer
of the ALU0 -> ALU1 path of sec. 3.1.1 (it would have to run in ALU0). Current handling: a one-unit
resource caps the throughput only ([D-7](#d-7)); the variable shift is treated as a producer of the
path (`isALUPipeOnly` counts `CortexA55UnitVarShift` as part of the ALU pipes). Measured:
`lslv x1, x2, x9; add x1, x1, #1` + loop branch 1.5 cycles/iteration (the pair issues together).
Expected impact: small, variable shifts are rare. How to settle: on hardware (the Cortex-A55 of the
Genio board), a micro-benchmark repeating the dependent pair `lslv; add` (and `add; lslv`, and two
independent `lslv` with an `add`), compared with the model.

<a id="u-2"></a>
**U-2. FCCMP (and FCSEL) reading integer flags.**
SOG: FP compare "FCCMP{E}, FCMP{E} 1, 1, 11" (sec. 4.11, p.26, l.1355); FP -> integer flag transfer
1 cycle (sec. 5.11, l.2544-2546); Table 2 "Conditional": a conditional instruction-1 can pair with a
flag-setting instruction-0 except when instruction-0 is NEON (p.11, l.452-457). Unclear: none of
these covers the case where instruction-1 is an FP instruction that reads the integer flags
(FCCMP, FCCMPE, FCSEL) after an integer flag setter. Current handling: `a55IsFlagBypassEligible`
looks only at the producer and at carry consumers, so `cmp` -> `fccmp`/`fcsel` is bypassed (same
cycle), while the flags written by FCCMP are FP flags ([D-8](#d-8)). Measured:
`cmp x1, x2; fccmp s0, s1, #0, eq; csel` + branch 2.0 cycles/iteration (cmp + fccmp together).
Expected impact: one cycle, rare. How to settle: on hardware, a micro-benchmark of
`cmp; fccmp; csel` (and `cmp; fcsel`) sequences.

<a id="u-3"></a>
**U-3. ALU -> MAC / DIV / AGU forwarding.**
*Status: decided 2026-10-31 to keep the effective ALU -> MAC / DIV / AGU latency at 1 (default
handling). The 2-cycle ALU -> AGU reading was implemented as an experimental option and removed
again the same day (project owner decision; see below). ALU -> MAC / DIV: not
implemented.*
SOG sec. 3.1.1 (p.9, l.349-351): "The ... (ALU) pipeline can forward results from the end of ex1 and
ex2 to earlier stages of both ALU pipelines, and to the iss stage of the Divide (DIV) and
Multiply-Accumulate (MAC) pipelines and the load/store Address Generation Units (AGUs)"; "Load-Store
instructions require their address operands at the iss stage" (l.357). Unclear: an ALU consumer
takes the result in ex1 (ALU -> ALU = exec latency 1), MAC multiplicands, DIV operands and AGU
address operands in iss, one stage earlier; ALU -> MAC/DIV/AGU may therefore be one cycle longer
(an effective 2 rather than 1). This is a reading of the stage names; the SOG gives no number. The text does not decide between the
readings: the same sentence names the ALU pipelines, DIV, MAC and the AGUs as targets of one
forwarding mechanism, and the separate "ex1 of ALU0 to ex1 of ALU1" path (l.351) suggests that the
ordinary ALU -> ALU forwarding does not reach ex1 in the same cycle, which fits an effective 1 for the
iss-stage consumers too; and the load-use latency 3 (data at wr, iss of the consumer in the same
cycle) uses the same stage convention. These are readings, not statements of the SOG.
Current handling (default): `ReadIM` and `ReadID` have advance 0 and `ReadAdrBase` has advance 1
only from loads, so the effective ALU -> MAC/DIV/AGU latency is 1 (test `SameCycleAluForwarding`:
`add w1; mul w1, w1, w1` 4 cycles; measured `add x1, x1, #8; ldr x2, [x1]` + branch 2.0). Depending
on the reading this is one cycle optimistic.

Removed experiment (2026-10-31): the 2-cycle ALU -> AGU reading was implemented as an issue-stage rule
behind the command-line option `--a55-alu-agu-latency2` (default off), compared on the research
evaluation, and then deleted together with the option at the project owner's decision (no tag remains
in the code). The rule, for reconstruction: in `A55DecoupledIssueStage::checkRegisterHazard`, after an
operand's register hazard had cleared, the instruction stalled one more cycle if (a) the operand was an
*address operand* of a load/store (the register operand of class GPR64sp, i.e. the base Rn of every
load, store, atomic, exclusive, PRFM and ASIMD structure load/store, and the index Rm of the
register-offset forms `...roW`/`...roX`; not the data registers and not the post-index register Xm of
the ASIMD forms), (b) its producer was an ALU-pipe instruction (the producer set of [D-6](#d-6): only
`CortexA55UnitALU`, CRC32W/CW excluded), and (c) that producer wrote back in this very cycle
(`RegisterFile::getElapsedCyclesFromWriteBack` = 0). So an address operand saw an ALU result at the
exec latency + 1: ALU basic -> AGU 2, ALU extend/shift -> AGU 3; ALU -> ALU, ALU -> store data, load
-> base, base update -> base and MAC -> base were unchanged, and ADRP -> LDR (then also LDUR) kept the waiver of
[C-5](#c-5). Measured: `add x1, x1, #8; ldr x2, [x1]` + branch 2.0 -> 3.0 with the option. The test
`AluToAguIsOneCycle` pins the default (ALU -> base 1). A negative
`ReadAdvance` is supported by the simulator (checked 2026-10-31: with `ReadIM` advance -1,
`add w1, w1, #1; mul w1, w1, w1` goes from 4.0 to 5.0 cycles/iteration; the Cortex-A520 model also uses
negative advances, `AArch64SchedA520.td`). An earlier version of this item said that LLVM's `ReadAdvance`
cannot be negative; that was wrong. How to settle: on hardware, `add x1, x1, #8; ldr x2, [x1]` chains.

<a id="u-4"></a>
**U-4. "Integer divides are serializing".**
SOG sec. 3.4 (p.12, l.522-523): "Integer divide instructions are serializing and do not allow
younger instructions to retire underneath to ensure that the integer divide results is retired
in-order." Sec. 4.5 note 1 (p.17, l.824-826): divides "block any subsequent divide operations until
complete". Unclear: whether younger instructions may issue (and execute) while a divide is in
flight, as long as they complete after it, or whether issue stops. Current handling:
`A55DecoupledIssueStage::canExecute` refuses to issue an instruction whose first write-back
(`findFirstWriteBackCycle`; for an instruction with no register result, its latency) would come
before the in-flight divide's (`DivWriteBackCycle`). Divides block each other through
`CortexA55UnitDiv`. With the best-case divide latency 3 ([D-1](#d-1)) the rule costs at most two
cycles. Test: `IntegerDivideIsSerializing` (`udiv` + three `add`: 4 cycles). Expected impact:
small at latency 3, larger if the divide latency is raised. How to settle: on hardware, `udiv`
followed by independent `add`s, with operands for early and late termination.

<a id="u-5"></a>
**U-5. Which resources an atomic occupies.**
SOG sec. 4.10 (p.23-25) gives latency, throughput and dual issue but, unlike other cores' guides,
no pipeline column (the A55 SOG has none anywhere). Unclear: whether an atomic blocks the load pipe,
the store pipe, or both, and for how long. Current handling: `CortexA55WriteAtomic<lat, occ>` holds
`CortexA55UnitLd` **and** `CortexA55UnitSt` for `1/T` cycles each (an atomic is a load and a store),
`BeginGroup`. Expected impact: atomics are rare in the evaluated loops (not verified). How to
settle: on hardware, LDADD interleaved with independent LDR or STR.

<a id="u-6"></a>
**U-6. ST4, one lane, S and D.**
SOG sec. 4.19 A64 (p.40, l.2289) has only "ASIMD store, 4 element, one lane, B/H 2, 1/2, 01"; S and
D element sizes have no row (AArch32 "VST4, 4 element, one lane 2, 1/2, 01", p.39, l.2232).
Current handling: `ST4i(32|64)` keep the former `CortexA55WriteVST2` (latency 5, store pipe 2
cycles, no `BeginGroup`), post-index with `WriteAdr`. Expected impact: negligible (rare). How to
settle: hardware measurement, or adopt the B/H row (2, 1/2, 01) by analogy.

<a id="u-7"></a>
**U-7. Q-forms of SHL, SSHR, USHR, SHRN2.**
SOG sec. 4.15 A64 (p.32, l.1760-1761): "ASIMD shift by immed SHL, SHRN(2), SLI, SRI, SSHR, USHR 2, 2,
11" with no asterisk, i.e. throughput 2 and both slots for the Q-forms too; the corresponding AArch32
row "VSHL, VSHR, VSHRN 2, 2*, 11*" (p.30, l.1638) has the asterisk. Unclear: whether the missing
asterisk in the A64 row is intended. Current handling: the A64 row literally (`AluVd_2` for every
form; Q-form throughput 2). Measured: four independent `shl v.4s` + branch 2.5 cycles/iteration,
against 4.0 for `add v.4s`. How to settle: hardware measurement of Q-form `shl` throughput.

<a id="u-8"></a>
**U-8. Which UBFM/SBFM encodings are "sign/zero extend".**
SOG sec. 4.7 A64 (p.18, l.907): "Sign/zero extend SXTB, UXTB 1, 2, 11" (only B; AArch32 lists SXTB,
SXTH, UXTB, UXTH, p.18, l.893). Unclear: whether SXTH/UXTH/SXTW and the X-register zero extensions
(`UBFM Xd, Xn, #0, #7/#15/#31`, printed as UBFX) are 1 or "Bitfield move" 2. Current handling:
`CortexA55IsExtendAlias` (immr = 0 and imms in {7, 15, 31}) gives 1 to all of them, for UBFM and
SBFM, W and X. Expected impact: small; SXTW and UXTH are common, so not negligible (not measured).
How to settle: hardware measurement of `sxtw` and `uxth` chains.

<a id="u-9"></a>
**U-9. When store data is needed.**
*Status: decided 2026-10-31: the natural reading is adopted (store data is needed one cycle later
than an ordinary operand: `ReadST` advance 1 from every producer, for all stores, [C-18](#c-18));
hardware measurement is still the way to settle it.*
SOG: stores "do not need to wait for data operands (which are required at wr)" (sec. 4.9, p.22,
l.1108-1109); a dedicated path "from the ex2 stage of the ALU0 to the ex2 stage of the store
pipeline" (sec. 3.1.1, l.351-352); but sec. 5.8 recommends separating a load and the corresponding
store "to avoid interlocks on the store source registers" ([I-6](#i-6)). Unclear: how many cycles
earlier than an ordinary operand the store data is needed, and whether an ALU instruction-0 can feed
the data of a store instruction-1 in the same cycle. Taken literally, "required at wr" would be two
stages after ex1 (an advance of 2 against an ALU operand); the ALU0 ex2 -> store ex2 path and FP
"required at f2" (one stage after the f1 operand read) point to one stage. The ASIMD stores (sec.
4.19, l.2193) give no stage. Current handling (since 2026-10-31): advance 1 for the data of every
store with an SOG row (before: register-offset stores only); with it, an ALU
instruction-0 and a store instruction-1 that stores its result pair in one cycle (an ALU result of
latency 1 is seen at 0). Not covered: atomics and STXR/STXP (no SOG row). How to settle: hardware
measurement of `ldr; str` and `add; str` with the stored register dependent.

<a id="u-10"></a>
**U-10. The flag bypass from a 2-cycle flag setter.**
SOG Table 2 "Conditional" (p.11, l.452-457) does not limit the flag-setting instruction-0 by
latency. Unclear: whether a conditional instruction-1 can pair with a flag setter of exec latency 2
(ADDS/SUBS with shifted or extended register, sec. 4.3 "ALU, extend and/or shift" 2). Current
handling: `a55IsFlagBypassEligible` ignores the producer's latency, so the pair issues together.
Measured: `adds x1, x2, x3, lsl #2; csel; csel` + branch 2.0 cycles/iteration. How to settle:
hardware measurement with `adds ..., lsl #2; b.eq` loops.

<a id="u-11"></a>
**U-11. The "(1/T)" base-update latency of writeback stores and ASIMD loads/stores.**
SOG: "(Store, writeback form) - (1/T) Same as before" with note 3 "update latency shown in
parentheses" (sec. 4.9, p.23, l.1182-1190); the same in sec. 4.14 (l.1557-1565), 4.18 (l.2183,
note 1 without "parentheses", l.2189) and 4.19 (l.2291-2295). Unclear: whether "(1/T)" means "the
reciprocal of the row's throughput" (the reading taken) and how it applies to rows with throughput
1 (latency 1) and to loads, where note 1 says only "shorter latency". Current handling:
`WriteAdr` 1, `CortexA55WriteAdr2` for STP Q (T = 1/2), `CortexA55WriteVAdr<1/T>` for the ASIMD
rows (e.g. LD1 four Q registers: 8, test `AsimdLoadsStores`).

<a id="u-12"></a>
**U-12. Exclusives without acquire/release.**
The A64 tables list LDAXR/LDAXP and STLXR/STLXP but not LDXR, LDXP, STXR, STXP. Current handling:
the stock classes; `mca-insts-info` reports LDXRX latency 3 on the load pipe, LDXPX latency 4 with
the load pipe 2 cycles, STXRX latency 4 on the load and store pipes (dual-issue bits not checked).
Not checked against any SOG row. How to settle: hardware measurement, or adopt the acquire/release
rows by analogy.

<a id="u-13"></a>
**U-13. Which pipe a NOP uses, and which hints are "Architectural NOP".**
SOG: the only mention of NOP is Table 2 (sec. 3.2, p.11, l.458), "Miscellaneous: IT, Architectural
NOP", read as "can be instruction-1"; Table 1 does not restrict it as instruction-0; sec. 4 has no
row for NOP or any other hint. Unclear: (a) which pipe, if any, a NOP occupies, and its throughput
(beyond the pairing rules, the SOG gives none); (b) whether "Architectural NOP" covers only `nop`
or also the hint-space instructions that the Cortex-A55 (Armv8.2) executes as NOPs, such as BTI
(Armv8.5), YIELD, SEV, SEVL, CSDB. *Status: the pointer-authentication hints (Armv8.3) are decided:
NOPs, 2026-11-02.* Current handling (`[a55_sog_n8]`, `[a55_pac_hint]`, [C-3](#c-3)): the `NOP`
opcode and the pointer-authentication hints PACIAZ, PACIBZ, PACIASP, PACIBSP,
PACIA1716, PACIB1716, AUTIAZ, AUTIBZ, AUTIASP, AUTIBSP, AUTIA1716, AUTIB1716 and XPACLRI have no
resource and no group bits (the Cortex-A55 has no FEAT_PAuth, so they execute as NOPs), so two of
them, or one and a branch, pair; every other hint keeps `WriteHint` on `CortexA55UnitB` (one per
cycle, no pairing with a branch): `HINT` (YIELD, WFE, WFI, SEV, SEVL, BTI, CSDB, ...). Side note found while checking: WFE
and WFI are `HINT` in LLVM, not `WriteSys`, so they do not get the `SingleIssue` that [C-3](#c-3)
attributes to the control instructions of Table 1 (recorded as [N-3](#n-3), not changed). Expected impact: small; NOPs
are mostly alignment padding. How to settle (the remaining hints): on hardware, loops of `nop` /
`bti c` pairs with a branch and with each other.

<a id="u-14"></a>
**U-14. Compare-and-branch (CBZ, CBNZ, TBZ, TBNZ) as instruction-1: which row applies.**
*Status: decided 2026-10-31 to keep the current handling.*
Table 2 (p.10, l.436-438): "Most branches can be dual-issued from this position except compare and
branch instructions." Sec. 4.2 A64 (p.14, l.676): "Compare and branch CBZ, CBNZ, TBZ, TBNZ 1, 1, 11"
(both slots). The AArch32 row "Compare and branch CBZ, CBNZ 1, 1, 01" (l.662) agrees with Table 2.
The two readings give `01` (slot 0 only, Table 2 and the AArch32 row) or `11` (sec. 4.2 A64).
Current handling: `11` (`WriteBr`, no `BeginGroup`), i.e. the A64 instruction table, which is the SOG's
entry for these exact A64 instructions. Changing it to `01` would follow Table 2. How to settle:
hardware measurement of `cbz` pairs and `add; cbz` in the second slot.

<a id="u-15"></a>
**U-15. Pointer chasing from LDAXP: which register.**
*Status: recorded 2026-10-31.*
SOG sec. 4.8 (p.21): "Load acquire exclusive, pair LDAXP 3(2) 1 00", note 1 (the pointer-chasing
path, "See section 3.3"). Table 3 (sec. 3.3, p.11, l.492-495) lists only "LDR & LDRT/LDTR" and
"LDRD/LDP (all variants)", with the per-register rule for LDP. Unclear: whether LDAXP follows the LDP
rule (first register of a 32-bit pair, second of a 64-bit pair) or both registers have the path.
Current handling: both registers (`CortexA55WriteLdAXP`, `CortexA55WriteLdAXPb` in the
`ReadAdrBase` list), as before. Expected impact: negligible (LDAXP feeding an address is rare). How
to settle: adopt the LDP rule by analogy (a project decision), or hardware measurement.

<a id="u-16"></a>
**U-16. Which load/store bases take the pointer-chasing path.**
*Status: recorded 2026-10-31.*
SOG sec. 3.3 (p.11, l.486-489): load data "can be forwarded ... to either the load or store AGU base
operand". The SOG does not list the consumer instructions. Current handling ([C-7](#c-7)): every load
and store form that has an SOG row, including the ASIMD LD1-LD4 / ST1-ST4 rows of sec. 4.18/4.19,
whose base is the same AGU base operand (the SOG does not name the ASIMD forms explicitly). Not
covered: atomics (sec. 4.10), PRFM, and the exclusives without acquire/release LDXR/LDXP/STXR/STXP
([U-12](#u-12)), whose base waits the full load latency. Expected impact: small (pointer chasing into
these is rare). How to settle: hardware measurement of `ldr x1, [x1]; ld1 {v0.8b}, [x1]` and
`ldr x1, [x0]; ldadd x2, x3, [x1]` chains, or a project decision to extend the rule.

---

## 7. Inconsistent within the SOG (I)

<a id="i-1"></a>
**I-1. Multiply and divide latencies: sec. 3.4 text vs. sec. 4.5 table.**
Sec. 3.4 (p.12, l.540-545): in AArch64 "32-bit multiplies take one cycle", "64-bit multiplies take
two or three cycles", "SMULH/UMULH takes four cycles", divides "up to 12" / "up to 20" cycles.
Sec. 4.5 (p.16-17, l.794-819): MADD 32-bit "3 (1)", 64-bit "4-5 (2-3)", SMULH/UMULH "6", throughput
1/3, divides "3 - 12 (11)" / "3 - 20 (19)". The text's multiply numbers match the table's
parenthesised accumulate latencies (1, 2-3) rather than the exec latencies; SMULH (4 vs. 6) matches
neither. Implementation: the table ([C-15](#c-15)).

<a id="i-2"></a>
**I-2. Load-use latency: 2 in sec. 3.3, "3 (2)" in the tables.**
Sec. 3.3 (p.11, l.484-485): load-use to the ALU "is two cycles". Sec. 4.8 (p.21): "3 (2)", with note 1
saying the number in parentheses is the pointer-chasing path (l.1088-1089); sec. 4.1 defines Exec
latency as the *minimum* latency seen by a dependent operation (l.640-641), which would be 2.
Implementation: 2 to ALU operands ([C-6](#c-6)), to load and store bases (from the loads of Table 3,
[C-7](#c-7)) and to store data ([C-18](#c-18)); 3 to everything else (shifter operands
[D-10](#d-10), multiplicands, FP).

<a id="i-3"></a>
**I-3. SLI/SRI in two rows of sec. 4.15.**
A64 (p.32, l.1760-1762): "ASIMD shift by immed SHL, SHRN(2), SLI, SRI, SSHR, USHR 2, 2, 11" and
"ASIMD shift by immed and insert SLI, SRI 2, 2*, 11*, note 1". The rows disagree for the Q-forms.
Implementation: the second row for the vector forms (`S(LI|RI)v..._shift`: `AluVd_2` / `AluVq_2`),
`AluVd_2` for the scalar SLId/SRId.

<a id="i-4"></a>
**I-4. Shift by immediate: 1 in AArch32, 2 in AArch64.**
AArch32 "Move, shift by immed ASR, LSL, LSR, ROR, RRX 1" (sec. 4.4, p.15, l.729); AArch64 has only
"Bitfield move, basic SBFM, UBFM 2" and "Bitfield extract EXTR 2" (sec. 4.7, p.18, l.905, 909), which
include the shift aliases; sec. 5.7 (p.43, l.2455-2456) says there are "no known cases where two
similar instructions with the same result behave differently". Implementation: 1 ([D-3](#d-3),
[D-4](#d-4)).

<a id="i-5"></a>
**I-5. "Exec latency 1" of ALU basic vs. the dedicated ALU0 -> ALU1 path.**
Sec. 4.3 (p.15, l.713) gives ALU basic 1, and sec. 4.1 defines Exec latency as the minimum latency
seen by a dependent operation (l.640-641). The dedicated ex1 -> ex1 path of sec. 3.1.1 (l.351)
lets a dependent instruction-1 execute in the same cycle (effective 0) if the path connects the two
instructions of a pair. Implementation: 1 in the table, 0 within a pair ([D-6](#d-6)).

<a id="i-6"></a>
**I-6. Store data: sec. 5.8 vs. sec. 4.9.**
Sec. 4.9 (p.22, l.1108-1109): stores "do not need to wait for data operands (which are required at
wr)". Sec. 5.8 (p.43, l.2478-2479): "Separate the load and corresponding store instruction by at
least one other instruction to avoid interlocks on the store source registers." If data were needed
only at wr, a load followed directly by the store of its result would hardly interlock.
Implementation: sec. 4.9 (store data needed one cycle later, for every store, [C-18](#c-18),
[U-9](#u-9)); with it a load directly followed by the store of its result still waits one cycle
(load latency 3, advance 1), so sec. 5.8's advice keeps a point.

<a id="i-7"></a>
**I-7. Pointer chasing from sign/zero-extending loads.**
Sec. 3.3 (p.11, l.489-490): the pointer-chasing path "In general ... is limited to load
instructions that do not require sign/zero extension, but more detail is provided in the following
table"; Table 3 (l.493): "LDR & LDRT/LDTR (all variants)". Sec. 4.8 A64 (p.21, l.1051-1063) gives
"3 (2)" (note 1) to rows that include LDRB, LDRH, LDRSB, LDRSH, LDRSW, LDURSW, LDTRSW.
Implementation: the tables (every scalar load is a producer); measured `ldrsw x1, [x1]` and
`ldrb w1, [x1]` chains: 2.0 cycles/iteration.

<a id="i-8"></a>
**I-8. ASIMD register move: 2 in A64, 1 in AArch32.**
Sec. 4.15 A64 "ASIMD logical MOV 2, 2*, 11*" (p.31, l.1709); A64 ORR (the instruction MOV is an
alias of) "1, 2*, 11*" (l.1707-1708); AArch32 sec. 4.17 "ASIMD move, register VMOV 1, 2*, 11*"
(p.34, l.1925); sec. 5.7 (l.2455-2456). Implementation: since 2026-10-31 the A64 MOV row, 2 for
the register-form ORR with Rn == Rm and 1 for any other ORR ([C-25](#c-25)); before, 1 for all.

<a id="i-9"></a>
**I-9. "Minimum" vs. "worst-case" latency.**
Sec. 4.1 (p.14, l.640-641): Exec latency is the *minimum* latency seen by a dependent operation.
Sec. 4.12 note 1 (p.27, l.1447) and sec. 4.17 note 3 (p.36, l.2038): "Latency number refers to the
worst-case latency from the result to a dependent instruction"; sec. 4.13 (p.27, l.1451-1452) and
sec. 4.18 (p.36, l.2042-2043): "the worst-case load-use latency". Implementation: the numbers as
printed.

<a id="i-10"></a>
**I-10. Misprints.**
Sec. 4.16 A64 "ASIMD FP convert, other FCVTAS, VCVTAU, ..." (p.33, l.1856): VCVTAU is an AArch32
mnemonic (FCVTAU meant). Sec. 5.10 (p.44, l.2535): "ldrp x0, [x0, #lo12 <const>]" (LDR meant).
Sec. 4.7 A64 "Reverse bytes REV, REV16, REVSH" (p.18, l.917): REVSH is AArch32; A64 has REV32.
Implementation: the evident meaning.

---

## 8. Reproducing the cycle counts quoted here

The cycles/iteration figures in sections 4-6 that are not from `tests/a55_sog_test.cpp` were
measured by the author on 2026-10-31 with a private build of the working tree (CMake, Release; not
`build/`), as follows:

```sh
# body: the instructions of one iteration, e.g. "ldr x1, [x0]" "str x2, [x1]"
{ echo '.text'; echo '.globl f'; echo '.type f,%function'; echo 'f:'; echo '1:';
  printf '  %s\n' "${body[@]}"; echo '  b 1b'; echo '  ret'; } > t.s
aarch64-linux-gnu-gcc -march=armv8.2-a+crypto+lse -nostdlib -static -Wl,-e,f t.s -o t
mca_tool --mcpu=cortex-a55 --mtriple=aarch64 --iterations=200 t
# cycles/iteration = cycles / (retired_instructions / length), first CSV row (the loop)
```

The loop branch `b 1b` is part of every body (so "four `fcmp`" is five instructions). The working
tree was being changed by other sessions at the time; re-measure before quoting a number.

---

## 9. Maintaining this document

* Update an item when the SOG reading or the implementation changes. IDs run from 1 without gaps
  within each category, in the order of the items; an item whose difference has been fixed becomes a
  C item, and the code, tests and other documents that name an ID are updated with it.
* When a deviation is introduced on purpose, add a D item, and tag the code with a comment that names
  the ID.
* Evaluation results (accuracy metrics, totals over benchmark sets) are not recorded in this document;
  cycles/iteration only when reproduced with `mca_tool` as in section 8.
* [docs.md](docs.md) holds the rest of the technical documentation (other cores, the MCA steady-state
  machinery, MLP, Facile, the scheduling-model plumbing); it links here for the Cortex-A55.
  [README.md](README.md) lists the documents.
