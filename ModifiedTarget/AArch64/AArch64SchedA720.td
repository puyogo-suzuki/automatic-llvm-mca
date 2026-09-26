//=- AArch64SchedA720.td - ARM Cortex-A720 Scheduling Definitions -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for ARM Cortex-A720 (P-Core).
// Reference & Source Attribution:
//   Arm Cortex-A720 Core Software Optimization Guide (109720)
//   https://developer.arm.com/documentation/109720/latest/
//
//===----------------------------------------------------------------------===//

def CortexA720Model : SchedMachineModel {
  let IssueWidth = 5;           // 5 MOPs per cycle decode / dispatch
  let MicroOpBufferSize = 160;  // Reorder Buffer (ROB)
  let LoadLatency = 3;          // L1 hit latency (3 cycles from Cortex-A720 SOG)
  let MispredictPenalty = 11;
  let CompleteModel = 0;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC]);
}

//===----------------------------------------------------------------------===//
// Define Cortex-A720 execution pipelines matching SOG Section 2.1 (13 Issue Pipelines)

// 2 Branch pipelines (Branch 0, Branch 1)
def A720UnitB0 : ProcResource<1> { let BufferSize = 16; }
def A720UnitB1 : ProcResource<1> { let BufferSize = 16; }
def A720UnitB  : ProcResource<2> { let BufferSize = 32; }

// 4 Integer pipelines (2 Single-Cycle, 2 Single/Multi-Cycle ALU/MUL/DIV)
def A720UnitI0 : ProcResource<1> { let BufferSize = 20; }
def A720UnitI1 : ProcResource<1> { let BufferSize = 20; }
def A720UnitM0 : ProcResource<1> { let BufferSize = 20; } // MUL/DIV
def A720UnitM1 : ProcResource<1> { let BufferSize = 20; } // MUL
def A720UnitI  : ProcResource<4> { let BufferSize = 80; }

def A720UnitMul : ProcResource<2> { let BufferSize = 40; } // M0 + M1

// 3 AGU Load/Store pipelines (2 Load/Store, 1 Load-only)
def A720UnitL0 : ProcResource<1> { let BufferSize = 24; }
def A720UnitL1 : ProcResource<1> { let BufferSize = 24; }
def A720UnitL2 : ProcResource<1> { let BufferSize = 24; }
def A720UnitLS : ProcResource<3> { let BufferSize = 72; }

// 2 Store Data pipelines
def A720UnitSD : ProcResource<2> { let BufferSize = 32; }

// 2 FP/ASIMD Vector pipelines
def A720UnitV0 : ProcResource<1> { let BufferSize = 24; }
def A720UnitV1 : ProcResource<1> { let BufferSize = 24; }
def A720UnitV  : ProcResource<2> { let BufferSize = 48; }

//===----------------------------------------------------------------------===//
// Zero-latency instructions.
//
// Arm Cortex-A720 Core Software Optimization Guide (109720, Issue 7.0)
// sec. 4.12 "Zero Latency Instructions":
//
//   "A subset of register-to-register move operations, move immediate
//    operations, predicates operations are executed with zero latency.  These
//    instructions do not utilize the scheduling and execution resources of the
//    machine.  These are as follows:
//      MOV Xd, #{12{1'b0},imm[3:0]}   MOV Xd, XZR
//      MOV Wd, #{12{1'b0},imm[3:0]}   MOV Wd, WZR
//      MOV Hd, WZR                    MOV Hd, XZR
//      MOV Sd, WZR                    MOV Dd, XZR
//      MOVI Dd, #0                    MOVI Vd.2D, #0
//      MOV Wd, Wn                     MOV Xd, Xn
//      FMOV Sd, Sn                    FMOV Dd, Dn
//      MOV Vd, Vn (vector)            MOV Zd.D, Zn.D
//      PTRUE                          PFALSE                 SETFFR"
//
//   "The MOV Wd, Wn, MOV Xd, Xn and FMOV Sd, Sn, FMOV Dd, Dn, MOV Vd, Vn
//    (vector), MOV Zd.D, Zn.D instructions may not be executed with zero
//    latency under certain conditions."
//
// Both halves of that quote are modelled the only way a static model can: the
// listed forms get Latency = 0 and NO ProcResource (which is literally the
// SOG's "do not utilize the scheduling and execution resources of the
// machine"), and the "may not be ... under certain conditions" caveat is
// accepted as unmodellable - the SOG names no condition, and a static
// scheduling model has nowhere to express one.  Upstream LLVM makes the same
// choice for the identically-worded caveat in the Neoverse N2/V1/N3 models.
//
// WHY THIS IS AN A720-LOCAL PREDICATE AND NOT stock LLVM's NeoverseZeroMove.
// AArch64SchedPredNeoverse.td's NeoverseZeroMove was written for Neoverse
// N2/V1, whose SOGs carry a SHORTER list than A720 sec. 4.12.  Reusing it
// verbatim here was checked opcode by opcode and is wrong in three places -
// two of them under-coverage, one over-coverage:
//
//   1. MOVZ IMMEDIATE RANGE (under-coverage, the substantive one).
//      NeoverseZeroMove requires CheckImmOperand<1, 0>, i.e. LITERALLY
//      MOV Wd/Xd, #0.  A720 sec. 4.12 says MOV Xd, #{12{1'b0},imm[3:0]}:
//      a 16-bit MOVZ immediate field whose top 12 bits are zero, i.e. ANY
//      value 0..15, not just 0.  Verified against the operand layout llvm-mc
//      actually produces (llvm-mc -triple=aarch64 -show-inst):
//        mov x2, #15  ->  <MCInst MOVZXi <Reg:X2> <Imm:15> <Imm:0>>
//      so operand 1 is the imm16 field and operand 2 the hw shift.
//      CheckImmOperandRange<1, 0, 15> is therefore the faithful encoding of
//      sec. 4.12; CheckImmOperand<1, 0> would silently drop 15/16ths of the
//      documented range.  The hw shift is still required to be 0
//      (CheckImmOperand<2, 0>): sec. 4.12 quotes a bare 16-bit field with no
//      shift, and a shifted MOVZ materialises a large constant, which is not a
//      value the rename stage can be assumed to fabricate for free.  That is
//      the deliberately conservative reading of an ambiguous line, flagged
//      here rather than left silent.
//
//   2. FMOV Sd, Sn / FMOV Dd, Dn (under-coverage).  NeoverseZeroMove contains
//      no FP-register-to-FP-register FMOV at all - its FMOV arm is
//      CheckOpcode<[FMOVWHr, FMOVXHr, FMOVWSr, FMOVXDr]> with CheckIsReg1Zero,
//      which is the GPR-SOURCED form (sec. 4.12's MOV Hd,WZR / MOV Hd,XZR /
//      MOV Sd,WZR / MOV Dd,XZR), a different instruction.  Confirmed with
//      llvm-mc: "fmov s4, s5" is FMOVSr and "fmov d6, d7" is FMOVDr, while
//      "fmov s2, wzr" is FMOVWSr and "fmov d3, xzr" is FMOVXDr.  N2 and V1
//      both give FMOVSr/FMOVDr an unconditional 2-cycle V write
//      (N2Write_2c_1V / V1Write_2c_1V); only the newer NeoverseN3 model, whose
//      core is A720's contemporary, makes them N3Write_0c.  A720's own SOG
//      lists them explicitly, so they are included here.
//      NOTE the deliberate exclusion of FMOVHr (FMOV Hd, Hn): sec. 4.12 lists
//      FMOV Sd,Sn and FMOV Dd,Dn and NOT the H form, so FMOVHr keeps its
//      normal latency.  NeoverseN3 draws the same line.
//
//   3. MOV Zd.D, Zn.D / PTRUE / PFALSE / SETFFR (over-coverage, excluded).
//      NeoverseZeroMove's last-but-one arm is ORR_ZZZ, and sec. 4.12's SVE
//      forms would additionally need PTRUE/PFALSE/SETFFR.  CortexA720Model
//      declares UnsupportedFeatures = SVEUnsupported.F (above), so no SVE
//      instruction carries scheduling data in this model at all and every SVE
//      arm would be unreachable.  They are omitted rather than written as dead
//      rows.
//
// Everything else in sec. 4.12 - ORR[WX]rs with Rn == ZR (which is how
// MOV Wd,WZR, MOV Xd,XZR, MOV Wd,Wn and MOV Xd,Xn all encode; verified with
// llvm-mc: "mov x5, x6" -> ORRXrs <Reg:X5> <Reg:XZR> <Reg:X6> <Imm:0>), the
// GPR-to-FP FMOVs, MOVID/MOVIv2d_ns with a zero immediate, and ORRv16i8 /
// ORRv8i8 with Rn == Rm - matches NeoverseZeroMove exactly, and those arms are
// transcribed with the same operand indices.
//
// CheckIsImmOperand<1> guards the MOVZ range test because a MOVZ operand can
// be a relocation expression rather than a literal (e.g. mov x0, #:abs_g0:sym);
// NeoverseZeroMove guards the same way.
//
// The predicate sits outside the `let SchedModel` block below: an
// MCSchedPredicate has no SchedModel field.  AArch64SchedPredNeoverse.td is
// deliberately NOT modified, so the eight Neoverse models that share
// NeoverseZeroMove are untouched by this.

def A720ZeroMove : MCSchedPredicate<
                     CheckAny<[
                       // MOV Xd, #{12{1'b0},imm[3:0]}
                       // MOV Wd, #{12{1'b0},imm[3:0]}
                       CheckAll<[CheckOpcode<[MOVZWi, MOVZXi]>,
                                 CheckIsImmOperand<1>,
                                 CheckImmOperandRange<1, 0, 15>,
                                 CheckImmOperand<2, 0>]>,
                       // MOV Wd, WZR
                       // MOV Xd, XZR
                       // MOV Wd, Wn
                       // MOV Xd, Xn
                       CheckAll<[CheckOpcode<[ORRWrs, ORRXrs]>,
                                 CheckIsReg1Zero,
                                 CheckImmOperand<3, 0>]>,
                       // MOV Hd, WZR
                       // MOV Hd, XZR
                       // MOV Sd, WZR
                       // MOV Dd, XZR
                       CheckAll<[CheckOpcode<[FMOVWHr, FMOVXHr,
                                              FMOVWSr, FMOVXDr]>,
                                 CheckIsReg1Zero]>,
                       // MOVI Dd, #0
                       // MOVI Vd.2D, #0
                       CheckAll<[CheckOpcode<[MOVID, MOVIv2d_ns]>,
                                 CheckImmOperand<1, 0>]>,
                       // MOV Vd, Vn (vector)
                       CheckAll<[CheckOpcode<[ORRv16i8, ORRv8i8]>,
                                 CheckSameRegOperand<1, 2>]>,
                       // FMOV Sd, Sn
                       // FMOV Dd, Dn
                       CheckOpcode<[FMOVSr, FMOVDr]>
                     ]>>;

//===----------------------------------------------------------------------===//
// Target-wide SchedWrites mappings for Cortex-A720 Model

let SchedModel = CortexA720Model in {

def : WriteRes<WriteImm, [A720UnitI]> { let Latency = 1; }
def : WriteRes<WriteI, [A720UnitI]> { let Latency = 1; }
def : WriteRes<WriteISReg, [A720UnitI]> { let Latency = 1; }
def : WriteRes<WriteIEReg, [A720UnitI]> { let Latency = 1; }
def : WriteRes<WriteExtr, [A720UnitI]> { let Latency = 1; }
def : WriteRes<WriteIS, [A720UnitI]> { let Latency = 1; }

def : WriteRes<WriteIM32, [A720UnitMul]> { let Latency = 2; }
def : WriteRes<WriteIM64, [A720UnitMul]> { let Latency = 3; }

// Integer Divide (SOG Table 3-4 variable latencies: 32-bit 5-12c [median 8], 64-bit 5-20c [median 12])
def : WriteRes<WriteID32, [A720UnitM0]> { let Latency = 8; }
def : WriteRes<WriteID64, [A720UnitM0]> { let Latency = 12; }

def : WriteRes<WriteBr, [A720UnitB]> { let Latency = 1; }
def : WriteRes<WriteBrReg, [A720UnitB]> { let Latency = 1; }

def : WriteRes<WriteLD, [A720UnitLS]> { let Latency = 3; }
def : WriteRes<WriteST, [A720UnitLS, A720UnitSD]> { let Latency = 1; }
def : WriteRes<WriteSTP, [A720UnitLS, A720UnitSD]> { let Latency = 1; }
def : WriteRes<WriteAdr, [A720UnitI]> { let Latency = 1; }
def : WriteRes<WriteLDIdx, [A720UnitLS]> { let Latency = 3; }
def : WriteRes<WriteSTIdx, [A720UnitLS, A720UnitSD]> { let Latency = 1; }

def : WriteRes<WriteLDHi, [A720UnitLS]> { let Latency = 3; }
def : WriteRes<WriteSys, [A720UnitI0]> { let Latency = 10; }
def : WriteRes<WriteBarrier, [A720UnitLS]> { let Latency = 10; }
def : WriteRes<WriteHint, [A720UnitI0]> { let Latency = 1; }

def : WriteRes<WriteF, [A720UnitV]> { let Latency = 3; }
def : WriteRes<WriteFCmp, [A720UnitV0]> { let Latency = 2; }
def : WriteRes<WriteFCvt, [A720UnitV]> { let Latency = 3; }
def : WriteRes<WriteFCopy, [A720UnitV]> { let Latency = 1; }
def : WriteRes<WriteFImm, [A720UnitV]> { let Latency = 1; }
def : WriteRes<WriteFMul, [A720UnitV]> { let Latency = 3; }

// Floating-Point Divide (SOG Table 3-12 variable latencies: 32-bit 7-10c [median 8], 64-bit 7-15c [median 11])
def : WriteRes<WriteFDiv, [A720UnitV0]> { let Latency = 10; }

def : WriteRes<WriteVd, [A720UnitV]> { let Latency = 2; }
def : WriteRes<WriteVq, [A720UnitV]> { let Latency = 2; }
def : WriteRes<WriteVLD, [A720UnitLS]> { let Latency = 3; }
def : WriteRes<WriteVST, [A720UnitLS, A720UnitV]> { let Latency = 1; }

def : WriteRes<WriteAtomic, []> { let Unsupported = 1; }

def : ReadAdvance<ReadI, 0>;
def : ReadAdvance<ReadISReg, 0>;
def : ReadAdvance<ReadIEReg, 0>;
def : ReadAdvance<ReadIM, 0>;
def : ReadAdvance<ReadIMA, 0>;
def : ReadAdvance<ReadID, 0>;
def : ReadAdvance<ReadST, 0>;
def : ReadAdvance<ReadExtrHi, 0>;
def : ReadAdvance<ReadAdrBase, 0>;
def : ReadAdvance<ReadVLD, 0>;

//===----------------------------------------------------------------------===//
// Zero-latency writes and the InstRW rows that bind A720ZeroMove (see the
// sec. 4.12 commentary above the predicate).
//
// A720Write_0c is the "renamed away" write: Latency 0 and SchedWriteRes<[]>,
// i.e. no ProcResource at all, matching sec. 4.12's "do not utilize the
// scheduling and execution resources of the machine".  It is used ONLY as the
// A720ZeroMove arm of the three variants below and nowhere else in this file,
// so within CortexA720Model "this write has Latency 0" is equivalent to "this
// instruction is a sec. 4.12 zero-latency form".  That property is what
// facile.cpp's getEdgeLatency() gate relies on; it is NOT a micro-op
// accounting marker of the N1Write_0c_0Z kind (which N1 uses for the second
// destination of a load pair, where the value genuinely is not available
// early), and no read-modify-write instruction such as MOVK is bound to it.
//
// The NoSchedPred fallback arms are the non-eliminated latencies from the
// A720 SOG's own instruction tables, not copies of N2's:
//   * A720Write_1c_1I  - Table 3-3 "ALU, basic" (ADD, ADC, AND, BIC, EON,
//     EOR, ORN, ORR, SUB, SBC): latency 1, throughput 4, pipeline I; and
//     Table 3-5 "Move immed" (MOVN, MOVK, MOVZ): latency 1, throughput 4,
//     pipeline I.  Covers ORR[WX]rs and MOVZ[WX]i.
//   * A720Write_3c_1M0 - Table 3-12 "FP transfer, from gen to low half of vec
//     reg" (FMOV): latency 3, throughput 1, pipeline M0.  Covers the
//     GPR-sourced FMOV[WX][HSD]r forms.
//   * A720Write_2c_1V  - Table 3-12 "FP move, register" (FMOV): latency 2,
//     throughput 2, pipeline V, with note 1 "Particular FMOV #0 or Register to
//     Register can be optimized in rename stage pipeline, execution latency
//     and throughput are then not representative" - the table's own pointer at
//     sec. 4.12.  Also Table 3-15 "ASIMD move, integer immed" (MOVI, MVNI):
//     latency 2, throughput 2, pipeline V, and Table 3-14 "ASIMD logical"
//     (AND, BIC, EOR, ORR): latency 2, throughput 2, pipeline V.  Covers
//     FMOV[SD]r, MOVID/MOVIv2d_ns and ORRv16i8/ORRv8i8.
// A720UnitI / A720UnitM0 / A720UnitV are this model's own pipelines declared
// above; N2's unit names and counts are different and are not reused.
//
// FMOVSr/FMOVDr are routed through a variant rather than bound straight to
// A720Write_0c (the way NeoverseN3 does it) purely so that A720ZeroMove stays
// the single place sec. 4.12's list is written down; the predicate arm for
// those two opcodes is unconditional, so the variant always resolves to 0c for
// them.

def A720Write_0c     : SchedWriteRes<[]>           { let Latency = 0; }
def A720Write_1c_1I  : SchedWriteRes<[A720UnitI]>  { let Latency = 1; }
def A720Write_2c_1V  : SchedWriteRes<[A720UnitV]>  { let Latency = 2; }
def A720Write_3c_1M0 : SchedWriteRes<[A720UnitM0]> { let Latency = 3; }

def A720Write_0or1c_1I : SchedWriteVariant<[
                      SchedVar<A720ZeroMove, [A720Write_0c]>,
                      SchedVar<NoSchedPred,  [A720Write_1c_1I]>]>;

def A720Write_0or2c_1V : SchedWriteVariant<[
                      SchedVar<A720ZeroMove, [A720Write_0c]>,
                      SchedVar<NoSchedPred,  [A720Write_2c_1V]>]>;

def A720Write_0or3c_1M0 : SchedWriteVariant<[
                      SchedVar<A720ZeroMove, [A720Write_0c]>,
                      SchedVar<NoSchedPred,  [A720Write_3c_1M0]>]>;

// MOV Wd/Xd, #{12{1'b0},imm[3:0]}
def : InstRW<[A720Write_0or1c_1I],  (instrs MOVZWi, MOVZXi)>;
// MOV Wd,WZR / MOV Xd,XZR / MOV Wd,Wn / MOV Xd,Xn
def : InstRW<[A720Write_0or1c_1I],  (instrs ORRWrs, ORRXrs)>;
// MOV Hd,WZR / MOV Hd,XZR / MOV Sd,WZR / MOV Dd,XZR
def : InstRW<[A720Write_0or3c_1M0], (instrs FMOVWHr, FMOVXHr,
                                            FMOVWSr, FMOVXDr)>;
// MOVI Dd,#0 / MOVI Vd.2D,#0
def : InstRW<[A720Write_0or2c_1V],  (instrs MOVID, MOVIv2d_ns)>;
// MOV Vd,Vn (vector)
def : InstRW<[A720Write_0or2c_1V],  (instrs ORRv16i8, ORRv8i8)>;
// FMOV Sd,Sn / FMOV Dd,Dn  (FMOVHr deliberately absent - not in sec. 4.12)
def : InstRW<[A720Write_0or2c_1V],  (instrs FMOVSr, FMOVDr)>;

} // SchedModel = CortexA720Model
