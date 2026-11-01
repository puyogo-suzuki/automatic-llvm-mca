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
  // Reorder Buffer (ROB) = 192 entries.  NOT from the SOG, which gives no ROB size.
  // Source: Cardyak's Microarchitecture Cheat Sheet (unofficial, secondary; the author
  // notes information on the 2023 Arm cores is scarce).  Another secondary source
  // (a 2026 blog) says 160, which is also the value stock LLVM's N2 model (the model
  // LLVM assigns to cortex-a720) carries, and which this file used before.
  // Sets the MLP window width (and MicroOpQueueSize) for the A720.
  // docs/a720-sog.md U-1.
  let MicroOpBufferSize = 192;
  // L1 hit latency: SOG Table 4-4 "Level-1 Cache Hit 4 core cycles" (l.3259) and the load tables
  // (4, l.792-850).  Was 3.  Not read by the --facile path (the load rows carry the latency).
  let LoadLatency = 4;
  let MispredictPenalty = 11;   // not from the SOG; not read by the --facile path (U-1)
  let CompleteModel = 0;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC]);
}

//===----------------------------------------------------------------------===//
// Define Cortex-A720 execution pipelines matching SOG Section 2.1 (13 Issue Pipelines)

// 2026-10-30 (user instruction, SOG-only values):
// the pipelines are now defined as the SOG's pipeline symbols (Table 3-1, a720sog.txt l.588-600) with
// ProcResGroups instead of independent ProcResource counters (before: A720UnitI was a separate
// ProcResource<4> and A720UnitMul a separate ProcResource<2>, not sharing any unit with I0/I1/M0/M1).
//   B = Branch 0/1; S = Integer single-cycle 0/1; M = Integer single/multicycle 0/1, M0 = multicycle 0;
//   I = S + M; L01 = Load/Store 0/1; L = L01 + Load 2; ID = integer store data 0/1; V = FP/ASIMD 0/1.
// "Text line references below: lN = line N of automatic-llvm-mca/tmp/a720sog.txt."
let SchedModel = CortexA720Model in {
def A720UnitB0 : ProcResource<1> { let BufferSize = 16; }
def A720UnitB1 : ProcResource<1> { let BufferSize = 16; }
def A720UnitB  : ProcResGroup<[A720UnitB0, A720UnitB1]> { let BufferSize = 32; }
def A720UnitI0 : ProcResource<1> { let BufferSize = 20; }   // S0
def A720UnitI1 : ProcResource<1> { let BufferSize = 20; }   // S1
def A720UnitM0 : ProcResource<1> { let BufferSize = 20; }   // M0 (MUL/DIV)
def A720UnitM1 : ProcResource<1> { let BufferSize = 20; }   // M1
def A720UnitS  : ProcResGroup<[A720UnitI0, A720UnitI1]> { let BufferSize = 40; }
def A720UnitM  : ProcResGroup<[A720UnitM0, A720UnitM1]> { let BufferSize = 40; }
def A720UnitI  : ProcResGroup<[A720UnitI0, A720UnitI1, A720UnitM0, A720UnitM1]> { let BufferSize = 80; }
def A720UnitL0 : ProcResource<1> { let BufferSize = 24; }
def A720UnitL1 : ProcResource<1> { let BufferSize = 24; }
def A720UnitL2 : ProcResource<1> { let BufferSize = 24; }
def A720UnitL01 : ProcResGroup<[A720UnitL0, A720UnitL1]> { let BufferSize = 48; }
def A720UnitLS  : ProcResGroup<[A720UnitL0, A720UnitL1, A720UnitL2]> { let BufferSize = 72; }
def A720UnitSD : ProcResource<2> { let BufferSize = 32; }   // ID (integer store data 0/1)
def A720UnitV0 : ProcResource<1> { let BufferSize = 24; }
def A720UnitV1 : ProcResource<1> { let BufferSize = 24; }
def A720UnitV  : ProcResGroup<[A720UnitV0, A720UnitV1]> { let BufferSize = 48; }
// [a720_dividers] docs/a720-sog.md D-3: not a pipeline.  The integer divider inside M0, held for
// the divide's occupancy while M0 itself is taken one cycle: "Integer divides are performed using
// an iterative algorithm and block any subsequent divide operations until complete" (sec. 3.5
// note 1, l.693) names later divides, not every M0 uOP; "Each issue pipeline can accept one uOP
// per cycle" (l.477).  The decision of the Cortex-A76 and Cortex-A78 audits.
def A720UnitIDiv : ProcResource<1>;
}

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

// ALU, basic / flagset 1, 4/cycle, I (l.635-641); Move immed, bitfield, extract, CLZ, REV, variable shift 1, I (l.767-777)
def : WriteRes<WriteImm, [A720UnitI]> { let Latency = 1; }
def : WriteRes<WriteI, [A720UnitI]> { let Latency = 1; }
def : WriteRes<WriteISReg, [A720UnitI]> { let Latency = 1; }   // per-form variants below (l.642-662)
def : WriteRes<WriteIEReg, [A720UnitM]> { let Latency = 2; }   // ALU, extend and shift 2, M (l.642); basic form below
def : WriteRes<WriteExtr, [A720UnitI]> { let Latency = 1; }    // EXTR 1, I (l.768)
def : WriteRes<WriteIS, [A720UnitI]> { let Latency = 1; }      // SBFM/UBFM 1, I (l.769)

// Multiply accumulate W/X/long 2(1), 1/cycle, M0 (l.685-690); MUL forms below
def : WriteRes<WriteIM32, [A720UnitM0]> { let Latency = 2; }
def : WriteRes<WriteIM64, [A720UnitM0]> { let Latency = 2; }

// [a720_div_best] docs/a720-sog.md C-10, D-1: "Divide, W-form SDIV, UDIV 5 to 12 1/12 to 1/5 M0",
// "Divide, X-form 5 to 20 1/20 to 1/5 M0" (l.683-684), note 1 "Early termination is possible,
// depending upon the data values" (l.694): the best case, latency 5 and the divider held 5 cycles
// (1/5), for both forms.  [a720_dividers] D-3: M0 one cycle, the divider (A720UnitIDiv) 5.  Was
// latency 8 / 12 with M0 held 8 / 12 cycles (a point inside the range, not a SOG value).
def : WriteRes<WriteID32, [A720UnitM0, A720UnitIDiv]> { let Latency = 5; let ReleaseAtCycles = [1, 5]; }
def : WriteRes<WriteID64, [A720UnitM0, A720UnitIDiv]> { let Latency = 5; let ReleaseAtCycles = [1, 5]; }

def : WriteRes<WriteBr, [A720UnitB]> { let Latency = 1; }     // l.613-619
def : WriteRes<WriteBrReg, [A720UnitB]> { let Latency = 1; }

// Loads 4, 3/cycle, L (l.790-838); stores 1, 2/cycle, L01 + ID (l.870-899)
// [a720_store_uops] docs/a720-sog.md C-7: "Stores uOPs are split into address and data uOPs"
// (l.867-868; FP and ASIMD stores l.1110, l.1740): two uOPs, the address on L01 and the data on ID
// (V for FP/ASIMD stores).  Was one.
def : WriteRes<WriteLD, [A720UnitLS]> { let Latency = 4; }
def : WriteRes<WriteST, [A720UnitL01, A720UnitSD]> { let Latency = 1; let NumMicroOps = 2; }
def : WriteRes<WriteSTP, [A720UnitL01, A720UnitSD]> { let Latency = 1; let NumMicroOps = 2; }
def : WriteRes<WriteAdr, [A720UnitI]> { let Latency = 1; }    // writeback: "L, I" for pre/post-index (l.797-802)
def : WriteRes<WriteLDIdx, [A720UnitLS]> { let Latency = 4; }
def : WriteRes<WriteSTIdx, [A720UnitL01, A720UnitSD]> { let Latency = 1; let NumMicroOps = 2; }
def : WriteRes<WriteSTX, [A720UnitL01, A720UnitSD]> { let Latency = 1; let NumMicroOps = 2; }

// [a720_ldp_hi] docs/a720-sog.md D-10: the second destination of a load pair has the row's
// latency, no pipeline and no uOP (the generic write; the LDP rows below use their own).  Was L
// for one more cycle and one uOP.
def : WriteRes<WriteLDHi, []> { let Latency = 4; let NumMicroOps = 0; }
// [a720_sys] docs/a720-sog.md D-12: the SOG has no rows for MRS, MSR, barriers and other system
// instructions (sec. 4.10 lists execution constraints without cycle counts, l.3080-3145).  Lowest
// estimate: latency 1, no pipeline (as the Cortex-A78 model).  Was latency 10 on S0 / on L.
def : WriteRes<WriteSys, []> { let Latency = 1; }
def : WriteRes<WriteBarrier, []> { let Latency = 1; }
// [a720_nop] docs/a720-sog.md C-3: NOP and the other hints (BTI, ...) take no execution resource,
// the decision made for the Cortex-A55, A520, A76 and A78.  Was S0 for one cycle.
def : WriteRes<WriteHint, []> { let Latency = 1; }

// FP: FABS/FADD/FSUB/FMIN/FMAX/FNEG/FCSEL 2, 2/cycle, V (l.963-996); FCMP/FCCMP 2, 2/cycle, V (l.965)
def : WriteRes<WriteF, [A720UnitV]> { let Latency = 2; }
def : WriteRes<WriteFCmp, [A720UnitV]> { let Latency = 2; }
def : WriteRes<WriteFCvt, [A720UnitV0]> { let Latency = 3; }  // FCVT* to gen / FCVT vec-vec 3, V0 (l.1014-1038); SCVTF below
def : WriteRes<WriteFCopy, [A720UnitV]> { let Latency = 3; }  // FMOV vec->gen 3, V (l.1045); gen->vec below
def : WriteRes<WriteFImm, [A720UnitV]> { let Latency = 2; }   // FMOV immed 2, V (l.1039)
def : WriteRes<WriteFMul, [A720UnitV]> { let Latency = 3; }   // FMUL 3, V (l.983)

// Floating-Point Divide / Square Root.  The scalar forms are given per form below
// (A720 SOG: H 5, S 7, D 12 cycles, "performed using a fully pipelined data path");
// this generic write is only the fallback for forms with no row below.
def : WriteRes<WriteFDiv, [A720UnitV0]> { let Latency = 10; }

// ASIMD default: arith basic, compare, logical, max/min, FP arith/abs/neg/compare 2, 2/cycle, V (l.1206 ff., l.1347 ff.)
def : WriteRes<WriteVd, [A720UnitV]> { let Latency = 2; }
def : WriteRes<WriteVq, [A720UnitV]> { let Latency = 2; }
def : WriteRes<WriteVLD, [A720UnitLS]> { let Latency = 6; }   // ASIMD load, 1 element, multiple, 1 reg: 6, L (l.1620)
def : WriteRes<WriteVST, [A720UnitL01, A720UnitV]> { let Latency = 2; let NumMicroOps = 2; } // FP/ASIMD store data on V (l.1128 ff.); [a720_store_uops]

def : WriteRes<WriteAtomic, []> { let Unsupported = 1; }

def : ReadAdvance<ReadI, 0>;
def : ReadAdvance<ReadISReg, 0>;
def : ReadAdvance<ReadIEReg, 0>;
def : ReadAdvance<ReadIM, 0>;
def : ReadAdvance<ReadIMA, 0>;
def : ReadAdvance<ReadID, 0>;
// [a720_from_a55_store_data] docs/a720-sog.md B-1.  NOT SOG-based: the A720 SOG says only that
// stores are split into address and data uOPs (l.867-868, l.1110, l.1740) and gives no operand
// timing.  The project decided for the Cortex-A55 (SOG-based there) and the Cortex-A520 (borrowed)
// that store data is read one cycle late, and applied it to the Cortex-A76 and A78 for
// consistency; it is applied here likewise.  ReadST is attached to the data operands of every
// store form below (the exclusives excepted).  It acts only on Facile's store -> load memory
// edges.  Was 0.
def : ReadAdvance<ReadST, 1>;
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

//===----------------------------------------------------------------------===//
// FP multiply-accumulate.
//
// A720 SOG: "FP multiply accumulate | FMADD, FMSUB, FNMADD, FNMSUB | 4 (2) | 2 | V",
// "ASIMD FP multiply accumulate | FMLA, FMLS | 4(2) | 2 | V" and
// "ASIMD FP multiply accumulate long | FMLAL(2), FMLSL(2) | 4(2) | 2 | V", with
// "FP multiply-accumulate pipelines support late-forwarding of accumulate operands
// from similar uOPs, allowing a typical sequence of multiply-accumulate uOPs to
// issue one every N cycles (accumulate latency N shown in parentheses)" and
// "late forwarding of the result from FP multiply uOPs to the accumulate operands
// of an FP multiply-accumulate uOP. The latter can potentially be issued 1 cycle
// after the FP multiply uOP has been issued".
// Result latency 4; the accumulate operand is needed 2 cycles early, which gives
// 2 from a previous multiply-accumulate and 3 - 2 = 1 from an FMUL (WriteFMul, 3).
// Without this the FMADD family fell into WriteFMul (latency 3, no early read).
def A720Wr_FMA : SchedWriteRes<[A720UnitV]> { let Latency = 4; }
// ASIMD FMUL/FMULX (and the scalar FMULX, D-5) have their own write (3, V; l.1439,
// docs/a720-sog.md C-20); SOG l.1492-1494 (ASIMD FMUL -> FMLA accumulate, "issued 1 cycle
// after") needs it in this ReadAdvance too.
def A720Wr_VFMUL : SchedWriteRes<[A720UnitV]> { let Latency = 3; }
def A720Rd_FMA : SchedReadAdvance<2, [WriteFMul, A720Wr_FMA, A720Wr_VFMUL]>;

def : InstRW<[A720Wr_FMA, ReadDefault, ReadDefault, A720Rd_FMA], (instregex "^FN?M(ADD|SUB)[HSD]rrr$")>;
def : InstRW<[A720Wr_FMA, A720Rd_FMA], (instregex "^FMLAv", "^FMLSv", "^FMLALv", "^FMLSLv", "^FML[AS]L2?lanev", "^FMLAL2v", "^FMLSL2v")>;

//===----------------------------------------------------------------------===//
// FP divide / square root, per form.
//
// A720 SOG: "FP divide, H-form | FDIV | 5 | 1 | V0", "S-form | 7 | 1 | V0",
// "D-form | 12 | 1 | V0", and the same three values for FSQRT, with the note "FP
// divide and square root operations are now performed using a fully pipelined data
// path": a fixed latency per form and a throughput of one per cycle, so the unit is
// occupied for one cycle (no ReleaseAtCycles).
def A720Write_5c_1V0  : SchedWriteRes<[A720UnitV0]> { let Latency = 5; }
def A720Write_7c_1V0  : SchedWriteRes<[A720UnitV0]> { let Latency = 7; }
def A720Write_12c_1V0 : SchedWriteRes<[A720UnitV0]> { let Latency = 12; }

def : InstRW<[A720Write_5c_1V0],  (instrs FDIVHrr, FSQRTHr)>;
def : InstRW<[A720Write_7c_1V0],  (instrs FDIVSrr, FSQRTSr)>;
def : InstRW<[A720Write_12c_1V0], (instrs FDIVDrr, FSQRTDr)>;

// The ASIMD forms (SOG "ASIMD FP divide / square root"): latency and throughput per
// form, for FDIV and FSQRT alike: D-form F16 8 cycles 1/4 (occupancy 4), D-form F32 8
// 1/2 (2), Q-form F16 12 1/8 (8), Q-form F32 10 1/4 (4), Q-form F64 13 1/2 (2).  The
// table's throughput column is used as printed, although the note under the table says
// the unit is fully pipelined (the scalar forms, whose table says 1, occupy it for one
// cycle).  docs/a720-sog.md C-20, I-2.
def A720Write_8c4_1V0  : SchedWriteRes<[A720UnitV0]> { let Latency = 8;  let ReleaseAtCycles = [4]; }
def A720Write_8c2_1V0  : SchedWriteRes<[A720UnitV0]> { let Latency = 8;  let ReleaseAtCycles = [2]; }
def A720Write_12c8_1V0 : SchedWriteRes<[A720UnitV0]> { let Latency = 12; let ReleaseAtCycles = [8]; }
def A720Write_10c4_1V0 : SchedWriteRes<[A720UnitV0]> { let Latency = 10; let ReleaseAtCycles = [4]; }
def A720Write_13c2_1V0 : SchedWriteRes<[A720UnitV0]> { let Latency = 13; let ReleaseAtCycles = [2]; }

def : InstRW<[A720Write_8c4_1V0],  (instrs FDIVv4f16, FSQRTv4f16)>;
def : InstRW<[A720Write_8c2_1V0],  (instrs FDIVv2f32, FSQRTv2f32)>;
def : InstRW<[A720Write_12c8_1V0], (instrs FDIVv8f16, FSQRTv8f16)>;
def : InstRW<[A720Write_10c4_1V0], (instrs FDIVv4f32, FSQRTv4f32)>;
def : InstRW<[A720Write_13c2_1V0], (instrs FDIVv2f64, FSQRTv2f64)>;


//===----------------------------------------------------------------------===//
// 2026-10-30: SOG instruction groups (lN = line of automatic-llvm-mca/tmp/a720sog.txt)

def A720Wr_1c_S    : SchedWriteRes<[A720UnitS]>  { let Latency = 1; }
def A720Wr_1c_I    : SchedWriteRes<[A720UnitI]>  { let Latency = 1; }
def A720Wr_2c_M    : SchedWriteRes<[A720UnitM]>  { let Latency = 2; }
def A720Wr_3c_M    : SchedWriteRes<[A720UnitM]>  { let Latency = 3; }
def A720Wr_1c_B_S  : SchedWriteRes<[A720UnitB, A720UnitS]> { let Latency = 1; }

// Address generation ADR/ADRP 1, 2/cycle, S (l.767)
def : InstRW<[A720Wr_1c_S], (instrs ADR, ADRP)>;
// Branch and link BL/BLR 1, 2/cycle, B + S (l.615-618)
def : InstRW<[A720Wr_1c_B_S], (instrs BL, BLR)>;

// Arithmetic shifted register (l.642-648): "Arithmetic, LSL shift, shift <= 4" 1, I;
// "Arithmetic, LSR/ASR/ROR shift or LSL shift > 4" 2, M.  (LSL #0 = ALU basic, 1, I.)
def A720Wr_ArithShift : SchedWriteVariant<[SchedVar<IsCheapLSL,  [A720Wr_1c_I]>,
                                           SchedVar<NoSchedPred, [A720Wr_2c_M]>]>;
def : InstRW<[A720Wr_ArithShift], (instregex "^(ADD|SUB)S?[WX]rs$")>;
// ALU, extend and shift (extended register) 2, M (l.642).
// [a720_ext] docs/a720-sog.md C-9: every extended-register form.  The variant on RegExtendedPred
// used before tested "extend operand != 0", which is false only for UXTB #0 - a real extend - so it
// gave `add x0, x1, w2, uxtb` the "ALU, basic" row 1 I.  Identity extends (UXTX/LSL #0, the
// `add x0, sp, x1` spelling): U-4.
def : InstRW<[A720Wr_2c_M], (instregex "^(ADD|SUB)S?[WX]rx(64)?$")>;
// Logical, shift, flagset ANDS/BICS 2, M (l.661); unshifted = ALU basic, flagset 1, I (l.638)
def A720Wr_LogicSFlag : SchedWriteVariant<[SchedVar<RegShiftedPred, [A720Wr_2c_M]>,
                                           SchedVar<NoSchedPred,    [A720Wr_1c_I]>]>;
def : InstRW<[A720Wr_LogicSFlag], (instregex "^(ANDS|BICS)[WX]rs$")>;
// (Logical, shift, no flagset is 1, I (l.659) = the WriteISReg default.)

// Multiply (l.685-693).  MADD/MSUB/SMADDL.. 2(1), 1/cycle, M0; note 3: "Multiply without accumulate
// when Ra is ZR ... MUL, MNEG, SMULL, SMNEGL, UMULL and UMNEGL can be executed on utilized pipeline M
// with an execution throughput of 2"; note 2: accumulate operand late-forwarded from similar uOPs,
// accumulate latency 1 (2 - 1).  SMULH/UMULH 3, 2/cycle, M; no accumulator forwarding.
def A720Wr_MAC_M0 : SchedWriteRes<[A720UnitM0]> { let Latency = 2; }
def A720Wr_MUL_M  : SchedWriteRes<[A720UnitM]>  { let Latency = 2; }
def A720Wr_MAC : SchedWriteVariant<[SchedVar<IsReg3ZeroPred, [A720Wr_MUL_M]>,
                                    SchedVar<NoSchedPred,    [A720Wr_MAC_M0]>]>;
def A720Rd_MAcc : SchedReadAdvance<1, [A720Wr_MAC_M0, A720Wr_MUL_M]>;
def : InstRW<[A720Wr_MAC, ReadDefault, ReadDefault, A720Rd_MAcc],
             (instrs MADDWrrr, MADDXrrr, MSUBWrrr, MSUBXrrr, SMADDLrrr, UMADDLrrr, SMSUBLrrr, UMSUBLrrr)>;
def : InstRW<[A720Wr_3c_M], (instrs SMULHrr, UMULHrr)>;

// [a720_crc] docs/a720-sog.md C-26: "CRC checksum ops CRC32, CRC32C 2 1 M0" with note 1 "late
// forwarding of the result from a producer uOP to a consumer uOP.  This results in a 1 cycle
// reduction in latency as seen by the consumer" (l.1875-1878).  The accumulated CRC (Rn) reads one
// cycle late from a CRC producer.  Was WriteISReg, 1 on I.
def A720Wr_CRC : SchedWriteRes<[A720UnitM0]> { let Latency = 2; }
def A720Rd_CRC : SchedReadAdvance<1, [A720Wr_CRC]>;
def : InstRW<[A720Wr_CRC, A720Rd_CRC], (instregex "^CRC32C?[BHWX]rr$")>;

// [a720_pac] docs/a720-sog.md C-11: Table 3-5 (l.712-750).  AUT* 1 2 M; PAC* 4 2 M; XPAC* 1 2 M;
// BLRAA/BRAA/RETAA forms 2 2 M, B; LDRAA/LDRAB 5 2 M, L, I, note 2 "Only Immed pre-index with
// write back use I pipes" (the base update, WriteAdr).  Note 1 (AUTH fail) is not modelled (D-1).
// Before: PACIASP/AUTIASP and the other hint-space forms took the hint row (S0), PACIA/AUTIA took
// WriteI, and BLRAA, BRAA, RETAA, LDRAA, XPACD/XPACI had no scheduling data at all (dropped).
def A720Wr_1c_M   : SchedWriteRes<[A720UnitM]> { let Latency = 1; }
def A720Wr_4c_M   : SchedWriteRes<[A720UnitM]> { let Latency = 4; }
def A720Wr_2c_M_B : SchedWriteRes<[A720UnitM, A720UnitB]> { let Latency = 2; }
def A720Wr_5c_M_L : SchedWriteRes<[A720UnitM, A720UnitLS]> { let Latency = 5; }
def : InstRW<[A720Wr_1c_M], (instrs AUTDA, AUTDB, AUTDZA, AUTDZB, AUTIA, AUTIB, AUTIA1716,
                                    AUTIB1716, AUTIASP, AUTIBSP, AUTIAZ, AUTIBZ, AUTIZA, AUTIZB,
                                    XPACD, XPACI, XPACLRI)>;
def : InstRW<[A720Wr_4c_M], (instrs PACDA, PACDB, PACDZA, PACDZB, PACGA, PACIA, PACIB, PACIA1716,
                                    PACIB1716, PACIASP, PACIBSP, PACIAZ, PACIBZ, PACIZA, PACIZB)>;
def : InstRW<[A720Wr_2c_M_B], (instrs BLRAA, BLRAAZ, BLRAB, BLRABZ, BRAA, BRAAZ, BRAB, BRABZ,
                                      RETAA, RETAB)>;
def : InstRW<[A720Wr_5c_M_L], (instrs LDRAAindexed, LDRABindexed)>;
def : InstRW<[WriteAdr, A720Wr_5c_M_L], (instrs LDRAAwriteback, LDRABwriteback)>;

// [a720_from_a55_store_data] docs/a720-sog.md B-1: the integer store forms whose stock Sched lists
// name no reads: same writes (Table 3-8, l.871-895), plus ReadST on the data.  The register-offset
// forms already read ReadST in their stock Sched lists.
def : InstRW<[WriteST, ReadST, ReadAdrBase], (instregex "^STR(BB|HH|W|X)ui$", "^STUR(BB|HH|W|X)i$",
                                                        "^STTR(B|H|W|X)i$", "^STL?LR(B|H|W|X)$")>;
def : InstRW<[WriteSTP, ReadST, ReadST, ReadAdrBase], (instregex "^STN?P(W|X)i$")>;
def : InstRW<[WriteAdr, WriteST, ReadST, ReadAdrBase], (instregex "^STR(BB|HH|W|X)(pre|post)$")>;
def : InstRW<[WriteAdr, WriteSTP, ReadST, ReadST, ReadAdrBase], (instregex "^STP(W|X)(pre|post)$")>;

// Loads (l.790-856).  Literal 5, L + S (l.790).  Load pair X-form / LDPSW 4, 3/2 per cycle (l.840-856):
// L group held 2 cycles.
def A720Wr_LD5_L_S : SchedWriteRes<[A720UnitLS, A720UnitS]> { let Latency = 5; }
def : InstRW<[A720Wr_LD5_L_S], (instregex "^LDR(W|X|SW)l$", "^PRFMl$")>;
def A720Wr_LDP_X   : SchedWriteRes<[A720UnitLS]> { let Latency = 4; let ReleaseAtCycles = [2]; }
// [a720_ldp_hi] D-10: the second destination of a pair: the row's latency, no uOP (was one uOP).
def A720Wr_LDHi_0  : SchedWriteRes<[]> { let Latency = 4; let NumMicroOps = 0; }
def A720Wr_LD_W    : SchedWriteRes<[A720UnitLS]> { let Latency = 4; }
def : InstRW<[A720Wr_LD_W, A720Wr_LDHi_0], (instregex "^LDN?PWi$")>;
def : InstRW<[A720Wr_LDP_X, A720Wr_LDHi_0], (instregex "^LDN?PXi$")>;
// LDPSW: 4, 3/2, "I, L" (l.842) - the I pipe was missing
def A720Wr_LDPSW : SchedWriteRes<[A720UnitLS, A720UnitI]> { let Latency = 4; let ReleaseAtCycles = [2, 1]; }
def : InstRW<[A720Wr_LDPSW, A720Wr_LDHi_0], (instrs LDPSWi)>;
def : InstRW<[WriteAdr, A720Wr_LD_W, A720Wr_LDHi_0], (instregex "^LDPW(pre|post)$")>;
def : InstRW<[WriteAdr, A720Wr_LDP_X, A720Wr_LDHi_0], (instregex "^LDPX(pre|post)$")>;
def : InstRW<[WriteAdr, A720Wr_LDPSW, A720Wr_LDHi_0], (instregex "^LDPSW(pre|post)$")>;

// FP / vector loads 6, 3/cycle, L (l.1052-1100): "an extra cycle is required to forward results to
// FP/ASIMD pipelines".  Vector pairs: S/D offset 3/cycle, Q offset and all pre/post 3/2.
def A720Wr_FLD    : SchedWriteRes<[A720UnitLS]> { let Latency = 6; }
def A720Wr_FLDP2  : SchedWriteRes<[A720UnitLS]> { let Latency = 6; let ReleaseAtCycles = [2]; }
def A720Wr_FLDHi  : SchedWriteRes<[]> { let Latency = 6; let NumMicroOps = 0; }   // [a720_ldp_hi] D-10
def : InstRW<[A720Wr_FLD], (instregex "^LDR[BHSDQ](ui|roW|roX|l)$", "^LDUR[BHSDQ]i$")>;
def : InstRW<[WriteAdr, A720Wr_FLD], (instregex "^LDR[BHSDQ](pre|post)$")>;
def : InstRW<[A720Wr_FLD, A720Wr_FLDHi], (instregex "^LDN?P[SD]i$")>;
def : InstRW<[A720Wr_FLDP2, A720Wr_FLDHi], (instregex "^LDN?PQi$")>;
def : InstRW<[WriteAdr, A720Wr_FLDP2, A720Wr_FLDHi], (instregex "^LDP[SDQ](pre|post)$")>;

// FP / vector stores 2, 2/cycle, L01 + V (l.1116-1180).
// [a720_store_uops] C-7: two uOPs (address on L01, data on V); three with the I of the scaled Q rows.
// [a720_from_a55_store_data] B-1: the data operands read ReadST, the base ReadAdrBase.
def A720Wr_FST : SchedWriteRes<[A720UnitL01, A720UnitV]> { let Latency = 2; let NumMicroOps = 2; }
def : InstRW<[A720Wr_FST, ReadST, ReadAdrBase], (instregex "^STR[BHSD](ui|roW|roX)$", "^STRQui$", "^STUR[BHSDQ]i$")>;
def : InstRW<[A720Wr_FST, ReadST, ReadST, ReadAdrBase], (instregex "^STN?P[SDQ]i$")>;
// STR Q, register offset: "scale, Q-form" and "extend, scale, Q-form" I, L01, V (l.1151-1152,
// l.1161-1162); "basic, Q-form" and "extend, Q-form" L01, V (l.1134-1135, l.1155-1156).
// [a720_strq_ro] docs/a720-sog.md C-18: "scale" is the S bit of the addressing mode
// (AArch64_AM::getMemDoShift), as for the Cortex-A76 and A78.  The stock ScaledIdxPred used before
// is "extend is not UXTX/LSL, or scaled", but the extend operand of a register-offset load/store
// holds the signed flag (0/1), not an AArch64_AM extend type, so the predicate was true for every
// form and gave the unscaled Q stores an I uOP.
def A720IsMemScaled : MCSchedPredicate<CheckMemScaled>;
def A720Wr_FST_I : SchedWriteRes<[A720UnitL01, A720UnitV, A720UnitI]> { let Latency = 2; let NumMicroOps = 3; }
def A720Wr_STRQro : SchedWriteVariant<[SchedVar<A720IsMemScaled, [A720Wr_FST_I]>, SchedVar<NoSchedPred, [A720Wr_FST]>]>;
def : InstRW<[A720Wr_STRQro, ReadST, ReadAdrBase], (instrs STRQroW, STRQroX)>;
// Pre/post-index: the I of "L01, V, I" is the base update (WriteAdr).
// [a720_str_pre] docs/a720-sog.md C-18, I-3: "Store vector reg, immed pre-index, B/H/S/D-form STR
// 3 2 L01, V, I" (l.1124-1125; checked in the PDF, p.27), against 2 for the post-index and the
// Q-form pre-index rows.  As printed.  Was 2.
def A720Wr_FST3 : SchedWriteRes<[A720UnitL01, A720UnitV]> { let Latency = 3; let NumMicroOps = 2; }
def : InstRW<[WriteAdr, A720Wr_FST3, ReadST, ReadAdrBase], (instregex "^STR[BHSD]pre$")>;
def : InstRW<[WriteAdr, A720Wr_FST, ReadST, ReadAdrBase], (instregex "^STR[BHSD]post$", "^STRQ(pre|post)$")>;
def : InstRW<[WriteAdr, A720Wr_FST, ReadST, ReadST, ReadAdrBase], (instregex "^STP[SDQ](pre|post)$")>;

// FP scalar (l.963-1046)
def A720Wr_3c_V0  : SchedWriteRes<[A720UnitV0]> { let Latency = 3; }
def A720Wr_3c_M0  : SchedWriteRes<[A720UnitM0]> { let Latency = 3; }
def A720Wr_5c_M0V : SchedWriteRes<[A720UnitM0, A720UnitV]> { let Latency = 5; }
def : InstRW<[A720Wr_3c_V0], (instregex "^FRINT([AIMNPXZ]|32X|32Z|64X|64Z)[HSD]r$")>;     // 3, V0 (l.988)
def : InstRW<[A720Wr_3c_M0], (instregex "^[SU]CVTF[SU][WX][HSD]ri?$")>;                   // 3, M0 (l.1013)
def : InstRW<[A720Wr_5c_M0V], (instrs FMOVXDHighr)>;                                        // gen->high half 5, M0 + V (l.1043)

// ASIMD (l.1189-1600)
def A720Wr_2c_V1  : SchedWriteRes<[A720UnitV1]> { let Latency = 2; }
def A720Wr_3c_V1  : SchedWriteRes<[A720UnitV1]> { let Latency = 3; }
def A720Wr_4c_V1  : SchedWriteRes<[A720UnitV1]> { let Latency = 4; }
def A720Wr_5c_V1V : SchedWriteRes<[A720UnitV1, A720UnitV]> { let Latency = 5; }
def A720Wr_6c_V1r2: SchedWriteRes<[A720UnitV1]> { let Latency = 6; let ReleaseAtCycles = [2]; }
def A720Wr_2c_V0  : SchedWriteRes<[A720UnitV0]> { let Latency = 2; }
def A720Wr_4c_V0  : SchedWriteRes<[A720UnitV0]> { let Latency = 4; }
def A720Wr_4c_V0r2: SchedWriteRes<[A720UnitV0]> { let Latency = 4; let ReleaseAtCycles = [2]; }
def A720Wr_6c_V0r4: SchedWriteRes<[A720UnitV0]> { let Latency = 6; let ReleaseAtCycles = [4]; }
def A720Wr_3c_V   : SchedWriteRes<[A720UnitV]>  { let Latency = 3; }
def A720Wr_2c_Vt1 : SchedWriteRes<[A720UnitV]>  { let Latency = 2; let ReleaseAtCycles = [2]; }
def A720Wr_4c_Vt1 : SchedWriteRes<[A720UnitV]>  { let Latency = 4; let ReleaseAtCycles = [2]; }
def A720Wr_5c_M0V_INS : SchedWriteRes<[A720UnitM0, A720UnitV]> { let Latency = 5; }

// Integer multiply (l.1272-1292): MUL/SQDMULH/SQRDMULH 4, V0; SMULL/UMULL/SQDMULL 4, V0;
// MLA/MLS and SMLAL/SMLSL/UMLAL/UMLSL 4(1), V0; SQRDMLAH/SH and SQDMLAL/SQDMLSL 4(2), V0
// (note 1: accumulate operands late-forwarded from similar uOPs); PMUL/PMULL 2, V0.
def A720Wr_VMUL : SchedWriteRes<[A720UnitV0]> { let Latency = 4; }
def A720Rd_VAcc1 : SchedReadAdvance<3, [A720Wr_VMUL]>;
def A720Rd_VAcc2 : SchedReadAdvance<2, [A720Wr_VMUL]>;
def : InstRW<[A720Wr_VMUL], (instregex "^(MUL|SQDMULH|SQRDMULH)v", "^(SMULL|UMULL|SQDMULL)v")>;
def : InstRW<[A720Wr_VMUL, A720Rd_VAcc1], (instregex "^(MLA|MLS)v", "^(SMLAL|SMLSL|UMLAL|UMLSL)v")>;
def : InstRW<[A720Wr_VMUL, A720Rd_VAcc2], (instregex "^(SQRDMLAH|SQRDMLSH)v", "^(SQDMLAL|SQDMLSL)v")>;
def : InstRW<[A720Wr_2c_V0], (instregex "^PMULL?v")>;
// Pairwise add and accumulate long, shift accumulate 4(1), V1 (l.1289-1293, note 2)
def A720Wr_VAccV1 : SchedWriteRes<[A720UnitV1]> { let Latency = 4; }
def A720Rd_VAccV1 : SchedReadAdvance<3, [A720Wr_VAccV1]>;
def : InstRW<[A720Wr_VAccV1, A720Rd_VAccV1], (instregex "^[SU]ADALPv", "^[SU]R?SRAv")>;
// Shifts (l.1294-1335): by immed basic / insert, by register basic 2, V1; complex 4, V1
def : InstRW<[A720Wr_2c_V1], (instregex "^(SHL|SHLL|SSHLL|SSHR|USHLL|USHR|SLI|SRI|SHRN)v",
                                        "^(SSHL|USHL)v")>;
def : InstRW<[A720Wr_4c_V1], (instregex "^(RSHRN|SQRSHRN|SQRSHRUN|SQSHRN|SQSHRUN|SRSHR|UQRSHRN|UQSHRN|URSHR)v",
                                        "^(SQSHL|UQSHL|SQSHLU)v.*_shift$",
                                        "^(SRSHL|SQRSHL|SQSHL|URSHL|UQRSHL|UQSHL)v[0-9]+i[0-9]+$")>;
// Reductions (l.1230-1235, l.1264-1270): 4H/4S 3, V1; 8B/8H 5, V1 + V; 16B 6, 1/2, V1
def : InstRW<[A720Wr_3c_V1],   (instregex "^(ADDV|SADDLV|UADDLV|SMAXV|SMINV|UMAXV|UMINV)v4i(16|32)v$")>;
def : InstRW<[A720Wr_5c_V1V],  (instregex "^(ADDV|SADDLV|UADDLV|SMAXV|SMINV|UMAXV|UMINV)v8i(8|16)v$")>;
def : InstRW<[A720Wr_6c_V1r2], (instregex "^(ADDV|SADDLV|UADDLV|SMAXV|SMINV|UMAXV|UMINV)v16i8v$")>;
// Narrowing saturating extract 4, V1 (l.1541-1543)
def : InstRW<[A720Wr_4c_V1], (instregex "^(SQXTN|SQXTUN|UQXTN)v")>;
// Transfers: DUP from gen 3, M0 (l.1536); UMOV/SMOV 2, 1/cycle, V (l.1589); INS from gen 5, M0 + V (l.1591)
def : InstRW<[A720Wr_3c_M0], (instregex "^DUPv.*gpr$")>;
def : InstRW<[A720Wr_2c_Vt1], (instregex "^[SU]MOVv")>;
def : InstRW<[A720Wr_5c_M0V_INS], (instregex "^INSvi(8|16|32|64)gpr$")>;
// ASIMD FP multiply FMUL/FMULX 3, V (l.1439)
def : InstRW<[A720Wr_VFMUL], (instregex "^FMULX?v")>;
// ASIMD FP pairwise FADDP/FMAXP/... 3, V (l.1413)
def : InstRW<[A720Wr_3c_V], (instregex "^F(ADDP|MAXP|MAXNMP|MINP|MINNMP)v")>;
// ASIMD FP max/min reduce (F32 / D-form F16) 4, 1/cycle, V (l.1417); Q-form F16 6, 2/3 (l.1435)
def : InstRW<[A720Wr_4c_Vt1], (instregex "^F(MAX|MIN)(NM)?Vv4i32v$", "^F(MAX|MIN)(NM)?Vv4i16v$")>;
// ASIMD FP convert / round per form (l.1354-1405, l.1444-1470):
//   D-form F32 (2s) and Q-form F64 (2d): 3, 1/cycle, V0; D-form F16 (4h) and Q-form F32 (4s): 4, 1/2;
//   Q-form F16 (8h): 6, 1/4.  FCVTL/FCVTN: F32<->F64 3, 1; F16<->F32 4, 1/2.
def : InstRW<[A720Wr_3c_V0],   (instregex "^(FCVT[AMNPZ][SU]|[SU]CVTF)v(2f32|2f64|1i32|1i64)$",
                                          "^FRINT([AIMNPXZ]|32X|32Z|64X|64Z)v(2f32|2f64)$",
                                          "^FCVTLv(2i32|4i32)$", "^FCVTNv(2i32|4i32)$", "^FCVTXNv")>;
def : InstRW<[A720Wr_4c_V0r2], (instregex "^(FCVT[AMNPZ][SU]|[SU]CVTF)v(4f16|4f32)$",
                                          "^FRINT([AIMNPXZ]|32X|32Z|64X|64Z)v(4f16|4f32)$",
                                          "^FCVTLv(4i16|8i16)$", "^FCVTNv(4i16|8i16)$")>;
def : InstRW<[A720Wr_6c_V0r4], (instregex "^(FCVT[AMNPZ][SU]|[SU]CVTF)v8f16$", "^FRINT[AIMNPXZ]v8f16$")>;


//===----------------------------------------------------------------------===//
// Further rows of Tables 3-15 to 3-21 (docs/a720-sog.md C-19 to C-25).
def A720Wr_ACC4_V1 : SchedWriteRes<[A720UnitV1]> { let Latency = 4; }
def A720Rd_ACC4_V1 : SchedReadAdvance<3, [A720Wr_ACC4_V1]>;
def : InstRW<[A720Wr_ACC4_V1, A720Rd_ACC4_V1], (instregex "^[SU]ABAv", "^[SU]ABALv")>;          // 4(1), V1 (l.1201-1203)
def A720Wr_DOT : SchedWriteRes<[A720UnitV]> { let Latency = 3; }
def A720Rd_DOT : SchedReadAdvance<2, [A720Wr_DOT]>;
def : InstRW<[A720Wr_DOT, A720Rd_DOT], (instregex "^(SDOT|UDOT|USDOT|SUDOT)(lane)?v", "^(SMMLA|UMMLA|USMMLA)")>; // 3(1), V (l.1240-1247)
def : InstRW<[A720Wr_3c_V], (instregex "^FCADDv")>;                                             // 3, V (l.1352)
def A720Wr_FCMLA : SchedWriteRes<[A720UnitV]> { let Latency = 4; }
def A720Rd_FCMLA : SchedReadAdvance<2, [A720Wr_FCMLA]>;
def : InstRW<[A720Wr_FCMLA, A720Rd_FCMLA], (instregex "^FCMLAv")>;                               // 4(2), V (l.1353)
def A720Wr_6c_Vr3 : SchedWriteRes<[A720UnitV]> { let Latency = 6; let ReleaseAtCycles = [3]; }
def : InstRW<[A720Wr_6c_Vr3], (instregex "^F(MAX|MIN)(NM)?Vv8i16v$")>;                        // Q-form F16 6, 2/3 (l.1435)
def : InstRW<[A720Wr_3c_V0], (instregex "^(URECPE|URSQRTE)v2i32$", "^(FRECPE|FRSQRTE)v(2f32|2f64|1i32|1i64)$", "^FRECPXv")>; // l.1547, 1551, 1559
def : InstRW<[A720Wr_4c_V0r2], (instregex "^(URECPE|URSQRTE)v4i32$", "^(FRECPE|FRSQRTE)v(4f16|4f32)$")>; // l.1549, 1554
def : InstRW<[A720Wr_6c_V0r4], (instregex "^(FRECPE|FRSQRTE)v8f16$")>;                         // l.1557
def A720Wr_4c_V : SchedWriteRes<[A720UnitV]> { let Latency = 4; }
def : InstRW<[A720Wr_4c_V], (instregex "^(FRECPS|FRSQRTS)(v|16|32|64)")>;                      // 4, 2, V (l.1560)
def A720Wr_4c_Vr2 : SchedWriteRes<[A720UnitV]> { let Latency = 4; let ReleaseAtCycles = [2]; }
def A720Wr_4c_Vr3 : SchedWriteRes<[A720UnitV]> { let Latency = 4; let ReleaseAtCycles = [3]; }
def A720Wr_6c_Vr4 : SchedWriteRes<[A720UnitV]> { let Latency = 6; let ReleaseAtCycles = [4]; }
def : InstRW<[A720Wr_4c_Vr2], (instregex "^TBLv(8|16)i8Three$", "^TBXv(8|16)i8Two$")>;         // l.1566, l.1583
def : InstRW<[A720Wr_4c_Vr3], (instregex "^TBLv(8|16)i8Four$")>;                               // l.1567
def : InstRW<[A720Wr_6c_Vr3], (instregex "^TBXv(8|16)i8Three$")>;                              // l.1585
def : InstRW<[A720Wr_6c_Vr4], (instregex "^TBXv(8|16)i8Four$")>;                               // l.1587
// Crypto SHA / SM (Table 3-21, l.1841-1864): V0, throughput 1
def A720Wr_2c_V0r1 : SchedWriteRes<[A720UnitV0]> { let Latency = 2; }
def A720Wr_4c_V0r1 : SchedWriteRes<[A720UnitV0]> { let Latency = 4; }
def : InstRW<[A720Wr_2c_V0r1], (instrs SHA1Hrr, SHA1SU0rrr, SHA1SU1rr, SHA256SU0rr, SHA256SU1rrr, SHA512H, SHA512H2, SHA512SU0, SHA512SU1,
                                         SM3PARTW1, SM3PARTW2, SM3SS1, SM3TT1A, SM3TT1B, SM3TT2A, SM3TT2B)>;
def : InstRW<[A720Wr_4c_V0r1], (instrs SHA1Crrr, SHA1Mrrr, SHA1Prrr, SHA256Hrrr, SHA256H2rrr, SM4E, SM4ENCKEY)>;
// BF16 (Table 3-17, l.1503-1510)
def A720Wr_BF42 : SchedWriteRes<[A720UnitV]> { let Latency = 4; }
def A720Wr_BF53 : SchedWriteRes<[A720UnitV]> { let Latency = 5; }
def A720Rd_BF : SchedReadAdvance<2, [A720Wr_BF42, A720Wr_BF53]>;
def : InstRW<[A720Wr_4c_V0r2], (instregex "^BFCVTN")>;
def : InstRW<[A720Wr_3c_V0], (instrs BFCVT)>;
def : InstRW<[A720Wr_BF42, A720Rd_BF], (instregex "^BFDOTv", "^BF16DOTlanev", "^BFMLAL[BT]")>;
def : InstRW<[A720Wr_BF53, A720Rd_BF], (instregex "^BFMMLA")>;
// Flag manipulation / FP condition flags: 1, 4/cycle, I (l.653-656)
def : InstRW<[A720Wr_1c_I], (instrs AXFLAG, XAFLAG, SETF8, SETF16, RMIF, CFINV)>;
// ASIMD FP convert, fixed-point forms (same rows as the other converts, by form; l.1373-1405)
def : InstRW<[A720Wr_3c_V0],   (instregex "^(FCVTZ[SU]|[SU]CVTF)v(2i32|2i64)_shift$", "^(FCVTZ[SU]|[SU]CVTF)[sd]$")>;
def : InstRW<[A720Wr_4c_V0r2], (instregex "^(FCVTZ[SU]|[SU]CVTF)v(4i16|4i32)_shift$")>;
def : InstRW<[A720Wr_6c_V0r4], (instregex "^(FCVTZ[SU]|[SU]CVTF)v8i16_shift$")>;

// [a720_frecpe_scalar] docs/a720-sog.md C-22: "ASIMD reciprocal and square root estimate, D-form F32
// and scalar forms FRECPE, FRSQRTE 3 1 V0" (l.1551-1553): the scalar H forms too.  They were 4, V0
// held 2 cycles (the D-form F16 row).
def : InstRW<[A720Wr_3c_V0], (instrs FRECPEv1f16, FRSQRTEv1f16)>;

// [a720_fp_cvt_scalar] docs/a720-sog.md D-4: FP conversions with the operand and the result in FP
// registers, scalar H form (`fcvtzs h0, h1`, `scvtf h0, h1`, also with #fbits).  No row names them;
// the S and D scalar forms take the "D-form F32 and Q-form F64" row (3 1 V0, l.1373-1383) above,
// and the H forms take the same row, the lowest the SOG gives for these mnemonics, as for the
// Cortex-A76 and A78 (D-4 there).  They were 4 / V0 held 2 (FCVT*, *CVTF h) and 2 on V (*CVTF
// v1i16, no row).
def : InstRW<[A720Wr_3c_V0], (instregex "^FCVT[AMNPZ][SU]v1f16$", "^[SU]CVTFv1i16$", "^(FCVTZ[SU]|[SU]CVTF)h$")>;

// [a720_asimd_scalar] docs/a720-sog.md D-5: the scalar forms of the ASIMD mnemonics take the row of
// their mnemonic (Table 3-15, l.1272-1330), as for the Cortex-A76 and A78.  They were the default
// 2 on V; FMULX scalar was 2 on V (ASIMD FP multiply FMULX 3 2 V, l.1439).
def : InstRW<[A720Wr_2c_V1], (instrs SHLd, SSHRd, USHRd, SLId, SRId)>;
def : InstRW<[A720Wr_4c_V1], (instregex "^[SU]RSHRd$", "^(SQSHL|UQSHL|SQSHLU)[bhsd]$",
                                        "^(SQSHRN|SQRSHRN|UQSHRN|UQRSHRN|SQSHRUN|SQRSHRUN)[bhs]$")>;
def : InstRW<[A720Wr_VAccV1, A720Rd_VAccV1], (instrs SSRAd, USRAd, SRSRAd, URSRAd)>;
def : InstRW<[A720Wr_VMUL, A720Rd_VAcc2], (instrs SQDMLALi16, SQDMLALi32, SQDMLSLi16, SQDMLSLi32)>;
def : InstRW<[A720Wr_VMUL], (instrs SQDMULLi16, SQDMULLi32)>;
def : InstRW<[A720Wr_VFMUL], (instrs FMULX16, FMULX32, FMULX64)>;

def A720Wr_VLS_6_LS1 : SchedWriteRes<[A720UnitLS]> { let Latency = 6; let ReleaseAtCycles = [1]; }
def A720Wr_VLS_6_LS2 : SchedWriteRes<[A720UnitLS]> { let Latency = 6; let ReleaseAtCycles = [2]; }
def A720Wr_VLS_6_LS3 : SchedWriteRes<[A720UnitLS]> { let Latency = 6; let ReleaseAtCycles = [3]; }
def A720Wr_VLS_7_LS4 : SchedWriteRes<[A720UnitLS]> { let Latency = 7; let ReleaseAtCycles = [4]; }
def A720Wr_VLS_8_LS1_V1 : SchedWriteRes<[A720UnitLS, A720UnitV]> { let Latency = 8; let ReleaseAtCycles = [1, 1]; }
def A720Wr_VLS_8_LS2_V1 : SchedWriteRes<[A720UnitLS, A720UnitV]> { let Latency = 8; let ReleaseAtCycles = [2, 1]; }
def A720Wr_VLS_8_LS1_V3 : SchedWriteRes<[A720UnitLS, A720UnitV]> { let Latency = 8; let ReleaseAtCycles = [1, 3]; }
def A720Wr_VLS_10_LS1_V3 : SchedWriteRes<[A720UnitLS, A720UnitV]> { let Latency = 10; let ReleaseAtCycles = [1, 3]; }
def A720Wr_VLS_8_LS6_V1 : SchedWriteRes<[A720UnitLS, A720UnitV]> { let Latency = 8; let ReleaseAtCycles = [6, 1]; }
def A720Wr_VLS_2_L011_V1 : SchedWriteRes<[A720UnitL01, A720UnitV]> { let Latency = 2; let NumMicroOps = 2; let ReleaseAtCycles = [1, 1]; }
def A720Wr_VLS_2_L012_V1 : SchedWriteRes<[A720UnitL01, A720UnitV]> { let Latency = 2; let NumMicroOps = 2; let ReleaseAtCycles = [2, 1]; }
def A720Wr_VLS_4_L012_V1 : SchedWriteRes<[A720UnitL01, A720UnitV]> { let Latency = 4; let NumMicroOps = 2; let ReleaseAtCycles = [2, 1]; }
def A720Wr_VLS_4_L013_V1 : SchedWriteRes<[A720UnitL01, A720UnitV]> { let Latency = 4; let NumMicroOps = 2; let ReleaseAtCycles = [3, 1]; }
def A720Wr_VLS_2_L013_V1 : SchedWriteRes<[A720UnitL01, A720UnitV]> { let Latency = 2; let NumMicroOps = 2; let ReleaseAtCycles = [3, 1]; }
def A720Wr_VLS_4_L014_V1 : SchedWriteRes<[A720UnitL01, A720UnitV]> { let Latency = 4; let NumMicroOps = 2; let ReleaseAtCycles = [4, 1]; }
// ASIMD load / store rows, one per SOG row of Tables 3-19 and 3-20 (line = SOG line)
// docs/a720-sog.md C-23, C-24, D-6.  [a720_store_uops] C-7: the store writes (A720Wr_VLS_*_L01*)
// have two uOPs (address and data).  [a720_from_a55_store_data] B-1: the data Vt of every store
// reads ReadST, the base Rn ReadAdrBase; for the one-lane forms the lane index takes the read in
// between (ReadDefault).
def : InstRW<[A720Wr_VLS_6_LS1], (instrs LD1Onev1d, LD1Onev2s, LD1Onev4h, LD1Onev8b)>;   // l.1620 ASIMD load, 1 element, multiple, 1 reg, D-form: 6 / 3 / L
def : InstRW<[WriteAdr, A720Wr_VLS_6_LS1], (instrs LD1Onev1d_POST, LD1Onev2s_POST, LD1Onev4h_POST, LD1Onev8b_POST)>;
def : InstRW<[A720Wr_VLS_6_LS1], (instrs LD1Onev16b, LD1Onev2d, LD1Onev4s, LD1Onev8h)>;   // l.1622 ASIMD load, 1 element, multiple, 1 reg, Q-form: 6 / 3 / L
def : InstRW<[WriteAdr, A720Wr_VLS_6_LS1], (instrs LD1Onev16b_POST, LD1Onev2d_POST, LD1Onev4s_POST, LD1Onev8h_POST)>;
def : InstRW<[A720Wr_VLS_6_LS2], (instrs LD1Twov1d, LD1Twov2s, LD1Twov4h, LD1Twov8b)>;   // l.1624 ASIMD load, 1 element, multiple, 2 reg, D-form: 6 / 3/2 / L
def : InstRW<[WriteAdr, A720Wr_VLS_6_LS2], (instrs LD1Twov1d_POST, LD1Twov2s_POST, LD1Twov4h_POST, LD1Twov8b_POST)>;
def : InstRW<[A720Wr_VLS_6_LS2], (instrs LD1Twov16b, LD1Twov2d, LD1Twov4s, LD1Twov8h)>;   // l.1626 ASIMD load, 1 element, multiple, 2 reg, Q-form: 6 / 3/2 / L
def : InstRW<[WriteAdr, A720Wr_VLS_6_LS2], (instrs LD1Twov16b_POST, LD1Twov2d_POST, LD1Twov4s_POST, LD1Twov8h_POST)>;
def : InstRW<[A720Wr_VLS_6_LS3], (instrs LD1Threev1d, LD1Threev2s, LD1Threev4h, LD1Threev8b)>;   // l.1628 ASIMD load, 1 element, multiple, 3 reg, D-form: 6 / 1 / L
def : InstRW<[WriteAdr, A720Wr_VLS_6_LS3], (instrs LD1Threev1d_POST, LD1Threev2s_POST, LD1Threev4h_POST, LD1Threev8b_POST)>;
def : InstRW<[A720Wr_VLS_6_LS3], (instrs LD1Threev16b, LD1Threev2d, LD1Threev4s, LD1Threev8h)>;   // l.1630 ASIMD load, 1 element, multiple, 3 reg, Q-form: 6 / 1 / L
def : InstRW<[WriteAdr, A720Wr_VLS_6_LS3], (instrs LD1Threev16b_POST, LD1Threev2d_POST, LD1Threev4s_POST, LD1Threev8h_POST)>;
def : InstRW<[A720Wr_VLS_7_LS4], (instrs LD1Fourv1d, LD1Fourv2s, LD1Fourv4h, LD1Fourv8b)>;   // l.1632 ASIMD load, 1 element, multiple, 4 reg, D-form: 7 / 3/4 / L
def : InstRW<[WriteAdr, A720Wr_VLS_7_LS4], (instrs LD1Fourv1d_POST, LD1Fourv2s_POST, LD1Fourv4h_POST, LD1Fourv8b_POST)>;
def : InstRW<[A720Wr_VLS_7_LS4], (instrs LD1Fourv16b, LD1Fourv2d, LD1Fourv4s, LD1Fourv8h)>;   // l.1634 ASIMD load, 1 element, multiple, 4 reg, Q-form: 7 / 3/4 / L
def : InstRW<[WriteAdr, A720Wr_VLS_7_LS4], (instrs LD1Fourv16b_POST, LD1Fourv2d_POST, LD1Fourv4s_POST, LD1Fourv8h_POST)>;
def : InstRW<[A720Wr_VLS_8_LS1_V1], (instrs LD1i16, LD1i32, LD1i8)>;   // l.1636 ASIMD load, 1 element, one lane, B/H/S: 8 / 2 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS1_V1], (instrs LD1i16_POST, LD1i32_POST, LD1i8_POST)>;
def : InstRW<[A720Wr_VLS_8_LS1_V1], (instrs LD1i64)>;   // l.1638 ASIMD load, 1 element, one lane, D: 8 / 2 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS1_V1], (instrs LD1i64_POST)>;
def : InstRW<[A720Wr_VLS_6_LS1], (instrs LD1Rv2s, LD1Rv4h, LD1Rv8b)>;   // l.1640 ASIMD load, 1 element, all lanes, D-form, B/H/S: 6 / 3 / L
def : InstRW<[WriteAdr, A720Wr_VLS_6_LS1], (instrs LD1Rv2s_POST, LD1Rv4h_POST, LD1Rv8b_POST)>;
def : InstRW<[A720Wr_VLS_6_LS1], (instrs LD1Rv1d)>;   // l.1642 ASIMD load, 1 element, all lanes, D-form, D: 6 / 3 / L
def : InstRW<[WriteAdr, A720Wr_VLS_6_LS1], (instrs LD1Rv1d_POST)>;
def : InstRW<[A720Wr_VLS_6_LS1], (instrs LD1Rv16b, LD1Rv2d, LD1Rv4s, LD1Rv8h)>;   // l.1644 ASIMD load, 1 element, all lanes, Q-form: 6 / 3 / L
def : InstRW<[WriteAdr, A720Wr_VLS_6_LS1], (instrs LD1Rv16b_POST, LD1Rv2d_POST, LD1Rv4s_POST, LD1Rv8h_POST)>;
def : InstRW<[A720Wr_VLS_8_LS1_V1], (instrs LD2Twov2s, LD2Twov4h, LD2Twov8b)>;   // l.1646 ASIMD load, 2 element, multiple, D-form, B/H/S: 8 / 2 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS1_V1], (instrs LD2Twov2s_POST, LD2Twov4h_POST, LD2Twov8b_POST)>;
def : InstRW<[A720Wr_VLS_8_LS2_V1], (instrs LD2Twov16b, LD2Twov4s, LD2Twov8h)>;   // l.1648 ASIMD load, 2 element, multiple, Q-form, B/H/S: 8 / 3/2 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS2_V1], (instrs LD2Twov16b_POST, LD2Twov4s_POST, LD2Twov8h_POST)>;
def : InstRW<[A720Wr_VLS_8_LS2_V1], (instrs LD2Twov2d)>;   // l.1650 ASIMD load, 2 element, multiple, Q-form, D: 8 / 3/2 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS2_V1], (instrs LD2Twov2d_POST)>;
def : InstRW<[A720Wr_VLS_8_LS1_V1], (instrs LD2i16, LD2i8)>;   // l.1652 ASIMD load, 2 element, one lane, B/H: 8 / 2 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS1_V1], (instrs LD2i16_POST, LD2i8_POST)>;
def : InstRW<[A720Wr_VLS_8_LS1_V1], (instrs LD2i32)>;   // l.1654 ASIMD load, 2 element, one lane, S: 8 / 2 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS1_V1], (instrs LD2i32_POST)>;
def : InstRW<[A720Wr_VLS_8_LS1_V1], (instrs LD2i64)>;   // l.1669 ASIMD load, 2 element, one lane, D: 8 / 2 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS1_V1], (instrs LD2i64_POST)>;
def : InstRW<[A720Wr_VLS_6_LS2], (instrs LD2Rv2s, LD2Rv4h, LD2Rv8b)>;   // l.1671 ASIMD load, 2 element, all lanes, D-form, B/H/S: 6 / 3/2 / L
def : InstRW<[WriteAdr, A720Wr_VLS_6_LS2], (instrs LD2Rv2s_POST, LD2Rv4h_POST, LD2Rv8b_POST)>;
def : InstRW<[A720Wr_VLS_6_LS2], (instrs LD2Rv1d)>;   // l.1673 ASIMD load, 2 element, all lanes, D-form, D: 6 / 3/2 / L
def : InstRW<[WriteAdr, A720Wr_VLS_6_LS2], (instrs LD2Rv1d_POST)>;
def : InstRW<[A720Wr_VLS_6_LS2], (instrs LD2Rv16b, LD2Rv2d, LD2Rv4s, LD2Rv8h)>;   // l.1675 ASIMD load, 2 element, all lanes, Q-form: 6 / 3/2 / L
def : InstRW<[WriteAdr, A720Wr_VLS_6_LS2], (instrs LD2Rv16b_POST, LD2Rv2d_POST, LD2Rv4s_POST, LD2Rv8h_POST)>;
def : InstRW<[A720Wr_VLS_8_LS1_V3], (instrs LD3Threev2s, LD3Threev4h, LD3Threev8b)>;   // l.1677 ASIMD load, 3 element, multiple, D-form, B/H/S: 8 / 2/3 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS1_V3], (instrs LD3Threev2s_POST, LD3Threev4h_POST, LD3Threev8b_POST)>;
def : InstRW<[A720Wr_VLS_10_LS1_V3], (instrs LD3Threev16b, LD3Threev4s, LD3Threev8h)>;   // l.1679 ASIMD load, 3 element, multiple, Q-form, B/H/S: 10 / 2/3 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_10_LS1_V3], (instrs LD3Threev16b_POST, LD3Threev4s_POST, LD3Threev8h_POST)>;
def : InstRW<[A720Wr_VLS_10_LS1_V3], (instrs LD3Threev2d)>;   // l.1681 ASIMD load, 3 element, multiple, Q-form, D: 10 / 2/3 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_10_LS1_V3], (instrs LD3Threev2d_POST)>;
def : InstRW<[A720Wr_VLS_8_LS1_V3], (instrs LD3i16, LD3i8)>;   // l.1683 ASIMD load, 3 element, one lane, B/H: 8 / 2/3 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS1_V3], (instrs LD3i16_POST, LD3i8_POST)>;
def : InstRW<[A720Wr_VLS_8_LS1_V3], (instrs LD3i32)>;   // l.1685 ASIMD load, 3 element, one lane, S: 8 / 2/3 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS1_V3], (instrs LD3i32_POST)>;
def : InstRW<[A720Wr_VLS_8_LS1_V3], (instrs LD3i64)>;   // l.1687 ASIMD load, 3 element, one lane, D: 8 / 2/3 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS1_V3], (instrs LD3i64_POST)>;
def : InstRW<[A720Wr_VLS_6_LS3], (instrs LD3Rv2s, LD3Rv4h, LD3Rv8b)>;   // l.1689 ASIMD load, 3 element, all lanes, D-form, B/H/S: 6 / 1 / L
def : InstRW<[WriteAdr, A720Wr_VLS_6_LS3], (instrs LD3Rv2s_POST, LD3Rv4h_POST, LD3Rv8b_POST)>;
def : InstRW<[A720Wr_VLS_6_LS3], (instrs LD3Rv1d)>;   // l.1691 ASIMD load, 3 element, all lanes, D-form, D: 6 / 1 / L
def : InstRW<[WriteAdr, A720Wr_VLS_6_LS3], (instrs LD3Rv1d_POST)>;
def : InstRW<[A720Wr_VLS_6_LS3], (instrs LD3Rv16b, LD3Rv4s, LD3Rv8h)>;   // l.1693 ASIMD load, 3 element, all lanes, Q-form, B/H/S: 6 / 1 / L
def : InstRW<[WriteAdr, A720Wr_VLS_6_LS3], (instrs LD3Rv16b_POST, LD3Rv4s_POST, LD3Rv8h_POST)>;
def : InstRW<[A720Wr_VLS_6_LS3], (instrs LD3Rv2d)>;   // l.1695 ASIMD load, 3 element, all lanes, Q-form, D: 6 / 1 / L
def : InstRW<[WriteAdr, A720Wr_VLS_6_LS3], (instrs LD3Rv2d_POST)>;
def : InstRW<[A720Wr_VLS_8_LS6_V1], (instrs LD4Fourv2s, LD4Fourv4h, LD4Fourv8b)>;   // l.1697 ASIMD load, 4 element, multiple, D-form, B/H/S: 8 / 1/2 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS6_V1], (instrs LD4Fourv2s_POST, LD4Fourv4h_POST, LD4Fourv8b_POST)>;
def : InstRW<[A720Wr_VLS_8_LS6_V1], (instrs LD4Fourv16b, LD4Fourv4s, LD4Fourv8h)>;   // l.1699 ASIMD load, 4 element, multiple, Q-form, B/H/S: 8 / 1/2 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS6_V1], (instrs LD4Fourv16b_POST, LD4Fourv4s_POST, LD4Fourv8h_POST)>;
def : InstRW<[A720Wr_VLS_8_LS6_V1], (instrs LD4Fourv2d)>;   // l.1701 ASIMD load, 4 element, multiple, Q-form, D: 8 / 1/2 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS6_V1], (instrs LD4Fourv2d_POST)>;
def : InstRW<[A720Wr_VLS_8_LS6_V1], (instrs LD4i16, LD4i8)>;   // l.1703 ASIMD load, 4 element, one lane, B/H: 8 / 1/2 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS6_V1], (instrs LD4i16_POST, LD4i8_POST)>;
def : InstRW<[A720Wr_VLS_8_LS6_V1], (instrs LD4i32)>;   // l.1705 ASIMD load, 4 element, one lane, S: 8 / 1/2 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS6_V1], (instrs LD4i32_POST)>;
def : InstRW<[A720Wr_VLS_8_LS6_V1], (instrs LD4i64)>;   // l.1707 ASIMD load, 4 element, one lane, D: 8 / 1/2 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS6_V1], (instrs LD4i64_POST)>;
def : InstRW<[A720Wr_VLS_8_LS1_V3], (instrs LD4Rv2s, LD4Rv4h, LD4Rv8b)>;   // l.1709 ASIMD load, 4 element, all lanes, D-form, B/H/S: 8 / 2/3 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS1_V3], (instrs LD4Rv2s_POST, LD4Rv4h_POST, LD4Rv8b_POST)>;
def : InstRW<[A720Wr_VLS_8_LS6_V1], (instrs LD4Rv1d)>;   // l.1711 ASIMD load, 4 element, all lanes, D-form, D: 8 / 1/2 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS6_V1], (instrs LD4Rv1d_POST)>;
def : InstRW<[A720Wr_VLS_8_LS1_V3], (instrs LD4Rv16b, LD4Rv4s, LD4Rv8h)>;   // l.1725 ASIMD load, 4 element, all lanes, Q-form, B/H/S: 8 / 2/3 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS1_V3], (instrs LD4Rv16b_POST, LD4Rv4s_POST, LD4Rv8h_POST)>;
def : InstRW<[A720Wr_VLS_8_LS6_V1], (instrs LD4Rv2d)>;   // l.1727 ASIMD load, 4 element, all lanes, Q-form, D (ASIMD load, wri: 8 / 1/2 / L, V
def : InstRW<[WriteAdr, A720Wr_VLS_8_LS6_V1], (instrs LD4Rv2d_POST)>;
def : InstRW<[A720Wr_VLS_2_L011_V1, ReadST, ReadAdrBase], (instrs ST1Onev1d, ST1Onev2s, ST1Onev4h, ST1Onev8b)>;   // l.1746 ASIMD store, 1 element, multiple, 1 reg, D-form: 2 / 2 / L01, V
def : InstRW<[WriteAdr, A720Wr_VLS_2_L011_V1, ReadST, ReadAdrBase], (instrs ST1Onev1d_POST, ST1Onev2s_POST, ST1Onev4h_POST, ST1Onev8b_POST)>;
def : InstRW<[A720Wr_VLS_2_L011_V1, ReadST, ReadAdrBase], (instrs ST1Onev16b, ST1Onev2d, ST1Onev4s, ST1Onev8h)>;   // l.1748 ASIMD store, 1 element, multiple, 1 reg, Q-form: 2 / 2 / L01, V
def : InstRW<[WriteAdr, A720Wr_VLS_2_L011_V1, ReadST, ReadAdrBase], (instrs ST1Onev16b_POST, ST1Onev2d_POST, ST1Onev4s_POST, ST1Onev8h_POST)>;
def : InstRW<[A720Wr_VLS_2_L011_V1, ReadST, ReadAdrBase], (instrs ST1Twov1d, ST1Twov2s, ST1Twov4h, ST1Twov8b)>;   // l.1750 ASIMD store, 1 element, multiple, 2 reg, D-form: 2 / 2 / L01, V
def : InstRW<[WriteAdr, A720Wr_VLS_2_L011_V1, ReadST, ReadAdrBase], (instrs ST1Twov1d_POST, ST1Twov2s_POST, ST1Twov4h_POST, ST1Twov8b_POST)>;
def : InstRW<[A720Wr_VLS_2_L011_V1, ReadST, ReadAdrBase], (instrs ST1Twov16b, ST1Twov2d, ST1Twov4s, ST1Twov8h)>;   // l.1752 ASIMD store, 1 element, multiple, 2 reg, Q-form: 2 / 2 / L01, V
def : InstRW<[WriteAdr, A720Wr_VLS_2_L011_V1, ReadST, ReadAdrBase], (instrs ST1Twov16b_POST, ST1Twov2d_POST, ST1Twov4s_POST, ST1Twov8h_POST)>;
def : InstRW<[A720Wr_VLS_2_L012_V1, ReadST, ReadAdrBase], (instrs ST1Threev1d, ST1Threev2s, ST1Threev4h, ST1Threev8b)>;   // l.1754 ASIMD store, 1 element, multiple, 3 reg, D-form: 2 / 1 / L01, V
def : InstRW<[WriteAdr, A720Wr_VLS_2_L012_V1, ReadST, ReadAdrBase], (instrs ST1Threev1d_POST, ST1Threev2s_POST, ST1Threev4h_POST, ST1Threev8b_POST)>;
def : InstRW<[A720Wr_VLS_2_L012_V1, ReadST, ReadAdrBase], (instrs ST1Threev16b, ST1Threev2d, ST1Threev4s, ST1Threev8h)>;   // l.1756 ASIMD store, 1 element, multiple, 3 reg, Q-form: 2 / 1 / L01, V
def : InstRW<[WriteAdr, A720Wr_VLS_2_L012_V1, ReadST, ReadAdrBase], (instrs ST1Threev16b_POST, ST1Threev2d_POST, ST1Threev4s_POST, ST1Threev8h_POST)>;
def : InstRW<[A720Wr_VLS_2_L012_V1, ReadST, ReadAdrBase], (instrs ST1Fourv1d, ST1Fourv2s, ST1Fourv4h, ST1Fourv8b)>;   // l.1758 ASIMD store, 1 element, multiple, 4 reg, D-form: 2 / 1 / L01, V
def : InstRW<[WriteAdr, A720Wr_VLS_2_L012_V1, ReadST, ReadAdrBase], (instrs ST1Fourv1d_POST, ST1Fourv2s_POST, ST1Fourv4h_POST, ST1Fourv8b_POST)>;
def : InstRW<[A720Wr_VLS_2_L012_V1, ReadST, ReadAdrBase], (instrs ST1Fourv16b, ST1Fourv2d, ST1Fourv4s, ST1Fourv8h)>;   // l.1760 ASIMD store, 1 element, multiple, 4 reg, Q-form: 2 / 1 / L01, V
def : InstRW<[WriteAdr, A720Wr_VLS_2_L012_V1, ReadST, ReadAdrBase], (instrs ST1Fourv16b_POST, ST1Fourv2d_POST, ST1Fourv4s_POST, ST1Fourv8h_POST)>;
def : InstRW<[A720Wr_VLS_2_L011_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST1i16, ST1i32, ST1i8)>;   // l.1762 ASIMD store, 1 element, one lane, B/H/S: 2 / 2 / L01, V
def : InstRW<[WriteAdr, A720Wr_VLS_2_L011_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST1i16_POST, ST1i32_POST, ST1i8_POST)>;
def : InstRW<[A720Wr_VLS_2_L011_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST1i64)>;   // l.1764 ASIMD store, 1 element, one lane, D: 2 / 2 / L01, V
def : InstRW<[WriteAdr, A720Wr_VLS_2_L011_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST1i64_POST)>;
def : InstRW<[A720Wr_VLS_2_L011_V1, ReadST, ReadAdrBase], (instrs ST2Twov2s, ST2Twov4h, ST2Twov8b)>;   // l.1766 ASIMD store, 2 element, multiple, D-form, B/H/S: 2 / 2 / V, L01
def : InstRW<[WriteAdr, A720Wr_VLS_2_L011_V1, ReadST, ReadAdrBase], (instrs ST2Twov2s_POST, ST2Twov4h_POST, ST2Twov8b_POST)>;
def : InstRW<[A720Wr_VLS_2_L011_V1, ReadST, ReadAdrBase], (instrs ST2Twov16b, ST2Twov4s, ST2Twov8h)>;   // l.1768 ASIMD store, 2 element, multiple, Q-form, B/H/S: 2 / 2 / V, L01
def : InstRW<[WriteAdr, A720Wr_VLS_2_L011_V1, ReadST, ReadAdrBase], (instrs ST2Twov16b_POST, ST2Twov4s_POST, ST2Twov8h_POST)>;
def : InstRW<[A720Wr_VLS_2_L011_V1, ReadST, ReadAdrBase], (instrs ST2Twov2d)>;   // l.1782 ASIMD store, 2 element, multiple, Q-form, D: 2 / 2 / V, L01
def : InstRW<[WriteAdr, A720Wr_VLS_2_L011_V1, ReadST, ReadAdrBase], (instrs ST2Twov2d_POST)>;
def : InstRW<[A720Wr_VLS_2_L011_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST2i16, ST2i32, ST2i8)>;   // l.1784 ASIMD store, 2 element, one lane, B/H/S: 2 / 2 / V, L01
def : InstRW<[WriteAdr, A720Wr_VLS_2_L011_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST2i16_POST, ST2i32_POST, ST2i8_POST)>;
def : InstRW<[A720Wr_VLS_2_L011_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST2i64)>;   // l.1786 ASIMD store, 2 element, one lane, D: 2 / 2 / V, L01
def : InstRW<[WriteAdr, A720Wr_VLS_2_L011_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST2i64_POST)>;
def : InstRW<[A720Wr_VLS_4_L012_V1, ReadST, ReadAdrBase], (instrs ST3Threev2s, ST3Threev4h, ST3Threev8b)>;   // l.1788 ASIMD store, 3 element, multiple, D-form, B/H/S: 4 / 1 / V, L01
def : InstRW<[WriteAdr, A720Wr_VLS_4_L012_V1, ReadST, ReadAdrBase], (instrs ST3Threev2s_POST, ST3Threev4h_POST, ST3Threev8b_POST)>;
def : InstRW<[A720Wr_VLS_4_L013_V1, ReadST, ReadAdrBase], (instrs ST3Threev16b, ST3Threev4s, ST3Threev8h)>;   // l.1790 ASIMD store, 3 element, multiple, Q-form, B/H/S: 4 / 2/3 / V, L01
def : InstRW<[WriteAdr, A720Wr_VLS_4_L013_V1, ReadST, ReadAdrBase], (instrs ST3Threev16b_POST, ST3Threev4s_POST, ST3Threev8h_POST)>;
def : InstRW<[A720Wr_VLS_2_L013_V1, ReadST, ReadAdrBase], (instrs ST3Threev2d)>;   // l.1792 ASIMD store, 3 element, multiple, Q-form, D: 2 / 2/3 / V, L01
def : InstRW<[WriteAdr, A720Wr_VLS_2_L013_V1, ReadST, ReadAdrBase], (instrs ST3Threev2d_POST)>;
def : InstRW<[A720Wr_VLS_2_L012_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST3i16, ST3i8)>;   // l.1794 ASIMD store, 3 element, one lane, B/H: 2 / 1 / V, L01
def : InstRW<[WriteAdr, A720Wr_VLS_2_L012_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST3i16_POST, ST3i8_POST)>;
def : InstRW<[A720Wr_VLS_2_L012_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST3i32)>;   // l.1796 ASIMD store, 3 element, one lane, S: 2 / 1 / V, L01
def : InstRW<[WriteAdr, A720Wr_VLS_2_L012_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST3i32_POST)>;
def : InstRW<[A720Wr_VLS_2_L012_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST3i64)>;   // l.1798 ASIMD store, 3 element, one lane, D: 2 / 1 / V, L01
def : InstRW<[WriteAdr, A720Wr_VLS_2_L012_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST3i64_POST)>;
def : InstRW<[A720Wr_VLS_4_L012_V1, ReadST, ReadAdrBase], (instrs ST4Fourv2s, ST4Fourv4h, ST4Fourv8b)>;   // l.1800 ASIMD store, 4 element, multiple, D-form, B/H/S: 4 / 1 / V, L01
def : InstRW<[WriteAdr, A720Wr_VLS_4_L012_V1, ReadST, ReadAdrBase], (instrs ST4Fourv2s_POST, ST4Fourv4h_POST, ST4Fourv8b_POST)>;
def : InstRW<[A720Wr_VLS_4_L014_V1, ReadST, ReadAdrBase], (instrs ST4Fourv16b, ST4Fourv4s, ST4Fourv8h)>;   // l.1802 ASIMD store, 4 element, multiple, Q-form, B/H/S: 4 / 1/2 / V, L01
def : InstRW<[WriteAdr, A720Wr_VLS_4_L014_V1, ReadST, ReadAdrBase], (instrs ST4Fourv16b_POST, ST4Fourv4s_POST, ST4Fourv8h_POST)>;
def : InstRW<[A720Wr_VLS_2_L012_V1, ReadST, ReadAdrBase], (instrs ST4Fourv2d)>;   // l.1804 ASIMD store, 4 element, multiple, Q-form, D: 2 / 1 / V, L01
def : InstRW<[WriteAdr, A720Wr_VLS_2_L012_V1, ReadST, ReadAdrBase], (instrs ST4Fourv2d_POST)>;
def : InstRW<[A720Wr_VLS_2_L012_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST4i16, ST4i32, ST4i8)>;   // l.1806 ASIMD store, 4 element, one lane, B/H/S: 2 / 1 / V, L01
def : InstRW<[WriteAdr, A720Wr_VLS_2_L012_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST4i16_POST, ST4i32_POST, ST4i8_POST)>;
def : InstRW<[A720Wr_VLS_2_L012_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST4i64)>;   // l.1808 ASIMD store, 4 element, one lane, D (ASIMD store, writeback : 2 / 1 / V, L01
def : InstRW<[WriteAdr, A720Wr_VLS_2_L012_V1, ReadST, ReadDefault, ReadAdrBase], (instrs ST4i64_POST)>;
// SOG rows with pipes inconsistent with throughput (Load group used): []
// covered 340 of 340 LDn/STn opcodes; not covered: []

} // SchedModel = CortexA720Model
