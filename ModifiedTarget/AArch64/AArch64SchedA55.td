//==- AArch64SchedCortexA55.td - ARM Cortex-A55 Scheduling Definitions -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for the ARM Cortex-A55 processors. Note
// that this schedule is currently used as the default for -mcpu=generic. As a
// result, some of the modelling decision made do not precisely model the
// Cortex-A55, instead aiming to be a good compromise between different cpus.
//
//===----------------------------------------------------------------------===//

// ===---------------------------------------------------------------------===//
// The following definitions describe the per-operand machine model.
// This works with MachineScheduler. See MCSchedModel.h for details.

// Cortex-A55 machine model for scheduling and other instruction cost heuristics.
def CortexA55Model : SchedMachineModel {
  let MicroOpBufferSize = 0;  // The Cortex-A55 is an in-order processor
  let IssueWidth = 2;         // It dual-issues under most circumstances
  let LoadLatency = 3;        // Cycles for loads to access the cache. The
                              // optimisation guide shows that most loads have
                              // a latency of 3, but some have a latency of 4
                              // or 5. Setting it 3 matches standard L1 hits. (CHANGED!)
  let MispredictPenalty = 8;  // A branch direction mispredict.
  let PostRAScheduler = 1;    // Enable PostRA scheduler pass.
  let CompleteModel = 0;      // Covers instructions applicable to Cortex-A55.

  list<Predicate> UnsupportedFeatures = [HasSVE, HasMTE];

  // FIXME: Remove when all errors have been fixed.
  let FullInstRWOverlapCheck = 0;
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available.

// Modeling each pipeline as a ProcResource using the BufferSize = 0 since the
// Cortex-A55 is in-order.

def CortexA55UnitALU    : ProcResource<2> { let BufferSize = 0; } // Int ALU
def CortexA55UnitMAC    : ProcResource<1> { let BufferSize = 0; } // Int MAC, 64-bi wide
def CortexA55UnitDiv    : ProcResource<1> { let BufferSize = 0; } // Int Division, not pipelined
def CortexA55UnitLd     : ProcResource<1> { let BufferSize = 0; } // Load pipe
def CortexA55UnitSt     : ProcResource<1> { let BufferSize = 0; } // Store pipe
def CortexA55UnitB      : ProcResource<1> { let BufferSize = 0; } // Branch
// [a55_varshift] Variable shift ASRV/LSLV/LSRV/RORV: "1, 1, 11" (SOG 4.4, l.758), i.e. throughput 1 against 2 for
//   ALU basic (and the AArch32 "Move, shift by register 1, 1, 11", l.733).  The SOG does not say which of the two
//   ALU pipes can do it; this one-unit resource only caps variable shifts at one per cycle.  A variable shift uses
//   one CortexA55UnitALU and this unit for one cycle each, so the other ALU stays free for any other ALU instruction.
def CortexA55UnitVarShift : ProcResource<1> { let BufferSize = 0; }

// The FP DIV/SQRT instructions execute totally differently from the FP ALU
// instructions, which can mostly be dual-issued; that's why for now we model
// them with 2 resources.
def CortexA55UnitFPALU  : ProcResource<2> { let BufferSize = 0; } // FP ALU
def CortexA55UnitFPMAC  : ProcResource<2> { let BufferSize = 0; } // FP MAC
def CortexA55UnitFPDIV  : ProcResource<1> { let BufferSize = 0; } // FP Div/SQRT, 64/128
// [a55_sog_n5] FP compare FCCMP{E}/FCMP{E}: "1, 1, 11" (SOG 4.11 p.26, l.1355; AArch32 VCMP likewise, l.1302), i.e.
//   throughput 1 against 2 for the other FP ALU rows.  Same representation as CortexA55UnitVarShift: the SOG does not
//   say which FP ALU unit compares; this one-unit resource only caps FP compares at one per cycle.  An FP compare
//   uses one CortexA55UnitFPALU and this unit for one cycle each, so it still pairs with another FP ALU instruction.
def CortexA55UnitFPCmp  : ProcResource<1> { let BufferSize = 0; }

//===----------------------------------------------------------------------===//
// Subtarget-specific SchedWrite types

let SchedModel = CortexA55Model in {

// These latencies are modeled without taking into account forwarding paths
// (the software optimisation guide lists latencies taking into account
// typical forwarding paths).
def : WriteRes<WriteImm, [CortexA55UnitALU]> { let Latency = 1; }    // MOVN, MOVZ (SOG: 1)
def : WriteRes<WriteI, [CortexA55UnitALU]> { let Latency = 1; }      // ALU (SOG: 1)
def IsA55UnshiftedReg : MCSchedPredicate<CheckNumOperands<3>>;

// [a55_planB] Shifted-reg ALU forms with shift amount 0 (e.g. add x0,x1,x2 / orr = mov / and / eor) are "ALU basic" 1
//   (SOG l.713-715); only a non-zero shift is "ALU extend and/or shift" 2 (l.716).  The 1-cycle form is a
//   1-cycle ALU producer for the same-cycle ALU0 -> ALU1 path of the issue stage [a55_alu_fwd].
def CortexA55WriteBfm1 : SchedWriteRes<[CortexA55UnitALU]> { let Latency = 1; }
def CortexA55WriteBfm2 : SchedWriteRes<[CortexA55UnitALU]> { let Latency = 2; }
def CortexA55WriteISReg1 : SchedWriteRes<[CortexA55UnitALU]> { let Latency = 1; }
def CortexA55WriteISReg2 : SchedWriteRes<[CortexA55UnitALU]> { let Latency = 2; }
def CortexA55WriteISRegV : SchedWriteVariant<[SchedVar<RegShiftedPred, [CortexA55WriteISReg2]>,
                                              SchedVar<NoSchedPred, [CortexA55WriteISReg1]>]>;
def : SchedAlias<WriteISReg, CortexA55WriteISRegV>;
def : WriteRes<WriteIEReg, [CortexA55UnitALU]> { let Latency = 2; }  // ALU of Extended-Reg (SOG: 2)
def : WriteRes<WriteExtr, [CortexA55UnitALU]> { let Latency = 2; }   // EXTR from a reg pair (SOG: 2); EXTR is
                                                                     // overridden by the InstRW [a55_ror_imm] below
def : WriteRes<WriteIS, [CortexA55UnitALU]> { let Latency = 1; }     // Shift/Scale (SOG: 1)

// MAC
// [a55_td_fix] SOG 4.5: MADD/MSUB 32-bit 3 (1) (l.800), SMADDL/UMADDL/SMSUBL/UMSUBL 3 (1) (l.816) -> WriteIM32 3.
//   MADD/MSUB 64-bit "4-5 (2-3)", throughput "1/3 - 1/2" (l.802) -> WriteIM64 4, ReleaseAtCycles 2 (best case).
//   Accumulator forwarding (the number in parentheses) = Latency - ReadIMA advance (2, unchanged).
def : WriteRes<WriteIM32, [CortexA55UnitMAC]> { let Latency = 3; }   // 32-bit Multiply
def : WriteRes<WriteIM64, [CortexA55UnitMAC]> { let Latency = 4; let ReleaseAtCycles = [2]; }   // 64-bit Multiply
// SMULH/UMULH: 6, throughput 1/3 (SOG l.819)
def CortexA55WriteIMH : SchedWriteRes<[CortexA55UnitMAC]> { let Latency = 6; let ReleaseAtCycles = [3]; }

// Div
// [a55_sog_impl #11] SOG 4.5 (p.16): SDIV/UDIV W-form "3 - 12 (11)", X-form "3 - 20 (19)", throughput
//   "1/12 (11) - 1/3" / "1/20 (19) - 1/3", dual-issue 01; note 1: iterative, early termination, blocks later divides.
//   Best case (project policy: range -> minimum latency, occupancy = best throughput): latency 3, DIV busy 3 cycles.
//   (The "serializing" rule of SOG 3.4 is modelled in src/a55/a55_issue_stage.cpp.)
def : WriteRes<WriteID32, [CortexA55UnitDiv]> {
  let Latency = 3; let ReleaseAtCycles = [3]; let BeginGroup = 1;
}
def : WriteRes<WriteID64, [CortexA55UnitDiv]> {
  let Latency = 3; let ReleaseAtCycles = [3]; let BeginGroup = 1;
}

// Load
def : WriteRes<WriteLD, [CortexA55UnitLd]> { let Latency = 3; }
def : WriteRes<WriteLDIdx, [CortexA55UnitLd]> { let Latency = 3; }
def : WriteRes<WriteLDHi, [CortexA55UnitLd]> { let Latency = 4; }

// Vector Load - Vector loads take 1-5 cycles to issue. For the WriteVecLd
//               below, choosing the median of 3 which makes the latency 6.
// An extra cycle is needed to get the swizzling right.
def : WriteRes<WriteVLD, [CortexA55UnitLd]> { let Latency = 6;
                                           let ReleaseAtCycles = [3]; }
def CortexA55WriteVLD1 : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 4; }
def CortexA55WriteVLD1SI : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 4; let SingleIssue = 1; }
def CortexA55WriteVLD2 : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 5;
                                                  let ReleaseAtCycles = [2]; }
def CortexA55WriteVLD3 : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 6;
                                                  let ReleaseAtCycles = [3]; }
def CortexA55WriteVLD4 : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 7;
                                                  let ReleaseAtCycles = [4]; }
def CortexA55WriteVLD5 : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 8;
                                                  let ReleaseAtCycles = [5]; }
def CortexA55WriteVLD6 : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 9;
                                                  let ReleaseAtCycles = [6]; }
def CortexA55WriteVLD7 : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 10;
                                                  let ReleaseAtCycles = [7]; }
def CortexA55WriteVLD8 : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 11;
                                                  let ReleaseAtCycles = [8]; }

def CortexA55WriteLDP1 : SchedWriteRes<[]> { let Latency = 3; }
// [a55_td_fix] first register of LDP W-form / LDPSW / LDP S-form: 3 (SOG l.1074, 1078, 1081, 1498); was VLD1(SI) = 4
//   LDP/LDNP W-form and LDPSW are dual-issue "00" (SOG 4.8 p.21) -> SingleIssue.
def CortexA55WriteLDPW3SI : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 3; let SingleIssue = 1; }
// [a55_sog_impl #4/#9] LDP/LDNP S-form (vector): 3, 1, dual-issue 01 (SOG 4.13 p.28) -> BeginGroup.
def CortexA55WriteLDPS3 : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 3; let BeginGroup = 1; }
// [a55_sog_n2] LDPSW "3, 1, 00" with no number in parentheses and no note 1 (SOG 4.8 p.21, l.1078): no pointer-
//   chasing path for either register, so LDPSW has writes of its own that are kept out of the ReadAdrBase list
//   (they stay in the ALU load-use lists: load-use 2, SOG 3.3 l.484).  Same values as CortexA55WriteLDPW3SI /
//   CortexA55WriteLDP1.
def CortexA55WriteLDPSW3SI : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 3; let SingleIssue = 1; }
def CortexA55WriteLDPSW3b  : SchedWriteRes<[]> { let Latency = 3; }

// [a55_sog_impl #9] Loads per SOG 4.8 (p.19-21) / 4.13 (p.27-28).  "Execution throughput" T < 1 is modelled
//   by holding the (single) load pipe for 1/T cycles; the dual-issue column by BeginGroup (01) / SingleIssue (00).
//   LDP/LDNP X-form "4 (3)", 1/2, 01; D-form 4, 1/2, 01: both registers 4, load pipe 2 cycles.
def CortexA55WriteLDPX4   : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 4; let ReleaseAtCycles = [2]; let BeginGroup = 1; }
def CortexA55WriteLDPX4b  : SchedWriteRes<[]> { let Latency = 4; }
//   LDP X-form pre/post-index "4 (3), 2", 1/2, dual-issue 00.
def CortexA55WriteLDPX4SI : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 4; let ReleaseAtCycles = [2]; let SingleIssue = 1; }
//   LDP/LDNP Q-form 6, 1/4, 01 (pre/post "6, 4").
def CortexA55WriteLDPQ6   : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 6; let ReleaseAtCycles = [4]; let BeginGroup = 1; }
def CortexA55WriteLDPQ6b  : SchedWriteRes<[]> { let Latency = 6; }
//   LDR/LDUR vector B/H-form 3, 1, 01.
def CortexA55WriteLdBH3   : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 3; let BeginGroup = 1; }
//   LDR/LDUR vector Q-form (incl. literal and pre/post) "4" / "4, 2", 1/2, 01.
def CortexA55WriteLdQ4    : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 4; let ReleaseAtCycles = [2]; let BeginGroup = 1; }
//   Load acquire (LDAR/LDLAR) and load acquire exclusive (LDAXR): "3 (2)", 1, 01.
def CortexA55WriteLdAcq   : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 3; let BeginGroup = 1; }
//   Load acquire exclusive pair LDAXP: "3(2)", 1, 00.
def CortexA55WriteLdAXP   : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 3; let SingleIssue = 1; }
//   [a55_sog_n2] its second register (was CortexA55WriteLDP1, which is now the no-fast-path second register of
//   LDP W): SOG 4.8 gives LDAXP "3(2)" with note 1 but Table 3 (sec. 3.3) lists only LDR/LDTR and LDRD/LDP, so
//   which register of LDAXP has the pointer-chasing path is not stated; both keep it, as before (docs U-15).
def CortexA55WriteLdAXPb  : SchedWriteRes<[]> { let Latency = 3; }
//   PRFM: 1, 1, 11.
def CortexA55WritePRFM    : SchedWriteRes<[CortexA55UnitLd]> { let Latency = 1; }

// Pre/Post Indexing - Performed as part of address generation
// [a55_sog_impl #9h] Base-register update latency (the number after the comma, SOG 4.8 note 2 / 4.13 note 2, and
//   "(1/T)" for stores, SOG 4.9 note 3 / 4.14 note 2 / 4.19 note 1): 1 for LDR/STR/STP (T = 1) and LDP W/S-form;
//   2 for LDP X/D-form, LDR Q and STP Q; 4 for LDP Q.  Was 0 (no SOG basis).
def : WriteRes<WriteAdr, []> { let Latency = 1; }
def CortexA55WriteAdr2 : SchedWriteRes<[]> { let Latency = 2; }
def CortexA55WriteAdr4 : SchedWriteRes<[]> { let Latency = 4; }

// Store
let RetireOOO = 1 in {
def : WriteRes<WriteST, [CortexA55UnitSt]> { let Latency = 1; }
def : WriteRes<WriteSTP, [CortexA55UnitSt]> { let Latency = 1; }
def : WriteRes<WriteSTIdx, [CortexA55UnitSt]> { let Latency = 1; }
}
def : WriteRes<WriteSTX, [CortexA55UnitSt]> { let Latency = 4; }
// [a55_sog_impl #9] SOG 4.9 (p.23): store release STLR/STLLR 2, 1/2, 01; store release exclusive STLXR/STLXP
//   4, 1/2, 01.  SOG 4.14 (p.29): STP/STNP Q-form 2, 1/2, 01 (writeback "(1/T)" = 2).
def CortexA55WriteSTLR  : SchedWriteRes<[CortexA55UnitSt]> { let Latency = 2; let ReleaseAtCycles = [2]; let BeginGroup = 1; }
def CortexA55WriteSTLXR : SchedWriteRes<[CortexA55UnitSt]> { let Latency = 4; let ReleaseAtCycles = [2]; let BeginGroup = 1; }
let RetireOOO = 1 in
def CortexA55WriteSTPQ  : SchedWriteRes<[CortexA55UnitSt]> { let Latency = 2; let ReleaseAtCycles = [2]; let BeginGroup = 1; }

// Vector Store - Similar to vector loads, can take 1-3 cycles to issue.
def : WriteRes<WriteVST, [CortexA55UnitSt]> { let Latency = 5;
                                          let ReleaseAtCycles = [2];}
def CortexA55WriteVST1 : SchedWriteRes<[CortexA55UnitSt]> { let Latency = 4; }
def CortexA55WriteVST2 : SchedWriteRes<[CortexA55UnitSt]> { let Latency = 5;
                                                  let ReleaseAtCycles = [2]; }
def CortexA55WriteVST3 : SchedWriteRes<[CortexA55UnitSt]> { let Latency = 6;
                                                  let ReleaseAtCycles = [3]; }
def CortexA55WriteVST4 : SchedWriteRes<[CortexA55UnitSt]> { let Latency = 5;
                                                  let ReleaseAtCycles = [4]; }

def : WriteRes<WriteAtomic, []> { let Unsupported = 1; }
// [a55_sog_impl #3] Armv8.1 atomics, SOG 4.10 (p.23-25).  The SOG names no pipeline: an atomic is modelled
//   on the load and the store pipe (it is a load and a store), each held for 1/T cycles.  All are dual-issue
//   01, CASP is 00.  ST<OP> is LD<OP> with Rt = WZR/XZR.
class CortexA55WriteAtomic<int lat, int occ> : SchedWriteRes<[CortexA55UnitLd, CortexA55UnitSt]> {
  let Latency = lat; let ReleaseAtCycles = [occ, occ]; let BeginGroup = 1;
}
class CortexA55WriteAtomicSI<int lat, int occ> : SchedWriteRes<[CortexA55UnitLd, CortexA55UnitSt]> {
  let Latency = lat; let ReleaseAtCycles = [occ, occ]; let SingleIssue = 1;
}
def CortexA55AtomicToZR : MCSchedPredicate<CheckAny<[CheckRegOperand<0, XZR>, CheckRegOperand<0, WZR>]>>;
//   LD<OP>{A}: 2, 1/2;  ST<OP>: 1, 1.
def CortexA55WriteLdOp : SchedWriteVariant<[SchedVar<CortexA55AtomicToZR, [CortexA55WriteAtomic<1, 1>]>,
                                            SchedVar<NoSchedPred,         [CortexA55WriteAtomic<2, 2>]>]>;
//   LD<OP>L / LD<OP>AL: 3, 1/3;  ST<OP>L: 2, 1/2.
def CortexA55WriteLdOpL : SchedWriteVariant<[SchedVar<CortexA55AtomicToZR, [CortexA55WriteAtomic<2, 2>]>,
                                             SchedVar<NoSchedPred,         [CortexA55WriteAtomic<3, 3>]>]>;

// Branch
// [a55_sog_impl #4] Dual-issue column: B/BL/CBZ/TBZ 11; BR/RET/BLR 10 (SOG 4.2 p.14) -> EndGroup.
//   Control instructions (MRS/MSR, WFI/WFE, barriers) "cannot be dual issued" (SOG 3.2 Table 1; not listed as
//   an instruction-1 in Table 2) -> SingleIssue.  Not WFE/WFI: they are HINT (WriteHint), left as is
//   (docs/a55-sog.md N-3).
def : WriteRes<WriteBr, [CortexA55UnitB]>;
def : WriteRes<WriteBrReg, [CortexA55UnitB]> { let EndGroup = 1; }
def : WriteRes<WriteSys, [CortexA55UnitB]> { let SingleIssue = 1; }
def : WriteRes<WriteBarrier, [CortexA55UnitB]> { let SingleIssue = 1; }
def : WriteRes<WriteHint, [CortexA55UnitB]>;
// [a55_sog_n8] NOP: SOG 3.2 Table 2 (p.11, l.458) "Miscellaneous: IT, Architectural NOP" can be instruction-1,
//   and Table 1 does not restrict it as instruction-0; Table 2 limits only branches against branches
//   (l.436-438).  So a NOP is not on the branch unit (it used to be, through WriteHint, and could not pair with
//   a branch or another NOP).  The SOG names no pipe for a NOP: no resource; the issue width still applies.
//   Only the NOP opcode (`nop`, 0xd503201f); the other hints (HINT: YIELD, WFE, WFI, SEV, BTI, ...) keep
//   WriteHint (docs/a55-sog.md U-13).
def CortexA55WriteNop : SchedWriteRes<[]> { let Latency = 1; }
def : InstRW<[CortexA55WriteNop], (instrs NOP)>;

// FP ALU
//   As WriteF result is produced in F5 and it can be mostly forwarded
//   to consumer at F1, the effectively latency is set as 4.
def : WriteRes<WriteF, [CortexA55UnitFPALU]> { let Latency = 4; }
def : WriteRes<WriteFCmp, [CortexA55UnitFPALU]> { let Latency = 3; }
def : WriteRes<WriteFCvt, [CortexA55UnitFPALU]> { let Latency = 4; }
def : WriteRes<WriteFCopy, [CortexA55UnitFPALU]> { let Latency = 3; }
def : WriteRes<WriteFImm, [CortexA55UnitFPALU]> { let Latency = 3; }

// NEON
class CortexA55WriteVd<int n, ProcResourceKind res> : SchedWriteRes<[res]> {
  let Latency = n;
}
class CortexA55WriteVq<int n, ProcResourceKind res> : SchedWriteRes<[res, res]> {
  let Latency = n;
  let BeginGroup = 1;
}
def CortexA55WriteDotScVq_4 : CortexA55WriteVq<4, CortexA55UnitFPALU>;
def CortexA55WriteDotVq_4 : CortexA55WriteVq<4, CortexA55UnitFPALU>;
def CortexA55WriteDotVd_4 : CortexA55WriteVd<4, CortexA55UnitFPALU>;
def CortexA55WriteMlaLVq_4 : CortexA55WriteVq<4, CortexA55UnitFPALU>;
def CortexA55WriteMlaIxVq_4 : CortexA55WriteVq<4, CortexA55UnitFPALU>;
def CortexA55WriteMlaVq_4 : CortexA55WriteVq<4, CortexA55UnitFPALU>;
def CortexA55WriteMlaVd_4 : CortexA55WriteVd<4, CortexA55UnitFPALU>;
def CortexA55WriteAluVq_4 : CortexA55WriteVq<4, CortexA55UnitFPALU>;
def CortexA55WriteAluVd_4 : CortexA55WriteVd<4, CortexA55UnitFPALU>;
def CortexA55WriteAluVd_3 : CortexA55WriteVd<3, CortexA55UnitFPALU>;
def CortexA55WriteAluVq_3 : CortexA55WriteVq<3, CortexA55UnitFPALU>;
def CortexA55WriteAluVd_2 : CortexA55WriteVd<2, CortexA55UnitFPALU>;
def CortexA55WriteAluVq_2 : CortexA55WriteVq<2, CortexA55UnitFPALU>;
def CortexA55WriteAluVd_1 : CortexA55WriteVd<1, CortexA55UnitFPALU>;
def CortexA55WriteAluVq_1 : CortexA55WriteVq<1, CortexA55UnitFPALU>;
def : SchedAlias<WriteVd, CortexA55WriteVd<4, CortexA55UnitFPALU>>;
def : SchedAlias<WriteVq, CortexA55WriteVq<4, CortexA55UnitFPALU>>;

// FP ALU specific new schedwrite definitions
def CortexA55WriteFPALU_F2 : SchedWriteRes<[CortexA55UnitFPALU]> { let Latency = 2;}
def CortexA55WriteFPALU_F3 : SchedWriteRes<[CortexA55UnitFPALU]> { let Latency = 3;}
def CortexA55WriteFPALU_F4 : SchedWriteRes<[CortexA55UnitFPALU]> { let Latency = 4;}
def CortexA55WriteFPALU_F5 : SchedWriteRes<[CortexA55UnitFPALU]> { let Latency = 5;}

// [a55_sog_impl] Throughput < 1 on the 2-wide FP/ASIMD pipes: an instruction with SOG throughput T holds one of
//   the two units for 2/T cycles (T = 1 -> 2 cycles, the same as CortexA55WriteVq).  Dual-issue 01 -> BeginGroup.
class CortexA55WriteVT<int lat, int occ, ProcResourceKind res> : SchedWriteRes<[res]> {
  let Latency = lat; let ReleaseAtCycles = [occ]; let BeginGroup = 1;
}
//   ASIMD SABA/SABAL, RADDHN/RSUBHN, SRSRA/URSRA, SADALP/UADALP: 4, 1/2, 01 (SOG 4.15 p.30-32).
def CortexA55WriteAluT2_4 : CortexA55WriteVT<4, 4, CortexA55UnitFPALU>;
//   Throughput 1 but dual-issue 11 (AESMC/AESIMC, SOG 4.20 p.40).
def CortexA55WriteAluT1_2_11 : SchedWriteRes<[CortexA55UnitFPALU, CortexA55UnitFPALU]> { let Latency = 2; }
//   FCSEL 2, 1, 01 (SOG 4.11 p.26).
def CortexA55WriteFCSEL : SchedWriteRes<[CortexA55UnitFPALU, CortexA55UnitFPALU]> { let Latency = 2; let BeginGroup = 1; }
//   ASIMD table lookup TBL "2+N-1", 1/N; TBX "2+N", 1/(N+1); 01 (SOG 4.17 p.35, N = table registers).
def CortexA55WriteTBL1 : CortexA55WriteVT<2, 2, CortexA55UnitFPALU>;
def CortexA55WriteTBL2 : CortexA55WriteVT<3, 4, CortexA55UnitFPALU>;
def CortexA55WriteTBL3 : CortexA55WriteVT<4, 6, CortexA55UnitFPALU>;
def CortexA55WriteTBL4 : CortexA55WriteVT<5, 8, CortexA55UnitFPALU>;
def CortexA55WriteTBX1 : CortexA55WriteVT<3, 4, CortexA55UnitFPALU>;
def CortexA55WriteTBX2 : CortexA55WriteVT<4, 6, CortexA55UnitFPALU>;
def CortexA55WriteTBX3 : CortexA55WriteVT<5, 8, CortexA55UnitFPALU>;
def CortexA55WriteTBX4 : CortexA55WriteVT<6, 10, CortexA55UnitFPALU>;
// [a55_sog_impl #7] Integer ops of latency 2 / 1 that are NOT 1-cycle "ALU basic" producers, so they stay out of
//   the ALU->ALU ReadAdvance lists below: RBIT 2 (SOG 4.7 p.18); CRC32B/H/X/CB/CH/CX 2, CRC32W/CW 1 (SOG 4.21 p.41).
def CortexA55WriteALU2 : SchedWriteRes<[CortexA55UnitALU]> { let Latency = 2; }
def CortexA55WriteCRC1 : SchedWriteRes<[CortexA55UnitALU]> { let Latency = 1; }

// FP Mul, Div, Sqrt. Div/Sqrt are not pipelined
// [a55_sog_impl #4/#5] FDIV/FSQRT: latency / hazard per SOG Table 4 (p.13) and 4.11 (p.26), dual-issue 01.
let RetireOOO = 1 in {
def : WriteRes<WriteFMul, [CortexA55UnitFPMAC]> { let Latency = 4; }
def : WriteRes<WriteFDiv, [CortexA55UnitFPDIV]> { let Latency = 22;
                                            let ReleaseAtCycles = [29]; }
def CortexA55WriteFMAC : SchedWriteRes<[CortexA55UnitFPMAC]> { let Latency = 4; }
// ASIMD FP multiply(-accumulate) Q-form ("2*, 11*", SOG 4.16 note 1) and by element (4, 1, 01): throughput 1, 01.
def CortexA55WriteFMACq : SchedWriteRes<[CortexA55UnitFPMAC, CortexA55UnitFPMAC]> { let Latency = 4; let BeginGroup = 1; }
def CortexA55WriteFDivHP : SchedWriteRes<[CortexA55UnitFPDIV]> { let Latency = 8;
                                                     let ReleaseAtCycles = [5]; let BeginGroup = 1; }
def CortexA55WriteFDivSP : SchedWriteRes<[CortexA55UnitFPDIV]> { let Latency = 13;
                                                     let ReleaseAtCycles = [10]; let BeginGroup = 1; }
def CortexA55WriteFDivDP : SchedWriteRes<[CortexA55UnitFPDIV]> { let Latency = 22;
                                                     let ReleaseAtCycles = [19]; let BeginGroup = 1; }
def CortexA55WriteFSqrtHP : SchedWriteRes<[CortexA55UnitFPDIV]> { let Latency = 8;
                                                      let ReleaseAtCycles = [5]; let BeginGroup = 1; }
def CortexA55WriteFSqrtSP : SchedWriteRes<[CortexA55UnitFPDIV]> { let Latency = 12;
                                                      let ReleaseAtCycles = [9]; let BeginGroup = 1; }
def CortexA55WriteFSqrtDP : SchedWriteRes<[CortexA55UnitFPDIV]> { let Latency = 22;
                                                      let ReleaseAtCycles = [19]; let BeginGroup = 1; }
}
//===----------------------------------------------------------------------===//
// Subtarget-specific SchedRead types.

// [a55_planB] Integer ReadAdvance limited per the SOG forwarding description (scratch/session_20261029/a55_alu_forwarding).
//   * advance 1 into a non-shifting ALU operand only from loads (load-use 2, SOG l.484).  The 1-cycle ALU
//     writes used to be in the same list (ALU0 ex1 -> ALU1 ex1 dedicated path, SOG l.351-352 -> same-cycle
//     dependent pair); that path is now an issue-stage rule [a55_alu_fwd].  Every other producer
//     (shift/extend ALU 2, bitfield 2, MUL, DIV, FP->GPR) is seen at its SOG exec latency (l.640).
//   * shifter operands (shifted/extended Rm, variable shifts, UBFM/SBFM/BFM Rn, EXTR) get no advance (InstRW below).
//   * address base: advance only from loads (pointer-chasing path, l.357-359); ALU -> AGU goes to iss (l.349-350).
def : ReadAdvance<ReadVLD, 0>;
def : ReadAdvance<ReadExtrHi, 0>;
// [a55_sog_impl #9g] the consumer side (ReadAdrBase on the base operand) is attached to every
//   LDR/LDUR/LDTR/LDP/LDAR form below, not only the register-offset ones, so "ldr x1,[x1]" sees the SOG
//   pointer-chasing latency 2 (SOG 3.3 p.11, 4.8 note 1); [a55_sog_n3] also to the base of every store form
//   with an SOG row and of the ASIMD LD1-LD4 / ST1-ST4 rows.
// [a55_sog_n2] The pointer-chasing producers are a list of their own, no longer the ALU load-use list below
//   (C-6).  SOG 3.3 Table 3 (p.11, l.492-495): "LDR & LDRT/LDTR (all variants)" (WriteLD, WriteLDIdx) and
//   "LDRD/LDP (all variants): Only the first register of the pair if 32-bit load or the second register of the
//   pair if 64-bit load" -> LDP/LDNP W-form: the first register only (CortexA55WriteLDPW3SI, not
//   CortexA55WriteLDP1); X-form: the second register only (CortexA55WriteLDPX4b, not CortexA55WriteLDPX4 /
//   CortexA55WriteLDPX4SI).  LDPSW: none (SOG 4.8 "3", no parentheses; CortexA55WriteLDPSW* not listed).
//   Load acquire (LDAR/LDAXR) and LDAXP: "3 (2)" with note 1 in SOG 4.8; for LDAXP both registers, as before
//   (U-15).  WriteLDHi: the stock second write of LDXP (no SOG row, U-12), unchanged.
def : ReadAdvance<ReadAdrBase, 1, [WriteLD, WriteLDIdx, WriteLDHi, CortexA55WriteLDPW3SI,
                 CortexA55WriteLDPX4b, CortexA55WriteLdAcq,
                 CortexA55WriteLdAXP, CortexA55WriteLdAXPb]>;
// [a55_sog_u9] Store data: advance 1 from every producer (SOG 4.9 data "required at wr", 4.14 "required at f2",
//   3.1.1 ALU0 ex2 -> store ex2).  ReadST is on the data of every store form (InstRW rows below), not only of the
//   register-offset stores (docs U-9, C-18).
def : ReadAdvance<ReadST, 1>;

// [a55_alu_fwd] ALU -> ALU: the dedicated "ex1 of ALU0 -> ex1 of ALU1" path (SOG 3.1.1, l.351-352) only exists
//   between the two instructions of one dual-issue pair (instruction-0 in ALU0, its dependent instruction-1 in
//   ALU1, same cycle).  It is no longer expressed as a ReadAdvance from the 1-cycle ALU writes (which encoded
//   the same-cycle pair only implicitly, and for every consumer reading ReadI); it is a rule of the issue stage
//   (A55DecoupledIssueStage::isSameCycleALUForward, src/a55/a55_issue_stage.cpp).  Across cycles an ALU result
//   is seen at its SOG exec latency (ALU basic 1, SOG 4.3).
//   The ReadI / CortexA55ReadNotShifted reads below keep the load-use advance only (load-use 2, SOG 3.3 l.484);
//   the issue stage also uses "this read takes the load-use advance" to recognise an operand consumed by the ALU
//   (not by the shifter, cf. CortexA55ReadShifted), i.e. one that can take the ALU0 -> ALU1 path.
def : ReadAdvance<ReadI, 1, [
                 WriteLD, WriteLDIdx, WriteLDHi, CortexA55WriteLDP1, CortexA55WriteLDPW3SI,
                 CortexA55WriteLDPX4, CortexA55WriteLDPX4b, CortexA55WriteLDPX4SI, CortexA55WriteLdAcq,
                 CortexA55WriteLdAXP, CortexA55WriteLdAXPb, CortexA55WriteLDPSW3SI, CortexA55WriteLDPSW3b]>;
// Shifted operand
def CortexA55ReadShifted : SchedReadAdvance<0>;
def CortexA55ReadNotShifted : SchedReadAdvance<1, [
                 WriteLD, WriteLDIdx, WriteLDHi, CortexA55WriteLDP1, CortexA55WriteLDPW3SI,
                 CortexA55WriteLDPX4, CortexA55WriteLDPX4b, CortexA55WriteLDPX4SI, CortexA55WriteLdAcq,
                 CortexA55WriteLdAXP, CortexA55WriteLdAXPb, CortexA55WriteLDPSW3SI, CortexA55WriteLDPSW3b]>;
def CortexA55ReadISReg : SchedReadVariant<[
        SchedVar<RegShiftedPred, [CortexA55ReadShifted]>,
        SchedVar<NoSchedPred, [CortexA55ReadNotShifted]>]>;
def : SchedAlias<ReadISReg, CortexA55ReadISReg>;

def CortexA55ReadIEReg : SchedReadVariant<[
        SchedVar<RegExtendedPred, [CortexA55ReadShifted]>,
        SchedVar<NoSchedPred, [CortexA55ReadNotShifted]>]>;
def : SchedAlias<ReadIEReg, CortexA55ReadIEReg>;

// MUL / DIV: [a55_planB B2] operands take the SOG exec latency (no advance, SOG l.640, l.800);
//   accumulator advance 2 only from a MAC producer (SOG l.354-356, 4.5 note 2: MADD W 3 (1)).
def : ReadAdvance<ReadIM, 0>;
def : ReadAdvance<ReadIMA, 2, [WriteIM32,WriteIM64]>;

// Div
def : ReadAdvance<ReadID, 0>;

// [a55_sog_impl #6b] The former FP MAC accumulate advance 2 (CortexA55ReadFMAC, citing "SOG Table 4 Note b")
//   is removed: note b is attached only to the non-fused AArch32 VMLA/VMLS/VRECPS/VRSQRTS ("8 (4b)"); the fused
//   VFMA/FMADD row of Table 4 (p.13), FMADD in 4.11 (p.26) and FMLA in 4.16 (p.33-34) give a plain 4 with no
//   accumulate latency, so an FMLA/FMADD accumulator chain is 4 cycles per step.
// [a55_sog_impl #6a] ASIMD integer multiply-accumulate "4 (1)" (SOG 4.15 p.31-32): the accumulate operand is
//   forwarded with latency 1 from "similar instructions" (note 2: MLA/MLS incl. by element, [SU]MLAL/[SU]MLSL)
//   and between dot products (note 3: SDOT/UDOT incl. by element) -> advance 4 - 1 = 3 from those producers only.
def CortexA55ReadMlaAcc : SchedReadAdvance<3, [CortexA55WriteMlaVd_4, CortexA55WriteMlaVq_4,
                                               CortexA55WriteMlaIxVq_4, CortexA55WriteMlaLVq_4]>;
def CortexA55ReadDotAcc : SchedReadAdvance<3, [CortexA55WriteDotVd_4, CortexA55WriteDotVq_4,
                                               CortexA55WriteDotScVq_4]>;

//===----------------------------------------------------------------------===//
// Subtarget-specific InstRWs.

//---
// Miscellaneous
//---
// [a55_sog_impl #9] Load pair (SOG 4.8 p.21, 4.13 p.28); LDNP has the same rows as LDP ("LDP, LDNP") and was
//   missing from these regexes.  The last read is the base register (pointer chasing, #9g).
def : InstRW<[CortexA55WriteLDPW3SI, CortexA55WriteLDP1, ReadAdrBase], (instregex "^LDN?PWi$")>;            // W: 3 (2), 1, 00
def : InstRW<[CortexA55WriteLDPSW3SI, CortexA55WriteLDPSW3b, ReadAdrBase], (instrs LDPSWi)>;           // LDPSW: 3, 1, 00 [a55_sog_n2]
def : InstRW<[CortexA55WriteLDPS3, CortexA55WriteLDP1, ReadAdrBase], (instregex "^LDN?PSi$")>;            // S: 3, 1, 01
def : InstRW<[CortexA55WriteLDPX4, CortexA55WriteLDPX4b, ReadAdrBase], (instregex "^LDN?P(X|D)i$")>;      // X: 4 (3), 1/2, 01; D: 4, 1/2, 01
def : InstRW<[CortexA55WriteLDPQ6, CortexA55WriteLDPQ6b, ReadAdrBase], (instregex "^LDN?PQi$")>;          // Q: 6, 1/4, 01
def : InstRW<[WriteAdr, CortexA55WriteLDPW3SI, CortexA55WriteLDP1, ReadAdrBase], (instregex "^LDPW(pre|post)$")>;  // "3 (2), 1", 00
def : InstRW<[WriteAdr, CortexA55WriteLDPSW3SI, CortexA55WriteLDPSW3b, ReadAdrBase], (instregex "^LDPSW(pre|post)$")>; // [a55_sog_n2]
def : InstRW<[WriteAdr, CortexA55WriteLDPS3, CortexA55WriteLDP1, ReadAdrBase], (instregex "^LDPS(pre|post)$")>;      // "3, 1", 01
def : InstRW<[CortexA55WriteAdr2, CortexA55WriteLDPX4SI, CortexA55WriteLDPX4b, ReadAdrBase], (instregex "^LDPX(pre|post)$")>; // "4 (3), 2", 00
def : InstRW<[CortexA55WriteAdr2, CortexA55WriteLDPX4, CortexA55WriteLDPX4b, ReadAdrBase], (instregex "^LDPD(pre|post)$")>;   // "4, 2", 01
def : InstRW<[CortexA55WriteAdr4, CortexA55WriteLDPQ6, CortexA55WriteLDPQ6b, ReadAdrBase], (instregex "^LDPQ(pre|post)$")>;   // "6, 4", 01
// [a55_sog_impl #9a/#9g] Single-register loads.  GPR and S/D-form: 3 (2), 1, 11 (pre/post ", 1");
//   vector B/H-form: 3, 1, 01; vector Q-form: 4, 1/2, 01 (pre/post "4, 2").
def : InstRW<[WriteLD, ReadAdrBase], (instregex "^LDR(W|X|BB|HH|SBW|SBX|SHW|SHX|SW|S|D)ui$",
                                                "^LDUR(W|X|BB|HH|SBW|SBX|SHW|SHX|SW|S|D)i$",
                                                "^LDTR(W|X|B|H|SBW|SBX|SHW|SHX|SW)i$")>;
def : InstRW<[WriteAdr, WriteLD, ReadAdrBase], (instregex "^LDR(W|X|BB|HH|SBW|SBX|SHW|SHX|SW|S|D)(pre|post)$")>;
def : InstRW<[CortexA55WriteLdBH3, ReadAdrBase], (instregex "^LDR(B|H)(ui|roW|roX)$", "^LDUR(B|H)i$")>;
def : InstRW<[WriteAdr, CortexA55WriteLdBH3, ReadAdrBase], (instregex "^LDR(B|H)(pre|post)$")>;
def : InstRW<[CortexA55WriteLdQ4, ReadAdrBase], (instregex "^LDRQ(ui|roW|roX)$", "^LDURQi$")>;
def : InstRW<[CortexA55WriteLdQ4], (instrs LDRQl)>;
def : InstRW<[CortexA55WriteAdr2, CortexA55WriteLdQ4, ReadAdrBase], (instregex "^LDRQ(pre|post)$")>;
// [a55_sog_impl #9i] load acquire 3 (2), 1, 01; load acquire exclusive pair LDAXP 3(2), 1, 00 (SOG 4.8 p.21).
def : InstRW<[CortexA55WriteLdAcq, ReadAdrBase], (instregex "^LDL?AX?R(B|H|W|X)$")>;
def : InstRW<[CortexA55WriteLdAXP, CortexA55WriteLdAXPb, ReadAdrBase], (instrs LDAXPW, LDAXPX)>;
// [a55_sog_impl #9k] PRFM: 1, 1, 11 (SOG 4.8 p.21).
def : InstRW<[CortexA55WritePRFM], (instrs PRFMui, PRFMl, PRFMroW, PRFMroX, PRFUMi)>;
// [a55_sog_impl #9j] STLR/STLLR 2, 1/2, 01; STLXR/STLXP 4, 1/2, 01 (SOG 4.9 p.23).
def : InstRW<[CortexA55WriteSTLR, ReadST, ReadAdrBase], (instregex "^STL?LR(B|H|W|X)$")>;
def : InstRW<[CortexA55WriteSTLXR, ReadST, ReadAdrBase], (instregex "^STLXR(B|H|W|X)$")>;
def : InstRW<[CortexA55WriteSTLXR, ReadST, ReadST, ReadAdrBase], (instregex "^STLXP(W|X)$")>;
// [a55_sog_impl #9b] STP/STNP Q-form 2, 1/2, 01; writeback "(1/T)" = 2 (SOG 4.14 p.29).
def : InstRW<[CortexA55WriteSTPQ, ReadST, ReadST, ReadAdrBase], (instrs STPQi, STNPQi)>;
def : InstRW<[CortexA55WriteAdr2, CortexA55WriteSTPQ, ReadST, ReadST, ReadAdrBase], (instrs STPQpre, STPQpost)>;
// [a55_sog_n3] Store base registers take the pointer-chasing path like load bases: SOG 3.3 (p.11, l.487-489) load
//   data "can be forwarded from the beginning of the wr pipeline stage to either the load or store AGU base
//   operand".  The stock Sched lists of these forms (AArch64InstrFormats.td: [WriteST], [WriteAdr, WriteST],
//   [WriteSTP], [WriteAdr, WriteSTP]) name no reads, so their base waited the full load latency; only the
//   register-offset stores ([WriteSTIdx, ReadST, ReadAdrBase]) had it.  The writes are the stock ones.  Reads in
//   operand order: Rt (, Rt2), Rn.  STLR/STLXR/STLXP and STP Q: rows above.
// [a55_sog_u9] The data operands (Rt, Rt2) read ReadST (advance 1 from every producer, below): the store data is
//   needed later than an ordinary operand.  SOG 4.9 (p.22, l.1108-1109) "Stores may issue to L1 at iss once their
//   address operands are available and do not need to wait for data operands (which are required at wr)";
//   3.1.1 (p.9, l.351-352) the dedicated path "from the ex2 stage of the ALU0 to the ex2 stage of the store
//   pipeline"; FP stores 4.14 (p.28, l.1521-1522) data "required at f2" (FP operands are read in f1, p.8);
//   5.7 (p.43, l.2453-2456) similar instructions behave alike.  The register-offset stores had ReadST already
//   (stock Sched list); now every store with an SOG row has it (docs U-9: the "natural reading", advance 1).
def : InstRW<[WriteST, ReadST, ReadAdrBase], (instregex "^STR(BB|HH|W|X|B|H|S|D|Q)ui$",
                                                             "^STUR(BB|HH|W|X|B|H|S|D|Q)i$",
                                                             "^STTR(B|H|W|X)i$")>;
def : InstRW<[WriteAdr, WriteST, ReadST, ReadAdrBase], (instregex "^STR(BB|HH|W|X|B|H|S|D|Q)(pre|post)$")>;
def : InstRW<[WriteSTP, ReadST, ReadST, ReadAdrBase], (instregex "^STN?P(W|X|S|D)i$")>;
def : InstRW<[WriteAdr, WriteSTP, ReadST, ReadST, ReadAdrBase], (instregex "^STP(W|X|S|D)(pre|post)$")>;
// [a55_sog_impl #3] Atomics (SOG 4.10 p.23-25); see CortexA55WriteLdOp above.
def : InstRW<[CortexA55WriteLdOp],  (instregex "^LD(ADD|CLR|EOR|SET|SMAX|SMIN|UMAX|UMIN)A?(B|H|W|X)$")>;
def : InstRW<[CortexA55WriteLdOpL], (instregex "^LD(ADD|CLR|EOR|SET|SMAX|SMIN|UMAX|UMIN)A?L(B|H|W|X)$")>;
def : InstRW<[CortexA55WriteAtomic<2, 2>], (instregex "^SWPA?(B|H|W|X)$")>;      // SWP{A}: 2, 1/2, 01
def : InstRW<[CortexA55WriteAtomic<3, 3>], (instregex "^SWPA?L(B|H|W|X)$")>;     // SWPL/SWPAL: 3, 1/3, 01
def : InstRW<[CortexA55WriteAtomic<4, 4>], (instregex "^CASA?(B|H|W|X)$")>;      // CAS{A}: 4, 1/4, 01
def : InstRW<[CortexA55WriteAtomic<3, 3>], (instregex "^CASA?L(B|H|W|X)$")>;     // CASL/CASAL: 3, 1/3, 01
def : InstRW<[CortexA55WriteAtomicSI<5, 5>], (instregex "^CASPA?(W|X)$")>;     // CASP{A}: 5, 1/5, 00
def : InstRW<[CortexA55WriteAtomicSI<4, 4>], (instregex "^CASPA?L(W|X)$")>;    // CASPL/CASPAL: 4, 1/4, 00
def : InstRW<[WriteI], (instrs COPY)>;
//---
// Vector Loads / Stores
//---
// [a55_sog_impl #9e/#9f] SOG 4.18 (p.37-38) and 4.19 (p.39-40), row by row: Exec latency, throughput T (the load
//   or store pipe is held 1/T cycles), dual-issue (all 01 -> BeginGroup, except LD1/ST1 one register: 11).
//   Writeback forms: base update latency "(1/T)" (CortexA55WriteVAdr).  The former values were the stock model's
//   "+1 for swizzling" latencies and a different occupancy table.  Stores retire out of order (SOG 4.19: "stores
//   are buffered and committed in the background").
// [a55_sog_n3] The last read of every row is the base register Rn (ReadAdrBase, pointer chasing): SOG 3.3 (p.11,
//   l.486-489) "The first stage (ex1) of both the load and store pipeline contains an AGU ... load data ... can be
//   forwarded ... to either the load or store AGU base operand".  The SOG does not name the ASIMD structure
//   loads/stores there (nor in 4.18/4.19); the statement is about the AGU base operand of the load/store pipes,
//   which these instructions use like any other load/store (docs C-7).  Operand order: LDn (multiple, LDnR): Rn
//   is use 0; LDn/STn one lane: Vt (tied), lane index, Rn = uses 0-2; STn multiple: Vt, Rn = uses 0-1.  The
//   post-index register Xm (use after Rn) gets nothing.  [a55_sog_u9] The store data Vt reads ReadST (advance 1):
//   SOG 4.19 (p.38, l.2193-2194) "Store instructions may issue once their address operands are available and do
//   not need to wait for data operands" (no stage given; taken as for the FP stores, 4.14 "required at f2", by
//   5.7).  For the one-lane loads the first read (the tied Vt) and the lane index stay ReadDefault.
class CortexA55WriteVLDx<int lat, int occ, bit bg> : SchedWriteRes<[CortexA55UnitLd]> {
  let Latency = lat; let ReleaseAtCycles = [occ]; let BeginGroup = bg;
}
class CortexA55WriteVSTx<int lat, int occ, bit bg> : SchedWriteRes<[CortexA55UnitSt]> {
  let Latency = lat; let ReleaseAtCycles = [occ]; let BeginGroup = bg; let RetireOOO = 1;
}
class CortexA55WriteVAdr<int lat> : SchedWriteRes<[]> { let Latency = lat; }
def : InstRW<[CortexA55WriteVLDx<3, 1, 0>, ReadAdrBase], (instregex "^LD1Onev(8b|4h|2s|1d)$")>; // LD1 1 reg D 3, 1, 11
def : InstRW<[CortexA55WriteVAdr<1>, CortexA55WriteVLDx<3, 1, 0>, ReadAdrBase], (instregex "^LD1Onev(8b|4h|2s|1d)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<4, 2, 1>, ReadAdrBase], (instregex "^LD1Onev(16b|8h|4s|2d)$")>; // LD1 1 reg Q 4, 1/2
def : InstRW<[CortexA55WriteVAdr<2>, CortexA55WriteVLDx<4, 2, 1>, ReadAdrBase], (instregex "^LD1Onev(16b|8h|4s|2d)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<4, 2, 1>, ReadAdrBase], (instregex "^LD1Twov(8b|4h|2s|1d)$")>; // LD1 2 reg D 4, 1/2
def : InstRW<[CortexA55WriteVAdr<2>, CortexA55WriteVLDx<4, 2, 1>, ReadAdrBase], (instregex "^LD1Twov(8b|4h|2s|1d)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<6, 4, 1>, ReadAdrBase], (instregex "^LD1Twov(16b|8h|4s|2d)$")>; // LD1 2 reg Q 6, 1/4
def : InstRW<[CortexA55WriteVAdr<4>, CortexA55WriteVLDx<6, 4, 1>, ReadAdrBase], (instregex "^LD1Twov(16b|8h|4s|2d)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<5, 3, 1>, ReadAdrBase], (instregex "^LD1Threev(8b|4h|2s|1d)$")>; // LD1 3 reg D 5, 1/3
def : InstRW<[CortexA55WriteVAdr<3>, CortexA55WriteVLDx<5, 3, 1>, ReadAdrBase], (instregex "^LD1Threev(8b|4h|2s|1d)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<8, 6, 1>, ReadAdrBase], (instregex "^LD1Threev(16b|8h|4s|2d)$")>; // LD1 3 reg Q 8, 1/6
def : InstRW<[CortexA55WriteVAdr<6>, CortexA55WriteVLDx<8, 6, 1>, ReadAdrBase], (instregex "^LD1Threev(16b|8h|4s|2d)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<6, 4, 1>, ReadAdrBase], (instregex "^LD1Fourv(8b|4h|2s|1d)$")>; // LD1 4 reg D 6, 1/4
def : InstRW<[CortexA55WriteVAdr<4>, CortexA55WriteVLDx<6, 4, 1>, ReadAdrBase], (instregex "^LD1Fourv(8b|4h|2s|1d)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<10, 8, 1>, ReadAdrBase], (instregex "^LD1Fourv(16b|8h|4s|2d)$")>; // LD1 4 reg Q 10, 1/8
def : InstRW<[CortexA55WriteVAdr<8>, CortexA55WriteVLDx<10, 8, 1>, ReadAdrBase], (instregex "^LD1Fourv(16b|8h|4s|2d)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<3, 1, 1>, ReadDefault, ReadDefault, ReadAdrBase], (instregex "^LD1i(8|16|32|64)$")>; // LD1 one lane 3, 1
def : InstRW<[CortexA55WriteVAdr<1>, CortexA55WriteVLDx<3, 1, 1>, ReadDefault, ReadDefault, ReadAdrBase], (instregex "^LD1i(8|16|32|64)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<3, 1, 1>, ReadAdrBase], (instregex "^LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)$")>; // LD1R 3, 1
def : InstRW<[CortexA55WriteVAdr<1>, CortexA55WriteVLDx<3, 1, 1>, ReadAdrBase], (instregex "^LD1Rv(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<4, 2, 1>, ReadAdrBase], (instregex "^LD2Twov(8b|4h|2s)$")>; // LD2 multiple D 4, 1/2
def : InstRW<[CortexA55WriteVAdr<2>, CortexA55WriteVLDx<4, 2, 1>, ReadAdrBase], (instregex "^LD2Twov(8b|4h|2s)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<6, 4, 1>, ReadAdrBase], (instregex "^LD2Twov(16b|8h|4s|2d)$")>; // LD2 multiple Q 6, 1/4
def : InstRW<[CortexA55WriteVAdr<4>, CortexA55WriteVLDx<6, 4, 1>, ReadAdrBase], (instregex "^LD2Twov(16b|8h|4s|2d)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<3, 1, 1>, ReadDefault, ReadDefault, ReadAdrBase], (instregex "^LD2i(8|16|32)$")>; // LD2 one lane B/H/S 3, 1
def : InstRW<[CortexA55WriteVAdr<1>, CortexA55WriteVLDx<3, 1, 1>, ReadDefault, ReadDefault, ReadAdrBase], (instregex "^LD2i(8|16|32)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<4, 2, 1>, ReadDefault, ReadDefault, ReadAdrBase], (instregex "^LD2i64$")>; // LD2 one lane D 4, 1/2
def : InstRW<[CortexA55WriteVAdr<2>, CortexA55WriteVLDx<4, 2, 1>, ReadDefault, ReadDefault, ReadAdrBase], (instregex "^LD2i64_POST$")>;
def : InstRW<[CortexA55WriteVLDx<3, 1, 1>, ReadAdrBase], (instregex "^LD2Rv(8b|4h|2s|16b|8h|4s)$")>; // LD2R B/H/S 3, 1
def : InstRW<[CortexA55WriteVAdr<1>, CortexA55WriteVLDx<3, 1, 1>, ReadAdrBase], (instregex "^LD2Rv(8b|4h|2s|16b|8h|4s)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<4, 2, 1>, ReadAdrBase], (instregex "^LD2Rv(1d|2d)$")>; // LD2R D 4, 1/2
def : InstRW<[CortexA55WriteVAdr<2>, CortexA55WriteVLDx<4, 2, 1>, ReadAdrBase], (instregex "^LD2Rv(1d|2d)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<6, 4, 1>, ReadAdrBase], (instregex "^LD3Threev(8b|4h)$")>; // LD3 multiple D-form B/H 6, 1/4
def : InstRW<[CortexA55WriteVAdr<4>, CortexA55WriteVLDx<6, 4, 1>, ReadAdrBase], (instregex "^LD3Threev(8b|4h)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<9, 7, 1>, ReadAdrBase], (instregex "^LD3Threev(16b|8h)$")>; // LD3 multiple Q-form B/H 9, 1/7
def : InstRW<[CortexA55WriteVAdr<7>, CortexA55WriteVLDx<9, 7, 1>, ReadAdrBase], (instregex "^LD3Threev(16b|8h)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<5, 3, 1>, ReadAdrBase], (instregex "^LD3Threev2s$")>; // LD3 multiple D-form S 5, 1/3
def : InstRW<[CortexA55WriteVAdr<3>, CortexA55WriteVLDx<5, 3, 1>, ReadAdrBase], (instregex "^LD3Threev2s_POST$")>;
def : InstRW<[CortexA55WriteVLDx<8, 6, 1>, ReadAdrBase], (instregex "^LD3Threev(4s|2d)$")>; // LD3 multiple Q-form S/D 8, 1/6
def : InstRW<[CortexA55WriteVAdr<6>, CortexA55WriteVLDx<8, 6, 1>, ReadAdrBase], (instregex "^LD3Threev(4s|2d)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<4, 2, 1>, ReadDefault, ReadDefault, ReadAdrBase], (instregex "^LD3i(8|16|32)$")>; // LD3 one lane B/H/S 4, 1/2
def : InstRW<[CortexA55WriteVAdr<2>, CortexA55WriteVLDx<4, 2, 1>, ReadDefault, ReadDefault, ReadAdrBase], (instregex "^LD3i(8|16|32)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<5, 3, 1>, ReadDefault, ReadDefault, ReadAdrBase], (instregex "^LD3i64$")>; // LD3 one lane D 5, 1/3
def : InstRW<[CortexA55WriteVAdr<3>, CortexA55WriteVLDx<5, 3, 1>, ReadDefault, ReadDefault, ReadAdrBase], (instregex "^LD3i64_POST$")>;
def : InstRW<[CortexA55WriteVLDx<4, 2, 1>, ReadAdrBase], (instregex "^LD3Rv(8b|4h|2s|16b|8h|4s)$")>; // LD3R B/H/S 4, 1/2
def : InstRW<[CortexA55WriteVAdr<2>, CortexA55WriteVLDx<4, 2, 1>, ReadAdrBase], (instregex "^LD3Rv(8b|4h|2s|16b|8h|4s)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<5, 3, 1>, ReadAdrBase], (instregex "^LD3Rv(1d|2d)$")>; // LD3R D 5, 1/3
def : InstRW<[CortexA55WriteVAdr<3>, CortexA55WriteVLDx<5, 3, 1>, ReadAdrBase], (instregex "^LD3Rv(1d|2d)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<7, 5, 1>, ReadAdrBase], (instregex "^LD4Fourv(8b|4h)$")>; // LD4 multiple D-form B/H 7, 1/5
def : InstRW<[CortexA55WriteVAdr<5>, CortexA55WriteVLDx<7, 5, 1>, ReadAdrBase], (instregex "^LD4Fourv(8b|4h)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<11, 9, 1>, ReadAdrBase], (instregex "^LD4Fourv(16b|8h)$")>; // LD4 multiple Q-form B/H 11, 1/9
def : InstRW<[CortexA55WriteVAdr<9>, CortexA55WriteVLDx<11, 9, 1>, ReadAdrBase], (instregex "^LD4Fourv(16b|8h)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<6, 4, 1>, ReadAdrBase], (instregex "^LD4Fourv2s$")>; // LD4 multiple D-form S 6, 1/4
def : InstRW<[CortexA55WriteVAdr<4>, CortexA55WriteVLDx<6, 4, 1>, ReadAdrBase], (instregex "^LD4Fourv2s_POST$")>;
def : InstRW<[CortexA55WriteVLDx<10, 8, 1>, ReadAdrBase], (instregex "^LD4Fourv(4s|2d)$")>; // LD4 multiple Q-form S/D 10, 1/8
def : InstRW<[CortexA55WriteVAdr<8>, CortexA55WriteVLDx<10, 8, 1>, ReadAdrBase], (instregex "^LD4Fourv(4s|2d)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<4, 2, 1>, ReadDefault, ReadDefault, ReadAdrBase], (instregex "^LD4i(8|16|32)$")>; // LD4 one lane B/H/S 4, 1/2
def : InstRW<[CortexA55WriteVAdr<2>, CortexA55WriteVLDx<4, 2, 1>, ReadDefault, ReadDefault, ReadAdrBase], (instregex "^LD4i(8|16|32)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<6, 4, 1>, ReadDefault, ReadDefault, ReadAdrBase], (instregex "^LD4i64$")>; // LD4 one lane D 6, 1/4
def : InstRW<[CortexA55WriteVAdr<4>, CortexA55WriteVLDx<6, 4, 1>, ReadDefault, ReadDefault, ReadAdrBase], (instregex "^LD4i64_POST$")>;
def : InstRW<[CortexA55WriteVLDx<4, 2, 1>, ReadAdrBase], (instregex "^LD4Rv(8b|4h|2s|16b|8h|4s)$")>; // LD4R B/H/S 4, 1/2
def : InstRW<[CortexA55WriteVAdr<2>, CortexA55WriteVLDx<4, 2, 1>, ReadAdrBase], (instregex "^LD4Rv(8b|4h|2s|16b|8h|4s)_POST$")>;
def : InstRW<[CortexA55WriteVLDx<6, 4, 1>, ReadAdrBase], (instregex "^LD4Rv(1d|2d)$")>; // LD4R D 6, 1/4
def : InstRW<[CortexA55WriteVAdr<4>, CortexA55WriteVLDx<6, 4, 1>, ReadAdrBase], (instregex "^LD4Rv(1d|2d)_POST$")>;
def : InstRW<[CortexA55WriteVSTx<1, 1, 0>, ReadST, ReadAdrBase], (instregex "^ST1Onev(8b|4h|2s|1d|16b|8h|4s|2d)$")>; // ST1 1 reg 1, 1, 11
def : InstRW<[CortexA55WriteVAdr<1>, CortexA55WriteVSTx<1, 1, 0>, ReadST, ReadAdrBase], (instregex "^ST1Onev(8b|4h|2s|1d|16b|8h|4s|2d)_POST$")>;
def : InstRW<[CortexA55WriteVSTx<1, 1, 1>, ReadST, ReadAdrBase], (instregex "^ST1Twov(8b|4h|2s|1d)$")>; // ST1 2 reg D 1, 1
def : InstRW<[CortexA55WriteVAdr<1>, CortexA55WriteVSTx<1, 1, 1>, ReadST, ReadAdrBase], (instregex "^ST1Twov(8b|4h|2s|1d)_POST$")>;
def : InstRW<[CortexA55WriteVSTx<2, 2, 1>, ReadST, ReadAdrBase], (instregex "^ST1Twov(16b|8h|4s|2d)$")>; // ST1 2 reg Q 2, 1/2
def : InstRW<[CortexA55WriteVAdr<2>, CortexA55WriteVSTx<2, 2, 1>, ReadST, ReadAdrBase], (instregex "^ST1Twov(16b|8h|4s|2d)_POST$")>;
def : InstRW<[CortexA55WriteVSTx<2, 2, 1>, ReadST, ReadAdrBase], (instregex "^ST1Threev(8b|4h|2s|1d)$")>; // ST1 3 reg D 2, 1/2
def : InstRW<[CortexA55WriteVAdr<2>, CortexA55WriteVSTx<2, 2, 1>, ReadST, ReadAdrBase], (instregex "^ST1Threev(8b|4h|2s|1d)_POST$")>;
def : InstRW<[CortexA55WriteVSTx<3, 3, 1>, ReadST, ReadAdrBase], (instregex "^ST1Threev(16b|8h|4s|2d)$")>; // ST1 3 reg Q 3, 1/3
def : InstRW<[CortexA55WriteVAdr<3>, CortexA55WriteVSTx<3, 3, 1>, ReadST, ReadAdrBase], (instregex "^ST1Threev(16b|8h|4s|2d)_POST$")>;
def : InstRW<[CortexA55WriteVSTx<2, 2, 1>, ReadST, ReadAdrBase], (instregex "^ST1Fourv(8b|4h|2s|1d)$")>; // ST1 4 reg D 2, 1/2
def : InstRW<[CortexA55WriteVAdr<2>, CortexA55WriteVSTx<2, 2, 1>, ReadST, ReadAdrBase], (instregex "^ST1Fourv(8b|4h|2s|1d)_POST$")>;
def : InstRW<[CortexA55WriteVSTx<4, 4, 1>, ReadST, ReadAdrBase], (instregex "^ST1Fourv(16b|8h|4s|2d)$")>; // ST1 4 reg Q 4, 1/4
def : InstRW<[CortexA55WriteVAdr<4>, CortexA55WriteVSTx<4, 4, 1>, ReadST, ReadAdrBase], (instregex "^ST1Fourv(16b|8h|4s|2d)_POST$")>;
def : InstRW<[CortexA55WriteVSTx<1, 1, 1>, ReadST, ReadDefault, ReadAdrBase], (instregex "^ST1i(8|16|32|64)$")>; // ST1 one lane 1, 1
def : InstRW<[CortexA55WriteVAdr<1>, CortexA55WriteVSTx<1, 1, 1>, ReadST, ReadDefault, ReadAdrBase], (instregex "^ST1i(8|16|32|64)_POST$")>;
def : InstRW<[CortexA55WriteVSTx<1, 1, 1>, ReadST, ReadAdrBase], (instregex "^ST2Twov(8b|4h|2s)$")>; // ST2 multiple D 1, 1
def : InstRW<[CortexA55WriteVAdr<1>, CortexA55WriteVSTx<1, 1, 1>, ReadST, ReadAdrBase], (instregex "^ST2Twov(8b|4h|2s)_POST$")>;
def : InstRW<[CortexA55WriteVSTx<2, 2, 1>, ReadST, ReadAdrBase], (instregex "^ST2Twov(16b|8h|4s|2d)$")>; // ST2 multiple Q 2, 1/2
def : InstRW<[CortexA55WriteVAdr<2>, CortexA55WriteVSTx<2, 2, 1>, ReadST, ReadAdrBase], (instregex "^ST2Twov(16b|8h|4s|2d)_POST$")>;
def : InstRW<[CortexA55WriteVSTx<1, 1, 1>, ReadST, ReadDefault, ReadAdrBase], (instregex "^ST2i(8|16|32)$")>; // ST2 one lane B/H/S 1, 1
def : InstRW<[CortexA55WriteVAdr<1>, CortexA55WriteVSTx<1, 1, 1>, ReadST, ReadDefault, ReadAdrBase], (instregex "^ST2i(8|16|32)_POST$")>;
def : InstRW<[CortexA55WriteVSTx<2, 2, 1>, ReadST, ReadDefault, ReadAdrBase], (instregex "^ST2i64$")>; // ST2 one lane D 2, 1/2
def : InstRW<[CortexA55WriteVAdr<2>, CortexA55WriteVSTx<2, 2, 1>, ReadST, ReadDefault, ReadAdrBase], (instregex "^ST2i64_POST$")>;
def : InstRW<[CortexA55WriteVSTx<3, 3, 1>, ReadST, ReadAdrBase], (instregex "^ST3Threev(8b|4h|2s|16b|8h|4s|2d)$")>; // ST3 multiple B/H/S and D 3, 1/3
def : InstRW<[CortexA55WriteVAdr<3>, CortexA55WriteVSTx<3, 3, 1>, ReadST, ReadAdrBase], (instregex "^ST3Threev(8b|4h|2s|16b|8h|4s|2d)_POST$")>;
def : InstRW<[CortexA55WriteVSTx<2, 2, 1>, ReadST, ReadDefault, ReadAdrBase], (instregex "^ST3i(8|16|32|64)$")>; // ST3 one lane 2, 1/2
def : InstRW<[CortexA55WriteVAdr<2>, CortexA55WriteVSTx<2, 2, 1>, ReadST, ReadDefault, ReadAdrBase], (instregex "^ST3i(8|16|32|64)_POST$")>;
def : InstRW<[CortexA55WriteVSTx<3, 3, 1>, ReadST, ReadAdrBase], (instregex "^ST4Fourv(8b|4h|2s)$")>; // ST4 multiple D-form B/H/S 3, 1/3
def : InstRW<[CortexA55WriteVAdr<3>, CortexA55WriteVSTx<3, 3, 1>, ReadST, ReadAdrBase], (instregex "^ST4Fourv(8b|4h|2s)_POST$")>;
def : InstRW<[CortexA55WriteVSTx<5, 5, 1>, ReadST, ReadAdrBase], (instregex "^ST4Fourv(16b|8h|4s)$")>; // ST4 multiple Q-form B/H/S 5, 1/5
def : InstRW<[CortexA55WriteVAdr<5>, CortexA55WriteVSTx<5, 5, 1>, ReadST, ReadAdrBase], (instregex "^ST4Fourv(16b|8h|4s)_POST$")>;
def : InstRW<[CortexA55WriteVSTx<4, 4, 1>, ReadST, ReadAdrBase], (instregex "^ST4Fourv2d$")>; // ST4 multiple Q-form D 4, 1/4
def : InstRW<[CortexA55WriteVAdr<4>, CortexA55WriteVSTx<4, 4, 1>, ReadST, ReadAdrBase], (instregex "^ST4Fourv2d_POST$")>;
def : InstRW<[CortexA55WriteVSTx<2, 2, 1>, ReadST, ReadDefault, ReadAdrBase], (instregex "^ST4i(8|16)$")>; // ST4 one lane B/H 2, 1/2
def : InstRW<[CortexA55WriteVAdr<2>, CortexA55WriteVSTx<2, 2, 1>, ReadST, ReadDefault, ReadAdrBase], (instregex "^ST4i(8|16)_POST$")>;
// ST4 one lane S/D: no SOG 4.19 row (only "ST4, one lane, B/H") -> left at the former values.
def : InstRW<[CortexA55WriteVST2, ReadST, ReadDefault, ReadAdrBase], (instregex "^ST4i(32|64)$")>;
def : InstRW<[WriteAdr, CortexA55WriteVST2, ReadST, ReadDefault, ReadAdrBase], (instregex "^ST4i(32|64)_POST$")>;

//---
// Floating Point Conversions, MAC, DIV, SQRT
//---
// [a55_sog_impl #10] ASIMD duplicate (SOG 4.17 p.35): gen reg 3, 2, 11 (no Q-form note) -> 3 for every form;
//   element (incl. the scalar DUPi* = "mov b/h/s/d, v.t[i]") 2, 2*, 11*.  (Was: Q-forms of both 2, D-forms 4.)
def : InstRW<[CortexA55WriteFPALU_F3], (instregex "^DUPv(8i8|16i8|4i16|8i16|2i32|4i32|2i64)gpr$")>;
def : InstRW<[CortexA55WriteAluVd_2], (instregex "^DUPv(8i8|4i16|2i32)lane$", "^DUPi(8|16|32|64)$")>;
def : InstRW<[CortexA55WriteAluVq_2], (instregex "^DUPv(16i8|8i16|4i32|2i64)lane$")>;
def : InstRW<[CortexA55WriteFPALU_F2], (instregex "^XTN")>;
// [a55_sog_n7] ASIMD extract narrow, saturating SQXTN(2), SQXTUN(2), UQXTN(2): 4, 2, 11 with no Q-form note
//   (SOG 4.17 p.35, l.1991-1992; AArch32 VQMOVN/VQMOVUN likewise 4, 2, 11, l.1931) -> one FP ALU unit, every form
//   (D, Q = the "2" forms, scalar).  (Was: the vector forms on the WriteVq default, throughput 1, slot 0 only.)
def : InstRW<[CortexA55WriteFPALU_F4], (instregex "^(SQXTN|SQXTUN|UQXTN)v")>;
def : InstRW<[CortexA55WriteFPALU_F3], (instregex "^FCVT[ALMNPZ][SU](S|U)?(W|X)")>;
// [a55_sog_impl #10] ASIMD FP convert, other (FCVT[AMNPZ][SU], [SU]CVTF vector/scalar-vector): 4, 2*, 11*
//   (SOG 4.16 p.33): Q-forms throughput 1, slot 0 only.  FCVTL/FCVTN/FCVTXN: 4, 1, 01 (SOG 4.16 p.33, 4.12 p.27).
def : InstRW<[CortexA55WriteFPALU_F4], (instregex "^FCVT[AMNPZ][SU]v(1i32|1i64|1f16|2f32|4f16|2i32_shift|4i16_shift)$")>;
def : InstRW<[CortexA55WriteAluVq_4], (instregex "^FCVT[AMNPZ][SU]v(4f32|2f64|8f16|4i32_shift|2i64_shift|8i16_shift)$")>;
def : InstRW<[CortexA55WriteAluVq_4], (instregex "^FCVT(L|N|XN)v")>;

def : InstRW<[CortexA55WriteFPALU_F5], (instregex "^(S|U)CVTF(S|U)(W|X)(H|S|D)")>; // [a55_td_fix] SOG l.1423: 5 (was F4)
def : InstRW<[CortexA55WriteFPALU_F4], (instregex "^(S|U)CVTF(h|s|d)")>;
def : InstRW<[CortexA55WriteFPALU_F4], (instregex "^(S|U)CVTFv(1i32|1i64|1i16|2f32|4f16|2i32_shift|4i16_shift)$")>;
def : InstRW<[CortexA55WriteAluVq_4], (instregex "^(S|U)CVTFv(4f32|2f64|8f16|4i32_shift|2i64_shift|8i16_shift)$")>;

// [a55_sog_impl #6b] FMADD/FMLA: 4 with no accumulate forwarding (see CortexA55ReadMlaAcc above).
//   ASIMD FMLA/FMLS: D-form 4, 2, 11; Q-form 4, 2*, 11* and by element 4, 1, 01 (SOG 4.16 p.34).
def : InstRW<[CortexA55WriteFMAC], (instregex "^FN?M(ADD|SUB).*")>;
def : InstRW<[CortexA55WriteFMAC], (instregex "^FML[AS]v(2f32|4f16)$", "^FML[AS]L")>;
def : InstRW<[CortexA55WriteFMACq], (instregex "^FML[AS]v(4f32|2f64|8f16)$", "^FML[AS]v.*_indexed$")>;
// [a55_sog_impl #10] ASIMD FP multiply by element (FMUL/FMULX, incl. scalar by element): 4, 1, 01 (SOG 4.16 p.34).
def : InstRW<[CortexA55WriteAluVq_4], (instregex "^FMULX?v.*_indexed$")>;
def : InstRW<[CortexA55WriteFDivHP], (instrs FDIVHrr)>;
def : InstRW<[CortexA55WriteFDivSP], (instrs FDIVSrr)>;
def : InstRW<[CortexA55WriteFDivDP], (instrs FDIVDrr)>;
def : InstRW<[CortexA55WriteFDivHP], (instregex "^FDIVv.*16$")>;
def : InstRW<[CortexA55WriteFDivSP], (instregex "^FDIVv.*32$")>;
def : InstRW<[CortexA55WriteFDivDP], (instregex "^FDIVv.*64$")>;
// [a55_sog_impl #5] FSQRT per precision (SOG Table 4 p.13, 4.11 p.26: H 8 / 1/5, S 12 / 1/9, D 22 / 1/19, 01;
//   "NEON and floating-point instructions of the same type have the same timing characteristics", 3.5.1).
//   The former "^.*SQRT.*(16|32|64)$" regexes missed the scalar FSQRTHr/Sr/Dr (left on WriteFDiv: 22 / 29) and
//   caught FRSQRTE/FRSQRTS/URSQRTE, which are ASIMD reciprocal (square root) estimate/step, 4, 2*, 11* (SOG 4.17
//   p.35), like FRECPE/FRECPS/FRECPX/URECPE.
def : InstRW<[CortexA55WriteFSqrtHP], (instrs FSQRTHr)>;
def : InstRW<[CortexA55WriteFSqrtSP], (instrs FSQRTSr)>;
def : InstRW<[CortexA55WriteFSqrtDP], (instrs FSQRTDr)>;
def : InstRW<[CortexA55WriteFSqrtHP], (instregex "^FSQRTv(4f16|8f16)$")>;
def : InstRW<[CortexA55WriteFSqrtSP], (instregex "^FSQRTv(2f32|4f32)$")>;
def : InstRW<[CortexA55WriteFSqrtDP], (instregex "^FSQRTv2f64$")>;
def : InstRW<[CortexA55WriteAluVd_4], (instregex "^(FRSQRTE|FRECPE|FRSQRTS|FRECPS|FRECPX)(16|32|64|v1f16|v1i16|v1i32|v1i64|v2f32|v4f16)$",
                                                 "^(URSQRTE|URECPE)v2i32$")>;
def : InstRW<[CortexA55WriteAluVq_4], (instregex "^(FRSQRTE|FRECPE|FRSQRTS|FRECPS)v(4f32|2f64|8f16)$",
                                                 "^(URSQRTE|URECPE)v4i32$")>;

// 4.15. Advanced SIMD integer instructions
// ASIMD absolute diff
def : InstRW<[CortexA55WriteAluVd_3], (instregex "[SU]ABDv(2i32|4i16|8i8)")>;
def : InstRW<[CortexA55WriteAluVq_3], (instregex "[SU]ABDv(16i8|4i32|8i16)")>;
// ASIMD absolute diff accum: 4, 1/2, 01 [a55_sog_impl #10]
def : InstRW<[CortexA55WriteAluT2_4], (instregex "[SU]ABAL?v")>;
// ASIMD absolute diff long
def : InstRW<[CortexA55WriteAluVq_3], (instregex "[SU]ABDLv")>;
// ASIMD arith #1
def : InstRW<[CortexA55WriteAluVd_2], (instregex "(ADD|SUB|NEG)v(1i64|2i32|4i16|8i8)",
  "[SU]R?HADDv(2i32|4i16|8i8)", "[SU]HSUBv(2i32|4i16|8i8)")>;
def : InstRW<[CortexA55WriteAluVq_2], (instregex "(ADD|SUB|NEG)v(2i64|4i32|8i16|16i8)",
  "[SU]R?HADDv(8i16|4i32|16i8)", "[SU]HSUBv(8i16|4i32|16i8)")>;
// ASIMD arith #2
def : InstRW<[CortexA55WriteAluVd_3], (instregex "ABSv(1i64|2i32|4i16|8i8)$",
  "[SU]ADDLPv(2i32_v1i64|4i16_v2i32|8i8_v4i16)$", 
  "([SU]QADD|[SU]QSUB|SQNEG|SUQADD|USQADD)v(1i16|1i32|1i64|1i8|2i32|4i16|8i8)$",
  "ADDPv(2i32|4i16|8i8)$")>;
def : InstRW<[CortexA55WriteAluVq_3], (instregex "ABSv(2i64|4i32|8i16|16i8)$",
  "[SU]ADDLPv(16i8_v8i16|4i32_v2i64|8i16_v4i32)$", 
  "([SU]QADD|[SU]QSUB|SQNEG|SUQADD|USQADD)v(16i8|2i64|4i32|8i16)$",
  "ADDPv(16i8|2i64|4i32|8i16)$")>;
// ASIMD arith #3
def : InstRW<[CortexA55WriteAluVq_3], (instregex  "SADDLv", "UADDLv", "SADDWv",
  "UADDWv", "SSUBLv", "USUBLv", "SSUBWv", "USUBWv", "ADDHNv", "SUBHNv")>;
// ASIMD arith #5: RADDHN/RSUBHN 4, 1/2, 01 [a55_sog_impl #10]
def : InstRW<[CortexA55WriteAluT2_4], (instregex "RADDHNv", "RSUBHNv")>;
// ASIMD arith, reduce
def : InstRW<[CortexA55WriteAluVq_3], (instregex  "ADDVv", "SADDLVv", "UADDLVv")>;
// ASIMD compare #1
def : InstRW<[CortexA55WriteAluVd_2], (instregex "CM(EQ|GE|GT|HI|HS|LE|LT)v(1i64|2i32|4i16|8i8)")>;
def : InstRW<[CortexA55WriteAluVq_2], (instregex "CM(EQ|GE|GT|HI|HS|LE|LT)v(2i64|4i32|8i16|16i8)")>;
// ASIMD compare #2
def : InstRW<[CortexA55WriteAluVd_3], (instregex "CMTSTv(1i64|2i32|4i16|8i8)")>;
def : InstRW<[CortexA55WriteAluVq_3], (instregex "CMTSTv(2i64|4i32|8i16|16i8)")>;
// ASIMD logical $1
def : InstRW<[CortexA55WriteAluVd_1], (instregex "(AND|EOR|NOT|ORN)v8i8",
  "ORRv(2i32|4i16)$", "BICv(2i32|4i16|8i8)$", "MVNIv(2i|2s|4i16)")>;
def : InstRW<[CortexA55WriteAluVq_1], (instregex "(AND|EOR|NOT|ORN)v16i8",
  "ORRv(4i32|8i16)$", "BICv(16i8|4i32|8i16)$", "MVNIv(4i32|4s|8i16)")>;
// [a55_sog_n6] ASIMD logical MOV 2, 2*, 11* (SOG 4.15 A64 p.31, l.1709), against ORR 1, 2*, 11* (l.1707-1708).
//   MOV Vd, Vn is ORR Vd, Vn, Vn: the register form of ORR with Rn == Rm takes the A64 MOV row (2); any other
//   ORR keeps the ORR row (1).  The AArch32 VMOV (register) row gives 1 (l.1925): docs/a55-sog.md I-8.
def CortexA55IsVecMovAlias : MCSchedPredicate<CheckSameRegOperand<1, 2>>;
def CortexA55WriteOrrVd : SchedWriteVariant<[SchedVar<CortexA55IsVecMovAlias, [CortexA55WriteAluVd_2]>,
                                             SchedVar<NoSchedPred, [CortexA55WriteAluVd_1]>]>;
def CortexA55WriteOrrVq : SchedWriteVariant<[SchedVar<CortexA55IsVecMovAlias, [CortexA55WriteAluVq_2]>,
                                             SchedVar<NoSchedPred, [CortexA55WriteAluVq_1]>]>;
def : InstRW<[CortexA55WriteOrrVd], (instrs ORRv8i8)>;
def : InstRW<[CortexA55WriteOrrVq], (instrs ORRv16i8)>;
// ASIMD max/min, basic
def : InstRW<[CortexA55WriteAluVd_2], (instregex "[SU](MIN|MAX)P?v(2i32|4i16|8i8)")>;
def : InstRW<[CortexA55WriteAluVq_2], (instregex "[SU](MIN|MAX)P?v(16i8|4i32|8i16)")>;
// SIMD max/min, reduce
def : InstRW<[CortexA55WriteAluVq_4], (instregex "[SU](MAX|MIN)Vv")>;
// ASIMD multiply, by element
def : InstRW<[CortexA55WriteAluVq_4], (instregex "MULv(2i32|4i16|4i32|8i16)_indexed$",
  "SQR?DMULHv(1i16|1i32|2i32|4i16|4i32|8i16)_indexed$")>;
// ASIMD multiply
def : InstRW<[CortexA55WriteAluVd_3], (instrs PMULv8i8)>;
def : InstRW<[CortexA55WriteAluVq_3], (instrs PMULv16i8)>;
// ASIMD multiply accumulate: "4 (1)" [a55_sog_impl #6a]; the first read is the accumulator (tied Rd).
def : InstRW<[CortexA55WriteMlaVd_4, CortexA55ReadMlaAcc], (instregex "ML[AS]v(2i32|4i16|8i8)$")>;
def : InstRW<[CortexA55WriteMlaVq_4, CortexA55ReadMlaAcc], (instregex "ML[AS]v(16i8|4i32|8i16)$")>;
def : InstRW<[CortexA55WriteMlaIxVq_4, CortexA55ReadMlaAcc], (instregex "ML[AS]v(2i32|4i16|4i32|8i16)_indexed$")>;
// ASIMD multiply accumulate half
def : InstRW<[CortexA55WriteAluVq_4], (instregex "SQRDML[AS]H[vi]")>;
// ASIMD multiply accumulate long
def : InstRW<[CortexA55WriteMlaLVq_4, CortexA55ReadMlaAcc], (instregex "[SU]ML[AS]Lv")>;
// ASIMD multiply accumulate long #2
def : InstRW<[CortexA55WriteAluVq_4], (instregex "SQDML[AS]L[iv]")>;
// ASIMD dot product
def : InstRW<[CortexA55WriteDotVd_4, CortexA55ReadDotAcc], (instregex "[SU]DOTv8i8")>;
def : InstRW<[CortexA55WriteDotVq_4, CortexA55ReadDotAcc], (instregex "[SU]DOTv16i8")>;
// ASIMD dot product, by scalar
def : InstRW<[CortexA55WriteDotScVq_4, CortexA55ReadDotAcc], (instregex "[SU]DOTlanev")>;
// ASIMD multiply long
def : InstRW<[CortexA55WriteAluVq_4], (instregex "[SU]MULLv", "SQDMULL[iv]")>;
// ASIMD polynomial (8x8) multiply long
def : InstRW<[CortexA55WriteAluVq_3], (instrs PMULLv8i8, PMULLv16i8)>;
// ASIMD pairwise add and accumulate: 4, 1/2, 01 [a55_sog_impl #10]
def : InstRW<[CortexA55WriteAluT2_4], (instregex "[SU]ADALPv")>;
// ASIMD shift accumulate
def : InstRW<[CortexA55WriteAluVd_3], (instregex "[SU]SRA(d|v2i32|v4i16|v8i8)")>;
def : InstRW<[CortexA55WriteAluVq_3], (instregex "[SU]SRAv(16i8|2i64|4i32|8i16)")>;
// ASIMD shift accumulate #2: SRSRA/URSRA 4, 1/2, 01 [a55_sog_impl #10]
def : InstRW<[CortexA55WriteAluT2_4], (instregex "[SU]RSRA[vd]")>;
// ASIMD shift by immed
def : InstRW<[CortexA55WriteAluVd_2], (instregex "SHLd$", "SHLv",
  "SLId$", "SRId$", "[SU]SHR[vd]", "SHRNv")>;
// ASIMD shift by immed
// SXTL and UXTL are aliases for SHLL
def : InstRW<[CortexA55WriteAluVq_2], (instregex "[US]?SHLLv")>;
// ASIMD shift by immed #2
def : InstRW<[CortexA55WriteAluVd_3], (instregex "[SU]RSHR(d|v2i32|v4i16|v8i8)",
  "RSHRNv(2i32|4i16|8i8)")>;
def : InstRW<[CortexA55WriteAluVq_3], (instregex "[SU]RSHRv(16i8|2i64|4i32|8i16)",
  "RSHRNv(16i8|4i32|8i16)")>;
// ASIMD shift by register
def : InstRW<[CortexA55WriteAluVd_2], (instregex "[SU]SHLv(1i64|2i32|4i16|8i8)")>;
def : InstRW<[CortexA55WriteAluVq_2], (instregex "[SU]SHLv(2i64|4i32|8i16|16i8)")>;
// ASIMD shift by register #2
def : InstRW<[CortexA55WriteAluVd_3], (instregex "[SU]RSHLv(1i64|2i32|4i16|8i8)")>;
def : InstRW<[CortexA55WriteAluVq_3], (instregex "[SU]RSHLv(2i64|4i32|8i16|16i8)")>;

//===----------------------------------------------------------------------===//
// [a55_td_fix] latencies aligned to the Cortex-A55 SOG v3 (automatic-llvm-mca/tmp/a55sog.txt)
//===----------------------------------------------------------------------===//
// SBFM/UBFM: "Bitfield move, basic" 2 (l.909); the SXTB/UXTB (and H/W) aliases
// (immr = 0, imms = 7/15/31): "Sign/zero extend" 1 (l.907).  BFM: "Bitfield move, insert" 2 (l.911).
def CortexA55IsExtendAlias : MCSchedPredicate<CheckAll<[CheckImmOperand<2, 0>,
                               CheckAny<[CheckImmOperand<3, 7>, CheckImmOperand<3, 15>, CheckImmOperand<3, 31>]>]>>;
// (CortexA55WriteBfm1/2 defined above, before the ReadAdvance lists [a55_planB])
//
// [a55_shift_imm] LSL/LSR/ASR by immediate (the UBFM/SBFM aliases) take 1 cycle.  NOTE: this is a project
//   decision, not the literal AArch64 table: SOG 4.7 lists only "Bitfield move, basic SBFM, UBFM 2" (l.909) and
//   has no AArch64 row for shift-by-immediate, whereas the AArch32 table 4.4 gives "Move, shift by immed
//   ASR/LSL/LSR/ROR 1, 2, 11" (l.729).  The AArch32 and AArch64 shifts are taken to use the same shifter (the
//   "shift" box in ex1 of both ALU pipes, SOG fig.1 l.271-281), so the AArch32 row is applied to the AArch64
//   aliases.  Other UBFM/SBFM aliases (UBFX/SBFX/UBFIZ/SBFIZ) keep "Bitfield move, basic" 2.
//   Alias conditions (operand 2 = immr, operand 3 = imms; W: size 32, X: size 64):
//     LSR #n = UBFM immr=n, imms=size-1        ASR #n = SBFM immr=n, imms=size-1   (n = 0..size-1)
//     LSL #n = UBFM immr=size-n, imms=size-1-n (n = 1..size-1), i.e. imms != size-1 and immr == imms+1.
//   (n = 0: UBFM #0,#size-1 is "LSR #0" -- also printed as LSL #0 -- a plain copy; it is caught by the LSR
//   condition.  For W it is also the existing extend predicate, immr=0, imms=31, which gives the same 1.)
//   A shift-immediate source still goes through the shifter (CortexA55ReadShifted): it cannot take the
//   same-cycle ALU0 -> ALU1 path as a consumer; as a 1-cycle ALU-pipe result it can as a producer [a55_alu_fwd].
class CortexA55IsShiftRightAlias<int size> : MCSchedPredicate<CheckImmOperand<3, !sub(size, 1)>>;
class CortexA55IsLslAlias<int size> : MCSchedPredicate<CheckAny<
        !foreach(s, !range(!sub(size, 1)), CheckAll<[CheckImmOperand<2, !add(s, 1)>, CheckImmOperand<3, s>]>)>>;
def CortexA55IsShiftRightAliasW : CortexA55IsShiftRightAlias<32>;
def CortexA55IsShiftRightAliasX : CortexA55IsShiftRightAlias<64>;
def CortexA55IsLslAliasW : CortexA55IsLslAlias<32>;
def CortexA55IsLslAliasX : CortexA55IsLslAlias<64>;
def CortexA55WriteShiftImm1 : SchedWriteRes<[CortexA55UnitALU]> { let Latency = 1; }
// (replaces the former CortexA55WriteXBFM = [extend alias -> 1, else 2] shared by all four opcodes)
// UBFM: LSR or LSL alias -> 1; SBFM: ASR alias -> 1; then the extend aliases (1) and the rest (2) as before.
def CortexA55WriteUBFMW : SchedWriteVariant<[SchedVar<CortexA55IsShiftRightAliasW, [CortexA55WriteShiftImm1]>,
                                             SchedVar<CortexA55IsLslAliasW, [CortexA55WriteShiftImm1]>,
                                             SchedVar<CortexA55IsExtendAlias, [CortexA55WriteBfm1]>,
                                             SchedVar<NoSchedPred, [CortexA55WriteBfm2]>]>;
def CortexA55WriteUBFMX : SchedWriteVariant<[SchedVar<CortexA55IsShiftRightAliasX, [CortexA55WriteShiftImm1]>,
                                             SchedVar<CortexA55IsLslAliasX, [CortexA55WriteShiftImm1]>,
                                             SchedVar<CortexA55IsExtendAlias, [CortexA55WriteBfm1]>,
                                             SchedVar<NoSchedPred, [CortexA55WriteBfm2]>]>;
def CortexA55WriteSBFMW : SchedWriteVariant<[SchedVar<CortexA55IsShiftRightAliasW, [CortexA55WriteShiftImm1]>,
                                             SchedVar<CortexA55IsExtendAlias, [CortexA55WriteBfm1]>,
                                             SchedVar<NoSchedPred, [CortexA55WriteBfm2]>]>;
def CortexA55WriteSBFMX : SchedWriteVariant<[SchedVar<CortexA55IsShiftRightAliasX, [CortexA55WriteShiftImm1]>,
                                             SchedVar<CortexA55IsExtendAlias, [CortexA55WriteBfm1]>,
                                             SchedVar<NoSchedPred, [CortexA55WriteBfm2]>]>;
// [a55_planB] the bitfield source goes through the shifter in ex1 (SOG fig.1, l.271-281): no advance.
def : InstRW<[CortexA55WriteUBFMW, CortexA55ReadShifted], (instrs UBFMWri)>;
def : InstRW<[CortexA55WriteUBFMX, CortexA55ReadShifted], (instrs UBFMXri)>;
def : InstRW<[CortexA55WriteSBFMW, CortexA55ReadShifted], (instrs SBFMWri)>;
def : InstRW<[CortexA55WriteSBFMX, CortexA55ReadShifted], (instrs SBFMXri)>;
//   BFM: operand 1 = Rd (tied, merged in the ALU), operand 2 = Rn (rotated by the shifter).
def : InstRW<[CortexA55WriteBfm2, ReadI, CortexA55ReadShifted], (instrs BFMWri, BFMXri)>;
// [a55_ror_imm] ROR by immediate (the EXTR alias with Rn == Rm) takes 1 cycle, for the same reason as LSL/LSR/ASR
//   by immediate above [a55_shift_imm]: a project decision applying the AArch32 row "Move, shift by immed
//   ASR/LSL/LSR/ROR 1, 2, 11" (SOG 4.4, l.729) to the AArch64 alias.  Read literally, the AArch64 table gives
//   "Bitfield extract EXTR 2, 2, 11" (SOG 4.7, l.905) for every EXTR, including ROR, i.e. 2.  An EXTR with
//   Rn != Rm has no AArch32 counterpart and keeps the SOG 4.7 value 2.
//   MC operands of EXTR{W,X}rri: 0 = Rd, 1 = Rn, 2 = Rm, 3 = lsb ("ror x0, x1, #n" = "extr x0, x1, x1, #n").
//   Like the shift-immediate aliases, both sources go through the shifter (CortexA55ReadShifted): never a consumer
//   of the same-cycle ALU0 -> ALU1 path; the 1-cycle form (CortexA55WriteShiftImm1) is a producer [a55_alu_fwd].
def CortexA55IsRorImmAlias : MCSchedPredicate<CheckSameRegOperand<1, 2>>;
def CortexA55WriteExtr2 : SchedWriteRes<[CortexA55UnitALU]> { let Latency = 2; }   // = WriteExtr above
def CortexA55WriteEXTR : SchedWriteVariant<[SchedVar<CortexA55IsRorImmAlias, [CortexA55WriteShiftImm1]>,
                                            SchedVar<NoSchedPred, [CortexA55WriteExtr2]>]>;
def : InstRW<[CortexA55WriteEXTR, CortexA55ReadShifted, CortexA55ReadShifted], (instrs EXTRWrri, EXTRXrri)>;
// [a55_planB] variable shifts: both the value and the amount feed the shifter (SOG l.758, fig.1): no advance.
// [a55_varshift] latency 1, throughput 1 (SOG 4.4 l.758): one ALU and CortexA55UnitVarShift for one cycle.
//   Still a 1-cycle ALU-pipe result, i.e. a producer of the same-cycle ALU0 -> ALU1 path (the issue stage counts
//   CortexA55UnitVarShift as part of the ALU pipes) [a55_alu_fwd].
def CortexA55WriteVarShift : SchedWriteRes<[CortexA55UnitALU, CortexA55UnitVarShift]> { let Latency = 1; }
def : InstRW<[CortexA55WriteVarShift, CortexA55ReadShifted, CortexA55ReadShifted], (instrs LSLVWr, LSLVXr, LSRVWr, LSRVXr, ASRVWr, ASRVXr, RORVWr, RORVXr)>;
// SMULH/UMULH 6, 1/3 (l.819)
def : InstRW<[CortexA55WriteIMH, ReadIM, ReadIM], (instrs SMULHrr, UMULHrr)>;
// FP move register 1 (l.1437), FP move immed 1 (l.1435), FP transfer from gen 2 (l.1439)
def CortexA55WriteFPALU_F1 : SchedWriteRes<[CortexA55UnitFPALU]> { let Latency = 1;}
def : InstRW<[CortexA55WriteFPALU_F1], (instrs FMOVHr, FMOVSr, FMOVDr, FMOVHi, FMOVSi, FMOVDi)>;
def : InstRW<[CortexA55WriteFPALU_F2], (instrs FMOVWHr, FMOVXHr, FMOVWSr, FMOVXDr, FMOVXDHighr)>;
// FP compare 1, 1, 11 (l.1355) [a55_sog_n5]: one FP ALU unit + CortexA55UnitFPCmp for one cycle (throughput 1).
def CortexA55WriteFCmp : SchedWriteRes<[CortexA55UnitFPALU, CortexA55UnitFPCmp]> { let Latency = 1; }
def : InstRW<[CortexA55WriteFCmp], (instregex "^FCMPE?[HSD]r[ri]$", "^FCCMPE?[HSD]rr$")>;
// FP select 2 (l.1374); throughput 1, dual-issue 01 [a55_sog_impl #4]
def : InstRW<[CortexA55WriteFCSEL], (instrs FCSELHrrr, FCSELSrrr, FCSELDrrr)>;
// ASIMD move, integer immed MOVI 1 (l.1995) / ASIMD move, FP immed FMOV 1 (l.1997)
def : InstRW<[CortexA55WriteFPALU_F1], (instrs MOVID)>;
def : InstRW<[CortexA55WriteAluVd_1], (instregex "^MOVIv(8b|2i32|4i16|2s)", "^FMOVv(2f32|4f16)_ns$")>;
def : InstRW<[CortexA55WriteAluVq_1], (instregex "^MOVIv(16b|4i32|8i16|4s|2d)")>;
// [a55_sog_impl #10] "ASIMD move, FP immed FMOV 1, 2, 11" has no Q-form note (unlike MOVI's "2*, 11*"):
//   the Q-form keeps throughput 2 and dual-issues from either slot.
def : InstRW<[CortexA55WriteFPALU_F1], (instregex "^FMOVv(2f64|4f32|8f16)_ns$")>;
// ASIMD insert, element to element INS 2 (l.1993); ASIMD transfer, gen reg to element INS 2 (l.2023)
def : InstRW<[CortexA55WriteFPALU_F2], (instregex "^INSvi(8|16|32|64)(lane|gpr)$")>;
// ASIMD bitwise insert BIF/BIT/BSL 2 (l.1977)
def : InstRW<[CortexA55WriteAluVd_2], (instregex "^(BSL|BIT|BIF)v8i8$")>;
def : InstRW<[CortexA55WriteAluVq_2], (instregex "^(BSL|BIT|BIF)v16i8$")>;

//===----------------------------------------------------------------------===//
// [a55_sog_impl] Further SOG v3 rows that were left on a default (WriteVd/WriteVq = 4, WriteI/WriteISReg = 1).
//   Q-forms of "2*, 11*" rows use CortexA55WriteVq (throughput 1, slot 0 only; SOG 4.15-4.17 note 1).
//===----------------------------------------------------------------------===//
// #7 RBIT 2 (SOG 4.7 p.18); CRC32B/H/X/CB/CH/CX 2, CRC32W/CW 1 (SOG 4.21 p.41).  Not 1-cycle "ALU basic"
//    producers: the issue stage's ALU0 -> ALU1 path [a55_alu_fwd] excludes CRC32W/CW explicitly (SOG 4.21 does
//    not place CRC in the ALU pipes); reads unchanged from the stock definitions.
def : InstRW<[CortexA55WriteALU2, ReadI], (instrs RBITWr, RBITXr)>;
def : InstRW<[CortexA55WriteALU2, ReadI, ReadISReg], (instrs CRC32Brr, CRC32Hrr, CRC32Xrr, CRC32CBrr, CRC32CHrr, CRC32CXrr)>;
def : InstRW<[CortexA55WriteCRC1, ReadI, ReadISReg], (instrs CRC32Wrr, CRC32CWrr)>;
// #10 ASIMD transfer, element to gen reg SMOV/UMOV: 2, 2, 11 (SOG 4.17 p.36).
def : InstRW<[CortexA55WriteFPALU_F2], (instregex "^[SU]MOVvi")>;
// #10 ASIMD extract EXT 2, 2*, 11*; reverse REV16/32/64 2, 2*, 11*; bit reverse RBIT 2, 2*, 11*;
//     count CLZ/CNT 2, 2*, 11*; CLS 3, 2*, 11* (SOG 4.17 p.35).
def : InstRW<[CortexA55WriteAluVd_2], (instrs EXTv8i8)>;
def : InstRW<[CortexA55WriteAluVq_2], (instrs EXTv16i8)>;
def : InstRW<[CortexA55WriteAluVd_2], (instregex "^REV(16|32|64)v(8i8|4i16|2i32)$", "^RBITv8i8$",
                                                 "^(CLZ|CNT)v(8i8|4i16|2i32)$")>;
def : InstRW<[CortexA55WriteAluVq_2], (instregex "^REV(16|32|64)v(16i8|8i16|4i32)$", "^RBITv16i8$",
                                                 "^(CLZ|CNT)v(16i8|8i16|4i32)$")>;
def : InstRW<[CortexA55WriteAluVd_3], (instregex "^CLSv(8i8|4i16|2i32)$")>;
def : InstRW<[CortexA55WriteAluVq_3], (instregex "^CLSv(16i8|8i16|4i32)$")>;
// #10 ASIMD transpose TRN1/TRN2 .2D 2, 1, 01; other 2, 2*, 11*; unzip/zip UZP1/2 ZIP1/2 2, 2*, 11* (SOG 4.17 p.36).
def : InstRW<[CortexA55WriteAluVd_2], (instregex "^(TRN|UZP|ZIP)(1|2)v(8i8|4i16|2i32)$")>;
def : InstRW<[CortexA55WriteAluVq_2], (instregex "^(TRN|UZP|ZIP)(1|2)v(16i8|8i16|4i32|2i64)$")>;
// #10 ASIMD FP compare FACGE/FACGT/FCMEQ/FCMGE/FCMGT/FCMLE/FCMLT: 2, 2*, 11* (SOG 4.16 p.33).
def : InstRW<[CortexA55WriteAluVd_2], (instregex "^F(AC|CM)(EQ|GE|GT|LE|LT)(16|32|64|v1i16rz|v1i32rz|v1i64rz|v2f32|v4f16|v2i32rz|v4i16rz)$")>;
def : InstRW<[CortexA55WriteAluVq_2], (instregex "^F(AC|CM)(EQ|GE|GT|LE|LT)(v4f32|v2f64|v8f16|v4i32rz|v2i64rz|v8i16rz)$")>;
// #10 ASIMD shift by immed: SQSHRN(2)/UQSHRN(2) 3, 2*, 11* (SOG 4.15 p.32, same row as RSHRN/SRSHR/URSHR);
//     shift by immed and insert SLI/SRI 2, 2*, 11* (vector forms; the scalar SLId/SRId are mapped above).
def : InstRW<[CortexA55WriteAluVd_3], (instregex "^[SU]QSHRN(b|h|s|v8i8_shift|v4i16_shift|v2i32_shift)$")>;
def : InstRW<[CortexA55WriteAluVq_3], (instregex "^[SU]QSHRN(v16i8_shift|v8i16_shift|v4i32_shift)$")>;
def : InstRW<[CortexA55WriteAluVd_2], (instregex "^S(LI|RI)v(8i8|4i16|2i32)_shift$")>;
def : InstRW<[CortexA55WriteAluVq_2], (instregex "^S(LI|RI)v(16i8|8i16|4i32|2i64)_shift$")>;
// #10 ASIMD table lookup TBL/TBX (SOG 4.17 p.35).
def : InstRW<[CortexA55WriteTBL1], (instrs TBLv8i8One, TBLv16i8One)>;
def : InstRW<[CortexA55WriteTBL2], (instrs TBLv8i8Two, TBLv16i8Two)>;
def : InstRW<[CortexA55WriteTBL3], (instrs TBLv8i8Three, TBLv16i8Three)>;
def : InstRW<[CortexA55WriteTBL4], (instrs TBLv8i8Four, TBLv16i8Four)>;
def : InstRW<[CortexA55WriteTBX1], (instrs TBXv8i8One, TBXv16i8One)>;
def : InstRW<[CortexA55WriteTBX2], (instrs TBXv8i8Two, TBXv16i8Two)>;
def : InstRW<[CortexA55WriteTBX3], (instrs TBXv8i8Three, TBXv16i8Three)>;
def : InstRW<[CortexA55WriteTBX4], (instrs TBXv8i8Four, TBXv16i8Four)>;
// #10 Cryptographic Extension (SOG 4.20 p.40-41): AESD/AESE 2, 1, 01; AESIMC/AESMC 2, 1, 11;
//     PMULL(2) 64x64 2, 1, 01; SHA1SU0 2, 1, 01; SHA1H/SHA1SU1 2, 1, 01; SHA256SU0 3, 1, 01.
//     (SHA1C/M/P, SHA256H/H2/SU1 4, 1, 01 are the WriteVq default.)  The AESE+AESMC / AESD+AESIMC dependent
//     pair dual-issue of SOG 3.5.2 is not modelled.
def : InstRW<[CortexA55WriteAluVq_2], (instrs AESErr, AESDrr, PMULLv1i64, PMULLv2i64, SHA1SU0rrr, SHA1Hrr, SHA1SU1rr)>;
def : InstRW<[CortexA55WriteAluT1_2_11], (instrs AESMCrr, AESIMCrr)>;
def : InstRW<[CortexA55WriteAluVq_3], (instrs SHA256SU0rr)>;

}
