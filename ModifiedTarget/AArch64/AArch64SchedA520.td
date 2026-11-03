//=- AArch64SchedA520.td - ARM Cortex-A520 Scheduling Definitions -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for ARM Cortex-A520 (E-Core).
// Reference & Source Attribution:
//   Arm Cortex-A520 Core Software Optimization Guide (PJDOC-1505342170-671342,
//   Issue 1.2), https://developer.arm.com/documentation/102553/latest/
//   "SOG l.N" below = line N of a text extraction of the guide (not part of the
//   repository).
//
// Every value is taken from the SOG; nothing is tuned to accuracy.
//
// What is NOT modelled (the SOG says nothing usable for a static model, or the effect
// cannot be expressed in a scheduling model; recorded in the README):
//   * Decoupling of the VPU from the integer pipeline (Figure 1 draws V0..V5 after ISS but
//     the text gives no queue depth or rule).  The tool's A55-only decoupled issue stage is
//     NOT applied to the A520; the A520 uses src/a520/a520_issue_stage.h, the stock in-order
//     stage without its write-back ordering rule (docs/a520-sog.md D-9, U-3).
//   * Multicycle-entry co-issue blocking (SOG l.3596-3606, docs/a520-sog.md U-2), AES fusion
//     (U-4), MOVPRFX fusion (SVE is not modelled).  The two-entry rule for >= 3 GPR sources or
//     destinations (SOG l.3567-3580) IS modelled as NumMicroOps = 2 (docs/a520-sog.md C-2).
//   * "Latency = 2 when the dependency is on Rm" for shifted/extended arithmetic is modelled
//     (negative ReadAdvance), but the SOG gives no such note for logical shifted forms.
//   * The 2x64-bit vector datapath option: values are the SOG's "A" column (2x128-bit
//     configuration, or scalar / non-Q forms of the 2x64-bit configuration, SOG l.541-544).
//     The board's configuration is not known (docs/a520-sog.md D-1).
//   * Integer divide (SOG l.660-661 note 2, l.683-689): latency and DIV occupancy are the best
//     case of the formula [N + bits/4], i.e. bits = 0: UDIV 3, SDIV 4 (docs/a520-sog.md D-13).
// docs/a520-sog.md compares this file with the SOG item by item.
//===----------------------------------------------------------------------===//

// Cortex-A520 SOG sec. 4.1 "Issue constraints", p.72 (SOG l.3585-3591): "A maximum of three
// issue queue entries can be co-issued per cycle (ignoring hazards) consisting of at most:
// Two ALU instructions / Two load instructions / One store instruction / Two VPU data
// processing instructions."  IssueWidth = 3.  The per-class limits are
// modelled by the resources below (ALU x2, Load group x2, LdSt x1 for stores, VPU x2).
def CortexA520Model : SchedMachineModel {
  let IssueWidth = 3;           // SOG l.3585: max three entries co-issued per cycle
  let MicroOpBufferSize = 0;    // In-order core
  let LoadLatency = 2;          // SOG l.846-941: integer load latency 2 (L1 hit)
  let MispredictPenalty = 10;   // not given in the SOG
  let CompleteModel = 0;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC]);
}

//===----------------------------------------------------------------------===//
// Pipelines (SOG sec. 2.1, Figure 1 and the table at l.457-515)
//   ALU0, ALU1, Branch, DIV, Load/Store, Load, MAC, PAC; VPU: VALU0, VALU1, VMAC0, VMAC1, VMC.
let SchedModel = CortexA520Model in {

def A520UnitALU   : ProcResource<2>;  // ALU0, ALU1
def A520UnitB     : ProcResource<1>;  // Branch
def A520UnitMAC   : ProcResource<1>;  // MAC
def A520UnitDIV   : ProcResource<1>;  // DIV
def A520UnitLdSt  : ProcResource<1>;  // Load/Store (the only store pipe)
def A520UnitLd    : ProcResource<1>;  // Load
def A520UnitLoad  : ProcResGroup<[A520UnitLdSt, A520UnitLd]>;  // loads: "Load/Store, Load"
def A520UnitVALU0 : ProcResource<1>;
def A520UnitVALU1 : ProcResource<1>;  // present in the 2x128-bit configuration
def A520UnitVALU  : ProcResGroup<[A520UnitVALU0, A520UnitVALU1]>;
// VMAC0/VMAC1.  Scalar FMUL/FMADD throughput 2 (SOG l.1134-1136) in either configuration
// (VMAC0 "supports 2x64-bit" operations, l.496-501), so two units.
def A520UnitVMAC  : ProcResource<2>;
// VMC "supports 2x64-bit or 1x128-bit operations" (l.504-506): scalar FDIV/FSQRT throughputs
// are printed as 2/N (l.1127-1168), i.e. two scalar operations in flight -> two units.
def A520UnitVMC   : ProcResource<2>;
// "Two VPU data processing instructions" per cycle (l.3594): every VPU write also takes one of these.
def A520UnitVPU   : ProcResource<2>;

//===----------------------------------------------------------------------===//
// Write types
// ALU
def A520Wr_1c_ALU      : SchedWriteRes<[A520UnitALU]> { let Latency = 1; }
def A520Wr_2c_ALU      : SchedWriteRes<[A520UnitALU]> { let Latency = 2; }
def A520Wr_1c_ALU_t1   : SchedWriteRes<[A520UnitALU]> { let Latency = 1; let ReleaseAtCycles = [2]; } // throughput 1
// Branch
def A520Wr_1c_B        : SchedWriteRes<[A520UnitB]>   { let Latency = 1; }
// MAC / DIV
def A520Wr_IMW         : SchedWriteRes<[A520UnitMAC]> { let Latency = 3; let NumMicroOps = 2; }        // l.663-665; 3 GPR sources -> 2 issue entries (l.3567-3580)
def A520Wr_IMX         : SchedWriteRes<[A520UnitMAC]> { let Latency = 4; let ReleaseAtCycles = [2]; let NumMicroOps = 2; }  // l.666 (1/2); 2 entries
def A520Wr_IML         : SchedWriteRes<[A520UnitMAC]> { let Latency = 2; let NumMicroOps = 2; }        // l.667; 2 entries
// [a520_mul_zr] MUL / MNEG / SMULL / UMULL / SMNEGL / UMNEGL are MADD / MSUB / SMADDL ... with the zero
// register as the accumulator: the zero register does not count as a GPR source, so these take one issue
// entry instead of two (docs/a520-sog.md D-15).
def A520Wr_IMW1        : SchedWriteRes<[A520UnitMAC]> { let Latency = 3; }
def A520Wr_IMX1        : SchedWriteRes<[A520UnitMAC]> { let Latency = 4; let ReleaseAtCycles = [2]; }
def A520Wr_IML1        : SchedWriteRes<[A520UnitMAC]> { let Latency = 2; }
def A520Wr_IMH         : SchedWriteRes<[A520UnitMAC]> { let Latency = 6; let ReleaseAtCycles = [4]; }  // l.671 (1/4)
// Divides block later divides until complete (note 2), so the occupancy is the latency; best case of
// [N + bits/4] with N = 3 for UDIV, 4 for SDIV (l.687-689), bits = 0: docs/a520-sog.md D-13.
def A520Wr_UDIV        : SchedWriteRes<[A520UnitDIV]> { let Latency = 3; let ReleaseAtCycles = [3]; }
def A520Wr_SDIV        : SchedWriteRes<[A520UnitDIV]> { let Latency = 4; let ReleaseAtCycles = [4]; }
// Loads / stores
def A520Wr_LD          : SchedWriteRes<[A520UnitLoad]> { let Latency = 2; }                             // l.846-941
def A520Wr_LDHi        : SchedWriteRes<[]> { let Latency = 2; let NumMicroOps = 0; }                    // 2nd reg of a pair (no extra issue entry)
def A520Wr_FLD         : SchedWriteRes<[A520UnitLoad]> { let Latency = 3; }                             // l.1224-1256
def A520Wr_FLD_t1      : SchedWriteRes<[A520UnitLoad]> { let Latency = 3; let ReleaseAtCycles = [2]; } // l.1257-1280 (pairs: 1)
def A520Wr_FLDHi       : SchedWriteRes<[]> { let Latency = 3; let NumMicroOps = 0; }
def A520Wr_ST          : SchedWriteRes<[A520UnitLdSt]> { let Latency = 1; }                             // l.969-1000 (1/cycle)
def A520Wr_STPQ        : SchedWriteRes<[A520UnitLdSt]> { let Latency = 2; let ReleaseAtCycles = [2]; } // l.1386 (STP Q: 2, 1/2)
def A520Wr_WB          : SchedWriteRes<[]> { let Latency = 1; let NumMicroOps = 0; }   // "Base register updates are done in parallel" (l.840, 953)
// [a520_from_a55_base_update] docs/a520-sog.md B-3: the base update of a writeback form is available after
//   1/T cycles of the row's A520 throughput T (rounded up; T >= 1 -> 1, i.e. A520Wr_WB / WriteAdr), the scheme
//   of the Cortex-A55 model (AArch64SchedA55.td [a55_sog_impl #9h], CortexA55WriteAdr2 / CortexA55WriteVAdr;
//   docs/a55-sog.md U-11), whose SOG prints "(1/T)" for the update latency.  The A520 SOG gives no update
//   latency ("done in parallel to the operation", l.840, l.953).  NOT SOG-based.  Used by STP Q pre/post
//   (T = 1/2) and the ASIMD LD/ST structure rows with T < 1.
class A520Wr_WBn<int lat> : SchedWriteRes<[]> { let Latency = lat; let NumMicroOps = 0; }
// VPU
def A520Wr_VALU_3      : SchedWriteRes<[A520UnitVALU, A520UnitVPU]> { let Latency = 3; }
def A520Wr_VALU_4      : SchedWriteRes<[A520UnitVALU, A520UnitVPU]> { let Latency = 4; }
def A520Wr_VALU0_3     : SchedWriteRes<[A520UnitVALU0, A520UnitVPU]> { let Latency = 3; }
def A520Wr_VALU0_4     : SchedWriteRes<[A520UnitVALU0, A520UnitVPU]> { let Latency = 4; }
def A520Wr_VALU0_5r5   : SchedWriteRes<[A520UnitVALU0, A520UnitVPU]> { let Latency = 5; let ReleaseAtCycles = [5, 1]; }
def A520Wr_VMAC_4      : SchedWriteRes<[A520UnitVMAC, A520UnitVPU]> { let Latency = 4; }
class A520Wr_VMC<int lat, int rac> : SchedWriteRes<[A520UnitVMC, A520UnitVPU]> {
  let Latency = lat; let ReleaseAtCycles = [rac, 1]; }

//===----------------------------------------------------------------------===//
// Generic SchedWrite defaults (the instruction groups below override them by InstRW)
def : WriteRes<WriteImm,   [A520UnitALU]> { let Latency = 1; }   // Move immed, l.815
def : WriteRes<WriteI,     [A520UnitALU]> { let Latency = 1; }   // Arithmetic/logical basic, l.593-635
def : WriteRes<WriteISReg, [A520UnitALU]> { let Latency = 1; }   // l.616, l.631 (Rm note: reads below)
def : WriteRes<WriteIEReg, [A520UnitALU]> { let Latency = 1; }   // l.616
def : WriteRes<WriteExtr,  [A520UnitALU]> { let Latency = 2; }   // EXTR 2 (ROR alias 1: variant below), l.783
def : WriteRes<WriteIS,    [A520UnitALU]> { let Latency = 2; }   // Bitfield move 2 (aliases 1: variant below), l.784-800
def : WriteRes<WriteIM32,  [A520UnitMAC]> { let Latency = 3; }   // l.663
def : WriteRes<WriteIM64,  [A520UnitMAC]> { let Latency = 4; let ReleaseAtCycles = [2]; } // l.666
def : WriteRes<WriteID32,  [A520UnitDIV]> { let Latency = 3; let ReleaseAtCycles = [3]; }  // UDIV, D-13
def : WriteRes<WriteID64,  [A520UnitDIV]> { let Latency = 3; let ReleaseAtCycles = [3]; }  // UDIV, D-13
def : WriteRes<WriteBr,    [A520UnitB]>   { let Latency = 1; }   // Branch pipeline, l.550-575
def : WriteRes<WriteBrReg, [A520UnitB]>   { let Latency = 1; }
def : WriteRes<WriteLD,    [A520UnitLoad]> { let Latency = 2; }
def : WriteRes<WriteLDIdx, [A520UnitLoad]> { let Latency = 2; }
def : WriteRes<WriteLDHi,  []>             { let Latency = 2; let NumMicroOps = 0; }   // second register of LDP (pair throughput 2, l.928-936)
def : WriteRes<WriteST,    [A520UnitLdSt]> { let Latency = 1; }
def : WriteRes<WriteSTP,   [A520UnitLdSt]> { let Latency = 1; }
def : WriteRes<WriteSTIdx, [A520UnitLdSt]> { let Latency = 1; }
// (WriteSTX is a WriteSequence<[WriteST, WriteLD]> in AArch64Schedule.td, which a WriteRes cannot override; the
//  exclusive stores have InstRW rows below, [a520_stx].)
def : WriteRes<WriteAdr,   []>             { let Latency = 1; let NumMicroOps = 0; }   // writeback in parallel, l.840
// [a520_sys] docs/a520-sog.md D-16: system instructions (MRS, MSR), barriers and the hints have no SOG row:
//   latency 1, no pipeline (the lower estimate, as the Cortex-A720 [a720_sys] and the Cortex-A78).
def : WriteRes<WriteSys,   []>             { let Latency = 1; }
def : WriteRes<WriteBarrier, []>           { let Latency = 1; }
def : WriteRes<WriteHint,  []>             { let Latency = 1; }
// [a520_nop] The NOP opcode takes no execution resource: the SOG has no row for it (docs/a520-sog.md D-14).
def A520Wr_Nop : SchedWriteRes<[]> { let Latency = 1; }
def : InstRW<[A520Wr_Nop], (instrs NOP)>;
// FP
def : WriteRes<WriteF,     [A520UnitVALU, A520UnitVPU]> { let Latency = 4; } // FABS/FADD/FSUB/FMIN/FMAX/FNEG/FRINT, l.1122-1151
def : WriteRes<WriteFCmp,  [A520UnitVALU0, A520UnitVPU]> { let Latency = 1; } // FCMP 1, throughput 1, l.1126
def : WriteRes<WriteFCvt,  [A520UnitVALU, A520UnitVPU]> { let Latency = 4; } // l.1185-1201
def : WriteRes<WriteFCopy, [A520UnitVALU0, A520UnitVPU]> { let Latency = 3; } // FMOV transfer vec->gen 3/1 (gen->vec below), l.1204-1205
def : WriteRes<WriteFImm,  [A520UnitVALU, A520UnitVPU]> { let Latency = 3; } // FMOV immed 3/2, l.1202
def : WriteRes<WriteFMul,  [A520UnitVMAC, A520UnitVPU]> { let Latency = 4; } // FMUL/FMADD 4/2, l.1134-1139
def : WriteRes<WriteFDiv,  [A520UnitVMC, A520UnitVPU]> { let Latency = 22; let ReleaseAtCycles = [19, 1]; } // fallback = D-form, l.1129
// ASIMD (default = "ASIMD arith, basic" and the other 3-cycle VALU rows, l.1433 ff.)
def : WriteRes<WriteVd,    [A520UnitVALU, A520UnitVPU]> { let Latency = 3; }
def : WriteRes<WriteVq,    [A520UnitVALU, A520UnitVPU]> { let Latency = 3; }
def : WriteRes<WriteVLD,   [A520UnitLoad]> { let Latency = 3; }  // ASIMD load, 1 element, multiple, 1 reg (l.1950 ff.)
def : WriteRes<WriteVST,   [A520UnitLdSt]> { let Latency = 1; }  // ASIMD store, 1 reg (l.2114 ff.)

def : WriteRes<WriteAtomic, []> { let Unsupported = 1; }

def : ReadAdvance<ReadI, 0>;
def : ReadAdvance<ReadISReg, 0>;
def : ReadAdvance<ReadIEReg, 0>;
def : ReadAdvance<ReadIM, 0>;
def : ReadAdvance<ReadIMA, 0>;
def : ReadAdvance<ReadID, 0>;
// [a520_from_a55_store_data] docs/a520-sog.md B-2: store data is read one cycle late (advance 1 from every
//   producer), borrowed from the Cortex-A55 (AArch64SchedA55.td [a55_sog_u9]; docs/a55-sog.md C-18, U-9).  The
//   A520 SOG gives no operand timing for stores (sec. 3.8, 3.15, 3.21 only "-" latencies).  ReadST is attached
//   to the data operands of every store form below (as on the A55), not only to the register-offset stores of
//   the stock Sched lists.  NOT SOG-based.
def : ReadAdvance<ReadST, 1>;
def : ReadAdvance<ReadExtrHi, 0>;
def : ReadAdvance<ReadAdrBase, 0>;
def : ReadAdvance<ReadVLD, 0>;

//===----------------------------------------------------------------------===//
// Integer instruction groups

// Address generation ADR/ADRP: 1, throughput 2, ALU (l.782).  Own write so that the
// sec. 4.5 pointer forwarding below can name it.
def A520Wr_ADRP : SchedWriteRes<[A520UnitALU]> { let Latency = 1; }
def : InstRW<[A520Wr_ADRP], (instrs ADRP)>;
def : InstRW<[A520Wr_1c_ALU], (instrs ADR)>;   // ADR: same row l.782, but sec. 4.5 names only adrp

// sec. 4.5 "A64 low latency pointer forwarding" (l.3664-3680): for "adrp x0, <const>;
// ldr x0, [x0, #lo12 <const>]" there are "dedicated forwarding paths that always allow this
// sequence to be executed without incurring a dependency-based stall".  Modelled as the
// base-register read of an unsigned-immediate load/store being one cycle early for an ADRP
// producer (ADRP latency 1 -> 0).  The SOG shows a load; "load-store addresses" in the same
// sentence is the reason stores are included.
def A520Rd_PtrFwd : SchedReadAdvance<1, [A520Wr_ADRP]>;
def : InstRW<[A520Wr_LD, A520Rd_PtrFwd],
             (instregex "^LDR(BB|HH|W|X|SBW|SBX|SHW|SHX|SW)ui$")>;
def : InstRW<[A520Wr_FLD, A520Rd_PtrFwd], (instregex "^LDR[BHSDQ]ui$")>;
// [a520_from_a55_store_data] B-2: the data Rt reads ReadST.
def : InstRW<[A520Wr_ST, ReadST, A520Rd_PtrFwd],
             (instregex "^STR(BB|HH|W|X|B|H|S|D|Q)ui$")>;

// "Arithmetic, extend and shift" ADD{S}/SUB{S}: latency 1, "Latency=2 when the dependency
// is on Rm" (l.616-620, note 1 l.639).  Shift amount 0 is "Arithmetic, basic" (l.593).
def A520Rd_RmLate      : SchedReadAdvance<-1>;
def A520Rd_ShiftedRm   : SchedReadVariant<[SchedVar<RegShiftedPred, [A520Rd_RmLate]>,
                                           SchedVar<NoSchedPred,    [ReadDefault]>]>;
def : InstRW<[A520Wr_1c_ALU, ReadDefault, A520Rd_ShiftedRm],
             (instregex "^(ADD|SUB)S?[WX]rs$")>;
// [a520_ext] docs/a520-sog.md D-6, U-10: every extended-register form takes the Rm note.  A variant on
//   RegExtendedPred ("extend operand != 0") is not used: it is false for UXTB #0, a real extend
//   (`add x1, x2, w3, uxtb`).  The identity extend (UXTX/LSL #0, the
//   `add x1, sp, x3` spelling) counts as an extend (U-10).
def : InstRW<[A520Wr_1c_ALU, ReadDefault, A520Rd_RmLate],
             (instregex "^(ADD|SUB)S?[WX]rx(64)?$")>;

// ADCS/SBCS throughput 1 (l.601); CCMN/CCMP throughput 1 (l.622).
def : InstRW<[A520Wr_1c_ALU_t1], (instregex "^(ADCS|SBCS)[WX]r$", "^CCM[NP][WX][ri]$")>;

// Bitfield move, basic: 2 (l.784); "Latency=1 for LSL (immediate), LSR (immediate) and UXTB
// aliases of UBFM; Latency=1 for SXTB and ASR (immediate) aliases of SBFM" (l.830-833).
// UBFM/SBFM operands: Rd, Rn, immr (2), imms (3).
//   LSR #s  = UBFM immr=s, imms=31/63       ASR #s = SBFM immr=s, imms=31/63
//   UXTB    = UBFMWri 0, 7                  SXTB   = SBFM[WX]ri 0, 7
//   LSL #s  = UBFM immr=(size-s), imms=(size-1-s), s >= 1
def A520BfmAlias1c : MCSchedPredicate<CheckAny<!listconcat(
    [CheckAll<[CheckOpcode<[UBFMWri, SBFMWri]>, CheckImmOperand<3, 31>]>,
     CheckAll<[CheckOpcode<[UBFMXri, SBFMXri]>, CheckImmOperand<3, 63>]>,
     CheckAll<[CheckOpcode<[UBFMWri, SBFMWri, SBFMXri]>, CheckImmOperand<2, 0>, CheckImmOperand<3, 7>]>],
    !foreach(s, !range(1, 32), CheckAll<[CheckOpcode<[UBFMWri]>, CheckImmOperand<2, !sub(32, s)>,
                                         CheckImmOperand<3, !sub(31, s)>]>),
    !foreach(s, !range(1, 64), CheckAll<[CheckOpcode<[UBFMXri]>, CheckImmOperand<2, !sub(64, s)>,
                                         CheckImmOperand<3, !sub(63, s)>]>))>>;
def A520Wr_BFM : SchedWriteVariant<[SchedVar<A520BfmAlias1c, [A520Wr_1c_ALU]>,
                                    SchedVar<NoSchedPred,    [A520Wr_2c_ALU]>]>;
def : InstRW<[A520Wr_BFM], (instrs UBFMWri, UBFMXri, SBFMWri, SBFMXri)>;
// Bitfield move, insert BFM: 2 (l.793)
def : InstRW<[A520Wr_2c_ALU], (instrs BFMWri, BFMXri)>;
// EXTR 2, "Latency=1 for ROR (immediate) alias of EXTR" (l.783, note l.828): ROR = EXTR Rd, Rn, Rn.
def A520IsRorImm : MCSchedPredicate<CheckSameRegOperand<1, 2>>;
def A520Wr_EXTR : SchedWriteVariant<[SchedVar<A520IsRorImm, [A520Wr_1c_ALU]>,
                                     SchedVar<NoSchedPred,  [A520Wr_2c_ALU]>]>;
def : InstRW<[A520Wr_EXTR], (instrs EXTRWrri, EXTRXrri)>;
// RBIT 2 (l.821)
def : InstRW<[A520Wr_2c_ALU], (instrs RBITWr, RBITXr)>;
// [a520_sog_fix] docs/a520-sog.md C-14: "Variable shift ASRV, LSLV, LSRV, RORV 1 2 ALU"
// (l.822-823).
def : InstRW<[A520Wr_1c_ALU], (instrs ASRVWr, ASRVXr, LSLVWr, LSLVXr, LSRVWr, LSRVXr, RORVWr, RORVXr)>;
// [a520_sog_fix] docs/a520-sog.md C-28: "CRC checksum ops CRC32, CRC32C 2 1 MAC" (l.2283-2284).
def A520Wr_CRC : SchedWriteRes<[A520UnitMAC]> { let Latency = 2; }
def : InstRW<[A520Wr_CRC], (instregex "^CRC32C?[BHWX]rr$")>;

// Multiply (l.663-675).  Note 1 (l.677-681): "dedicated forwarding path in the accumulate
// portion ... a typical sequence of multiply-accumulate instructions can issue one every 2
// cycles": accumulate operand read early so that MAC->MAC accumulate latency is 2
// (W: 3-1, X: 4-2; long: 2-0).  Not for consumers of multiply high.
// [a520_mac_fwd] docs/a520-sog.md C-5, U-7: note 1 says "the result of one MAC operation" to "a following MAC
//   operation", whatever its form, so every MAC write (W, X, long; also with the zero register as the
//   accumulator, i.e. MUL / SMULL ...) is a producer for the accumulator of every MADD / MSUB form.  The advance
//   is the consumer's (W 1, X 2; the long forms 0: their own chain is already 2).
def A520Rd_MAccW : SchedReadAdvance<1, [A520Wr_IMW, A520Wr_IMW1, A520Wr_IMX, A520Wr_IMX1, A520Wr_IML, A520Wr_IML1]>;
def A520Rd_MAccX : SchedReadAdvance<2, [A520Wr_IMW, A520Wr_IMW1, A520Wr_IMX, A520Wr_IMX1, A520Wr_IML, A520Wr_IML1]>;
// MC operands of MADD/MSUB/SMADDL...: 0 = Rd, 1 = Rn, 2 = Rm, 3 = Ra.
def A520AccIsZR : MCSchedPredicate<CheckAny<[CheckRegOperand<3, WZR>, CheckRegOperand<3, XZR>]>>;
def A520Wr_MADDW : SchedWriteVariant<[SchedVar<A520AccIsZR, [A520Wr_IMW1]>, SchedVar<NoSchedPred, [A520Wr_IMW]>]>;
def A520Wr_MADDX : SchedWriteVariant<[SchedVar<A520AccIsZR, [A520Wr_IMX1]>, SchedVar<NoSchedPred, [A520Wr_IMX]>]>;
def A520Wr_MADDL : SchedWriteVariant<[SchedVar<A520AccIsZR, [A520Wr_IML1]>, SchedVar<NoSchedPred, [A520Wr_IML]>]>;
def : InstRW<[A520Wr_MADDW, ReadDefault, ReadDefault, A520Rd_MAccW], (instrs MADDWrrr, MSUBWrrr)>;
def : InstRW<[A520Wr_MADDX, ReadDefault, ReadDefault, A520Rd_MAccX], (instrs MADDXrrr, MSUBXrrr)>;
def : InstRW<[A520Wr_MADDL], (instrs SMADDLrrr, UMADDLrrr, SMSUBLrrr, UMSUBLrrr)>;
def : InstRW<[A520Wr_IMH], (instrs SMULHrr, UMULHrr)>;
def : InstRW<[A520Wr_UDIV], (instrs UDIVWr, UDIVXr)>;
def : InstRW<[A520Wr_SDIV], (instrs SDIVWr, SDIVXr)>;

// Load pair (l.928-945): signed offset throughput 2 (one Load-pipe slot, second register
// free); pre/post-index throughput 1.
def : InstRW<[A520Wr_LD, A520Wr_LDHi], (instregex "^LDN?P(W|X|SW)i$")>;
// LDP W/X/SW pre/post-index: 3 GPR destinations (base, Rt, Rt2) -> 2 issue entries (l.3567-3580)
def A520Wr_LD_t1_2e : SchedWriteRes<[A520UnitLoad]> { let Latency = 2; let ReleaseAtCycles = [2]; let NumMicroOps = 2; }
def : InstRW<[A520Wr_WB, A520Wr_LD_t1_2e, A520Wr_LDHi], (instregex "^LDP(W|X|SW)(pre|post)$")>;
// Stores with 3 GPR sources -> 2 issue entries: STR (register offset) of a GPR, STP of GPRs (l.3567-3580)
def A520Wr_ST_2e : SchedWriteRes<[A520UnitLdSt]> { let Latency = 1; let NumMicroOps = 2; }
// [a520_store_zr] docs/a520-sog.md D-15: the zero register is not a GPR source, so a store whose data (Rt, Rt2)
//   or index register is WZR/XZR has at most two GPR sources and takes one entry (A520Wr_ST).  MC operands:
//   STR ro: 0 = Rt, 1 = Rn, 2 = Rm; STP/STNP offset: 0 = Rt, 1 = Rt2, 2 = Rn; STP pre/post: 0 = written-back
//   base, 1 = Rt, 2 = Rt2, 3 = Rn.
def A520StRoZR   : MCSchedPredicate<CheckAny<[CheckRegOperand<0, WZR>, CheckRegOperand<0, XZR>,
                                              CheckRegOperand<2, WZR>, CheckRegOperand<2, XZR>]>>;
def A520StPZR    : MCSchedPredicate<CheckAny<[CheckRegOperand<0, WZR>, CheckRegOperand<0, XZR>,
                                              CheckRegOperand<1, WZR>, CheckRegOperand<1, XZR>]>>;
def A520StPIdxZR : MCSchedPredicate<CheckAny<[CheckRegOperand<1, WZR>, CheckRegOperand<1, XZR>,
                                              CheckRegOperand<2, WZR>, CheckRegOperand<2, XZR>]>>;
def A520Wr_STRo   : SchedWriteVariant<[SchedVar<A520StRoZR,   [A520Wr_ST]>, SchedVar<NoSchedPred, [A520Wr_ST_2e]>]>;
def A520Wr_STP    : SchedWriteVariant<[SchedVar<A520StPZR,    [A520Wr_ST]>, SchedVar<NoSchedPred, [A520Wr_ST_2e]>]>;
def A520Wr_STPIdx : SchedWriteVariant<[SchedVar<A520StPIdxZR, [A520Wr_ST]>, SchedVar<NoSchedPred, [A520Wr_ST_2e]>]>;
// [a520_from_a55_store_data] B-2: the data operands (Rt, Rt2) read ReadST; reads in operand order Rt (, Rt2), Rn.
def : InstRW<[A520Wr_STRo, ReadST, ReadAdrBase], (instregex "^STR(BB|HH|W|X)ro[WX]$")>;
def : InstRW<[A520Wr_STP, ReadST, ReadST, ReadAdrBase], (instregex "^STN?P(W|X)i$")>;
def : InstRW<[A520Wr_WB, A520Wr_STPIdx, ReadST, ReadST, ReadAdrBase], (instregex "^STP(W|X)(pre|post)$")>;
// [a520_from_a55_store_data] B-2: the store forms whose stock Sched lists name no reads ([WriteST],
//   [WriteAdr, WriteST], [WriteSTP], [WriteAdr, WriteSTP]): same writes, plus ReadST on the data (the A55's
//   rows of AArch64SchedA55.td, [a55_sog_u9]).  Release stores as on the A55 (stock write WriteST; the
//   exclusive stores: [a520_stx] below); STXR/STXP and the atomics have no ReadST, as on the A55.
def : InstRW<[WriteST, ReadST, ReadAdrBase], (instregex "^STUR(BB|HH|W|X|B|H|S|D|Q)i$", "^STTR(B|H|W|X)i$")>;
def : InstRW<[WriteAdr, WriteST, ReadST, ReadAdrBase], (instregex "^STR(BB|HH|W|X|B|H|S|D|Q)(pre|post)$")>;
def : InstRW<[WriteSTP, ReadST, ReadST, ReadAdrBase], (instregex "^STN?P(S|D)i$")>;
def : InstRW<[WriteAdr, WriteSTP, ReadST, ReadST, ReadAdrBase], (instregex "^STP(S|D)(pre|post)$")>;
def : InstRW<[WriteST, ReadST, ReadAdrBase], (instregex "^STL?LR(B|H|W|X)$")>;
// [a520_stx] docs/a520-sog.md N-1, U-6: the exclusive stores have no SOG row.  They take the store row of
//   Table 3-7 ("- / 1 / Load/Store": latency 1 for the status result, the Load/Store pipe) and the entries of
//   the sec. 4.1 rule (C-2): STXR/STLXR read Rt and Rn (one entry), STXP/STLXP read Rt, Rt2 and Rn (two
//   entries; one when Rt or Rt2 is the zero register, D-15).  MC operands: 0 = Ws, 1 = Rt, (2 = Rt2,) then Rn.
//   (The stock WriteSTX, a WriteSequence of WriteST and WriteLD, would give all four two entries, latency 3 and a
//   load-pipe slot.)  STXR/STXP read no ReadST, as on the A55 (B-2).
def A520Wr_STXP : SchedWriteVariant<[SchedVar<A520StPIdxZR, [A520Wr_ST]>, SchedVar<NoSchedPred, [A520Wr_ST_2e]>]>;
def : InstRW<[A520Wr_ST], (instregex "^STXR(B|H|W|X)$")>;
def : InstRW<[A520Wr_STXP], (instregex "^STXP(W|X)$")>;
def : InstRW<[A520Wr_ST, ReadST, ReadAdrBase], (instregex "^STLXR(B|H|W|X)$")>;
def : InstRW<[A520Wr_STXP, ReadST, ReadST, ReadAdrBase], (instregex "^STLXP(W|X)$")>;

//===----------------------------------------------------------------------===//
// FP / vector loads and stores
// Load vector reg: 3, throughput 2 (l.1224-1256).  (ui forms are above, with sec. 4.5.)
def : InstRW<[A520Wr_FLD], (instregex "^LDR[BHSDQ](roW|roX|l)$", "^LDUR[BHSDQ]i$")>;
def : InstRW<[A520Wr_WB, A520Wr_FLD], (instregex "^LDR[BHSDQ](pre|post)$")>;
// Load vector pair: 3, throughput 1 (l.1257-1280).
def : InstRW<[A520Wr_FLD_t1, A520Wr_FLDHi], (instregex "^LDN?P[SDQ]i$")>;
def : InstRW<[A520Wr_WB, A520Wr_FLD_t1, A520Wr_FLDHi], (instregex "^LDP[SDQ](pre|post)$")>;
// Store vector pair, Q-form: 2, throughput 1/2 (l.1386-1396).
// [a520_from_a55_store_data] B-2: Rt, Rt2 read ReadST.
def : InstRW<[A520Wr_STPQ, ReadST, ReadST, ReadAdrBase], (instregex "^STN?PQi$")>;
// [a520_from_a55_base_update] B-3: base update 1/T = 2 (T = 1/2).
def : InstRW<[A520Wr_WBn<2>, A520Wr_STPQ, ReadST, ReadST, ReadAdrBase], (instregex "^STPQ(pre|post)$")>;

//===----------------------------------------------------------------------===//
// FP scalar (Table 3-11 l.1122-1170, Table 3-12 l.1185-1205)
def : InstRW<[A520Wr_VALU0_3], (instregex "^FCSEL[HSD]rrr$")>;              // FCSEL 3 / 1, l.1152
def : InstRW<[A520Wr_VALU0_5r5], (instregex "^FCCMPE?[HSD]rr$")>;          // FCCMP 5 / (1/5), l.1125
def : InstRW<[A520Wr_VALU0_3], (instrs FMOVHr, FMOVSr, FMOVDr)>;            // FP move, register 3 / 1, l.1203
def : InstRW<[A520Wr_VALU_3], (instregex "^FMOV[WX][HSD]r$", "^FMOVXDHighr$")>; // gen->vec 3 / 2, l.1204
def : InstRW<[A520Wr_VALU0_3], (instregex "^FMOV[HSD][WX]r$", "^FMOVDXHighr$")>; // vec->gen 3 / 1, l.1205
def : InstRW<[A520Wr_VALU0_4], (instregex "^FCVT[AMNPZ][SU][SU][WX][HSD]ri?$", "^FJCVTZS$")>; // l.1187-1199
def : InstRW<[A520Wr_VALU_4], (instregex "^[SU]CVTF[SU][WX][HSD]ri?$", "^FCVT[HSD][HSD]r$")>; // l.1185, 1200
// FDIV / FSQRT per form (l.1127-1129, l.1153-1168): two units (see A520UnitVMC), throughput 2/N.
def : InstRW<[A520Wr_VMC<8, 5>],   (instrs FDIVHrr)>;
def : InstRW<[A520Wr_VMC<13, 10>], (instrs FDIVSrr)>;
def : InstRW<[A520Wr_VMC<22, 19>], (instrs FDIVDrr)>;
def : InstRW<[A520Wr_VMC<11, 5>],  (instrs FSQRTHr)>;
def : InstRW<[A520Wr_VMC<14, 9>],  (instrs FSQRTSr)>;
def : InstRW<[A520Wr_VMC<25, 19>], (instrs FSQRTDr)>;

//===----------------------------------------------------------------------===//
// ASIMD ("A" column).  Default 3-cycle VALU (WriteVd/WriteVq) covers arith basic, compare,
// logical, max/min, shift basic, bitwise, extract EXT, move, reverse, zip, transpose, etc.

// sec. 4.7 "SIMD MAC forwarding" (l.3694-3708): MUL, MLA, MLS, UMULL(2), SMULL(2), UMLAL(2),
// SMLAL(2), UMLSL(2), SMLSL(2), UDOT, SDOT have a dedicated accumulator forwarding path; "the
// latency between the above instructions will be 1 cycle" when two consecutive instructions
// share the destination/accumulator register, element size and register size.  Modelled as the
// accumulator operand reading the result of these instructions 3 cycles early (4 -> 1).  The
// "consecutive" and "same element/register size" conditions cannot be checked by a scheduling
// model and are not.
def A520Wr_VMUL : SchedWriteRes<[A520UnitVMAC, A520UnitVPU]> { let Latency = 4; }
def A520Rd_VMACFwd : SchedReadAdvance<3, [A520Wr_VMUL]>;
// Multiply / dot product, 4 (l.1497-1570)
def : InstRW<[A520Wr_VMUL], (instregex "^MULv", "^(SMULL|UMULL)v")>;
def : InstRW<[A520Wr_VMUL, A520Rd_VMACFwd], (instregex "^(MLA|MLS)v", "^(SMLAL|UMLAL|SMLSL|UMLSL)v",
                                                       "^(SDOT|UDOT)v")>;
def : InstRW<[A520Wr_VMAC_4], (instregex "^(SQDMULH|SQRDMULH)v",
                                         "^(SQDMULL|SQDMLAL|SQDMLSL)v", "^(USDOT|SUDOT)v")>;
def : InstRW<[A520Wr_VALU_4], (instregex "^PMUL(v|Lv(8|16)i8)")>;                     // PMUL 4, VALU (l.1560-1567)
// 4-cycle VALU rows: ADDHN etc. (l.1447), CMTST (l.1495), SHRN family (l.1596-1609),
// XTN / SQXTN (l.1861-1864), INS element (l.1866)
def : InstRW<[A520Wr_VALU_4], (instregex "^(ADD|SUB|RSUB)HNv", "^CMTSTv",
                                         "^(SHRN|RSHRN|SQSHRN|UQSHRN|SQRSHRN|UQRSHRN|SQSHRUN|SQRSHRUN)v",
                                         "^(XTN|SQXTN|UQXTN|SQXTUN)v", "^INSvi(8|16|32|64)lane$")>;
// Reductions: 4, throughput 1 (l.1482-1488, l.1516-1540)
def : InstRW<[A520Wr_VALU0_4], (instregex "^ADDVv4i(16|32)v$", "^(SADDLV|UADDLV|SMAXV|UMAXV|SMINV|UMINV)v",
                                          "^F(MAX|MIN)(NM)?Vv")>;
// [a520_sog_fix] docs/a520-sog.md C-21: "ASIMD arith, reduce ADDV 3 1" (l.1485-1486) for the
// forms other than 4H/4S (8B, 16B, 8H); the 4H/4S row (l.1482) is 4 (Reductions above).
def : InstRW<[A520Wr_VALU0_3], (instrs ADDVv8i8v, ADDVv16i8v, ADDVv8i16v)>;
// Transfers: UMOV/SMOV 3 / 1, INS from gen 3 / 1, DUP from gen 3 / 1 (l.1856, l.1926-1929)
def : InstRW<[A520Wr_VALU0_3], (instregex "^[SU]MOVv", "^INSvi(8|16|32|64)gpr$", "^DUPv.*gpr$")>;
// ASIMD FP: arith/abs/neg/max/min/round/convert 4 (l.1636-1745), multiply / FMLA 4 VMAC (l.1746-1752)
def : InstRW<[A520Wr_VALU_4], (instregex "^F(ABS|ABD|NEG|ADD|SUB|MAX|MIN|MAXNM|MINNM|ADDP|MAXP|MINP|MAXNMP|MINNMP)v",
                                         "^FRINT([AIMNPXZ]|32X|32Z|64X|64Z)v",
                                         "^FCVT[AMNPZ][SU]v", "^[SU]CVTFv", "^FCVT(L|N|XN)v")>;
def : InstRW<[A520Wr_VMAC_4], (instregex "^FMULX?v", "^FML[AS]v", "^FML[AS]L2?v",
                                         "^FRECP[ES]v", "^FRSQRT[ES]v", "^FRECPXv")>;
// ASIMD FP divide / square root (l.1709-1718, l.1800-1809; the Q-form F64 FDIV row prints pipeline "VALU", read as VMC like the other rows): two VMC units, throughput as printed.
def : InstRW<[A520Wr_VMC<8, 5>],   (instrs FDIVv4f16)>;
def : InstRW<[A520Wr_VMC<13, 10>], (instrs FDIVv2f32)>;
def : InstRW<[A520Wr_VMC<8, 10>],  (instrs FDIVv8f16)>;
def : InstRW<[A520Wr_VMC<13, 20>], (instrs FDIVv4f32)>;
def : InstRW<[A520Wr_VMC<22, 38>], (instrs FDIVv2f64)>;
def : InstRW<[A520Wr_VMC<8, 5>],   (instrs FSQRTv4f16)>;
def : InstRW<[A520Wr_VMC<12, 9>],  (instrs FSQRTv2f32)>;
def : InstRW<[A520Wr_VMC<8, 10>],  (instrs FSQRTv8f16)>;
def : InstRW<[A520Wr_VMC<12, 18>], (instrs FSQRTv4f32)>;
def : InstRW<[A520Wr_VMC<22, 38>], (instrs FSQRTv2f64)>;


//===----------------------------------------------------------------------===//
// Further rows from the row-wise SOG check (docs/a520-sog.md).
def : InstRW<[A520Wr_VALU_4], (instrs FABD16, FABD32, FABD64)>;                                    // FP absolute value FABD 4, l.1122
def : InstRW<[A520Wr_VALU_4], (instregex "^(SCVTF|UCVTF)[hsd]$", "^FCVTZ[SU][hsd]$")>;             // scalar-SIMD fixed-point converts: ASIMD FP convert 4 (l.1661 ff.)
def A520Wr_ALU_t1q : SchedWriteRes<[A520UnitALU]> { let Latency = 1; let ReleaseAtCycles = [4]; }  // throughput 1/2
def A520Wr_ALU2_t1q : SchedWriteRes<[A520UnitALU]> { let Latency = 2; let ReleaseAtCycles = [4]; }
def : InstRW<[A520Wr_ALU_t1q], (instrs AXFLAG, XAFLAG, CFINV)>;                                    // l.794, l.810 (1/2)
def : InstRW<[A520Wr_ALU2_t1q], (instrs SETF8, SETF16)>;                                           // l.808 (2, 1/2)
def : InstRW<[A520Wr_1c_ALU_t1], (instrs RMIF)>;                                                   // l.810 (1, 1)
// ASIMD integer (A column)
def A520Wr_VALU_6r7 : SchedWriteRes<[A520UnitVALU, A520UnitVPU]> { let Latency = 6; let ReleaseAtCycles = [7, 1]; }
def A520Wr_VALU_6r4 : SchedWriteRes<[A520UnitVALU, A520UnitVPU]> { let Latency = 6; let ReleaseAtCycles = [4, 1]; }
def A520Wr_VALU_6r5 : SchedWriteRes<[A520UnitVALU, A520UnitVPU]> { let Latency = 6; let ReleaseAtCycles = [5, 1]; }
def A520Wr_VALU_8r5 : SchedWriteRes<[A520UnitVALU, A520UnitVPU]> { let Latency = 8; let ReleaseAtCycles = [5, 1]; }
def : InstRW<[A520Wr_VALU_6r7], (instregex "^[SU]ABAv")>;                                         // 6, 2/7 (l.1427)
def : InstRW<[A520Wr_VALU_6r4], (instregex "^[SU]ABALv")>;                                        // 6, 1/2 (l.1429)
def : InstRW<[A520Wr_VALU_4], (instregex "^(SQABS|SQADD|SQNEG|SQSUB|SUQADD|UQADD|UQSUB|USQADD)v")>; // arith, complex 4 (l.1447)
def : InstRW<[A520Wr_VALU_8r5], (instregex "^RADDHNv")>;                                          // 8, 2/5 (l.1458)
def : InstRW<[A520Wr_VALU_6r5], (instregex "^[SU]ADALPv")>;                                       // 6, 2/5 (l.1571)
def : InstRW<[A520Wr_VALU_8r5], (instregex "^[SU]RSRAv")>;                                        // 8, 2/5 (l.1586)
def : InstRW<[A520Wr_VALU_4], (instregex "^(SQSHL|SQSHLU|UQSHL|SQRSHL|UQRSHL)v")>;                // shift complex 4 (l.1601, l.1615)
def A520Wr_VMAC_4t1 : SchedWriteRes<[A520UnitVMAC, A520UnitVPU]> { let Latency = 4; let ReleaseAtCycles = [2, 1]; }
def : InstRW<[A520Wr_VMAC_4t1], (instregex "^(SQRDMLAH|SQRDMLSH)v")>;                             // 4, 1 (l.1547-1552)
def : InstRW<[A520Wr_VMUL, A520Rd_VMACFwd], (instregex "^(SDOT|UDOT)lanev")>;                      // dot product lane forms: same row l.1497 and sec. 4.7
def : InstRW<[A520Wr_VMAC_4], (instregex "^(USDOT|SUDOT)lanev", "^FML[AS]L2?lanev", "^FCADDv", "^FCMLAv",
                                         "^(URECPE|URSQRTE)v", "^BFMLAL[BT]")>;                   // l.1498, l.1750, l.1649-1651, l.1872, l.1828
// TBL / TBX by table count (l.1906-1925)
def A520Wr_V_8r5  : SchedWriteRes<[A520UnitVALU, A520UnitVPU]> { let Latency = 8;  let ReleaseAtCycles = [5, 1]; }
def A520Wr_V_12r10: SchedWriteRes<[A520UnitVALU, A520UnitVPU]> { let Latency = 12; let ReleaseAtCycles = [10, 1]; }
def A520Wr_V_16r18: SchedWriteRes<[A520UnitVALU, A520UnitVPU]> { let Latency = 16; let ReleaseAtCycles = [18, 1]; }
def A520Wr_V_20r26: SchedWriteRes<[A520UnitVALU, A520UnitVPU]> { let Latency = 20; let ReleaseAtCycles = [26, 1]; }
def : InstRW<[A520Wr_VALU_4],   (instregex "^TBLv(8|16)i8One$")>;
def : InstRW<[A520Wr_V_8r5],    (instregex "^TBLv(8|16)i8Two$", "^TBXv(8|16)i8One$")>;
def : InstRW<[A520Wr_V_12r10],  (instregex "^TBLv(8|16)i8Three$", "^TBXv(8|16)i8Two$")>;
def : InstRW<[A520Wr_V_16r18],  (instregex "^TBLv(8|16)i8Four$", "^TBXv(8|16)i8Three$")>;
def : InstRW<[A520Wr_V_20r26],  (instregex "^TBXv(8|16)i8Four$")>;
// Crypto (Table 3-21, l.2224-2275).  Crypto0/Crypto1 pipes ("Crypto1 present only in 2x128-bit", l.474-481).
def A520UnitCrypto : ProcResource<2>;
def A520Wr_AES   : SchedWriteRes<[A520UnitCrypto, A520UnitVPU]> { let Latency = 3; }
def A520Wr_VMC4  : SchedWriteRes<[A520UnitVMC, A520UnitVPU]> { let Latency = 4; }
def A520Wr_VMC3  : SchedWriteRes<[A520UnitVMC, A520UnitVPU]> { let Latency = 3; }
def A520Wr_VMC9  : SchedWriteRes<[A520UnitVMC, A520UnitVPU]> { let Latency = 9; let ReleaseAtCycles = [18, 1]; }
def A520Wr_V3t1  : SchedWriteRes<[A520UnitVALU0, A520UnitVPU]> { let Latency = 3; }
def : InstRW<[A520Wr_AES], (instrs AESDrr, AESErr, AESIMCrr, AESMCrr)>;
def : InstRW<[A520Wr_VMC4], (instrs PMULLv1i64, PMULLv2i64, SHA1Crrr, SHA1Mrrr, SHA1Prrr, SHA256Hrrr, SHA256H2rrr, SHA256SU0rr, SHA256SU1rrr)>;
def : InstRW<[A520Wr_VMC3], (instrs SHA1SU0rrr, SHA1SU1rr)>;
def : InstRW<[A520Wr_V3t1], (instrs SHA1Hrr)>;
def : InstRW<[A520Wr_VMC9], (instrs SHA512H, SHA512H2, SHA512SU0, SHA512SU1, RAX1, SM3PARTW1, SM3PARTW2, SM3SS1,
                                     SM3TT1A, SM3TT1B, SM3TT2A, SM3TT2B, SM4E, SM4ENCKEY)>;
// BF16 (Table 3-17, l.1822-1841): BFCVTN / BFCVT 4, VALU; BFDOT 10, VMAC + VALU
def A520Wr_BFDOT : SchedWriteRes<[A520UnitVMAC, A520UnitVALU, A520UnitVPU]> { let Latency = 10; }
def : InstRW<[A520Wr_VALU_4], (instregex "^BFCVTN", "^BFCVT$")>;
def : InstRW<[A520Wr_BFDOT], (instregex "^BFDOTv", "^BF16DOTlanev")>;
// [a520_sog_fix] docs/a520-sog.md C-23: "ASIMD matrix multiply accumulate BFMMLA 14, 15
// 1,1/2 VMAC, VALU" (l.1826-1827): A column, latency 14, throughput 1 (two units of each
// resource, so each is held 2 cycles).
def A520Wr_BFMMLA : SchedWriteRes<[A520UnitVMAC, A520UnitVALU, A520UnitVPU]> {
  let Latency = 14; let ReleaseAtCycles = [2, 2, 1]; }
def : InstRW<[A520Wr_BFMMLA], (instrs BFMMLA)>;
// [a520_sog_fix] docs/a520-sog.md C-21: "ASIMD matrix multiply-accumulate SMMLA, UMMLA,
// USMMLA 4 2,1 VALU" (l.1505-1507).
def : InstRW<[A520Wr_VALU_4], (instrs SMMLA, UMMLA, USMMLA)>;
// [a520_sog_fix] docs/a520-sog.md C-27: "Crypto SHA3 ops BCAX, EOR3 3 / XAR 4, 2,1 VALU"
// (l.2262-2263).
def : InstRW<[A520Wr_VALU_4], (instrs XAR)>;

def A520Wr_VLS_3_Load1 : SchedWriteRes<[A520UnitLoad]> { let Latency = 3; let ReleaseAtCycles = [1]; }
def A520Wr_VLS_3_LdSt1 : SchedWriteRes<[A520UnitLdSt]> { let Latency = 3; let ReleaseAtCycles = [1]; }
def A520Wr_VLS_4_LdSt2 : SchedWriteRes<[A520UnitLdSt]> { let Latency = 4; let ReleaseAtCycles = [2]; }
def A520Wr_VLS_4_LdSt1 : SchedWriteRes<[A520UnitLdSt]> { let Latency = 4; let ReleaseAtCycles = [1]; }
def A520Wr_VLS_5_LdSt3 : SchedWriteRes<[A520UnitLdSt]> { let Latency = 5; let ReleaseAtCycles = [3]; }
def A520Wr_VLS_5_LdSt4 : SchedWriteRes<[A520UnitLdSt]> { let Latency = 5; let ReleaseAtCycles = [4]; }
def A520Wr_VLS_6_LdSt4 : SchedWriteRes<[A520UnitLdSt]> { let Latency = 6; let ReleaseAtCycles = [4]; }
def A520Wr_VLS_1_LdSt1 : SchedWriteRes<[A520UnitLdSt]> { let Latency = 1; let ReleaseAtCycles = [1]; }
def A520Wr_VLS_1_LdSt2 : SchedWriteRes<[A520UnitLdSt]> { let Latency = 1; let ReleaseAtCycles = [2]; }
def A520Wr_VLS_1_LdSt3 : SchedWriteRes<[A520UnitLdSt]> { let Latency = 1; let ReleaseAtCycles = [3]; }
def A520Wr_VLS_1_LdSt4 : SchedWriteRes<[A520UnitLdSt]> { let Latency = 1; let ReleaseAtCycles = [4]; }
def A520Wr_VLS_1_LdSt6 : SchedWriteRes<[A520UnitLdSt]> { let Latency = 1; let ReleaseAtCycles = [6]; }
def A520Wr_VLS_1_LdSt8 : SchedWriteRes<[A520UnitLdSt]> { let Latency = 1; let ReleaseAtCycles = [8]; }
// ASIMD load / store rows, one InstRW per SOG row (line = SOG line)
// [a520_from_a55_store_data] B-2: the ST1-ST4 rows read the data Vt with ReadST (advance 1), the base Rn with
//   ReadAdrBase; for the one-lane forms the lane index takes the read in between (ReadDefault), as in the
//   A55's ST1-ST4 rows.  The post-index register Xm gets nothing.
// [a520_from_a55_base_update] B-3: the post-index rows whose throughput T is below 1 (the LdSt pipe held N = 1/T
//   cycles) write the base with A520Wr_WBn<N>; the others keep A520Wr_WB (1).
def : InstRW<[A520Wr_VLS_3_Load1], (instrs LD1Onev1d, LD1Onev2s, LD1Onev4h, LD1Onev8b)>;   // l.1959 ASIMD load, 1 element, multiple, 1 reg, D-form: 3 / 2 / Load/Store,Load
def : InstRW<[A520Wr_WB, A520Wr_VLS_3_Load1], (instrs LD1Onev1d_POST, LD1Onev2s_POST, LD1Onev4h_POST, LD1Onev8b_POST)>;
def : InstRW<[A520Wr_VLS_3_Load1], (instrs LD1Onev16b, LD1Onev2d, LD1Onev4s, LD1Onev8h)>;   // l.1962 ASIMD load, 1 element, multiple, 1 reg, Q-form: 3 / 2 / Load/Store,Load
def : InstRW<[A520Wr_WB, A520Wr_VLS_3_Load1], (instrs LD1Onev16b_POST, LD1Onev2d_POST, LD1Onev4s_POST, LD1Onev8h_POST)>;
def : InstRW<[A520Wr_VLS_3_LdSt1], (instrs LD1Twov1d, LD1Twov2s, LD1Twov4h, LD1Twov8b)>;   // l.1965 ASIMD load, 1 element, multiple, 2 reg, D-form: 3 / 1 / Load/Store
def : InstRW<[A520Wr_WB, A520Wr_VLS_3_LdSt1], (instrs LD1Twov1d_POST, LD1Twov2s_POST, LD1Twov4h_POST, LD1Twov8b_POST)>;
def : InstRW<[A520Wr_VLS_3_LdSt1], (instrs LD1Twov16b, LD1Twov2d, LD1Twov4s, LD1Twov8h)>;   // l.1968 ASIMD load, 1 element, multiple, 2 reg, Q-form: 3 / 1 / Load/Store
def : InstRW<[A520Wr_WB, A520Wr_VLS_3_LdSt1], (instrs LD1Twov16b_POST, LD1Twov2d_POST, LD1Twov4s_POST, LD1Twov8h_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt2], (instrs LD1Threev1d, LD1Threev2s, LD1Threev4h, LD1Threev8b)>;   // l.1971 ASIMD load, 1 element, multiple, 3 reg, D-form: 4 / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_4_LdSt2], (instrs LD1Threev1d_POST, LD1Threev2s_POST, LD1Threev4h_POST, LD1Threev8b_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt2], (instrs LD1Threev16b, LD1Threev2d, LD1Threev4s, LD1Threev8h)>;   // l.1974 ASIMD load, 1 element, multiple, 3 reg, Q-form: 4 / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_4_LdSt2], (instrs LD1Threev16b_POST, LD1Threev2d_POST, LD1Threev4s_POST, LD1Threev8h_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt2], (instrs LD1Fourv1d, LD1Fourv2s, LD1Fourv4h, LD1Fourv8b)>;   // l.1977 ASIMD load, 1 element, multiple, 4 reg, D-form: 4 / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_4_LdSt2], (instrs LD1Fourv1d_POST, LD1Fourv2s_POST, LD1Fourv4h_POST, LD1Fourv8b_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt2], (instrs LD1Fourv16b, LD1Fourv2d, LD1Fourv4s, LD1Fourv8h)>;   // l.1980 ASIMD load, 1 element, multiple, 4 reg, Q-form: 4 / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_4_LdSt2], (instrs LD1Fourv16b_POST, LD1Fourv2d_POST, LD1Fourv4s_POST, LD1Fourv8h_POST)>;
def : InstRW<[A520Wr_VLS_3_Load1], (instrs LD1i16, LD1i32, LD1i8)>;   // l.1994 ASIMD load, 1 element, one lane, B/H/S: 3 / 2 / Load/Store
def : InstRW<[A520Wr_WB, A520Wr_VLS_3_Load1], (instrs LD1i16_POST, LD1i32_POST, LD1i8_POST)>;
def : InstRW<[A520Wr_VLS_3_Load1], (instrs LD1i64)>;   // l.1997 ASIMD load, 1 element, one lane, D: 3 / 2 / Load/Store,Load
def : InstRW<[A520Wr_WB, A520Wr_VLS_3_Load1], (instrs LD1i64_POST)>;
def : InstRW<[A520Wr_VLS_3_Load1], (instrs LD1Rv2s, LD1Rv4h, LD1Rv8b)>;   // l.1999 ASIMD load, 1 element, all lanes, D-form, B/H/S: 3 / 2 / Load/Store,Load
def : InstRW<[A520Wr_WB, A520Wr_VLS_3_Load1], (instrs LD1Rv2s_POST, LD1Rv4h_POST, LD1Rv8b_POST)>;
def : InstRW<[A520Wr_VLS_3_Load1], (instrs LD1Rv1d)>;   // l.2002 ASIMD load, 1 element, all lanes, D-form, D: 3 / 2 / Load/Store,Load
def : InstRW<[A520Wr_WB, A520Wr_VLS_3_Load1], (instrs LD1Rv1d_POST)>;
def : InstRW<[A520Wr_VLS_3_Load1], (instrs LD1Rv16b, LD1Rv2d, LD1Rv4s, LD1Rv8h)>;   // l.2005 ASIMD load, 1 element, all lanes, Q-form: 3 / 2 / Load/Store,Load
def : InstRW<[A520Wr_WB, A520Wr_VLS_3_Load1], (instrs LD1Rv16b_POST, LD1Rv2d_POST, LD1Rv4s_POST, LD1Rv8h_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt1], (instrs LD2Twov2s, LD2Twov4h, LD2Twov8b)>;   // l.2008 ASIMD load, 2 element, multiple, D-form, B/H/S: 4 / 1 / Load/Store
def : InstRW<[A520Wr_WB, A520Wr_VLS_4_LdSt1], (instrs LD2Twov2s_POST, LD2Twov4h_POST, LD2Twov8b_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt2], (instrs LD2Twov16b, LD2Twov4s, LD2Twov8h)>;   // l.2011 ASIMD load, 2 element, multiple, Q-form, B/H/S: 4 / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_4_LdSt2], (instrs LD2Twov16b_POST, LD2Twov4s_POST, LD2Twov8h_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt1], (instrs LD2Twov2d)>;   // l.2014 ASIMD load, 2 element, multiple, Q-form, D: 4 / 1 / Load/Store
def : InstRW<[A520Wr_WB, A520Wr_VLS_4_LdSt1], (instrs LD2Twov2d_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt2], (instrs LD2i16, LD2i8)>;   // l.2017 ASIMD load, 2 element, one lane, B/H: 4 / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_4_LdSt2], (instrs LD2i16_POST, LD2i8_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt2], (instrs LD2i32)>;   // l.2020 ASIMD load, 2 element, one lane, S: 4 / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_4_LdSt2], (instrs LD2i32_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt2], (instrs LD2i64)>;   // l.2022 ASIMD load, 2 element, one lane, D: 4 / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_4_LdSt2], (instrs LD2i64_POST)>;
def : InstRW<[A520Wr_VLS_3_LdSt1], (instrs LD2Rv2s, LD2Rv4h, LD2Rv8b)>;   // l.2024 ASIMD load, 2 element, all lanes, D-form, B/H/S: 3 / 1 / Load/Store
def : InstRW<[A520Wr_WB, A520Wr_VLS_3_LdSt1], (instrs LD2Rv2s_POST, LD2Rv4h_POST, LD2Rv8b_POST)>;
def : InstRW<[A520Wr_VLS_3_LdSt1], (instrs LD2Rv1d)>;   // l.2027 ASIMD load, 2 element, all lanes, D-form, D: 3 / 1 / Load/Store
def : InstRW<[A520Wr_WB, A520Wr_VLS_3_LdSt1], (instrs LD2Rv1d_POST)>;
def : InstRW<[A520Wr_VLS_3_LdSt1], (instrs LD2Rv16b, LD2Rv2d, LD2Rv4s, LD2Rv8h)>;   // l.2030 ASIMD load, 2 element, all lanes, Q-form: 3 / 1 / Load/Store
def : InstRW<[A520Wr_WB, A520Wr_VLS_3_LdSt1], (instrs LD2Rv16b_POST, LD2Rv2d_POST, LD2Rv4s_POST, LD2Rv8h_POST)>;
def : InstRW<[A520Wr_VLS_5_LdSt3], (instrs LD3Threev2s, LD3Threev4h, LD3Threev8b)>;   // l.2045 ASIMD load, 3 element, multiple, D-form, B/H/S: 5 / 1/3 / Load/Store
def : InstRW<[A520Wr_WBn<3>, A520Wr_VLS_5_LdSt3], (instrs LD3Threev2s_POST, LD3Threev4h_POST, LD3Threev8b_POST)>;
def : InstRW<[A520Wr_VLS_5_LdSt3], (instrs LD3Threev16b, LD3Threev4s, LD3Threev8h)>;   // l.2048 ASIMD load, 3 element, multiple, Q-form, B/H/S: 5 / 1/3 / Load/Store
def : InstRW<[A520Wr_WBn<3>, A520Wr_VLS_5_LdSt3], (instrs LD3Threev16b_POST, LD3Threev4s_POST, LD3Threev8h_POST)>;
def : InstRW<[A520Wr_VLS_5_LdSt3], (instrs LD3Threev2d)>;   // l.2051 ASIMD load, 3 element, multiple, Q-form, D: 5 / 1/3 / Load/Store
def : InstRW<[A520Wr_WBn<3>, A520Wr_VLS_5_LdSt3], (instrs LD3Threev2d_POST)>;
def : InstRW<[A520Wr_VLS_5_LdSt3], (instrs LD3i16, LD3i8)>;   // l.2054 ASIMD load, 3 element, one lane, B/H: 5 / 1/3 / Load/Store
def : InstRW<[A520Wr_WBn<3>, A520Wr_VLS_5_LdSt3], (instrs LD3i16_POST, LD3i8_POST)>;
def : InstRW<[A520Wr_VLS_5_LdSt3], (instrs LD3i32)>;   // l.2057 ASIMD load, 3 element, one lane, S: 5 / 1/3 / Load/Store
def : InstRW<[A520Wr_WBn<3>, A520Wr_VLS_5_LdSt3], (instrs LD3i32_POST)>;
def : InstRW<[A520Wr_VLS_5_LdSt3], (instrs LD3i64)>;   // l.2059 ASIMD load, 3 element, one lane, D: 5 / 1/3 / Load/Store
def : InstRW<[A520Wr_WBn<3>, A520Wr_VLS_5_LdSt3], (instrs LD3i64_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt2], (instrs LD3Rv2s, LD3Rv4h, LD3Rv8b)>;   // l.2061 ASIMD load, 3 element, all lanes, D-form, B/H/S: 4 / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_4_LdSt2], (instrs LD3Rv2s_POST, LD3Rv4h_POST, LD3Rv8b_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt2], (instrs LD3Rv1d)>;   // l.2064 ASIMD load, 3 element, all lanes, D-form, D: 4 / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_4_LdSt2], (instrs LD3Rv1d_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt2], (instrs LD3Rv16b, LD3Rv4s, LD3Rv8h)>;   // l.2067 ASIMD load, 3 element, all lanes, Q-form, B/H/S: 4 / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_4_LdSt2], (instrs LD3Rv16b_POST, LD3Rv4s_POST, LD3Rv8h_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt2], (instrs LD3Rv2d)>;   // l.2070 ASIMD load, 3 element, all lanes, Q-form, D: 4 / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_4_LdSt2], (instrs LD3Rv2d_POST)>;
def : InstRW<[A520Wr_VLS_5_LdSt3], (instrs LD4Fourv2s, LD4Fourv4h, LD4Fourv8b)>;   // l.2073 ASIMD load, 4 element, multiple, D-form, B/H/S: 5 / 1/3 / Load/Store
def : InstRW<[A520Wr_WBn<3>, A520Wr_VLS_5_LdSt3], (instrs LD4Fourv2s_POST, LD4Fourv4h_POST, LD4Fourv8b_POST)>;
def : InstRW<[A520Wr_VLS_5_LdSt3], (instrs LD4Fourv16b, LD4Fourv4s, LD4Fourv8h)>;   // l.2076 ASIMD load, 4 element, multiple, Q-form, B/H/S: 5 / 1/3 / Load/Store
def : InstRW<[A520Wr_WBn<3>, A520Wr_VLS_5_LdSt3], (instrs LD4Fourv16b_POST, LD4Fourv4s_POST, LD4Fourv8h_POST)>;
def : InstRW<[A520Wr_VLS_5_LdSt4], (instrs LD4Fourv2d)>;   // l.2079 ASIMD load, 4 element, multiple, Q-form, D: 5 / 1/4 / Load/Store
def : InstRW<[A520Wr_WBn<4>, A520Wr_VLS_5_LdSt4], (instrs LD4Fourv2d_POST)>;
def : InstRW<[A520Wr_VLS_6_LdSt4], (instrs LD4i16, LD4i8)>;   // l.2082 ASIMD load, 4 element, one lane, B/H: 6 / 1/4 / Load/Store
def : InstRW<[A520Wr_WBn<4>, A520Wr_VLS_6_LdSt4], (instrs LD4i16_POST, LD4i8_POST)>;
def : InstRW<[A520Wr_VLS_6_LdSt4], (instrs LD4i32, LD4i64)>;   // l.2096 ASIMD load, 4 element, one lane, S ASIMD load, 4 element, on: 6 / 1/4 / Load/Store
def : InstRW<[A520Wr_WBn<4>, A520Wr_VLS_6_LdSt4], (instrs LD4i32_POST, LD4i64_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt2], (instrs LD4Rv2s, LD4Rv4h, LD4Rv8b)>;   // l.2100 ASIMD load, 4 element, all lanes, D-form, B/H/S: 4 / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_4_LdSt2], (instrs LD4Rv2s_POST, LD4Rv4h_POST, LD4Rv8b_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt2], (instrs LD4Rv1d)>;   // l.2103 ASIMD load, 4 element, all lanes, D-form, D: 4 / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_4_LdSt2], (instrs LD4Rv1d_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt2], (instrs LD4Rv16b, LD4Rv4s, LD4Rv8h)>;   // l.2106 ASIMD load, 4 element, all lanes, Q-form, B/H/S: 4 / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_4_LdSt2], (instrs LD4Rv16b_POST, LD4Rv4s_POST, LD4Rv8h_POST)>;
def : InstRW<[A520Wr_VLS_4_LdSt2], (instrs LD4Rv2d)>;   // l.2109 ASIMD load, 4 element, all lanes, Q-form, D: 4 / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_4_LdSt2], (instrs LD4Rv2d_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt1, ReadST, ReadAdrBase], (instrs ST1Onev1d, ST1Onev2s, ST1Onev4h, ST1Onev8b)>;   // l.2121 ASIMD store, 1 element, multiple, 1 reg, D-form: - / 1 / Load/Store
def : InstRW<[A520Wr_WB, A520Wr_VLS_1_LdSt1, ReadST, ReadAdrBase], (instrs ST1Onev1d_POST, ST1Onev2s_POST, ST1Onev4h_POST, ST1Onev8b_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt1, ReadST, ReadAdrBase], (instrs ST1Onev16b, ST1Onev2d, ST1Onev4s, ST1Onev8h)>;   // l.2124 ASIMD store, 1 element, multiple, 1 reg, Q-form: - / 1 / Load/Store
def : InstRW<[A520Wr_WB, A520Wr_VLS_1_LdSt1, ReadST, ReadAdrBase], (instrs ST1Onev16b_POST, ST1Onev2d_POST, ST1Onev4s_POST, ST1Onev8h_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt1, ReadST, ReadAdrBase], (instrs ST1Twov1d, ST1Twov2s, ST1Twov4h, ST1Twov8b)>;   // l.2127 ASIMD store, 1 element, multiple, 2 reg, D-form: - / 1 / Load/Store
def : InstRW<[A520Wr_WB, A520Wr_VLS_1_LdSt1, ReadST, ReadAdrBase], (instrs ST1Twov1d_POST, ST1Twov2s_POST, ST1Twov4h_POST, ST1Twov8b_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt2, ReadST, ReadAdrBase], (instrs ST1Twov16b, ST1Twov2d, ST1Twov4s, ST1Twov8h)>;   // l.2130 ASIMD store, 1 element, multiple, 2 reg, Q-form: - / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_1_LdSt2, ReadST, ReadAdrBase], (instrs ST1Twov16b_POST, ST1Twov2d_POST, ST1Twov4s_POST, ST1Twov8h_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt3, ReadST, ReadAdrBase], (instrs ST1Threev1d, ST1Threev2s, ST1Threev4h, ST1Threev8b)>;   // l.2133 ASIMD store, 1 element, multiple, 3 reg, D-form: - / 1/3 / Load/Store
def : InstRW<[A520Wr_WBn<3>, A520Wr_VLS_1_LdSt3, ReadST, ReadAdrBase], (instrs ST1Threev1d_POST, ST1Threev2s_POST, ST1Threev4h_POST, ST1Threev8b_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt3, ReadST, ReadAdrBase], (instrs ST1Threev16b, ST1Threev2d, ST1Threev4s, ST1Threev8h)>;   // l.2148 ASIMD store, 1 element, multiple, 3 reg, Q-form: - / 1/3 / Load/Store
def : InstRW<[A520Wr_WBn<3>, A520Wr_VLS_1_LdSt3, ReadST, ReadAdrBase], (instrs ST1Threev16b_POST, ST1Threev2d_POST, ST1Threev4s_POST, ST1Threev8h_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt2, ReadST, ReadAdrBase], (instrs ST1Fourv1d, ST1Fourv2s, ST1Fourv4h, ST1Fourv8b)>;   // l.2151 ASIMD store, 1 element, multiple, 4 reg, D-form: - / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_1_LdSt2, ReadST, ReadAdrBase], (instrs ST1Fourv1d_POST, ST1Fourv2s_POST, ST1Fourv4h_POST, ST1Fourv8b_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt4, ReadST, ReadAdrBase], (instrs ST1Fourv16b, ST1Fourv2d, ST1Fourv4s, ST1Fourv8h)>;   // l.2154 ASIMD store, 1 element, multiple, 4 reg, Q-form: - / 1/4 / Load/Store
def : InstRW<[A520Wr_WBn<4>, A520Wr_VLS_1_LdSt4, ReadST, ReadAdrBase], (instrs ST1Fourv16b_POST, ST1Fourv2d_POST, ST1Fourv4s_POST, ST1Fourv8h_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt1, ReadST, ReadDefault, ReadAdrBase], (instrs ST1i16, ST1i32, ST1i8)>;   // l.2157 ASIMD store, 1 element, one lane, B/H/S: - / 1 / Load/Store
def : InstRW<[A520Wr_WB, A520Wr_VLS_1_LdSt1, ReadST, ReadDefault, ReadAdrBase], (instrs ST1i16_POST, ST1i32_POST, ST1i8_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt1, ReadST, ReadDefault, ReadAdrBase], (instrs ST1i64)>;   // l.2160 ASIMD store, 1 element, one lane, D: - / 1 / Load/Store
def : InstRW<[A520Wr_WB, A520Wr_VLS_1_LdSt1, ReadST, ReadDefault, ReadAdrBase], (instrs ST1i64_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt1, ReadST, ReadAdrBase], (instrs ST2Twov2s, ST2Twov4h, ST2Twov8b)>;   // l.2162 ASIMD store, 2 element, multiple, D-form, B/H/S: - / 1 / Load/Store
def : InstRW<[A520Wr_WB, A520Wr_VLS_1_LdSt1, ReadST, ReadAdrBase], (instrs ST2Twov2s_POST, ST2Twov4h_POST, ST2Twov8b_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt2, ReadST, ReadAdrBase], (instrs ST2Twov16b, ST2Twov4s, ST2Twov8h)>;   // l.2165 ASIMD store, 2 element, multiple, Q-form, B/H/S: - / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_1_LdSt2, ReadST, ReadAdrBase], (instrs ST2Twov16b_POST, ST2Twov4s_POST, ST2Twov8h_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt2, ReadST, ReadAdrBase], (instrs ST2Twov2d)>;   // l.2168 ASIMD store, 2 element, multiple, Q-form, D: - / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_1_LdSt2, ReadST, ReadAdrBase], (instrs ST2Twov2d_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt1, ReadST, ReadDefault, ReadAdrBase], (instrs ST2i16, ST2i32, ST2i8)>;   // l.2171 ASIMD store, 2 element, one lane, B/H/S: - / 1 / Load/Store
def : InstRW<[A520Wr_WB, A520Wr_VLS_1_LdSt1, ReadST, ReadDefault, ReadAdrBase], (instrs ST2i16_POST, ST2i32_POST, ST2i8_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt1, ReadST, ReadDefault, ReadAdrBase], (instrs ST2i64)>;   // l.2174 ASIMD store, 2 element, one lane, D: - / 1 / Load/Store
def : InstRW<[A520Wr_WB, A520Wr_VLS_1_LdSt1, ReadST, ReadDefault, ReadAdrBase], (instrs ST2i64_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt4, ReadST, ReadAdrBase], (instrs ST3Threev2s, ST3Threev4h, ST3Threev8b)>;   // l.2176 ASIMD store, 3 element, multiple, D-form, B/H/S: - / 1/4 / Load/Store
def : InstRW<[A520Wr_WBn<4>, A520Wr_VLS_1_LdSt4, ReadST, ReadAdrBase], (instrs ST3Threev2s_POST, ST3Threev4h_POST, ST3Threev8b_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt6, ReadST, ReadAdrBase], (instrs ST3Threev16b, ST3Threev4s, ST3Threev8h)>;   // l.2179 ASIMD store, 3 element, multiple, Q-form, B/H/S: - / 1/6 / Load/Store
def : InstRW<[A520Wr_WBn<6>, A520Wr_VLS_1_LdSt6, ReadST, ReadAdrBase], (instrs ST3Threev16b_POST, ST3Threev4s_POST, ST3Threev8h_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt3, ReadST, ReadAdrBase], (instrs ST3Threev2d)>;   // l.2182 ASIMD store, 3 element, multiple, Q-form, D: - / 1/3 / Load/Store
def : InstRW<[A520Wr_WBn<3>, A520Wr_VLS_1_LdSt3, ReadST, ReadAdrBase], (instrs ST3Threev2d_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt2, ReadST, ReadDefault, ReadAdrBase], (instrs ST3i16, ST3i32, ST3i8)>;   // l.2185 ASIMD store, 3 element, one lane, B/H/S: - / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_1_LdSt2, ReadST, ReadDefault, ReadAdrBase], (instrs ST3i16_POST, ST3i32_POST, ST3i8_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt2, ReadST, ReadDefault, ReadAdrBase], (instrs ST3i64)>;   // l.2199 ASIMD store, 3 element, one lane, D: - / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_1_LdSt2, ReadST, ReadDefault, ReadAdrBase], (instrs ST3i64_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt4, ReadST, ReadAdrBase], (instrs ST4Fourv2s, ST4Fourv4h, ST4Fourv8b)>;   // l.2201 ASIMD store, 4 element, multiple, D-form, B/H/S: - / 1/4 / Load/Store
def : InstRW<[A520Wr_WBn<4>, A520Wr_VLS_1_LdSt4, ReadST, ReadAdrBase], (instrs ST4Fourv2s_POST, ST4Fourv4h_POST, ST4Fourv8b_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt8, ReadST, ReadAdrBase], (instrs ST4Fourv16b, ST4Fourv4s, ST4Fourv8h)>;   // l.2204 ASIMD store, 4 element, multiple, Q-form, B/H/S: - / 1/8 / Load/Store
def : InstRW<[A520Wr_WBn<8>, A520Wr_VLS_1_LdSt8, ReadST, ReadAdrBase], (instrs ST4Fourv16b_POST, ST4Fourv4s_POST, ST4Fourv8h_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt4, ReadST, ReadAdrBase], (instrs ST4Fourv2d)>;   // l.2207 ASIMD store, 4 element, multiple, Q-form, D: - / 1/4 / Load/Store
def : InstRW<[A520Wr_WBn<4>, A520Wr_VLS_1_LdSt4, ReadST, ReadAdrBase], (instrs ST4Fourv2d_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt2, ReadST, ReadDefault, ReadAdrBase], (instrs ST4i16, ST4i32, ST4i8)>;   // l.2210 ASIMD store, 4 element, one lane, B/H/S: - / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_1_LdSt2, ReadST, ReadDefault, ReadAdrBase], (instrs ST4i16_POST, ST4i32_POST, ST4i8_POST)>;
def : InstRW<[A520Wr_VLS_1_LdSt2, ReadST, ReadDefault, ReadAdrBase], (instrs ST4i64)>;   // l.2213 ASIMD store, 4 element, one lane, D: - / 1/2 / Load/Store
def : InstRW<[A520Wr_WBn<2>, A520Wr_VLS_1_LdSt2, ReadST, ReadDefault, ReadAdrBase], (instrs ST4i64_POST)>;
// SOG rows with pipes inconsistent with throughput (Load group used): [1994]
// covered 340 of 340 LDn/STn opcodes; not covered: []

//===----------------------------------------------------------------------===//
// [a520_pac] Pointer authentication, SOG sec. 3.5 Table 3-4 (l.702-773) and sec. 4.6
// (l.3686-3689); docs/a520-sog.md C-13 and C-7.  PAC pipeline: one unit
// (fig. 1, l.421; table l.487).
def A520UnitPAC : ProcResource<1>;
// "Compute pointer authentication code ... 5 1 PAC" (PACDA..., PACGA, PACIA..., incl. the
// hint forms PACIASP, PACIAZ, PACIA1716, ...).
def A520Wr_PAC : SchedWriteRes<[A520UnitPAC]> { let Latency = 5; }
def : InstRW<[A520Wr_PAC], (instrs PACDA, PACDB, PACDZA, PACDZB, PACGA, PACIA, PACIB, PACIZA, PACIZB,
                                   PACIA1716, PACIB1716, PACIASP, PACIBSP, PACIAZ, PACIBZ)>;
// "Authenticate instruction address ... 5 1 PAC" (AUTIA..., incl. the hint forms).
def A520Wr_AUT : SchedWriteRes<[A520UnitPAC]> { let Latency = 5; }
def : InstRW<[A520Wr_AUT], (instrs AUTIA, AUTIB, AUTIA1716, AUTIB1716, AUTIASP, AUTIBSP, AUTIAZ, AUTIBZ,
                                   AUTIZA, AUTIZB)>;
// "Strip pointer authentication code XPACD, XPACI, XPACLRI 5 1/5 PAC".
def A520Wr_XPAC : SchedWriteRes<[A520UnitPAC]> { let Latency = 5; let ReleaseAtCycles = [5]; }
def : InstRW<[A520Wr_XPAC], (instrs XPACD, XPACI, XPACLRI)>;
// sec. 4.6 "AUT* RET forwarding": "any variant of the AUT instruction will be dual issued with
// the directly following RET instruction.  The latency of the AUT instruction for the
// dependency of the LR does not apply": RET reads an AUT* result 5 cycles early (latency 0).
// "directly following" is not checked (docs/a520-sog.md D-11).
def A520Rd_AutRet : SchedReadAdvance<5, [A520Wr_AUT]>;
def : InstRW<[WriteBrReg, A520Rd_AutRet], (instrs RET)>;
// "Branch, return, with pointer authentication RETA* - 1 Branch".
def : InstRW<[A520Wr_1c_B], (instrs RETAA, RETAB)>;
// "Branch, register, with pointer authentication BRAA* - 1 Branch, PAC" and "Branch and
// link, register, with pointer authentication BLRAA* 1 1 Branch, PAC".
def A520Wr_BPAC : SchedWriteRes<[A520UnitB, A520UnitPAC]> { let Latency = 1; }
def : InstRW<[A520Wr_BPAC], (instrs BRAA, BRAAZ, BRAB, BRABZ, BLRAA, BLRAAZ, BLRAB, BLRABZ)>;

//===----------------------------------------------------------------------===//
// [a520_from_a55_atomics] docs/a520-sog.md B-1: Armv8.1 LSE atomics, borrowed from the Cortex-A55.
//   The A520 SOG has no row for any atomic (sec. 4.1, l.3596-3606, only lists "atomics with acquire/release"
//   among the multicycle entries); WriteAtomic stays Unsupported, so without these rows InstrBuilder drops them.
//   NOT SOG-based: latency, occupancy and group bits are those of AArch64SchedA55.td ([a55_sog_impl #3],
//   A55 SOG sec. 4.10; docs/a55-sog.md C-19, U-5): the atomic holds the A520's load pipe and its Load/Store
//   pipe (a load and a store; the A55 holds its load and its store pipe) for 1/T cycles each, BeginGroup
//   ("01"), CASP SingleIssue ("00").  Issue entries follow the A520 SOG (sec. 4.1, l.3574-3578, C-2): CAS and
//   CASP read three or more GPRs (Rs, Rt, Rn) -> 2 entries; LD<OP>/SWP read Rs and Rn -> 1 entry.
//   ST<OP> is LD<OP> with Rt = WZR/XZR.  Not covered (still dropped, as on the A55): CAST/CASPT (LSUI),
//   LDCLRP/LDSETP/SWPP (LSE128), RCW*.
class A520Wr_Atomic<int lat, int occ, int uops> : SchedWriteRes<[A520UnitLd, A520UnitLdSt]> {
  let Latency = lat; let ReleaseAtCycles = [occ, occ]; let NumMicroOps = uops; let BeginGroup = 1;
}
class A520Wr_AtomicSI<int lat, int occ, int uops> : SchedWriteRes<[A520UnitLd, A520UnitLdSt]> {
  let Latency = lat; let ReleaseAtCycles = [occ, occ]; let NumMicroOps = uops; let SingleIssue = 1;
}
def A520AtomicToZR : MCSchedPredicate<CheckAny<[CheckRegOperand<0, XZR>, CheckRegOperand<0, WZR>]>>;
//   LD<OP>{A}: 2, 1/2;  ST<OP>: 1, 1.
def A520Wr_LdOp  : SchedWriteVariant<[SchedVar<A520AtomicToZR, [A520Wr_Atomic<1, 1, 1>]>,
                                      SchedVar<NoSchedPred,    [A520Wr_Atomic<2, 2, 1>]>]>;
//   LD<OP>L / LD<OP>AL: 3, 1/3;  ST<OP>L: 2, 1/2.
def A520Wr_LdOpL : SchedWriteVariant<[SchedVar<A520AtomicToZR, [A520Wr_Atomic<2, 2, 1>]>,
                                      SchedVar<NoSchedPred,    [A520Wr_Atomic<3, 3, 1>]>]>;
def : InstRW<[A520Wr_LdOp],  (instregex "^LD(ADD|CLR|EOR|SET|SMAX|SMIN|UMAX|UMIN)A?(B|H|W|X)$")>;
def : InstRW<[A520Wr_LdOpL], (instregex "^LD(ADD|CLR|EOR|SET|SMAX|SMIN|UMAX|UMIN)A?L(B|H|W|X)$")>;
def : InstRW<[A520Wr_Atomic<2, 2, 1>], (instregex "^SWPA?(B|H|W|X)$")>;      // SWP{A}: 2, 1/2
def : InstRW<[A520Wr_Atomic<3, 3, 1>], (instregex "^SWPA?L(B|H|W|X)$")>;     // SWPL/SWPAL: 3, 1/3
def : InstRW<[A520Wr_Atomic<4, 4, 2>], (instregex "^CASA?(B|H|W|X)$")>;      // CAS{A}: 4, 1/4
def : InstRW<[A520Wr_Atomic<3, 3, 2>], (instregex "^CASA?L(B|H|W|X)$")>;     // CASL/CASAL: 3, 1/3
def : InstRW<[A520Wr_AtomicSI<5, 5, 2>], (instregex "^CASPA?(W|X)$")>;       // CASP{A}: 5, 1/5
def : InstRW<[A520Wr_AtomicSI<4, 4, 2>], (instregex "^CASPA?L(W|X)$")>;      // CASPL/CASPAL: 4, 1/4

} // SchedModel = CortexA520Model
