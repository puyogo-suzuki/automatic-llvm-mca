//=- AArch64SchedNeoverseV1.td - NeoverseV1 Scheduling Model -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the scheduling model for the Arm Neoverse V1 processors.
//
// References:
// - "Arm Neoverse V1 Software Optimization Guide"
// - "Arm Neoverse V1 Platform: Unleashing a new performance tier for Arm-based computing"
//   https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/neoverse-v1-platform-a-new-performance-tier-for-arm
// - "Neoverse V1"
//   https://en.wikichip.org/wiki/arm_holdings/microarchitectures/neoverse_v1

//
//===----------------------------------------------------------------------===//

// LOCAL USE: compiled into the tool's own tables and installed ONLY for
// cortex-x1 / cortex-x1c (src/cpu/cpu_traits.cpp, CpuModel::X1); neoverse-v1 and
// neoverse-512tvb, which LLVM also maps to NeoverseV1Model, keep libLLVM's stock model.
// So the machine parameters and the rows below are the Cortex-X1's: they follow the
// Cortex-X1 Core Software Optimization Guide (PJDOC-466751330-12804, Issue 4.0).
// docs/x1-sog.md compares them item by item; the tags [x1_...] name the items
// (docs/x1-sog.md section 2.4).  Issue width 16 uOPs/cycle; the 8-Mop rename width
// lives in cpu_traits.cpp; decode width (5) and scheduler entries (150) are not
// modelled (docs/x1-sog.md U-1, D-12).
def NeoverseV1Model : SchedMachineModel {
  let IssueWidth            =  16; // uOPs dispatched per cycle (X1 SOG sec. 4.1).
  let MicroOpBufferSize     = 224; // Cortex-X1 re-order buffer (owner's value, docs/x1-sog.md U-1).
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  11; // Cycles cost of branch mispredicted.
  let LoopMicroOpBufferSize =  16; // NOTE: Copied from Cortex-A57.
  let CompleteModel         =   1;

  // [x1_sve] docs/x1-sog.md D-8: no SVE on the Cortex-X1.
  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCPA,
                                                    HasCSSC, HasMOPS, HasMOPS_GO]);
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available on Neoverse V1.
// Instructions are first fetched and then decoded into internal macro-ops
// (MOPs).  From there, the MOPs proceed through register renaming and dispatch
// stages.  A MOP can be split into one or more micro-ops further down the
// pipeline, after the decode stage.  Once dispatched, micro-ops wait for their
// operands and issue out-of-order to one of the issue pipelines.  Each issue
// pipeline can accept one micro-op per cycle.

let SchedModel = NeoverseV1Model in {

// Define the issue ports.
def V1UnitB   : ProcResource<2>;  // Branch 0/1
def V1UnitS   : ProcResource<2>;  // Integer single cycle 0/1
def V1UnitM0  : ProcResource<1>;  // Integer multicycle 0
def V1UnitM1  : ProcResource<1>;  // Integer multicycle 1
def V1UnitL01 : ProcResource<2>;  // Load/Store 0/1
def V1UnitL2  : ProcResource<1>;  // Load 2
def V1UnitD   : ProcResource<2>;  // Store data 0/1
def V1UnitV0  : ProcResource<1>;  // FP/ASIMD 0
def V1UnitV1  : ProcResource<1>;  // FP/ASIMD 1
def V1UnitV2  : ProcResource<1>;  // FP/ASIMD 2
def V1UnitV3  : ProcResource<1>;  // FP/ASIMD 3
def V1UnitFlg : ProcResource<3>;  // Flags

def V1UnitI   : ProcResGroup<[V1UnitS,
                              V1UnitM0, V1UnitM1]>;   // Integer units
def V1UnitM   : ProcResGroup<[V1UnitM0, V1UnitM1]>;   // Integer multicycle units
def V1UnitL   : ProcResGroup<[V1UnitL01, V1UnitL2]>;  // Load units
def V1UnitV   : ProcResGroup<[V1UnitV0, V1UnitV1,
                              V1UnitV2, V1UnitV3]>;   // FP/ASIMD units
def V1UnitV01 : ProcResGroup<[V1UnitV0, V1UnitV1]>;   // FP/ASIMD 0/1 units
def V1UnitV02 : ProcResGroup<[V1UnitV0, V1UnitV2]>;   // FP/ASIMD 0/2 units
def V1UnitV13 : ProcResGroup<[V1UnitV1, V1UnitV3]>;   // FP/ASIMD 1/3 units

// [x1_dividers] docs/x1-sog.md D-3: the iterative dividers.  An integer divide "block[s] any
// subsequent divide operations until complete" (sec. 3.6 note 1, l.791-792) and an FP divide /
// square root blocks "subsequent similar operations to the same pipeline until complete"
// (l.1295-1296, l.1931-1932); every issue pipeline "can accept one µOP per cycle" (l.388-389).  So
// a divide takes its pipeline (M0; V0 or V2) for one cycle and holds a divider, not the pipeline,
// for its occupancy (the decision of the Cortex-A78, docs/a78-sog.md D-3).  FP divide and square
// root issue to V0 and V2 ("V02", l.1217-1245, Table 2-1 l.469-474): one divider in each.
def X1UnitIDiv : ProcResource<1>;  // integer divider, in the M0 pipeline
def X1UnitFDiv : ProcResource<2>;  // FP divide / square root units, in the V0 and V2 pipelines

// Define commonly used read types.

// No generic forwarding is provided for these types.
def : ReadAdvance<ReadI,       0>;
def : ReadAdvance<ReadISReg,   0>;
def : ReadAdvance<ReadIEReg,   0>;
def : ReadAdvance<ReadIM,      0>;
def : ReadAdvance<ReadIMA,     0>;
def : ReadAdvance<ReadID,      0>;
def : ReadAdvance<ReadExtrHi,  0>;
def : ReadAdvance<ReadAdrBase, 0>;
// [x1_from_a55_store_data] docs/x1-sog.md B-1: the data operand of a store is read one cycle late
// (advance 1 from every producer).  The X1 SOG says only that stores are split into address and
// data µOPs (l.1106-1107, l.1460, l.2260); it gives no operand timing.  This is the project's
// decision for the Cortex-A55 / A520, applied to the Cortex-A76, A78 and A720 for consistency
// (docs/a78-sog.md B-1), and applied here likewise: ReadST is attached to the data operands of
// every store form (STXR/STXP and the atomics excepted, as there).  NOT SOG-based.  It acts only
// on Facile's store -> load memory edges.
def : ReadAdvance<ReadST,      1>;
def : ReadAdvance<ReadVLD,     0>;

def : WriteRes<WriteAtomic,  []> { let Unsupported = 1; }
def : WriteRes<WriteBarrier, []> { let Latency = 1; }
def : WriteRes<WriteHint,    []> { let Latency = 1; }


//===----------------------------------------------------------------------===//
// Define generic 0 micro-op types

def V1Write_0c : SchedWriteRes<[]> { let Latency = 0; }

// [x1_ldpsw] docs/x1-sog.md C-12: "Load pair, signed immed offset, signed words LDPSW
// 5 1.5 I, L" (l.1002-1003, l.1023-1024): one I uOP and the load pipes held two cycles
// (1.5 per cycle over the three L pipes).  Stock V1 models a single L slot (3 per
// cycle).  Expressed as occupancy, not as an extra uOP, so NumMicroOps stays 2.
def V1Write_5c_1I_2L : SchedWriteRes<[V1UnitI, V1UnitL]> {
  let Latency = 5;
  let NumMicroOps = 2;
  let ReleaseAtCycles = [1, 2];
}

// [x1_ldp_hi] docs/x1-sog.md D-10: second (high) destination of a load pair that
// shares the first write's L micro-op: no extra uOP and no resource, but the register
// is NOT available at cycle 0 - it comes out of the same L1 access as the low half
// ("the maximum latency to load all the registers written by the instruction",
// l.961-962, l.1375-1376).  Stock V1 uses a latency-0, zero-uOP write here, which is
// indistinguishable from a rename-time zero-latency MOV once facile trusts
// Latency == 0 for the X1 (facile.cpp: isZeroLatencyMovTarget()), and would make the
// high half of LDPW/LDPSW/LDP[SDQ] free.  Give it the latency of the load instead.
let Latency = 4, NumMicroOps = 0 in
def V1Write_4c_0Z : SchedWriteRes<[]>;
let Latency = 5, NumMicroOps = 0 in
def V1Write_5c_0Z : SchedWriteRes<[]>;
let Latency = 6, NumMicroOps = 0 in
def V1Write_6c_0Z : SchedWriteRes<[]>;

//===----------------------------------------------------------------------===//
// Define generic 1 micro-op types

def V1Write_1c_1B      : SchedWriteRes<[V1UnitB]>   { let Latency = 1; }
def V1Write_1c_1I      : SchedWriteRes<[V1UnitI]>   { let Latency = 1; }
def V1Write_1c_1I_1Flg : SchedWriteRes<[V1UnitI, V1UnitFlg]>   { let Latency = 1; }
def V1Write_4c_1L      : SchedWriteRes<[V1UnitL]>   { let Latency = 4; }
def V1Write_6c_1L      : SchedWriteRes<[V1UnitL]>   { let Latency = 6; }
def V1Write_2c_1M      : SchedWriteRes<[V1UnitM]>   { let Latency = 2; }
def V1Write_2c_1M_1Flg : SchedWriteRes<[V1UnitM, V1UnitFlg]>   { let Latency = 2; }
def V1Write_3c_1M      : SchedWriteRes<[V1UnitM]>   { let Latency = 3; }
def V1Write_2c_1M0     : SchedWriteRes<[V1UnitM0]>  { let Latency = 2; }
def V1Write_3c_1M0     : SchedWriteRes<[V1UnitM0]>  { let Latency = 3; }
def V1Write_5c_1M0     : SchedWriteRes<[V1UnitM0]>  { let Latency = 5; }
// [x1_div_best] docs/x1-sog.md C-10, D-1, D-3: integer divide, the best value of "5 to 12,
// 1/12 to 1/5" (W) and "5 to 20, 1/20 to 1/5" (X) (l.703-704): latency 5, M0 one cycle, the
// divider held 5 cycles.  (Stock V1: latency 12 / 20 with M0 held 5 cycles.)
def X1Write_5c_1M0_IDiv : SchedWriteRes<[V1UnitM0, X1UnitIDiv]> { let Latency = 5;
                                                                 let ReleaseAtCycles = [1, 5]; }
def V1Write_2c_1V      : SchedWriteRes<[V1UnitV]>   { let Latency = 2; }
def V1Write_4c_1V      : SchedWriteRes<[V1UnitV]>   { let Latency = 4; }
def V1Write_2c_1V0     : SchedWriteRes<[V1UnitV0]>  { let Latency = 2; }
def V1Write_3c_1V0     : SchedWriteRes<[V1UnitV0]>  { let Latency = 3; }
def V1Write_4c_1V0     : SchedWriteRes<[V1UnitV0]>  { let Latency = 4; }
def V1Write_2c_1V01    : SchedWriteRes<[V1UnitV01]> { let Latency = 2; }
def V1Write_3c_1V01    : SchedWriteRes<[V1UnitV01]> { let Latency = 3; }
def V1Write_2c_1V02    : SchedWriteRes<[V1UnitV02]> { let Latency = 2; }
def V1Write_3c_1V02    : SchedWriteRes<[V1UnitV02]> { let Latency = 3; }
def V1Write_4c_1V02    : SchedWriteRes<[V1UnitV02]> { let Latency = 4; }
// FP divide / square root (scalar and ASIMD): every form has its own write
// ("V1WriteFDS_<lat>c<occ>", next to the InstRW rows).  V1Write_10c7_1V02 is only the
// WriteFDiv fallback for a form with no row.  [x1_dividers] D-3: V02 one cycle, the
// occupancy on an FP divide unit.
def V1Write_10c7_1V02  : SchedWriteRes<[V1UnitV02, X1UnitFDiv]> { let Latency = 7;
                                                                  let ReleaseAtCycles = [1, 7]; }
def V1Write_2c_1V1     : SchedWriteRes<[V1UnitV1]>  { let Latency = 2; }
def V1Write_2c_1V13    : SchedWriteRes<[V1UnitV13]> { let Latency = 2; }
def V1Write_4c_1V13    : SchedWriteRes<[V1UnitV13]> { let Latency = 4; }

//===----------------------------------------------------------------------===//
// Define generic 2 micro-op types

let Latency = 1, NumMicroOps = 2 in
def V1Write_1c_1B_1S     : SchedWriteRes<[V1UnitB, V1UnitS]>;
let Latency = 6, NumMicroOps = 2 in
def V1Write_6c_1B_1M0    : SchedWriteRes<[V1UnitB, V1UnitM0]>;
let Latency = 3, NumMicroOps = 2 in
def V1Write_3c_1I_1M     : SchedWriteRes<[V1UnitI, V1UnitM]>;
let Latency = 5, NumMicroOps = 2 in
def V1Write_5c_1I_1L     : SchedWriteRes<[V1UnitI, V1UnitL]>;
let Latency = 7, NumMicroOps = 2 in
def V1Write_7c_1I_1L     : SchedWriteRes<[V1UnitI, V1UnitL]>;
let Latency = 6, NumMicroOps = 2 in
def V1Write_6c_2L        : SchedWriteRes<[V1UnitL, V1UnitL]>;
let Latency = 8, NumMicroOps = 2 in
def V1Write_8c_1L_1V     : SchedWriteRes<[V1UnitL, V1UnitV]>;
let Latency = 1, NumMicroOps = 2 in
def V1Write_1c_1L01_1D   : SchedWriteRes<[V1UnitL01, V1UnitD]>;
let Latency = 2, NumMicroOps = 2 in
def V1Write_2c_1L01_1V01 : SchedWriteRes<[V1UnitL01, V1UnitV01]>;
let Latency = 4, NumMicroOps = 2 in
def V1Write_4c_1L01_1V01 : SchedWriteRes<[V1UnitL01, V1UnitV01]>;
let Latency = 9, NumMicroOps = 2 in
def V1Write_9c_1M0_1L    : SchedWriteRes<[V1UnitM0, V1UnitL]>;
let Latency = 5, NumMicroOps = 2 in
def V1Write_5c_1M0_1V    : SchedWriteRes<[V1UnitM0, V1UnitV]>;
let Latency = 4, NumMicroOps = 2 in
def V1Write_4c_2V        : SchedWriteRes<[V1UnitV, V1UnitV]>;
let Latency = 4, NumMicroOps = 2 in
def V1Write_4c_2V01      : SchedWriteRes<[V1UnitV01, V1UnitV01]>;
let Latency = 4, NumMicroOps = 2 in
def V1Write_4c_2V02      : SchedWriteRes<[V1UnitV02, V1UnitV02]>;
let Latency = 4, NumMicroOps = 2 in
def V1Write_4c_1V13_1V   : SchedWriteRes<[V1UnitV13, V1UnitV]>;
let Latency = 4, NumMicroOps = 2 in
def V1Write_4c_2V13      : SchedWriteRes<[V1UnitV13, V1UnitV13]>;

//===----------------------------------------------------------------------===//
// Define generic 3 micro-op types

let Latency = 2, NumMicroOps = 3 in
def V1Write_2c_1I_1L01_1V01 : SchedWriteRes<[V1UnitI, V1UnitL01, V1UnitV01]>;
let Latency = 8, NumMicroOps = 3 in
def V1Write_8c_1L_2V        : SchedWriteRes<[V1UnitL, V1UnitV, V1UnitV]>;
let Latency = 6, NumMicroOps = 3 in
def V1Write_6c_3L           : SchedWriteRes<[V1UnitL, V1UnitL, V1UnitL]>;
let Latency = 2, NumMicroOps = 3 in
def V1Write_2c_2L01_1V01    : SchedWriteRes<[V1UnitL01, V1UnitL01, V1UnitV01]>;
let Latency = 6, NumMicroOps = 3 in
def V1Write_6c_3V           : SchedWriteRes<[V1UnitV, V1UnitV, V1UnitV]>;
let Latency = 4, NumMicroOps = 3 in
def V1Write_4c_3V01         : SchedWriteRes<[V1UnitV01, V1UnitV01, V1UnitV01]>;
let Latency = 6, NumMicroOps = 3 in
def V1Write_6c_3V01         : SchedWriteRes<[V1UnitV01, V1UnitV01, V1UnitV01]>;

//===----------------------------------------------------------------------===//
// Define generic 4 micro-op types

let Latency = 7, NumMicroOps = 4 in
def V1Write_7c_4L        : SchedWriteRes<[V1UnitL, V1UnitL, V1UnitL, V1UnitL]>;
let Latency = 8, NumMicroOps = 4 in
def V1Write_8c_2L_2V        : SchedWriteRes<[V1UnitL, V1UnitL,
                                             V1UnitV, V1UnitV]>;
let Latency = 2, NumMicroOps = 4 in
def V1Write_2c_2L01_2V01    : SchedWriteRes<[V1UnitL01, V1UnitL01,
                                             V1UnitV01, V1UnitV01]>;
let Latency = 4, NumMicroOps = 4 in
def V1Write_4c_2L01_2V01    : SchedWriteRes<[V1UnitL01, V1UnitL01,
                                             V1UnitV01, V1UnitV01]>;

//===----------------------------------------------------------------------===//
// Define generic 5 micro-op types

let Latency = 8, NumMicroOps = 5 in
def V1Write_8c_2L_3V            : SchedWriteRes<[V1UnitL, V1UnitL,
                                                 V1UnitV, V1UnitV, V1UnitV]>;
let Latency = 6, NumMicroOps = 5 in
def V1Write_6c_5V01             : SchedWriteRes<[V1UnitV01, V1UnitV01,
                                                 V1UnitV01, V1UnitV01, V1UnitV01]>;

//===----------------------------------------------------------------------===//
// Define generic 6 micro-op types

let Latency = 8, NumMicroOps = 6 in
def V1Write_8c_3L_3V      : SchedWriteRes<[V1UnitL, V1UnitL, V1UnitL,
                                           V1UnitV, V1UnitV, V1UnitV]>;
let Latency = 2, NumMicroOps = 6 in
def V1Write_2c_3L01_3V01  : SchedWriteRes<[V1UnitL01, V1UnitL01, V1UnitL01,
                                           V1UnitV01, V1UnitV01, V1UnitV01]>;
let Latency = 5, NumMicroOps = 6 in
def V1Write_5c_3L01_3V01  : SchedWriteRes<[V1UnitL01, V1UnitL01, V1UnitL01,
                                           V1UnitV01, V1UnitV01, V1UnitV01]>;
let Latency = 6, NumMicroOps = 6 in
def V1Write_6c_3L01_3V01  : SchedWriteRes<[V1UnitL01, V1UnitL01, V1UnitL01,
                                           V1UnitV01, V1UnitV01, V1UnitV01]>;

//===----------------------------------------------------------------------===//
// Define generic 7 micro-op types

let Latency = 8, NumMicroOps = 7 in
def V1Write_8c_3L_4V         : SchedWriteRes<[V1UnitL, V1UnitL, V1UnitL,
                                              V1UnitV, V1UnitV, V1UnitV, V1UnitV]>;

//===----------------------------------------------------------------------===//
// Define generic 8 micro-op types

let Latency = 2, NumMicroOps = 8 in
def V1Write_2c_4L01_4V01  : SchedWriteRes<[V1UnitL01, V1UnitL01,
                                           V1UnitL01, V1UnitL01,
                                           V1UnitV01, V1UnitV01,
                                           V1UnitV01, V1UnitV01]>;
let Latency = 4, NumMicroOps = 8 in
def V1Write_4c_4L01_4V01  : SchedWriteRes<[V1UnitL01, V1UnitL01,
                                           V1UnitL01, V1UnitL01,
                                           V1UnitV01, V1UnitV01,
                                           V1UnitV01, V1UnitV01]>;

//===----------------------------------------------------------------------===//
// Define generic 12 micro-op types

let Latency = 7, NumMicroOps = 12 in
def V1Write_7c_6L01_6V01 : SchedWriteRes<[V1UnitL01, V1UnitL01, V1UnitL01,
                                          V1UnitL01, V1UnitL01, V1UnitL01,
                                          V1UnitV01, V1UnitV01, V1UnitV01,
                                          V1UnitV01, V1UnitV01, V1UnitV01]>;

//===----------------------------------------------------------------------===//
// Define predicate-controlled types

def V1Write_0or1c_1I : SchedWriteVariant<[
                      SchedVar<NeoverseZeroMove, [V1Write_0c]>,
                      SchedVar<NoSchedPred,      [V1Write_1c_1I]>]>;

//===----------------------------------------------------------------------===//
// Cortex-X1 writes and predicates (docs/x1-sog.md).  Each is used by the rows below that
// name its tag.

// [x1_extr] C-11: "Bitfield extract, one reg EXTR 1 4 I", note 1: "One reg form is when Rn==Rm
// or imm==0, all other forms are considered two regs" (l.890-891, l.903).  The stock
// IsRORImmIdiomPred checks only Rn == Rm.
def X1IsExtrOneRegPred : MCSchedPredicate<
                           CheckAll<[CheckOpcode<[EXTRWrri, EXTRXrri]>,
                                     CheckAny<[CheckSameRegOperand<1, 2>,
                                               CheckImmOperand<3, 0>]>]>>;
// [x1_reg_offset_h] [x1_store_rows] "Scaled" = the S bit of the register-offset addressing mode
// (AArch64_AM::getMemDoShift), as for the Cortex-A76, A78 and A720.
def X1IsMemScaled : MCSchedPredicate<CheckMemScaled>;

// [x1_logical_flagset] C-9: "ALU, basic, flagset ... ANDS, BICS 1 3 I" (l.565-567) for the
// unshifted register form, "Logical, shift, flagset ANDS, BICS 2 2 M" (l.590) for a shift.
def X1Write_LogicalS : SchedWriteVariant<[
                         SchedVar<NeoverseNoLSL, [V1Write_1c_1I_1Flg]>,
                         SchedVar<NoSchedPred,   [V1Write_2c_1M_1Flg]>]>;

// [x1_store_rows] C-13, C-17: register-offset stores with an I µOP.
def X1Write_2c_1I_1L01_1D : SchedWriteRes<[V1UnitI, V1UnitL01, V1UnitD]> { let Latency = 2;
                                                                         let NumMicroOps = 3; }

// [x1_v_throughput] C-19, C-20, D-2: a V-pipe row whose throughput T is below what its µOPs
// give: the group held n/T cycles in total (n = 4 for V, 2 for V02; D-2); the µOP count is
// that of the write it replaces.
class X1WriteV<int lat, int uops, int vcyc> : SchedWriteRes<[V1UnitV]> {
  let Latency = lat;
  let NumMicroOps = uops;
  let ReleaseAtCycles = [vcyc];
}
class X1WriteV02<int lat, int uops, int vcyc> : SchedWriteRes<[V1UnitV02]> {
  let Latency = lat;
  let NumMicroOps = uops;
  let ReleaseAtCycles = [vcyc];
}

// [x1_ls_throughput] C-21, C-22, D-6: a load or store whose throughput T is below what its µOPs
// give on the three load pipes L (two store pipes L01).  The µOP count and the V cycles are those
// of the write it replaces; the L (L01) pipes are held 3/T (2/T, rounded up) cycles in total.
// Which pipeline limits is not stated; L is taken, as for the Cortex-A76 and A78.
class X1WriteLDV<int lat, int uops, int lcyc, int vcyc> : SchedWriteRes<[V1UnitL, V1UnitV]> {
  let Latency = lat;
  let NumMicroOps = uops;
  let ReleaseAtCycles = [lcyc, vcyc];
}
class X1WriteSTV<int lat, int uops, int lcyc, int vcyc> : SchedWriteRes<[V1UnitL01, V1UnitV01]> {
  let Latency = lat;
  let NumMicroOps = uops;
  let ReleaseAtCycles = [lcyc, vcyc];
}

// [x1_pac_hint] D-13: the pointer-authentication instructions of the HINT space are NOPs on a
// core without FEAT_PAuth.  The X1 implements Armv8.2-A with RAS, LDAPR and the dot product
// (sec. 2, l.373-376), no pointer authentication, and its SOG has no PAC row; LLVM gives cortex-x1
// no PAuth feature, cortex-x1c has it.  Without the feature they take the NOP treatment (WriteHint:
// no pipeline, C-3); with it (cortex-x1c, no X1C SOG consulted, U-12) the stock Neoverse V1 rows.
def X1HasPAuth : FeatureSchedPredicate<FeaturePAuth>;
def X1Write_Hint : SchedWriteRes<[]> { let Latency = 1; }
def X1Write_PACHint : SchedWriteVariant<[
                        SchedVar<X1HasPAuth,  [V1Write_5c_1M0]>,
                        SchedVar<NoSchedPred, [X1Write_Hint]>]>;
def X1Write_XPACHint : SchedWriteVariant<[
                         SchedVar<X1HasPAuth,  [V1Write_2c_1M0]>,
                         SchedVar<NoSchedPred, [X1Write_Hint]>]>;

//===----------------------------------------------------------------------===//
// Define forwarded types

// NOTE: SOG, p. 20, n. 2: Accumulator forwarding is not supported for
// consumers of 64 bit multiply high operations?
def V1Wr_IM   : SchedWriteRes<[V1UnitM]>  { let Latency = 2; }
def V1Wr_IMA  : SchedWriteRes<[V1UnitM0]> { let Latency = 2; }
def V1WriteIM : SchedWriteVariant<
                  [SchedVar<NeoverseMULIdiomPred, [V1Wr_IM]>,
                   SchedVar<NoSchedPred,          [V1Wr_IMA]>]>;
// [x1_maddl] docs/x1-sog.md C-10, I-5, U-5: "Multiply accumulate long SMADDL, SMSUBL, UMADDL,
// UMSUBL 2(1) 2 M0" (l.708-709): a throughput of 2 on the single M0 pipe is not possible; the
// row is taken as the Cortex-A78's sibling row "2(1) 2 M" (on both M pipes), the lower estimate.
// The zero-accumulator forms (SMULL, UMULL, SMNEGL, UMNEGL: "Multiply long 2 2 M", l.711-712)
// stay V1Wr_IM.  The accumulator forwarding (note 2, l.793-795) is taken to work from either kind
// to either kind (as for the A78, U-5).
def X1Wr_IMAL  : SchedWriteRes<[V1UnitM]> { let Latency = 2; }
def X1WriteIML : SchedWriteVariant<
                   [SchedVar<NeoverseMULIdiomPred, [V1Wr_IM]>,
                    SchedVar<NoSchedPred,          [X1Wr_IMAL]>]>;
// [x1_mul_acc_fwd] docs/x1-sog.md U-5: "late-forwarding of accumulate operands from similar µOPs"
// (note 2, l.793-795) is taken to include the multiplies without an accumulator (MUL, MNEG and
// SMULL, UMULL, SMNEGL, UMNEGL, i.e. V1Wr_IM, the zero-accumulator arm of the variants above): their
// result reaches the accumulator of a following MADD/MSUB/SMADDL-family one cycle early, the lower
// estimate and the reading of the Cortex-A55, A520, A76 and A720 models.  SMULH/UMULH are excluded
// by the note.
def V1Rd_IMA : SchedReadAdvance<1, [V1Wr_IMA, X1Wr_IMAL, V1Wr_IM]>;

def V1Wr_FMA : SchedWriteRes<[V1UnitV]> { let Latency = 4; }
def V1Rd_FMA : SchedReadAdvance<2, [WriteFMul, V1Wr_FMA]>;

def V1Wr_ADA : SchedWriteRes<[V1UnitV13]> { let Latency = 4; }
def V1Rd_ADA : SchedReadAdvance<3, [V1Wr_ADA]>;

// X1 SOG Table 3-24: ASIMD dot product "2 (1), 4, V" (stock V1: 3 (1)).
def V1Wr_VDOT : SchedWriteRes<[V1UnitV]> { let Latency = 2; }
def V1Rd_VDOT : SchedReadAdvance<1, [V1Wr_VDOT]>;

def V1Wr_VMMA : SchedWriteRes<[V1UnitV]> { let Latency = 3; }
def V1Rd_VMMA : SchedReadAdvance<2, [V1Wr_VMMA]>;

def V1Wr_VMA : SchedWriteRes<[V1UnitV02]> { let Latency = 4; }
def V1Wr_VMAL : SchedWriteRes<[V1UnitV02]> { let Latency = 4; }
// [x1_mul_acc_fwd] docs/x1-sog.md U-5: ASIMD "Multiply-accumulate pipelines support late-forwarding
// of accumulate operands from similar µOPs" (Table 3-24 note 1, l.1758-1760) on the MLA/MLS and
// [SU]MLAL/[SU]MLSL rows "4(1)" (l.1630, l.1633-1636).  Taken to include the multiplies of the same
// mnemonics without an accumulator (ASIMD multiply MUL "4 2 V02", l.1628-1629; multiply long SMULL,
// UMULL "3 2 V02", l.1643-1645) and the other form (MLA <-> MLAL), as the integer forwarding above
// and the Cortex-A720 model; the lower estimate.  SQDMULH, SQRDMULH and SQDMULL stay
// V1Write_4c_1V02 / V1Write_3c_1V02: their accumulating forms (SQRDMLAH, SQDMLAL) have no forwarding.
def X1Wr_VMUL  : SchedWriteRes<[V1UnitV02]> { let Latency = 4; }
def X1Wr_VMULL : SchedWriteRes<[V1UnitV02]> { let Latency = 3; }
def V1Rd_VMA  : SchedReadAdvance<3, [V1Wr_VMA, V1Wr_VMAL, X1Wr_VMUL, X1Wr_VMULL]>;
def V1Rd_VMAL : SchedReadAdvance<3, [V1Wr_VMA, V1Wr_VMAL, X1Wr_VMUL, X1Wr_VMULL]>;

def V1Wr_VSA : SchedWriteRes<[V1UnitV13]> { let Latency = 4; }
def V1Rd_VSA : SchedReadAdvance<3, [V1Wr_VSA]>;

def V1Wr_FCMA : SchedWriteRes<[V1UnitV]> { let Latency = 4; }
def V1Rd_FCMA : SchedReadAdvance<2, [V1Wr_FCMA]>;

def V1Wr_FPM : SchedWriteRes<[V1UnitV]> { let Latency = 3; }
def V1Wr_FPMA : SchedWriteRes<[V1UnitV]> { let Latency = 4; }
def V1Rd_FPMA : SchedReadAdvance<2, [V1Wr_FPM, V1Wr_FPMA]>;

def V1Wr_FPMAL : SchedWriteRes<[V1UnitV]> { let Latency = 5; }
def V1Rd_FPMAL : SchedReadAdvance<3, [V1Wr_FPMAL]>;

def V1Wr_BFD : SchedWriteRes<[V1UnitV]> { let Latency = 4; }
def V1Rd_BFD : SchedReadAdvance<2, [V1Wr_BFD]>;

def V1Wr_BFMMA : SchedWriteRes<[V1UnitV]> { let Latency = 5; }
def V1Rd_BFMMA : SchedReadAdvance<2, [V1Wr_BFMMA]>;

def V1Wr_BFMLA : SchedWriteRes<[V1UnitV]> { let Latency = 4; }
def V1Rd_BFMLA : SchedReadAdvance<2, [V1Wr_BFMLA]>;

def V1Wr_CRC : SchedWriteRes<[V1UnitM0]> { let Latency = 2; }
def V1Rd_CRC : SchedReadAdvance<1, [V1Wr_CRC]>;

// Miscellaneous Instructions
// -----------------------------------------------------------------------------

// COPY
def : InstRW<[V1Write_1c_1I], (instrs COPY)>;

// MSR
def : WriteRes<WriteSys, []> { let Latency = 1; }


// Branch Instructions
// -----------------------------------------------------------------------------

// Branch, immed
// Compare and branch
def : SchedAlias<WriteBr, V1Write_1c_1B>;

// Branch, register
def : SchedAlias<WriteBrReg, V1Write_1c_1B>;

// Branch and link, immed
// Branch and link, register
def : InstRW<[V1Write_1c_1B_1S], (instrs BL, BLR)>;


// Arithmetic and Logical Instructions
// -----------------------------------------------------------------------------

// ALU, basic
// Conditional compare
// Conditional select
// Logical, basic
// Address generation
// Count leading
// Reverse bits/bytes
// Move immediate
def : SchedAlias<WriteI, V1Write_1c_1I>;

// ALU, basic, flagset
def : InstRW<[V1Write_1c_1I_1Flg],
             (instregex "^(ADD|SUB)S[WX]r[ir]$",
                        "^(ADC|SBC)S[WX]r$",
                        "^ANDS[WX]ri$",
                        "^(AND|BIC)S[WX]rr$")>;
def : InstRW<[V1Write_0or1c_1I], (instregex "^MOVZ[WX]i$")>;

// ALU, extend and shift
def : SchedAlias<WriteIEReg, V1Write_2c_1M>;

// Arithmetic, LSL shift, shift <= 4
// Arithmetic, LSR/ASR/ROR shift or LSL shift > 4
def V1WriteISReg : SchedWriteVariant<
                     [SchedVar<IsCheapLSL,  [V1Write_1c_1I]>,
                      SchedVar<NoSchedPred, [V1Write_2c_1M]>]>;
def              : SchedAlias<WriteISReg, V1WriteISReg>;

// Arithmetic, flagset, LSL shift, shift <= 4
// Arithmetic, flagset, LSR/ASR/ROR shift or LSL shift > 4
def V1WriteISRegS : SchedWriteVariant<
                      [SchedVar<IsCheapLSL,  [V1Write_1c_1I_1Flg]>,
                       SchedVar<NoSchedPred, [V1Write_2c_1M_1Flg]>]>;
def               : InstRW<[V1WriteISRegS],
                           (instregex "^(ADD|SUB)S[WX]rs$")>;

// [x1_ext_flagset] docs/x1-sog.md C-9, U-4: "ALU, extend and shift ADD{S}, SUB{S} 2 2 M"
// (l.568) for every extended-register form, as the non-flag-setting forms (WriteIEReg).  No
// IsCheapLSL variant: it would read the extend operand as a shift and give UXTB (extend operand 0)
// the basic row (1, I).
def               : InstRW<[V1Write_2c_1M_1Flg],
                           (instregex "^(ADD|SUB)S[WX]rx(64)?$")>;

// Logical, shift, no flagset
def : InstRW<[V1Write_1c_1I], (instregex "^(AND|BIC|EON|EOR|ORN)[WX]rs$")>;
def : InstRW<[V1Write_0or1c_1I], (instregex "^ORR[WX]rs$")>;

// Logical, shift, flagset
// [x1_logical_flagset] docs/x1-sog.md C-9: the unshifted register form (TST, ANDS, BICS
// Rd, Rn, Rm) is "ALU, basic, flagset" (1 3 I), unlike the shifted forms (2 on M).
def : InstRW<[X1Write_LogicalS], (instregex "^(AND|BIC)S[WX]rs$")>;

// Flag manipulation instructions
def : InstRW<[V1Write_1c_1I_1Flg], (instrs SETF8, SETF16, RMIF, CFINV)>;


// Divide and multiply instructions
// -----------------------------------------------------------------------------

// Divide
// [x1_div_best] docs/x1-sog.md C-10, D-1, D-3: latency 5, M0 one cycle, the divider held 5 cycles
// (the best value of each range).
def : SchedAlias<WriteID32, X1Write_5c_1M0_IDiv>;
def : SchedAlias<WriteID64, X1Write_5c_1M0_IDiv>;

def : SchedAlias<WriteIM32, V1Write_2c_1M>;
def : SchedAlias<WriteIM64, V1Write_2c_1M>;

// Multiply
// Multiply accumulate, W-form
// Multiply accumulate, X-form
def : InstRW<[V1WriteIM, ReadIM, ReadIM, V1Rd_IMA],
             (instregex "^M(ADD|SUB)[WX]rrr$")>;

// Multiply accumulate long
// Multiply long
// [x1_maddl] on M, 2 per cycle (I-5).
def : InstRW<[X1WriteIML, ReadIM, ReadIM, V1Rd_IMA],
             (instregex "^(S|U)M(ADD|SUB)Lrrr$")>;
// Multiply high
def : InstRW<[V1Write_3c_1M, ReadIM, ReadIM], (instrs SMULHrr, UMULHrr)>;


// Pointer Authentication Instructions (v8.3 PAC)
// -----------------------------------------------------------------------------

// Authenticate data address
// Authenticate instruction address
// Compute pointer authentication code for data address
// Compute pointer authentication code, using generic key
// Compute pointer authentication code for instruction address
def : InstRW<[V1Write_5c_1M0],
             (instregex "^AUT(D[AB]|DZ[AB]|I[AB]|IZ[AB]|I[AB]171615|I[AB]SPPC[ir])$",
                        "^PAC(D[AB]|DZ[AB]|GA|I[AB]|IZ[AB]|I[AB]171615|I[AB]SPPC|NBI[AB]SPPC|M)$",
                        "^(PAC|AUTPAC|AUTx16x17|AUTxMxN)$")>;
// [x1_pac_hint] docs/x1-sog.md D-13: the HINT-space forms are NOPs without FEAT_PAuth.
def : InstRW<[X1Write_PACHint], (instrs PACIAZ, PACIBZ, PACIASP, PACIBSP, PACIA1716, PACIB1716,
                                        AUTIAZ, AUTIBZ, AUTIASP, AUTIBSP, AUTIA1716, AUTIB1716)>;

// Branch and link, register, with pointer authentication
// Branch, register, with pointer authentication
// Branch, return, with pointer authentication
def : InstRW<[V1Write_6c_1B_1M0], (instregex "^BL?RA[AB]Z?$",
                                             "^E?RETA[AB]$")>;

// Load register, with pointer authentication
def : InstRW<[V1Write_9c_1M0_1L], (instregex "^LDRA[AB](indexed|writeback)")>;

// Strip pointer authentication code
def : InstRW<[V1Write_2c_1M0], (instrs XPACD, XPACI)>;
def : InstRW<[X1Write_XPACHint], (instrs XPACLRI)>;  // [x1_pac_hint]


// Miscellaneous data-processing instructions
// -----------------------------------------------------------------------------

// Bitfield extract, one reg
// Bitfield extract, two regs
// [x1_extr] docs/x1-sog.md C-11: one register when Rn == Rm or imm == 0.
def V1WriteExtr : SchedWriteVariant<
                    [SchedVar<X1IsExtrOneRegPred, [V1Write_1c_1I]>,
                     SchedVar<NoSchedPred,       [V1Write_3c_1I_1M]>]>;
def : SchedAlias<WriteExtr, V1WriteExtr>;

// Bitfield move, basic
// Variable shift
def : SchedAlias<WriteIS, V1Write_1c_1I>;

// Bitfield move, insert
def : InstRW<[V1Write_2c_1M], (instregex "^BFM[WX]ri$")>;

// Move immediate
def : SchedAlias<WriteImm, V1Write_1c_1I>;


// Load instructions
// -----------------------------------------------------------------------------

// Load register, immed offset
def : SchedAlias<WriteLD, V1Write_4c_1L>;

// Load register, immed offset, index
// Load register, register offset, basic / scale by 4/8 / extend / extend, scale by 4/8
// ("4 3 L", l.984-995: no I uOP)
def : SchedAlias<WriteLDIdx, V1Write_4c_1L>;
def : SchedAlias<WriteAdr,   V1Write_1c_1I>;

// [x1_reg_offset_h] docs/x1-sog.md C-12: "Load register, register offset, scale by 2" and
// "..., extend, scale by 2" LDRH, LDRSH: "5 3 I, L" (l.989-990, l.996-997); without the scale
// they are "basic" / "extend" (4 3 L).
def X1WriteLDRH : SchedWriteVariant<[
                    SchedVar<X1IsMemScaled, [V1Write_5c_1I_1L]>,
                    SchedVar<NoSchedPred,   [V1Write_4c_1L]>]>;
def : InstRW<[X1WriteLDRH, ReadAdrBase], (instregex "^LDRHHro[WX]$", "^LDRSH[WX]ro[WX]$")>;

// Load pair, immed offset
// (LDP X-form "4 1.5 L", l.1000-1001: WriteLD and WriteLDHi each take one L uOP.)
def : SchedAlias<WriteLDHi, V1Write_4c_1L>;
def : InstRW<[V1Write_4c_1L, V1Write_4c_0Z], (instrs LDPWi, LDNPWi)>;
def : InstRW<[WriteAdr, V1Write_4c_1L, V1Write_4c_0Z],
             (instrs LDPWpost, LDPWpre)>;

// Load pair, signed immed offset, signed words
def : InstRW<[V1Write_5c_1I_2L, V1Write_5c_0Z], (instrs LDPSWi)>;

// Load pair, immed post or pre-index, signed words
def : InstRW<[WriteAdr, V1Write_5c_1I_2L, V1Write_5c_0Z],
             (instrs LDPSWpost, LDPSWpre)>;


// Store instructions
// -----------------------------------------------------------------------------

// Store register, immed offset
def : SchedAlias<WriteST, V1Write_1c_1L01_1D>;

// Store register, immed offset, index
// Store register, register offset, basic / scaled by 4/8 / extend / extend, scale by 4/8
// ("1 2 L01, D", l.1131-1140: no I uOP)
def : SchedAlias<WriteSTIdx, V1Write_1c_1L01_1D>;

// Store pair, immed offset
def : SchedAlias<WriteSTP, V1Write_1c_1L01_1D>;

// [x1_store_rows] docs/x1-sog.md C-13, I-3: "Store register, register offset, scaled by 2" and
// "..., extend, scale by 1" (sic) STRH: "2 2 I, L01, D" (l.1135-1136, l.1141-1142).
def X1WriteSTRH : SchedWriteVariant<[
                    SchedVar<X1IsMemScaled, [X1Write_2c_1I_1L01_1D]>,
                    SchedVar<NoSchedPred,   [V1Write_1c_1L01_1D]>]>;
def : InstRW<[X1WriteSTRH, ReadST, ReadAdrBase], (instregex "^STRHHro[WX]$")>;

// [x1_from_a55_store_data] docs/x1-sog.md B-1: the integer store forms whose stock Sched lists
// name no reads: same writes, plus ReadST on the data.
def : InstRW<[WriteST, ReadST, ReadAdrBase], (instregex "^STR(BB|HH|W|X)ui$", "^STUR(BB|HH|W|X)i$",
                                                        "^STTR(B|H|W|X)i$", "^STL?LR(B|H|W|X)$")>;
def : InstRW<[WriteSTP, ReadST, ReadST, ReadAdrBase], (instregex "^STN?P(W|X)i$")>;
def : InstRW<[WriteSTX, ReadST, ReadAdrBase], (instregex "^STLXR(B|H|W|X)$")>;
def : InstRW<[WriteSTX, ReadST, ReadST, ReadAdrBase], (instregex "^STLXP(W|X)$")>;
def : InstRW<[WriteAdr, WriteST, ReadST, ReadAdrBase], (instregex "^STR(BB|HH|W|X)(pre|post)$")>;
def : InstRW<[WriteAdr, WriteSTP, ReadST, ReadST, ReadAdrBase], (instregex "^STP(W|X)(pre|post)$")>;


// FP data processing instructions
// -----------------------------------------------------------------------------

// FP absolute value
// FP arithmetic
// FP min/max
// FP negate
def : SchedAlias<WriteF, V1Write_2c_1V>;

// FP compare
def : SchedAlias<WriteFCmp, V1Write_2c_1V0>;

// FP divide
// FP square root
def : SchedAlias<WriteFDiv, V1Write_10c7_1V02>;

// [x1_fdiv_best] docs/x1-sog.md C-14, C-19, D-1: FP divide / square root, one write per (best
// latency, occupancy).  X1 SOG, FP and ASIMD divide / square root tables (l.1217-1245,
// l.1814-1818, l.1857-1866).  The SOG throughput is for the V0/V2 pair, so the occupancy of ONE
// unit is ceil(2 / best throughput) (D-2):
//   scalar H 8/7 -> 2, S 8/7 -> 2, D 4/7 -> 4 (FSQRT likewise);
//   ASIMD D-form F16 2/7 -> 7, D-form F32 4/7 -> 4, Q-form F16 1/5 (FDIV) / 2/11 (FSQRT)
//   -> 10 / 11, Q-form F32 2/7 -> 7, Q-form F64 2/7 -> 7.
// Best latency: 7, except Q-form F16 (FDIV 10, FSQRT 11).  [x1_dividers] D-3: the occupancy is
// on an FP divide unit (X1UnitFDiv); V02 is taken for one cycle.
def V1WriteFDS_7c2   : SchedWriteRes<[V1UnitV02, X1UnitFDiv]> { let Latency = 7;  let ReleaseAtCycles = [1, 2]; }
def V1WriteFDS_7c4   : SchedWriteRes<[V1UnitV02, X1UnitFDiv]> { let Latency = 7;  let ReleaseAtCycles = [1, 4]; }
def V1WriteFDS_7c7   : SchedWriteRes<[V1UnitV02, X1UnitFDiv]> { let Latency = 7;  let ReleaseAtCycles = [1, 7]; }
def V1WriteFDS_10c10 : SchedWriteRes<[V1UnitV02, X1UnitFDiv]> { let Latency = 10; let ReleaseAtCycles = [1, 10]; }
def V1WriteFDS_11c11 : SchedWriteRes<[V1UnitV02, X1UnitFDiv]> { let Latency = 11; let ReleaseAtCycles = [1, 11]; }

// FP divide, H-form
// FP square root, H-form
def : InstRW<[V1WriteFDS_7c2], (instrs FDIVHrr, FSQRTHr)>;

// FP divide, S-form
// FP square root, S-form
def : InstRW<[V1WriteFDS_7c2], (instrs FDIVSrr, FSQRTSr)>;

// FP divide, D-form
def : InstRW<[V1WriteFDS_7c4], (instrs FDIVDrr)>;

// FP square root, D-form
def : InstRW<[V1WriteFDS_7c4], (instrs FSQRTDr)>;

// FP multiply
def : WriteRes<WriteFMul, [V1UnitV]> { let Latency = 3; }

// FP multiply accumulate
def : InstRW<[V1Wr_FMA, ReadDefault, ReadDefault, V1Rd_FMA],
             (instregex "^FN?M(ADD|SUB)[HSD]rrr$")>;

// FP round to integral
def : InstRW<[V1Write_3c_1V02], (instregex "^FRINT[AIMNPXZ][HSD]r$",
                                           "^FRINT(32|64)[XZ][SD]r$")>;

// FP select
// X1 SOG Table 3-16: FCSEL "2, 2, V02" (stock V1: V01).
def : InstRW<[V1Write_2c_1V02], (instregex "^FCSEL[HSD]rrr$")>;


// FP miscellaneous instructions
// -----------------------------------------------------------------------------

// FP convert, from gen to vec reg
def : InstRW<[V1Write_3c_1M0], (instregex "^[SU]CVTF[SU][WX][HSD]ri$")>;

// FP convert, from vec to gen reg
// X1 SOG Table 3-18: "3, 1, V02" - either of V0/V2 but only ONE per cycle
// (stock V1: V0 only).  Expressed as a 2-cycle occupancy of the V02 group (D-2).
// [x1_fp_cvt] docs/x1-sog.md C-15: also the fixed-point forms (#fbits, "ri"), not WriteFCvt
// (V02 one cycle, 2 per cycle).
def V1Write_3c2_1V02 : SchedWriteRes<[V1UnitV02]> { let Latency = 3;
                                                    let ReleaseAtCycles = [2]; }
def : InstRW<[V1Write_3c2_1V02], (instregex "^FCVT[AMNPZ][SU][SU][WX][HSD]ri?$")>;

// FP convert, Javascript from vec to gen reg
def : InstRW<[V1Write_3c_1V0], (instrs FJCVTZS)>;

// FP convert, from vec to vec reg
def : SchedAlias<WriteFCvt, V1Write_3c_1V02>;

// FP move, immed
def : SchedAlias<WriteFImm, V1Write_2c_1V>;

// FP move, register
def : InstRW<[V1Write_2c_1V], (instrs FMOVHr, FMOVSr, FMOVDr)>;

// FP transfer, from gen to low half of vec reg
// (FMOV from WZR/XZR is not among the zero-latency MOVs of sec. 4.15: C-4.)
def : InstRW<[V1Write_3c_1M0], (instrs FMOVWHr, FMOVXHr, FMOVWSr, FMOVXDr)>;

// FP transfer, from gen to high half of vec reg
def : InstRW<[V1Write_5c_1M0_1V], (instrs FMOVXDHighr)>;

// FP transfer, from vec to gen reg
def : SchedAlias<WriteFCopy, V1Write_2c_1V1>;


// FP load instructions
// -----------------------------------------------------------------------------

// Load vector reg, literal, S/D/Q forms
// Load vector reg, unscaled immed
// Load vector reg, unsigned immed
def : InstRW<[V1Write_6c_1L, ReadAdrBase], (instregex "^LDR[SDQ]l$",
                                                      "^LDUR[BHSDQ]i$",
                                                      "^LDR[BHSDQ]ui$")>;

// Load vector reg, immed post-index
// Load vector reg, immed pre-index
def : InstRW<[WriteAdr, V1Write_6c_1L],
             (instregex "^LDR[BHSDQ](post|pre)$")>;

// Load vector reg, register offset, basic
// Load vector reg, register offset, scale, S/D-form
// Load vector reg, register offset, extend
// Load vector reg, register offset, extend, scale, S/D-form
// (The "L, I" of the "basic" row, l.1391-1392, is not modelled: docs/x1-sog.md I-4.)
def : InstRW<[V1Write_6c_1L, ReadAdrBase], (instregex "^LDR[BSD]ro[WX]$")>;

// Load vector reg, register offset, scale, H/Q-form
// Load vector reg, register offset, extend, scale, H/Q-form
// [x1_reg_offset_h] docs/x1-sog.md C-16, I-4: "7 3 I, L" only with the scale (l.1407-1408,
// l.1413-1414); without it LDR H/Q is "basic" (X index, I-4) or "extend" (W index), 6 3 L
// (l.1409-1410).
def X1WriteLDRHQro : SchedWriteVariant<[
                       SchedVar<X1IsMemScaled, [V1Write_7c_1I_1L]>,
                       SchedVar<NoSchedPred,   [V1Write_6c_1L]>]>;
def : InstRW<[X1WriteLDRHQro, ReadAdrBase], (instregex "^LDR[HQ]ro[WX]$")>;

// Load vector pair, immed offset, S/D-form
def : InstRW<[V1Write_6c_1L, V1Write_6c_0Z], (instregex "^LDN?P[SD]i$")>;

// Load vector pair, immed offset, Q-form
// [x1_ldp_hi] docs/x1-sog.md C-16, D-10: "6 3/2 L" (l.1417-1418): both registers 6 (the second
// is V1Write_6c_0Z, not WriteLDHi with latency 4).  Two L uOPs.
def : InstRW<[V1Write_6c_2L, V1Write_6c_0Z], (instrs LDPQi, LDNPQi)>;

// Load vector pair, immed post-index, S/D-form
// Load vector pair, immed pre-index, S/D-form
def : InstRW<[WriteAdr, V1Write_6c_1L, V1Write_6c_0Z],
             (instregex "^LDP[SD](pre|post)$")>;

// Load vector pair, immed post-index, Q-form
// Load vector pair, immed pre-index, Q-form
// [x1_ldp_hi] "6 3/2 L, I" (l.1421-1422, l.1425-1426).
def : InstRW<[WriteAdr, V1Write_6c_2L, V1Write_6c_0Z],
             (instrs LDPQpost, LDPQpre)>;


// FP store instructions
// -----------------------------------------------------------------------------
// [x1_from_a55_store_data] docs/x1-sog.md B-1: the data operands read ReadST, the base
// ReadAdrBase.

// Store vector reg, unscaled immed, B/H/S/D/Q-form
def : InstRW<[V1Write_2c_1L01_1V01, ReadST, ReadAdrBase], (instregex "^STUR[BHSDQ]i$")>;

// Store vector reg, immed post-index, B/H/S/D/Q-form
// Store vector reg, immed pre-index, B/H/S/D/Q-form
def : InstRW<[WriteAdr, V1Write_2c_1L01_1V01, ReadST, ReadAdrBase],
             (instregex "^STR[BHSDQ](pre|post)$")>;

// Store vector reg, unsigned immed, B/H/S/D/Q-form
def : InstRW<[V1Write_2c_1L01_1V01, ReadST, ReadAdrBase], (instregex "^STR[BHSDQ]ui$")>;

// Store vector reg, register offset, basic, B/S/D-form
// Store vector reg, register offset, scale, S/D-form
// Store vector reg, register offset, extend, B/S/D-form
// Store vector reg, register offset, extend, scale, S/D-form
def : InstRW<[V1Write_2c_1L01_1V01, ReadST, ReadAdrBase],
             (instregex "^STR[BSD]ro[WX]$")>;

// Store vector reg, register offset, basic, H/Q-form
// Store vector reg, register offset, scale, H/Q-form
// Store vector reg, register offset, extend, H/Q-form
// Store vector reg, register offset, extend, scale, H/Q-form
// [x1_store_rows] docs/x1-sog.md C-17: "2 2 I, L01, V01" only with the scale ("scale, H-form",
// "scale, Q-form", "extend, scale, H-form", "extend, scale, Q-form", l.1486-1501); "basic" and
// "extend" H/Q are "2 2 L01, V01" (l.1482-1485, l.1493-1495).
def X1WriteSTRHQro : SchedWriteVariant<[
                       SchedVar<X1IsMemScaled, [V1Write_2c_1I_1L01_1V01]>,
                       SchedVar<NoSchedPred,   [V1Write_2c_1L01_1V01]>]>;
def : InstRW<[X1WriteSTRHQro, ReadST, ReadAdrBase],
             (instregex "^STR[HQ]ro[WX]$")>;

// Store vector pair, immed offset, S/D/Q-form
// (Q-form "2 2 L01, V01", l.1515-1516: one V01 uOP, as printed; docs/x1-sog.md I-7.)
def : InstRW<[V1Write_2c_1L01_1V01, ReadST, ReadST, ReadAdrBase], (instregex "^STN?P[SDQ]i$")>;

// Store vector pair, immed post-index, S/D-form
// Store vector pair, immed pre-index, S/D-form
def : InstRW<[WriteAdr, V1Write_2c_1L01_1V01, ReadST, ReadST, ReadAdrBase],
             (instregex "^STP[SD](pre|post)$")>;

// Store vector pair, immed post-index, Q-form
// Store vector pair, immed pre-index, Q-form
def : InstRW<[WriteAdr, V1Write_2c_2L01_1V01, ReadST, ReadST, ReadAdrBase],
             (instrs STPQpre, STPQpost)>;


// ASIMD integer instructions
// -----------------------------------------------------------------------------

// ASIMD absolute diff
// ASIMD absolute diff long
// ASIMD arith, basic
// ASIMD arith, complex
// ASIMD arith, pair-wise
// ASIMD compare
// ASIMD logical
// ASIMD max/min, basic and pair-wise
def : SchedAlias<WriteVd, V1Write_2c_1V>;
def : SchedAlias<WriteVq, V1Write_2c_1V>;

// ASIMD absolute diff accum
// ASIMD absolute diff accum long
// ASIMD pairwise add and accumulate long
def : InstRW<[V1Wr_ADA, V1Rd_ADA], (instregex "^[SU]ABAL?v", "^[SU]ADALPv")>;

// ASIMD arith, reduce, 4H/4S
// ASIMD max/min, reduce, 4H/4S
def : InstRW<[V1Write_2c_1V13], (instregex "^(ADD|[SU]ADDL)Vv4(i16|i32)v$",
                                           "^[SU](MAX|MIN)Vv4(i16|i32)v$")>;

// ASIMD arith, reduce, 8B/8H
// ASIMD max/min, reduce, 8B/8H
def : InstRW<[V1Write_4c_1V13_1V], (instregex "^(ADD|[SU]ADDL)Vv8(i8|i16)v$",
                                              "^[SU](MAX|MIN)Vv8(i8|i16)v$")>;

// ASIMD arith, reduce, 16B
// ASIMD max/min, reduce, 16B
def : InstRW<[V1Write_4c_2V13], (instregex "^(ADD|[SU]ADDL)Vv16i8v$",
                                           "[SU](MAX|MIN)Vv16i8v$")>;

// ASIMD dot product
// ASIMD dot product using signed and unsigned integers
def : InstRW<[V1Wr_VDOT, V1Rd_VDOT],
             (instregex "^([SU]|SU|US)DOT(lane)?v(8|16)i8$")>;

// ASIMD matrix multiply-accumulate
def : InstRW<[V1Wr_VMMA, V1Rd_VMMA], (instrs SMMLA, UMMLA, USMMLA)>;

// ASIMD multiply
// [x1_mul_acc_fwd] MUL has a write of its own (U-5).
def : InstRW<[X1Wr_VMUL],      (instregex "^MULv")>;
def : InstRW<[V1Write_4c_1V02], (instregex "^SQ(R)?DMULHv")>;

// ASIMD multiply accumulate
def : InstRW<[V1Wr_VMA, V1Rd_VMA], (instregex "^MLAv", "^MLSv")>;

// ASIMD multiply accumulate long
def : InstRW<[V1Wr_VMAL, V1Rd_VMAL], (instregex "^[SU]MLALv", "^[SU]MLSLv")>;

// ASIMD multiply accumulate high
def : InstRW<[V1Write_4c_1V02], (instregex "^SQRDMLAHv", "^SQRDMLSHv")>;

// ASIMD multiply accumulate saturating long
def : InstRW<[V1Write_4c_1V02], (instregex "^SQDML[AS]L[iv]")>;

// ASIMD multiply/multiply long (8x8) polynomial
def : InstRW<[V1Write_3c_1V01], (instregex "^PMULL?v(8|16)i8$")>;

// Crypto polynomial (64x64) multiply long
// X1 SOG Table 3-34: PMULL(2) "2, 2, V01".
def : InstRW<[V1Write_2c_1V01], (instregex "^PMULLv(1i64|2i64)$")>;

// ASIMD multiply long
// [x1_mul_acc_fwd] SMULL, UMULL have a write of their own (U-5).
def : InstRW<[X1Wr_VMULL],     (instregex "^[SU]MULLv")>;
// [x1_asimd_scalar] docs/x1-sog.md D-5: the scalar SQDMULL (i16/i32) as well.
def : InstRW<[V1Write_3c_1V02], (instregex "^SQDMULLv", "^SQDMULLi(16|32)$")>;

// ASIMD shift accumulate
// [x1_asimd_scalar] D-5: the scalar forms (SSRA Dd, ... "d") as well.
def : InstRW<[V1Wr_VSA, V1Rd_VSA], (instregex "^[SU]SRAv", "^[SU]RSRAv", "^[SU]R?SRAd$")>;

// ASIMD shift by immed, complex
// ASIMD shift by register, complex
// [x1_asimd_scalar] D-5: the scalar SRSHR, URSHR (Dd) and the scalar saturating narrowing shifts
// (SQSHRN, SQRSHRN, UQSHRN, UQRSHRN, SQSHRUN, SQRSHRUN; Bd/Hd/Sd) as well.
def : InstRW<[V1Write_4c_1V13],
             (instregex "^RSHRNv", "^SQRSHRU?Nv", "^(SQSHLU?|UQSHL)[bhsd]$",
                        "^(SQSHLU?|UQSHL)(v8i8|v16i8|v4i16|v8i16|v2i32|v4i32|v2i64)_shift$",
                        "^SQSHU?RNv", "^SQSHRUNv", "^[SU]RSHRv", "^UQR?SHRNv", 
                        "^[SU]Q?RSHLv", "^[SU]QSHLv",
                        "^[SU]RSHRd$", "^(SQ|UQ)R?SHRN[bhs]$", "^SQR?SHRUN[bhs]$")>;

// ASIMD shift by immed, basic
// ASIMD shift by immed and insert, basic
// ASIMD shift by register, basic
// [x1_asimd_scalar] D-5: the scalar SHL, SSHR, USHR, SLI, SRI (Dd) as well.
def : InstRW<[V1Write_2c_1V13], (instregex "^SHLL?v", "^SHRNv", "^[SU]SHLLv",
                                          "^[SU]SHRv", "^S[LR]Iv", "^[SU]SHLv",
                                          "^SHLd$", "^[SU]SHRd$", "^S[LR]Id$")>;


// ASIMD FP instructions
// -----------------------------------------------------------------------------

// ASIMD FP absolute value/difference
// ASIMD FP arith, normal
// ASIMD FP compare
// ASIMD FP max/min, normal
// ASIMD FP max/min, pairwise
// ASIMD FP negate
// Covered by "SchedAlias (WriteV[dq]...)" above

// ASIMD FP complex add
def : InstRW<[V1Write_4c_1V], (instregex "^FCADD(v[48]f16|v[24]f32|v2f64)$")>;

// ASIMD FP complex multiply add
def : InstRW<[V1Wr_FCMA, V1Rd_FCMA], (instregex "^FCMLAv")>;

// ASIMD FP multiply
// [x1_asimd_scalar] docs/x1-sog.md D-5: the scalar FMULX (Hd/Sd/Dd) as well.
def : InstRW<[V1Wr_FPM], (instregex "^FMULX?v", "^FMULX(16|32|64)$")>;

// ASIMD FP multiply accumulate
def : InstRW<[V1Wr_FPMA, V1Rd_FPMA], (instregex "^FML[AS]v")>;

// ASIMD FP multiply accumulate long
def : InstRW<[V1Wr_FPMAL, V1Rd_FPMAL], (instregex "^FML[AS]L2?v")>;

// ASIMD FP convert, long (F16 to F32)
def : InstRW<[V1Write_4c_2V02], (instregex "^FCVTLv[48]i16$")>;

// ASIMD FP convert, long (F32 to F64)
def : InstRW<[V1Write_3c_1V02], (instregex "^FCVTLv[24]i32$")>;

// ASIMD FP convert, narrow (F32 to F16)
def : InstRW<[V1Write_4c_2V02], (instregex "^FCVTNv[48]i16$")>;

// ASIMD FP convert, narrow (F64 to F32)
def : InstRW<[V1Write_3c_1V02], (instregex "^FCVTNv[24]i32$",
                                           "^FCVTXN(v[24]f32|v1i64)$")>;

// ASIMD FP convert, other, D-form F32 and Q-form F64
// [x1_fp_cvt_scalar] docs/x1-sog.md D-4: every scalar form with the operand and the result in FP
// registers (H, S, D; also with #fbits) takes this row, the lowest the SOG gives for these
// mnemonics (as for the Cortex-A78, docs/a78-sog.md D-4), not the D-form F16 /
// Q-form F32 row (4, V02 two uOPs) for the S forms or the Q-form F16 row (6) for the H forms.
def : InstRW<[V1Write_3c_1V02], (instregex "^FCVT[AMNPZ][SU]v2f(32|64)$",
                                           "^FCVT[AMNPZ][SU]v2i(32|64)_shift$",
                                           "^FCVT[AMNPZ][SU]v1(i64|i32|f16)$",
                                           "^FCVTZ[SU][dsh]$",
                                           "^[SU]CVTFv2f(32|64)$",
                                           "^[SU]CVTFv2i(32|64)_shift$",
                                           "^[SU]CVTFv1i(64|32|16)$",
                                           "^[SU]CVTF[dsh]$")>;

// ASIMD FP convert, other, D-form F16 and Q-form F32
def : InstRW<[V1Write_4c_2V02], (instregex "^FCVT[AMNPZ][SU]v4f(16|32)$",
                                           "^FCVT[AMNPZ][SU]v4i(16|32)_shift$",
                                           "^[SU]CVTFv4f(16|32)$",
                                           "^[SU]CVTFv4i(16|32)_shift$")>;

// ASIMD FP convert, other, Q-form F16
// [x1_v_throughput] docs/x1-sog.md C-19: "6 1/2 V02" (l.1808-1813): V02 held 4 cycles in total
// (2 / (1/2), D-2).
def : InstRW<[X1WriteV02<6, 4, 4>], (instregex "^FCVT[AMNPZ][SU]v8f16$",
                                               "^FCVT[AMNPZ][SU]v8i16_shift$",
                                               "^[SU]CVTFv8f16$",
                                               "^[SU]CVTFv8i16_shift$")>;

// ASIMD FP divide, D-form, F16
// ASIMD FP square root, D-form, F16
def : InstRW<[V1WriteFDS_7c7], (instrs FDIVv4f16, FSQRTv4f16)>;

// ASIMD FP divide, D-form, F32
// ASIMD FP square root, D-form, F32
def : InstRW<[V1WriteFDS_7c4], (instrs FDIVv2f32, FSQRTv2f32)>;

// ASIMD FP divide, Q-form, F32
// ASIMD FP square root, Q-form, F32
def : InstRW<[V1WriteFDS_7c7], (instrs FDIVv4f32, FSQRTv4f32)>;

// ASIMD FP divide, Q-form, F16
def : InstRW<[V1WriteFDS_10c10], (instrs FDIVv8f16)>;

// ASIMD FP divide, Q-form, F64
def : InstRW<[V1WriteFDS_7c7], (instrs FDIVv2f64)>;

// ASIMD FP square root, Q-form, F16
def : InstRW<[V1WriteFDS_11c11], (instrs FSQRTv8f16)>;

// ASIMD FP square root, Q-form, F64
def : InstRW<[V1WriteFDS_7c7], (instrs FSQRTv2f64)>;

// ASIMD FP max/min, reduce, F32 and D-form F16
def : InstRW<[V1Write_4c_2V], (instregex "^F(MAX|MIN)(NM)?Vv4(i16|i32)v$")>;

// ASIMD FP max/min, reduce, Q-form F16
def : InstRW<[V1Write_6c_3V], (instregex "^F(MAX|MIN)(NM)?Vv8i16v$")>;

// ASIMD FP round, D-form F32 and Q-form F64
def : InstRW<[V1Write_3c_1V02], (instregex "^FRINT[AIMNPXZ]v2f(32|64)$")>;

// ASIMD FP round, D-form F16 and Q-form F32
def : InstRW<[V1Write_4c_2V02], (instregex "^FRINT[AIMNPXZ]v4f(16|32)$")>;

// ASIMD FP round, Q-form F16
// [x1_v_throughput] C-19: "6 1/2 V02" (l.1853-1856): V02 held 4 cycles.
def : InstRW<[X1WriteV02<6, 4, 4>], (instregex "^FRINT[AIMNPXZ]v8f16$")>;


// ASIMD BF instructions
// -----------------------------------------------------------------------------

// ASIMD convert, F32 to BF16
def : InstRW<[V1Write_4c_1V02], (instrs BFCVTN, BFCVTN2)>;

// ASIMD dot product
def : InstRW<[V1Wr_BFD, V1Rd_BFD], (instregex "^BF(DOT|16DOTlane)v[48]bf16$")>;

// ASIMD matrix multiply accumulate
def : InstRW<[V1Wr_BFMMA, V1Rd_BFMMA], (instrs BFMMLA)>;

// ASIMD multiply accumulate long
def : InstRW<[V1Wr_BFMLA, V1Rd_BFMLA], (instregex "^BFMLAL[BT](Idx)?$")>;

// Scalar convert, F32 to BF16
def : InstRW<[V1Write_3c_1V02], (instrs BFCVT)>;


// ASIMD miscellaneous instructions
// -----------------------------------------------------------------------------

// ASIMD bit reverse
// ASIMD bitwise insert
// ASIMD count
// ASIMD duplicate, element
// ASIMD extract
// ASIMD extract narrow
// ASIMD insert, element to element
// ASIMD move, FP immed
// ASIMD move, integer immed
// ASIMD reverse
// ASIMD table lookup, 1 or 2 table regs
// ASIMD table lookup extension, 1 table reg
// ASIMD transfer, element to gen reg
// ASIMD transpose
// ASIMD unzip/zip
// Covered by "SchedAlias (WriteV[dq]...)" above

// ASIMD duplicate, gen reg
def : InstRW<[V1Write_3c_1M0],
             (instregex "^DUP((v16|v8)i8|(v8|v4)i16|(v4|v2)i32|v2i64)gpr$")>;

// ASIMD extract narrow, saturating
def : InstRW<[V1Write_4c_1V13], (instregex "^[SU]QXTNv", "^SQXTUNv")>;

// ASIMD reciprocal and square root estimate, D-form U32
// ASIMD reciprocal and square root estimate, D-form F32 and scalar forms
// [x1_v_throughput] docs/x1-sog.md C-20: "D-form F32 and scalar forms 3 2 V02" (l.1966-1968):
// also the scalar H forms (not the D-form F16 row, 4).
def : InstRW<[V1Write_3c_1V02], (instrs URECPEv2i32,
                                        URSQRTEv2i32,
                                        FRECPEv1i32, FRECPEv2f32, FRECPEv1i64, FRECPEv1f16,
                                        FRSQRTEv1i32, FRSQRTEv2f32, FRSQRTEv1i64, FRSQRTEv1f16)>;

// ASIMD reciprocal and square root estimate, Q-form U32
// ASIMD reciprocal and square root estimate, D-form F16 and Q-form F32
// [x1_v_throughput] C-20: "4 1 V02" (l.1964-1965, l.1969-1971): V02 held 2 cycles.
def : InstRW<[X1WriteV02<4, 1, 2>], (instrs URECPEv4i32,
                                            URSQRTEv4i32,
                                            FRECPEv4f16, FRECPEv4f32,
                                            FRSQRTEv4f16, FRSQRTEv4f32)>;
// Q-form F64: no SOG row (docs/x1-sog.md U-8); the model's row (4, V02 one cycle) is kept.
def : InstRW<[V1Write_4c_1V02], (instrs FRECPEv2f64, FRSQRTEv2f64)>;

// ASIMD reciprocal and square root estimate, Q-form F16
// [x1_v_throughput] C-20: "6 1/2 V02" (l.1972-1973): V02 held 4 cycles.
def : InstRW<[X1WriteV02<6, 2, 4>], (instrs FRECPEv8f16,
                                            FRSQRTEv8f16)>;

// ASIMD reciprocal exponent
def : InstRW<[V1Write_3c_1V02], (instrs FRECPXv1f16, FRECPXv1i32, FRECPXv1i64)>;

// ASIMD reciprocal step
def : InstRW<[V1Write_4c_1V], (instregex "^FRECPS(16|32|64)$", "^FRECPSv",
                                         "^FRSQRTS(16|32|64)$", "^FRSQRTSv")>;

// ASIMD table lookup, 1 or 2 table regs
// ASIMD table lookup extension, 1 table reg
// X1 SOG Table 3-28: "TBL 1 or 2 regs / TBX 1 reg: 2, 2, V01" - a single uOP
// on V0/V1 (stock V1: two uOPs, throughput 1).
def : InstRW<[V1Write_2c_1V01], (instregex "^TBLv(8|16)i8(One|Two)$",
                                           "^TBXv(8|16)i8One$")>;

// ASIMD table lookup, 3 table regs
// ASIMD table lookup extension, 2 table reg
def : InstRW<[V1Write_4c_2V01], (instrs TBLv8i8Three, TBLv16i8Three,
                                        TBXv8i8Two, TBXv16i8Two)>;

// ASIMD table lookup, 4 table regs
def : InstRW<[V1Write_4c_3V01], (instrs TBLv8i8Four, TBLv16i8Four)>;

// ASIMD table lookup extension, 3 table reg
def : InstRW<[V1Write_6c_3V01], (instrs TBXv8i8Three, TBXv16i8Three)>;

// ASIMD table lookup extension, 4 table reg
def : InstRW<[V1Write_6c_5V01], (instrs TBXv8i8Four, TBXv16i8Four)>;

// ASIMD transfer, element to gen reg
// [x1_v_throughput] docs/x1-sog.md C-20: "UMOV, SMOV 2 1 V" (l.2002-2003): V held 4 cycles in
// total (4 / 1, D-2).
def : InstRW<[X1WriteV<2, 1, 4>], (instregex "^SMOVvi(((8|16)to(32|64))|32to64)$",
                                         "^UMOVvi(8|16|32|64)$")>;

// ASIMD transfer, gen reg to element
def : InstRW<[V1Write_5c_1M0_1V], (instregex "^INSvi(8|16|32|64)gpr$")>;


// ASIMD load instructions
// -----------------------------------------------------------------------------

// ASIMD load, 1 element, multiple, 1 reg
def : InstRW<[V1Write_6c_1L],
             (instregex "^LD1Onev(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, V1Write_6c_1L],
             (instregex "^LD1Onev(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 2 reg
def : InstRW<[V1Write_6c_2L],
             (instregex "^LD1Twov(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, V1Write_6c_2L],
             (instregex "^LD1Twov(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 3 reg
def : InstRW<[V1Write_6c_3L],
             (instregex "^LD1Threev(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, V1Write_6c_3L],
             (instregex "^LD1Threev(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 4 reg, D-form
def : InstRW<[V1Write_6c_2L],
             (instregex "^LD1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, V1Write_6c_2L],
             (instregex "^LD1Fourv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, multiple, 4 reg, Q-form
def : InstRW<[V1Write_7c_4L],
             (instregex "^LD1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V1Write_7c_4L],
             (instregex "^LD1Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, one lane
// ASIMD load, 1 element, all lanes
def : InstRW<[V1Write_8c_1L_1V],
             (instregex "^LD1(i|Rv)(8|16|32|64)$",
                        "^LD1Rv(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, V1Write_8c_1L_1V],
             (instregex "^LD1i(8|16|32|64)_POST$",
                        "^LD1Rv(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 2 element, multiple, D-form
def : InstRW<[V1Write_8c_1L_2V],
             (instregex "^LD2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, V1Write_8c_1L_2V],
             (instregex "^LD2Twov(8b|4h|2s)_POST$")>;
                        
// ASIMD load, 2 element, multiple, Q-form
def : InstRW<[V1Write_8c_2L_2V],
             (instregex "^LD2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V1Write_8c_2L_2V],
             (instregex "^LD2Twov(16b|8h|4s|2d)_POST$")>;
                        
// ASIMD load, 2 element, one lane
// ASIMD load, 2 element, all lanes
def : InstRW<[V1Write_8c_1L_2V],
             (instregex "^LD2i(8|16|32|64)$",
                        "^LD2Rv(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, V1Write_8c_1L_2V],
             (instregex "^LD2i(8|16|32|64)_POST$",
                        "^LD2Rv(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;
                        
// ASIMD load, 3 element, multiple, D-form
// ASIMD load, 3 element, one lane
// ASIMD load, 3 element, all lanes
def : InstRW<[V1Write_8c_2L_3V],
             (instregex "^LD3Threev(8b|4h|2s)$",
                        "^LD3i(8|16|32|64)$",
                        "^LD3Rv(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, V1Write_8c_2L_3V],
             (instregex "^LD3Threev(8b|4h|2s)_POST$",
                        "^LD3i(8|16|32|64)_POST$",
                        "^LD3Rv(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 3 element, multiple, Q-form
def : InstRW<[V1Write_8c_3L_3V],
             (instregex "^LD3Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V1Write_8c_3L_3V],
             (instregex "^LD3Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 4 element, multiple, D-form
// ASIMD load, 4 element, one lane
// ASIMD load, 4 element, all lanes
def : InstRW<[V1Write_8c_3L_4V],
             (instregex "^LD4Fourv(8b|4h|2s)$",
                        "^LD4i(8|16|32|64)$",
                        "^LD4Rv(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, V1Write_8c_3L_4V],
             (instregex "^LD4Fourv(8b|4h|2s)_POST$",
                        "^LD4i(8|16|32|64)_POST$",
                        "^LD4Rv(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 4 element, multiple, Q-form
// [x1_ls_throughput] docs/x1-sog.md C-21, D-6: "9 1/2 L, V" (l.2169-2172): L held 6 cycles in
// total (3 / (1/2)), 8 uOPs and 4 V cycles.
def : InstRW<[X1WriteLDV<9, 8, 6, 4>],
             (instregex "^LD4Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, X1WriteLDV<9, 8, 6, 4>],
             (instregex "^LD4Fourv(16b|8h|4s|2d)_POST$")>;


// ASIMD store instructions
// -----------------------------------------------------------------------------
// [x1_from_a55_store_data] docs/x1-sog.md B-1: the data Vt reads ReadST, the base Rn ReadAdrBase; for
// the one-lane forms the lane index takes the read in between (ReadDefault).  The data side is
// V01 (Table 2-1), not the "V" of the table: docs/x1-sog.md I-6.

// ASIMD store, 1 element, multiple, 1 reg
// ASIMD store, 1 element, multiple, 2 reg, D-form
def : InstRW<[V1Write_2c_1L01_1V01, ReadST, ReadAdrBase],
             (instregex "^ST1Onev(8b|16b|4h|8h|2s|4s|1d|2d)$",
                        "^ST1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, V1Write_2c_1L01_1V01, ReadST, ReadAdrBase],
             (instregex "^ST1Onev(8b|16b|4h|8h|2s|4s|1d|2d)_POST$",
                        "^ST1Twov(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 2 reg, Q-form
// ASIMD store, 1 element, multiple, 3 reg, D-form
// ASIMD store, 1 element, multiple, 4 reg, D-form
def : InstRW<[V1Write_2c_2L01_2V01, ReadST, ReadAdrBase],
             (instregex "^ST1Twov(16b|8h|4s|2d)$",
                        "^ST1Threev(8b|4h|2s|1d)$",
                        "^ST1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, V1Write_2c_2L01_2V01, ReadST, ReadAdrBase],
             (instregex "^ST1Twov(16b|8h|4s|2d)_POST$",
                        "^ST1Threev(8b|4h|2s|1d)_POST$",
                        "^ST1Fourv(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 3 reg, Q-form
def : InstRW<[V1Write_2c_3L01_3V01, ReadST, ReadAdrBase],
             (instregex "^ST1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V1Write_2c_3L01_3V01, ReadST, ReadAdrBase],
             (instregex "^ST1Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 4 reg, Q-form
def : InstRW<[V1Write_2c_4L01_4V01, ReadST, ReadAdrBase],
             (instregex "^ST1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V1Write_2c_4L01_4V01, ReadST, ReadAdrBase],
             (instregex "^ST1Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, one lane
// ASIMD store, 2 element, multiple, D-form
// ASIMD store, 2 element, one lane
def : InstRW<[V1Write_4c_1L01_1V01, ReadST, ReadDefault, ReadAdrBase],
             (instregex "^ST1i(8|16|32|64)$",
                        "^ST2Twov(8b|4h|2s)$",
                        "^ST2i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, V1Write_4c_1L01_1V01, ReadST, ReadDefault, ReadAdrBase],
             (instregex "^ST1i(8|16|32|64)_POST$",
                        "^ST2Twov(8b|4h|2s)_POST$",
                        "^ST2i(8|16|32|64)_POST$")>;

// ASIMD store, 2 element, multiple, Q-form
// ASIMD store, 3 element, multiple, D-form
// ASIMD store, 3 element, one lane
// ASIMD store, 4 element, one lane, D
def : InstRW<[V1Write_4c_2L01_2V01, ReadST, ReadDefault, ReadAdrBase],
             (instregex "^ST2Twov(16b|8h|4s|2d)$",
                        "^ST3Threev(8b|4h|2s)$",
                        "^ST3i(8|16|32|64)$",
                        "^ST4i64$")>;
def : InstRW<[WriteAdr, V1Write_4c_2L01_2V01, ReadST, ReadDefault, ReadAdrBase],
             (instregex "^ST2Twov(16b|8h|4s|2d)_POST$",
                        "^ST3Threev(8b|4h|2s)_POST$",
                        "^ST3i(8|16|32|64)_POST$",
                        "^ST4i64_POST$")>;

// ASIMD store, 3 element, multiple, Q-form
def : InstRW<[V1Write_5c_3L01_3V01, ReadST, ReadAdrBase],
             (instregex "^ST3Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, V1Write_5c_3L01_3V01, ReadST, ReadAdrBase],
             (instregex "^ST3Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 4 element, multiple, D-form
def : InstRW<[V1Write_6c_3L01_3V01, ReadST, ReadAdrBase],
             (instregex "^ST4Fourv(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, V1Write_6c_3L01_3V01, ReadST, ReadAdrBase],
             (instregex "^ST4Fourv(8b|4h|2s)_POST$")>;

// ASIMD store, 4 element, multiple, Q-form, B/H/S
def : InstRW<[V1Write_7c_6L01_6V01, ReadST, ReadAdrBase],
             (instregex "^ST4Fourv(16b|8h|4s)$")>;
def : InstRW<[WriteAdr, V1Write_7c_6L01_6V01, ReadST, ReadAdrBase],
             (instregex "^ST4Fourv(16b|8h|4s)_POST$")>;

// ASIMD store, 4 element, multiple, Q-form, D
def : InstRW<[V1Write_4c_4L01_4V01, ReadST, ReadAdrBase],
             (instrs ST4Fourv2d)>;
def : InstRW<[WriteAdr, V1Write_4c_4L01_4V01, ReadST, ReadAdrBase],
             (instrs ST4Fourv2d_POST)>;

// ASIMD store, 4 element, one lane, B/H/S
// [x1_ls_throughput] docs/x1-sog.md C-22, D-6: "6 4/3 V, L01" (l.2323-2326): L01 and V01 like every
// other store, L01 held 2 / (4/3) = 1.5, rounded up to 2 cycles (D-2), V01 2 cycles, 6 uOPs.
def : InstRW<[X1WriteSTV<6, 6, 2, 2>, ReadST, ReadDefault, ReadAdrBase],
             (instregex "^ST4i(8|16|32)$")>;
def : InstRW<[WriteAdr, X1WriteSTV<6, 6, 2, 2>, ReadST, ReadDefault, ReadAdrBase],
             (instregex "^ST4i(8|16|32)_POST$")>;


// Cryptography extensions
// -----------------------------------------------------------------------------

// Crypto polynomial (64x64) multiply long
// Covered by "SchedAlias (WriteV[dq]...)" above

// Crypto AES ops
// Cortex-X1 SOG Table 3-34: AESD/AESE/AESIMC/AESMC are "2, 2, V01" (latency 2,
// throughput 2, the V0/V1 pipes only).  Stock V1 uses the 4-wide V group
// (4 per cycle), which the X1 does not have for AES.
// [x1_aes] docs/x1-sog.md U-6: no forwarding from AESE/AESD to AESMC/AESIMC (no read advance,
// unlike the stock advance of 2); the SOG gives no number for an adjacent dependent pair (sec. 4.7,
// l.2604-2605), the decision of the Cortex-A78 (docs/a78-sog.md U-6).
def           : InstRW<[V1Write_2c_1V01], (instrs AESDrr, AESErr, AESMCrr, AESIMCrr)>;

// Crypto SHA1 hash acceleration op
// Crypto SHA1 schedule acceleration ops
// Crypto SHA256 schedule acceleration ops
// Crypto SHA512 hash acceleration ops
// Crypto SM3 ops
def : InstRW<[V1Write_2c_1V0], (instregex "^SHA1(H|SU[01])rrr?$",
                                          "^SHA256SU[01]rrr?$",
                                          "^SHA512(H2?|SU[01])$",
                                          "^SM3(PARTW(1|2SM3SS1)|TT[12][AB])$")>;

// Crypto SHA1 hash acceleration ops
// Crypto SHA256 hash acceleration ops
// Crypto SM4 ops
def : InstRW<[V1Write_4c_1V0], (instregex "^SHA1[CMP]rrr$",
                                          "^SHA256H2?rrr$",
                                          "^SM4E(KEY)?$")>;

// Crypto SHA3 ops
def : InstRW<[V1Write_2c_1V0], (instrs BCAX, EOR3, RAX1, XAR)>;


// CRC instruction
// -----------------------------------------------------------------------------

// CRC checksum ops
def : InstRW<[V1Wr_CRC, V1Rd_CRC], (instregex "^CRC32C?[BHWX]rr$")>;


// SVE instructions
// -----------------------------------------------------------------------------
// [x1_sve] docs/x1-sog.md D-8: the Cortex-X1 implements no SVE (Armv8.2-A with the extensions
// listed in sec. 2, l.373-376; LLVM's cortex-x1 has no SVE feature, so the disassembler does not
// decode SVE encodings for it).  SVE is in UnsupportedFeatures and the stock Neoverse V1 SVE rows
// are not carried.

}
