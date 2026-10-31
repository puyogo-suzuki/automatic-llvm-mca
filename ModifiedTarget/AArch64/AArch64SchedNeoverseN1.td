//=- AArch64SchedNeoverseN1.td - NeoverseN1 Scheduling Model -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the scheduling model for the Arm Neoverse N1 processors.
//
// References:
// - "Arm Neoverse N1 Software Optimization Guide"
// - https://en.wikichip.org/wiki/arm_holdings/microarchitectures/neoverse_n1
//
// In this project the model is installed ONLY for cortex-a76 / cortex-a76ae
// (src/cpu/cpu_traits.cpp, src/cpu/sched_model.cpp); neoverse-n1 and cortex-a77
// keep libLLVM's stock model.  It has been compared with the "Arm Cortex-A76
// Software Optimization Guide" (PJDOC-466751330-7215, Version 10.0), item by
// item, in docs/a76-sog.md; "l.N" below is line N of the text copy
// tmp/a76sog.txt, and [a76_*] tags name the items of that document.
//
//===----------------------------------------------------------------------===//

def NeoverseN1Model : SchedMachineModel {
  let IssueWidth            =   4; // Set to actual 4-wide decode/dispatch bandwidth of Cortex-A76
  let MicroOpBufferSize     = 128; // NOTE: Copied from Cortex-A76.
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  11; // Cycles cost of branch mispredicted.
  let LoopMicroOpBufferSize =  16; // NOTE: Copied from Cortex-A57.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    SVEUnsupported.F,
                                                    [HasMTE, HasCSSC, HasMOPS, HasMOPS_GO]);
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available on Neoverse N1.
// Instructions are first fetched and then decoded into internal macro-ops
// (MOPs).  From there, the MOPs proceed through register renaming and dispatch
// stages.  A MOP can be split into one or more micro-ops further down the
// pipeline, after the decode stage.  Once dispatched, micro-ops wait for their
// operands and issue out-of-order to one of the issue pipelines.  Each issue
// pipeline can accept one micro-op per cycle.

let SchedModel = NeoverseN1Model in {

// Define the issue ports.
// [a76_store_data_pipes] docs/a76-sog.md C-2, C-12: the A76 has eight execution
// pipelines, B, S0, S1, M, V0, V1, L0, L1 (l.225-226 and the diagram of sec. 2.1), and the
// legend of the instruction tables (sec. 3.2, l.299-309) defines "D" (the store-data pipelines
// of the integer stores) as "Integer single Cycle 1 and Integer multicycle", i.e. S1 and M.
// This model used to have two store-data pipelines of its own ("N1UnitD : ProcResource<2>"),
// which do not exist in the SOG and let integer store data take no integer-pipe bandwidth.
// S is split into S0 and S1 so that D can be the group {S1, M}.
def N1UnitB  : ProcResource<1>;  // Branch
def N1UnitS0 : ProcResource<1>;  // Integer single cycle 0
def N1UnitS1 : ProcResource<1>;  // Integer single cycle 1
def N1UnitM  : ProcResource<1>;  // Integer multicycle
def N1UnitL  : ProcResource<2>;  // Load/Store 0/1
def N1UnitV0 : ProcResource<1>;  // FP/ASIMD 0
def N1UnitV1 : ProcResource<1>;  // FP/ASIMD 1

def N1UnitI : ProcResGroup<[N1UnitS0, N1UnitS1, N1UnitM]>;  // Integer units ("I")
def N1UnitD : ProcResGroup<[N1UnitS1, N1UnitM]>;            // Store data ("D"): S1 and M
def N1UnitV : ProcResGroup<[N1UnitV0, N1UnitV1]>;           // FP/ASIMD units

// [a76_dividers] docs/a76-sog.md D-3: the iterative dividers.  An integer divide
// "block[s] any subsequent divide operations until complete" (l.482, note 1 of
// sec. 3.6) and an FP divide / square root blocks "subsequent similar operations
// to the same pipeline" (l.922-923, l.1508-1509); every pipeline "can accept and
// complete one uop per cycle" (l.226).  So a divide takes its pipeline (M, V0) for
// one cycle and holds the divider, not the pipeline, for its occupancy.
def N1UnitIDiv : ProcResource<1>;  // integer divider, in the M pipeline
def N1UnitFDiv : ProcResource<1>;  // FP divide / square root unit, in the V0 pipeline

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
// [a76_from_a55_store_data] docs/a76-sog.md B-1: the data operand of a store is read one
// cycle late (advance 1 from every producer).  The A76 SOG says only that stores are split into
// address and data uops at dispatch (l.738-739, l.1077, l.1827); it gives no operand timing.  This
// is the project's decision for the in-order cores (Cortex-A55, SOG-based there, docs/a55-sog.md
// C-18; Cortex-A520, borrowed, docs/a520-sog.md B-2), applied to the A76 for consistency: ReadST is
// attached to the data operands of every store form (STXR/STXP and the atomics excepted, as there).
// NOT SOG-based.  It acts only on Facile's store -> load memory edges (a store has no register result
// but its base update).
def : ReadAdvance<ReadST,      1>;
def : ReadAdvance<ReadVLD,     0>;

def : WriteRes<WriteAtomic,  []> { let Unsupported = 1; }
def : WriteRes<WriteBarrier, []> { let Latency = 1; }
def : WriteRes<WriteHint,    []> { let Latency = 1; }


//===----------------------------------------------------------------------===//
// Define generic 0 micro-op types

// [a76_ldp_hi] docs/a76-sog.md C-11, C-15: the second destination of LDP W / LDNP W / LDPSW.  It
// takes no uop of its own, but it is loaded by the same access as the first one, so it has the
// row's latency (4, l.631-632; LDPSW 5, l.635-642).  This used to be N1Write_0c_0Z (latency 0),
// which Facile (getEdgeLatency, per-def latency, 1-cycle floor) turned into a 1-cycle edge.
let Latency = 4, NumMicroOps = 0 in
def N1Write_4c_0Z : SchedWriteRes<[]>;
let Latency = 5, NumMicroOps = 0 in
def N1Write_5c_0Z : SchedWriteRes<[]>;

//===----------------------------------------------------------------------===//
// Define generic 1 micro-op types

def N1Write_1c_1B     : SchedWriteRes<[N1UnitB]>  { let Latency = 1; }
def N1Write_1c_1I     : SchedWriteRes<[N1UnitI]>  { let Latency = 1; }
def N1Write_2c_1M     : SchedWriteRes<[N1UnitM]>  { let Latency = 2; }
def N1Write_3c_1M     : SchedWriteRes<[N1UnitM]>  { let Latency = 3; }
def N1Write_4c3_1M    : SchedWriteRes<[N1UnitM]>  { let Latency = 4;
                                                    let ReleaseAtCycles = [3]; }
def N1Write_5c3_1M    : SchedWriteRes<[N1UnitM]>  { let Latency = 5;
                                                    let ReleaseAtCycles = [3]; }
// [a76_div_best] docs/a76-sog.md C-9, D-1: integer divide, W-form "5 to 12, 1/12 to 1/5", X-form
// "5 to 20, 1/20 to 1/5", M (l.428-429), early termination "depending upon the data values"
// (l.483).  The best value of each range: latency 5 and the divider held 5 cycles, for both forms
// (this used to be the worst-case latency 12 / 20 with M held 5 cycles).  The divider is its own
// resource ([a76_dividers]); the M pipeline is taken for one cycle.
def N1Write_5c5_1M_IDiv : SchedWriteRes<[N1UnitM, N1UnitIDiv]> { let Latency = 5;
                                                                 let ReleaseAtCycles = [1, 5]; }
def N1Write_4c_1L     : SchedWriteRes<[N1UnitL]>  { let Latency = 4; }
def N1Write_5c_1L     : SchedWriteRes<[N1UnitL]>  { let Latency = 5; }
def N1Write_7c_1L     : SchedWriteRes<[N1UnitL]>  { let Latency = 7; }
def N1Write_2c_1V     : SchedWriteRes<[N1UnitV]>  { let Latency = 2; }
def N1Write_3c_1V     : SchedWriteRes<[N1UnitV]>  { let Latency = 3; }
def N1Write_4c_1V     : SchedWriteRes<[N1UnitV]>  { let Latency = 4; }
def N1Write_5c_1V     : SchedWriteRes<[N1UnitV]>  { let Latency = 5; }
def N1Write_2c_1V0    : SchedWriteRes<[N1UnitV0]> { let Latency = 2; }
def N1Write_3c_1V0    : SchedWriteRes<[N1UnitV0]> { let Latency = 3; }
def N1Write_4c_1V0    : SchedWriteRes<[N1UnitV0]> { let Latency = 4; }
def N1Write_7c7_1V0   : SchedWriteRes<[N1UnitV0]> { let Latency = 7;
                                                    let ReleaseAtCycles = [7]; }
// FP divide / square root.  Every form has its own write below ("N1WriteFDS_<lat>c<occ>",
// next to the InstRW rows): the A76 SOG gives a latency RANGE and a throughput RANGE that
// depend on the operands, and the BEST of each is used - the shortest latency, and the
// unit occupied for ceil(1 / best throughput) cycles.  N1Write_10c7_1V0 is only the
// WriteFDiv fallback for a form with no row.
// [a76_dividers]: V0 for one cycle, the FP divide unit for the occupancy.
def N1Write_10c7_1V0  : SchedWriteRes<[N1UnitV0, N1UnitFDiv]> { let Latency = 7;
                                                    let ReleaseAtCycles = [1, 7]; }
def N1Write_2c_1V1    : SchedWriteRes<[N1UnitV1]> { let Latency = 2; }
def N1Write_3c_1V1    : SchedWriteRes<[N1UnitV1]> { let Latency = 3; }
def N1Write_4c_1V1    : SchedWriteRes<[N1UnitV1]> { let Latency = 4; }

//===----------------------------------------------------------------------===//
// Define generic 2 micro-op types

let Latency = 1, NumMicroOps = 2 in
def N1Write_1c_1B_1I   : SchedWriteRes<[N1UnitB, N1UnitI]>;
let Latency = 3, NumMicroOps = 2 in
def N1Write_3c_1I_1M   : SchedWriteRes<[N1UnitI, N1UnitM]>;
let Latency = 2, NumMicroOps = 2 in
def N1Write_2c_1I_1L   : SchedWriteRes<[N1UnitI, N1UnitL]>;
let Latency = 5, NumMicroOps = 2 in
def N1Write_5c_1I_1L   : SchedWriteRes<[N1UnitI, N1UnitL]>;
// [a76_ldp_hi] LDPSW: "5 1 I, L" (l.635-642): throughput 1 although one I and one L uop would
// allow 2 per cycle; the load pipe is held two cycles (as LDP X, which has two L uops).
let Latency = 5, NumMicroOps = 2 in
def N1Write_5c_1I_2L   : SchedWriteRes<[N1UnitI, N1UnitL]> { let ReleaseAtCycles = [1, 2]; }
let Latency = 6, NumMicroOps = 2 in
def N1Write_6c_1I_1L   : SchedWriteRes<[N1UnitI, N1UnitL]>;
let Latency = 7, NumMicroOps = 2 in
def N1Write_7c_1I_1L   : SchedWriteRes<[N1UnitI, N1UnitL]>;
let Latency = 5, NumMicroOps = 2 in
def N1Write_5c_1M_1V   : SchedWriteRes<[N1UnitM, N1UnitV]>;
let Latency = 6, NumMicroOps = 2 in
def N1Write_6c_1M_1V0  : SchedWriteRes<[N1UnitM, N1UnitV0]>;
let Latency = 5, NumMicroOps = 2 in
def N1Write_5c_2L      : SchedWriteRes<[N1UnitL, N1UnitL]>;
let Latency = 1, NumMicroOps = 2 in
def N1Write_1c_1L_1D   : SchedWriteRes<[N1UnitL, N1UnitD]>;
let Latency = 2, NumMicroOps = 2 in
def N1Write_2c_1L_1V   : SchedWriteRes<[N1UnitL, N1UnitV]>;
let Latency = 4, NumMicroOps = 2 in
def N1Write_4c_1L_1V   : SchedWriteRes<[N1UnitL, N1UnitV]>;
let Latency = 7, NumMicroOps = 2 in
def N1Write_7c_1L_1V   : SchedWriteRes<[N1UnitL, N1UnitV]>;
let Latency = 4, NumMicroOps = 2 in
def N1Write_4c_1V0_1V1 : SchedWriteRes<[N1UnitV0, N1UnitV1]>;
let Latency = 4, NumMicroOps = 2 in
def N1Write_4c_2V0     : SchedWriteRes<[N1UnitV0, N1UnitV0]>;
let Latency = 5, NumMicroOps = 2 in
def N1Write_5c_2V0     : SchedWriteRes<[N1UnitV0, N1UnitV0]>;
let Latency = 6, NumMicroOps = 2 in
def N1Write_6c_2V1     : SchedWriteRes<[N1UnitV1, N1UnitV1]>;
let Latency = 5, NumMicroOps = 2 in
def N1Write_5c_1V1_1V  : SchedWriteRes<[N1UnitV1, N1UnitV]>;

//===----------------------------------------------------------------------===//
// Define generic 3 micro-op types

// [a76_store_rows] latency 2 as the name says (SOG l.1103-1104: 2); it was 7, a typo.
let Latency = 2, NumMicroOps = 3 in
def N1Write_2c_1I_1L_1V : SchedWriteRes<[N1UnitI, N1UnitL, N1UnitV]>;
let Latency = 1, NumMicroOps = 3 in
def N1Write_1c_2L_1D    : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitD]>;
let Latency = 2, NumMicroOps = 3 in
def N1Write_2c_1L_2V    : SchedWriteRes<[N1UnitL, N1UnitV, N1UnitV]>;
let Latency = 6, NumMicroOps = 3 in
def N1Write_6c_3L       : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitL]>;
let Latency = 4, NumMicroOps = 3 in
def N1Write_4c_3V       : SchedWriteRes<[N1UnitV, N1UnitV, N1UnitV]>;
let Latency = 6, NumMicroOps = 3 in
def N1Write_6c_3V       : SchedWriteRes<[N1UnitV, N1UnitV, N1UnitV]>;
let Latency = 8, NumMicroOps = 3 in
def N1Write_8c_3V       : SchedWriteRes<[N1UnitV, N1UnitV, N1UnitV]>;

//===----------------------------------------------------------------------===//
// Define generic 4 micro-op types

let Latency = 2, NumMicroOps = 4 in
def N1Write_2c_2I_2L : SchedWriteRes<[N1UnitI, N1UnitI, N1UnitL, N1UnitL]>;
let Latency = 6, NumMicroOps = 4 in
def N1Write_6c_4L    : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitL, N1UnitL]>;
let Latency = 2, NumMicroOps = 4 in
def N1Write_2c_2L_2V : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitV, N1UnitV]>;
// [a76_store_rows] latency 3 as the name says (SOG ST1 2 reg Q-form and 4 reg D-form: 3, l.1839-1846);
// it was 2, a typo.
let Latency = 3, NumMicroOps = 4 in
def N1Write_3c_2L_2V : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitV, N1UnitV]>;
let Latency = 5, NumMicroOps = 4 in
def N1Write_5c_2L_2V : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitV, N1UnitV]>;
let Latency = 7, NumMicroOps = 4 in
def N1Write_7c_2L_2V : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitV, N1UnitV]>;
let Latency = 4, NumMicroOps = 4 in
def N1Write_4c_4V    : SchedWriteRes<[N1UnitV, N1UnitV, N1UnitV, N1UnitV]>;
let Latency = 6, NumMicroOps = 4 in
def N1Write_6c_4V0   : SchedWriteRes<[N1UnitV0, N1UnitV0, N1UnitV0, N1UnitV0]>;

//===----------------------------------------------------------------------===//
// Define generic 5 micro-op types

let Latency = 3, NumMicroOps = 5 in
def N1Write_3c_2L_3V : SchedWriteRes<[N1UnitL, N1UnitL,
                                      N1UnitV, N1UnitV, N1UnitV]>;
let Latency = 7, NumMicroOps = 5 in
def N1Write_7c_2L_3V : SchedWriteRes<[N1UnitL, N1UnitL,
                                      N1UnitV, N1UnitV, N1UnitV]>;
let Latency = 6, NumMicroOps = 5 in
def N1Write_6c_5V    : SchedWriteRes<[N1UnitV, N1UnitV, N1UnitV, N1UnitV, N1UnitV]>;

//===----------------------------------------------------------------------===//
// Define generic 6 micro-op types

let Latency = 3, NumMicroOps = 6 in
def N1Write_3c_4L_2V : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitL, N1UnitL,
                                      N1UnitV, N1UnitV]>;
let Latency = 4, NumMicroOps = 6 in
def N1Write_4c_3L_3V : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitL,
                                      N1UnitV, N1UnitV, N1UnitV]>;
let Latency = 5, NumMicroOps = 6 in
def N1Write_5c_3L_3V : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitL,
                                      N1UnitV, N1UnitV, N1UnitV]>;
let Latency = 6, NumMicroOps = 6 in
def N1Write_6c_3L_3V : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitL,
                                      N1UnitV, N1UnitV, N1UnitV]>;
let Latency = 7, NumMicroOps = 6 in
def N1Write_7c_3L_3V : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitL,
                                      N1UnitV, N1UnitV, N1UnitV]>;
let Latency = 8, NumMicroOps = 6 in
def N1Write_8c_3L_3V : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitL,
                                      N1UnitV, N1UnitV, N1UnitV]>;

//===----------------------------------------------------------------------===//
// Define generic 7 micro-op types

let Latency = 8, NumMicroOps = 7 in
def N1Write_8c_3L_4V : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitL,
                                      N1UnitV, N1UnitV, N1UnitV, N1UnitV]>;

//===----------------------------------------------------------------------===//
// Define generic 8 micro-op types

let Latency = 5, NumMicroOps = 8 in
def N1Write_5c_4L_4V  : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitL, N1UnitL, 
                                       N1UnitV, N1UnitV, N1UnitV, N1UnitV]>;
let Latency = 6, NumMicroOps = 8 in
def N1Write_6c_4L_4V  : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitL, N1UnitL, 
                                       N1UnitV, N1UnitV, N1UnitV, N1UnitV]>;
let Latency = 8, NumMicroOps = 8 in
def N1Write_8c_4L_4V  : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitL, N1UnitL,
                                       N1UnitV, N1UnitV, N1UnitV, N1UnitV]>;
let Latency = 10, NumMicroOps = 8 in
def N1Write_10c_4L_4V : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitL, N1UnitL,
                                       N1UnitV, N1UnitV, N1UnitV, N1UnitV]>;

//===----------------------------------------------------------------------===//
// Define generic 12 micro-op types

let Latency = 9, NumMicroOps = 12 in
def N1Write_9c_6L_6V : SchedWriteRes<[N1UnitL, N1UnitL, N1UnitL,
                                      N1UnitL, N1UnitL, N1UnitL, 
                                      N1UnitV, N1UnitV, N1UnitV,
                                      N1UnitV, N1UnitV, N1UnitV]>;

//===----------------------------------------------------------------------===//
// [a76_ls_throughput] docs/a76-sog.md C-20, C-21: a load or store with "L, V" pipelines whose
// throughput T is below what its uops give on two L and two V pipes.  The uop count and the V cycles
// are those of the write it replaces; the L pipes are held 2/T cycles in total, so the row's
// throughput is reached.  Which of the two pipelines limits is not stated; L is taken
// (docs/a76-sog.md D-7).
class N1WriteLS<int lat, int uops, int lcyc, int vcyc> : SchedWriteRes<[N1UnitL, N1UnitV]> {
  let Latency = lat;
  let NumMicroOps = uops;
  let ReleaseAtCycles = [lcyc, vcyc];
}

//===----------------------------------------------------------------------===//
// Define forwarded types

def N1Wr_IM32 : SchedWriteRes<[N1UnitM]> { let Latency = 2; }
def N1Wr_IM64 : SchedWriteRes<[N1UnitM]> { let Latency = 4;
                                           let ReleaseAtCycles = [3]; }
def N1Rd_IMA  : SchedReadAdvance<1, [N1Wr_IM32, N1Wr_IM64]>;

def N1Wr_MH   : SchedWriteRes<[N1UnitM]> { let Latency = 5;
                                           let ReleaseAtCycles = [4]; }
def N1Rd_MH : SchedReadAdvance<2, [N1Wr_MH]>;

def N1Wr_FMA : SchedWriteRes<[N1UnitV]> { let Latency = 4; }
// [a76_fmul_fwd] docs/a76-sog.md C-13: "late forwarding of the result from FP multiply uops to
// the accumulate operands of an FP multiply-accumulate uop ... 1 cycle after the FP multiply uop has
// been issued" (note 2 of sec. 3.11, l.925-927).  The advance named WriteFMul, which this model
// aliases (SchedAlias) to N1Write_3c_1V; a ReadAdvance does not see through the alias, so the scalar
// FMUL -> FMADD accumulator edge kept the full 3 cycles.  The scalar FP multiplies now have a write
// of their own, N1Wr_FMul, named in the advance.
def N1Wr_FMul : SchedWriteRes<[N1UnitV]> { let Latency = 3; }
def N1Rd_FMA : SchedReadAdvance<2, [WriteFMul, N1Wr_FMul, N1Wr_FMA]>;

def N1Wr_VA : SchedWriteRes<[N1UnitV1]> { let Latency = 4; }
def N1Rd_VA : SchedReadAdvance<3, [N1Wr_VA]>;

def N1Wr_VMA  : SchedWriteRes<[N1UnitV0]> { let Latency = 4; }
def N1Wr_VMAQ : SchedWriteRes<[N1UnitV0, N1UnitV0]> { let Latency = 5;
                                                      let NumMicroOps = 2; }
def N1Rd_VMA  : SchedReadAdvance<3, [N1Wr_VMA, N1Wr_VMAQ]>;

def N1Wr_VMAL : SchedWriteRes<[N1UnitV0]> { let Latency = 4; }
def N1Rd_VMAL : SchedReadAdvance<3, [N1Wr_VMAL]>;

def N1Wr_VPA : SchedWriteRes<[N1UnitV1]> { let Latency = 4; }
def N1Rd_VPA : SchedReadAdvance<3, [N1Wr_VPA]>;

def N1Wr_VSA : SchedWriteRes<[N1UnitV1]> { let Latency = 4; }
def N1Rd_VSA : SchedReadAdvance<3, [N1Wr_VSA]>;

def N1Wr_VFM  : SchedWriteRes<[N1UnitV]> { let Latency = 3; }
def N1Wr_VFMA : SchedWriteRes<[N1UnitV]> { let Latency = 4; }
def N1Rd_VFMA : SchedReadAdvance<2, [N1Wr_VFM, N1Wr_VFMA]>;

def N1Wr_VFMAL : SchedWriteRes<[N1UnitV]> { let Latency = 5; }
def N1Rd_VFMAL : SchedReadAdvance<3, [N1Wr_VFMAL]>;

def N1Wr_CRC : SchedWriteRes<[N1UnitM]> { let Latency = 2; }
def N1Rd_CRC : SchedReadAdvance<1, [N1Wr_CRC]>;

// Miscellaneous Instructions
// -----------------------------------------------------------------------------

def : InstRW<[WriteI], (instrs COPY)>;

// Convert floating-point condition flags
// Flag manipulation instructions
def : WriteRes<WriteSys, []> { let Latency = 1; }


// Branch Instructions
// -----------------------------------------------------------------------------

// Branch, immed
// Compare and branch
def : SchedAlias<WriteBr, N1Write_1c_1B>;

// Branch, register
def : SchedAlias<WriteBrReg, N1Write_1c_1B>;

// Branch and link, immed
// Branch and link, register
// [a76_blr_lr] docs/a76-sog.md C-7 (and X-2): "Branch and link, register (reg == lr) BLR 2 1 I, B"
// (l.321-322), against 1 for reg != lr (l.319).
def N1IsBlrLR : MCSchedPredicate<CheckRegOperand<0, LR>>;
let Latency = 2, NumMicroOps = 2 in
def N1Write_2c_1B_1I : SchedWriteRes<[N1UnitB, N1UnitI]>;
def N1WriteBLR : SchedWriteVariant<[
                   SchedVar<N1IsBlrLR,   [N1Write_2c_1B_1I]>,
                   SchedVar<NoSchedPred, [N1Write_1c_1B_1I]>]>;
def : InstRW<[N1Write_1c_1B_1I], (instrs BL)>;
def : InstRW<[N1WriteBLR], (instrs BLR)>;


// Arithmetic and Logical Instructions
// -----------------------------------------------------------------------------

// ALU, basic
// ALU, basic, flagset
// Conditional compare
// Conditional select
// Logical, basic
// Address generation
// Count leading
// Reverse bits/bytes
// Move immediate
def : SchedAlias<WriteI, N1Write_1c_1I>;

// ALU, extend and shift
def : SchedAlias<WriteIEReg, N1Write_2c_1M>;

// Arithmetic, LSL shift, shift <= 4
// Arithmetic, flagset, LSL shift, shift <= 4
// Arithmetic, LSR/ASR/ROR shift or LSL shift > 4
def N1WriteISReg : SchedWriteVariant<[
                     SchedVar<IsCheapLSL,  [N1Write_1c_1I]>,
                     SchedVar<NoSchedPred, [N1Write_2c_1M]>]>;
def              : SchedAlias<WriteISReg, N1WriteISReg>;

// Logical, shift, no flagset
def : InstRW<[N1Write_1c_1I],
             (instregex "^(AND|BIC|EON|EOR|ORN|ORR)[WX]rs$")>;

// Logical, shift, flagset
// [a76_logic_flagset] docs/a76-sog.md C-8: ANDS/BICS with a shifted register are
// "Logical, shift, flagset 2 1 M" (l.362), but without a shift (shift amount 0, e.g. TST Xn, Xm)
// they are "Logical, basic AND{S}, BIC{S}, ... 1 3 I" (l.359).  This row used to give every
// ANDS/BICS (register) 2 cycles on M.
def N1WriteLogicFlagset : SchedWriteVariant<[
                            SchedVar<RegShiftedPred, [N1Write_2c_1M]>,
                            SchedVar<NoSchedPred,    [N1Write_1c_1I]>]>;
def : InstRW<[N1WriteLogicFlagset], (instregex "^(AND|BIC)S[WX]rs$")>;


// Divide and multiply instructions
// -----------------------------------------------------------------------------

// Divide ([a76_div_best], [a76_dividers])
def : SchedAlias<WriteID32, N1Write_5c5_1M_IDiv>;
def : SchedAlias<WriteID64, N1Write_5c5_1M_IDiv>;

// Multiply accumulate, W-form
// Multiply accumulate, X-form
// Multiply accumulate long
def : SchedAlias<WriteIM32, N1Write_2c_1M>;
def : SchedAlias<WriteIM64, N1Write_4c3_1M>;

def : InstRW<[N1Wr_IM32, ReadIM, ReadIM, N1Rd_IMA],
             (instregex "^M(ADD|SUB)Wrrr$")>;
def : InstRW<[N1Wr_IM64, ReadIM, ReadIM, N1Rd_IMA],
             (instregex "^M(ADD|SUB)Xrrr$")>;
def : InstRW<[N1Wr_IM32, ReadIM, ReadIM, N1Rd_IMA],
             (instregex "^(S|U)M(ADD|SUB)Lrrr$")>;

// Multiply high
def : InstRW<[N1Wr_MH, ReadIM, ReadIM, N1Rd_MH], (instrs SMULHrr, UMULHrr)>;


// Miscellaneous data-processing instructions
// -----------------------------------------------------------------------------

// Bitfield extract, one reg
// Bitfield extract, two regs
def N1WriteExtr : SchedWriteVariant<[
                    SchedVar<IsRORImmIdiomPred, [N1Write_1c_1I]>,
                    SchedVar<NoSchedPred,       [N1Write_3c_1I_1M]>]>;
def : SchedAlias<WriteExtr, N1WriteExtr>;

// Bitfield move, basic
// Variable shift
def : SchedAlias<WriteIS, N1Write_1c_1I>;

// Bitfield move, insert
def : InstRW<[N1Write_2c_1M], (instregex "^BFM[WX]ri$")>;

// Move immediate
def : SchedAlias<WriteImm, N1Write_1c_1I>;

// Load instructions
// -----------------------------------------------------------------------------

// Load register, immed offset
def : SchedAlias<WriteLD, N1Write_4c_1L>;

// Load register, immed offset, index
def : SchedAlias<WriteLDIdx, N1Write_4c_1L>;
def : SchedAlias<WriteAdr,   N1Write_1c_1I>;

// [a76_reg_offset_h] docs/a76-sog.md C-11, C-15: "Load register, register offset, scale by 2"
// and "..., extend, scale by 2" LDRH, LDRSH: "5 2 I, L" (l.623-624, l.629-630); without the scale
// they are "basic" / "extend" (4 2 L, l.619-620, l.625-626).  Every register-offset LDRH/LDRSH
// used to be 4 on L.  "Scaled" = the S bit of the addressing mode (AArch64_AM::getMemDoShift).
def N1IsMemScaled : MCSchedPredicate<CheckMemScaled>;
def N1WriteLDRH : SchedWriteVariant<[
                    SchedVar<N1IsMemScaled, [N1Write_5c_1I_1L]>,
                    SchedVar<NoSchedPred,   [N1Write_4c_1L]>]>;
def : InstRW<[N1WriteLDRH, ReadAdrBase], (instregex "^LDRHHro[WX]$", "^LDRSH[WX]ro[WX]$")>;

// Load pair, immed offset
def : SchedAlias<WriteLDHi, N1Write_4c_1L>;

// Load pair, immed offset, W-form
def : InstRW<[N1Write_4c_1L, N1Write_4c_0Z], (instrs LDPWi, LDNPWi)>;

// Load pair, signed immed offset, signed words
def : InstRW<[N1Write_5c_1I_2L, N1Write_5c_0Z], (instrs LDPSWi)>;

// Load pair, immed post or pre-index, signed words
def : InstRW<[WriteAdr, N1Write_5c_1I_2L, N1Write_5c_0Z],
             (instrs LDPSWpost, LDPSWpre)>;


// Store instructions
// -----------------------------------------------------------------------------

// Store register, immed offset
def : SchedAlias<WriteST, N1Write_1c_1L_1D>;

// Store register, immed offset, index
def : SchedAlias<WriteSTIdx, N1Write_1c_1L_1D>;

// Store pair, immed offset
def : SchedAlias<WriteSTP, N1Write_1c_2L_1D>;

// Store pair, immed offset, W-form
// [a76_store_rows] docs/a76-sog.md C-12, C-16: "Store pair, immed offset, W-form STP, STNP
// 1 2 L, D" (l.777-778); STNP W used to take the X-form's two L uops (throughput 1).
def : InstRW<[N1Write_1c_1L_1D, ReadST, ReadST, ReadAdrBase], (instrs STPWi, STNPWi)>;
// [a76_from_a55_store_data] the integer store forms whose stock Sched lists name no reads: same
// writes, plus ReadST on the data.
def : InstRW<[WriteST, ReadST, ReadAdrBase], (instregex "^STR(BB|HH|W|X)ui$", "^STUR(BB|HH|W|X)i$",
                                                        "^STTR(B|H|W|X)i$", "^STL?LR(B|H|W|X)$")>;
def : InstRW<[WriteSTP, ReadST, ReadST, ReadAdrBase], (instrs STPXi, STNPXi)>;
def : InstRW<[WriteSTX, ReadST, ReadAdrBase], (instregex "^STLXR(B|H|W|X)$")>;
def : InstRW<[WriteSTX, ReadST, ReadST, ReadAdrBase], (instregex "^STLXP(W|X)$")>;

// [a76_store_rows] "Store register, register offset, scaled by 2" and "..., extend, scale by 1"
// (sic, I-4) STRH: "2 3/2 I, L, D" (l.769-770, l.775-776); the other register-offset
// stores are "1 2 L, D".  Every register-offset STRH used to be 1 on L and D.
let Latency = 2, NumMicroOps = 3 in
def N1Write_2c_1I_1L_1D : SchedWriteRes<[N1UnitI, N1UnitL, N1UnitD]>;
def N1WriteSTRH : SchedWriteVariant<[
                    SchedVar<N1IsMemScaled, [N1Write_2c_1I_1L_1D]>,
                    SchedVar<NoSchedPred,   [N1Write_1c_1L_1D]>]>;
def : InstRW<[N1WriteSTRH, ReadST, ReadAdrBase], (instregex "^STRHHro[WX]$")>;

// [a76_store_wb] docs/a76-sog.md D-6, I-3: the pre- and post-index forms of the integer stores
// are "Store register, immed post-index / pre-index ... 1 2 L, D" and "Store pair, immed
// post-index / pre-index ... 1 1 L, D" (l.744-747, l.780-787): no I pipeline, and STR keeps
// throughput 2, which with the data on S1/M ([a76_store_data_pipes]) leaves no integer pipe for a
// base-update uop.  The base update keeps latency 1 (the "(1)" of the FP/ASIMD writeback notes,
// l.1165-1166) and one uop, but uses no pipeline.  It used to be WriteAdr (1 cycle on I).
def N1Write_1c_STWB : SchedWriteRes<[]> { let Latency = 1; let NumMicroOps = 1; }
def : InstRW<[N1Write_1c_STWB, WriteST, ReadST, ReadAdrBase], (instregex "^STR(BB|HH|W|X)(pre|post)$")>;
def : InstRW<[N1Write_1c_STWB, WriteSTP, ReadST, ReadST, ReadAdrBase], (instregex "^STP(W|X)(pre|post)$")>;


// FP data processing instructions
// -----------------------------------------------------------------------------

// FP absolute value
// FP arithmetic
// FP min/max
// FP negate
// FP select
def : SchedAlias<WriteF, N1Write_2c_1V>;

// FP compare
def : SchedAlias<WriteFCmp, N1Write_2c_1V0>;

// FP divide
// FP square root
def : SchedAlias<WriteFDiv, N1Write_10c7_1V0>;

// FP divide / square root, one write per (best latency, occupancy).  A76 SOG, FP and
// ASIMD divide / square root tables; best throughput -> occupancy in cycles:
//   scalar H 4/7 -> 2, S 4/7 -> 2, D 2/7 -> 4 (FSQRT D: 2/7 as well);
//   ASIMD D-form F16 1/7 -> 7, D-form F32 2/7 -> 4, Q-form F16 1/10 (FDIV) / 1/11 (FSQRT)
//   -> 10 / 11, Q-form F32 1/7 -> 7, Q-form F64 1/7 -> 7.
// Best latency: 7, except Q-form F16 (FDIV 10, FSQRT 11).
// [a76_dividers] docs/a76-sog.md D-3: the occupancy is on the FP divide unit N1UnitFDiv;
// the V0 pipeline is taken for one cycle (it used to be held for the whole occupancy).
def N1WriteFDS_7c2   : SchedWriteRes<[N1UnitV0, N1UnitFDiv]> { let Latency = 7;  let ReleaseAtCycles = [1, 2]; }
def N1WriteFDS_7c4   : SchedWriteRes<[N1UnitV0, N1UnitFDiv]> { let Latency = 7;  let ReleaseAtCycles = [1, 4]; }
def N1WriteFDS_7c7   : SchedWriteRes<[N1UnitV0, N1UnitFDiv]> { let Latency = 7;  let ReleaseAtCycles = [1, 7]; }
def N1WriteFDS_10c10 : SchedWriteRes<[N1UnitV0, N1UnitFDiv]> { let Latency = 10; let ReleaseAtCycles = [1, 10]; }
def N1WriteFDS_11c11 : SchedWriteRes<[N1UnitV0, N1UnitFDiv]> { let Latency = 11; let ReleaseAtCycles = [1, 11]; }

// FP divide, H-form
// FP square root, H-form
def : InstRW<[N1WriteFDS_7c2], (instrs FDIVHrr, FSQRTHr)>;

// FP divide, S-form
// FP square root, S-form
def : InstRW<[N1WriteFDS_7c2], (instrs FDIVSrr, FSQRTSr)>;

// FP divide, D-form
def : InstRW<[N1WriteFDS_7c4], (instrs FDIVDrr)>;

// FP square root, D-form
def : InstRW<[N1WriteFDS_7c4], (instrs FSQRTDr)>;

// FP multiply
def : SchedAlias<WriteFMul, N1Write_3c_1V>;
// [a76_fmul_fwd]
def : InstRW<[N1Wr_FMul], (instregex "^FN?MUL[HSD]rr$", "^FMULX(16|32|64)$")>;

// FP multiply accumulate
def : InstRW<[N1Wr_FMA, ReadDefault, ReadDefault, N1Rd_FMA], (instregex "^FN?M(ADD|SUB)[HSD]rrr$")>;

// FP round to integral
def : InstRW<[N1Write_3c_1V0], (instregex "^FRINT[AIMNPXZ][HSD]r$",
                                          "^FRINT(32|64)[XZ][SD]r$")>;


// FP miscellaneous instructions
// -----------------------------------------------------------------------------

// FP convert, from vec to vec reg
// FP convert, Javascript from vec to gen reg
def : SchedAlias<WriteFCvt, N1Write_3c_1V>;

// FP convert, from gen to vec reg
def : InstRW<[N1Write_6c_1M_1V0], (instregex "^[SU]CVTF[SU][WX][HSD]ri$")>;

// FP convert, from vec to gen reg
// [a76_fp_cvt] docs/a76-sog.md C-14, C-18: the fixed-point forms (FCVTZS/FCVTZU Rd, Fn, #fbits,
// "...ri") are FCVTZS/FCVTZU too (l.938-941: 4 1 V0, V1); they used to fall to WriteFCvt (3, V).
def : InstRW<[N1Write_4c_1V0_1V1], (instregex "^FCVT[AMNPZ][SU][SU][WX][HSD]r$",
                                              "^FCVTZ[SU]S[WX][HSD]ri$")>;

// [a76_fp_cvt] "FP convert, from vec to vec reg FCVT, FCVTXN 3 1 V0" (l.936): the precision
// conversions and the scalar FCVTXN used to be on V (2 per cycle; FCVTXN latency 2).
def : InstRW<[N1Write_3c_1V0], (instregex "^FCVT[HSD][HSD]r$", "^FCVTXNv1i64$")>;

// [a76_fp_cvt_scalar] docs/a76-sog.md D-4: FCVT[AMNPZ][SU], SCVTF, UCVTF with the
// operand and the result in FP/ASIMD registers (scalar forms, integer and fixed-point) have no row
// of their own.  They are a "convert, from vec to vec reg" (l.936: 3 1 V0), and the ASIMD rows of
// the same mnemonics give 3 1 V0 for two lanes, more for four and eight (l.1377-1402); one lane
// is taken as the two-lane row.  They used to be 2 on V (the stock class).
def : InstRW<[N1Write_3c_1V0], (instregex "^FCVT[AMNPZ][SU]v1(i32|i64|f16)$",
                                          "^[SU]CVTFv1(i16|i32|i64)$",
                                          "^FCVTZ[SU][hsd]$", "^[SU]CVTF[hsd]$")>;

// FP move, immed
def : SchedAlias<WriteFImm, N1Write_2c_1V>;

// FP move, register
def : InstRW<[N1Write_2c_1V], (instrs FMOVHr, FMOVSr, FMOVDr)>;

// FP transfer, from gen to low half of vec reg
// FP transfer, from gen to high half of vec reg
def : InstRW<[N1Write_3c_1M], (instrs FMOVWHr, FMOVXHr, FMOVWSr, FMOVXDr,
                                      FMOVXDHighr)>;

// FP transfer, from vec to gen reg
def : SchedAlias<WriteFCopy, N1Write_2c_1V1>;


// FP load instructions
// -----------------------------------------------------------------------------

// Load vector reg, literal, S/D/Q forms
// Load vector reg, unscaled immed
def : InstRW<[N1Write_5c_1L, ReadAdrBase], (instregex "^LDR[SDQ]l$",
                                                      "^LDUR[BHSDQ]i$")>;

// Load vector reg, immed post-index
// Load vector reg, immed pre-index
def : InstRW<[WriteAdr, N1Write_5c_1L],
             (instregex "^LDR[BHSDQ](post|pre)$")>;

// Load vector reg, unsigned immed
def : InstRW<[N1Write_5c_1I_1L], (instregex "^LDR[BHSDQ]ui$")>;

// Load vector reg, register offset, basic
// Load vector reg, register offset, scale, S/D-form
// Load vector reg, register offset, extend
// Load vector reg, register offset, extend, scale, S/D-form
def : InstRW<[N1Write_5c_1I_1L, ReadAdrBase], (instregex "^LDR[BSD]ro[WX]$")>;

// Load vector reg, register offset, scale, H/Q-form
// Load vector reg, register offset, extend, scale, H/Q-form
// [a76_reg_offset_h] docs/a76-sog.md C-11, C-15: only the scaled H/Q forms are "6 2 I, L"
// (l.1015-1016, l.1032-1033); unscaled they are "basic" / "extend" (5 2 L, I, l.1011-1012,
// l.1017-1018).  Every H/Q register-offset load used to be 6.
def N1WriteLDRHQ : SchedWriteVariant<[
                     SchedVar<N1IsMemScaled, [N1Write_6c_1I_1L]>,
                     SchedVar<NoSchedPred,   [N1Write_5c_1I_1L]>]>;
def : InstRW<[N1WriteLDRHQ, ReadAdrBase], (instregex "^LDR[HQ]ro[WX]$")>;

// [a76_ldp_hi] docs/a76-sog.md C-11, C-15: the second register of an FP/ASIMD load pair has the
// row's latency (S/D 5, Q 7; l.1034-1045).  It used to be WriteLDHi (4, the integer load).  The
// Q-form with an immediate offset is "7 1 L" (l.1036-1037), without the I uop it used to have; the
// regular expression "^LDPN?[HQ]i$" also missed LDNP Q (LDNPQi, which had 4 and 4).
// Load vector pair, immed offset, S/D-form
def : InstRW<[N1Write_5c_1I_1L, N1Write_5c_1L], (instregex "^LDN?P[SD]i$")>;

// Load vector pair, immed offset, Q-form
def : InstRW<[N1Write_7c_1L, N1Write_7c_1L], (instrs LDPQi, LDNPQi)>;

// Load vector pair, immed post-index, S/D-form
// Load vector pair, immed pre-index, S/D-form
def : InstRW<[WriteAdr, N1Write_5c_1L, N1Write_5c_1L],
             (instregex "^LDP[SD](pre|post)$")>;

// Load vector pair, immed post-index, Q-form
// Load vector pair, immed pre-index, Q-form
def : InstRW<[WriteAdr, N1Write_7c_1L, N1Write_7c_1L],
             (instrs LDPQpost, LDPQpre)>;


// FP store instructions
// -----------------------------------------------------------------------------

// Store vector reg, unscaled immed, B/H/S/D-form
def : InstRW<[N1Write_2c_1I_1L, ReadST, ReadAdrBase], (instregex "^STUR[BHSD]i$")>;

// Store vector reg, unscaled immed, Q-form
def : InstRW<[N1Write_2c_2I_2L, ReadST, ReadAdrBase], (instrs STURQi)>;

// Store vector reg, immed post-index, B/H/S/D-form
// Store vector reg, immed pre-index, B/H/S/D-form
def : InstRW<[WriteAdr, N1Write_2c_1L_1V, ReadST, ReadAdrBase], (instregex "^STR[BHSD](pre|post)$")>;

// Store vector reg, immed pre-index, Q-form
// Store vector reg, immed post-index, Q-form
def : InstRW<[WriteAdr, N1Write_2c_2L_2V, ReadST, ReadAdrBase], (instrs STRQpre, STRQpost)>;

// Store vector reg, unsigned immed, B/H/S/D-form
def : InstRW<[N1Write_2c_1L_1V, ReadST, ReadAdrBase], (instregex "^STR[BHSD]ui$")>;

// Store vector reg, unsigned immed, Q-form
def : InstRW<[N1Write_2c_2L_2V, ReadST, ReadAdrBase], (instrs STRQui)>;

// Store vector reg, register offset, basic, B/S/D-form
// Store vector reg, register offset, scale, B/S/D-form
// Store vector reg, register offset, extend, B/S/D-form
// Store vector reg, register offset, extend, scale, B/S/D-form
def : InstRW<[N1Write_2c_1L_1V, ReadST, ReadAdrBase], (instregex "^STR[BSD]ro[WX]$")>;

// Store vector reg, register offset, basic, H-form
// Store vector reg, register offset, scale, H-form
// Store vector reg, register offset, extend, H-form
// Store vector reg, register offset, extend, scale, H-form
// Store vector reg, register offset, basic, Q-form
// Store vector reg, register offset, scale, Q-form
// Store vector reg, register offset, extend, Q-form
// Store vector reg, register offset, extend, scale, Q-form
// [a76_store_rows] docs/a76-sog.md C-12, C-16: only the SCALED H and Q forms have the I uop
// ("scale, H-form 2 2 I, L, V", "scale, Q-form 2 1 I, L, V", and the "extend, scale" rows,
// l.1103-1108, l.1113-1118); "basic" and "extend" without scale are "L, V" for every size
// (l.1099-1102, l.1109-1112).  The H form used to have the I uop always, the Q form never.
let Latency = 2, NumMicroOps = 5 in
def N1Write_2c_1I_2L_2V : SchedWriteRes<[N1UnitI, N1UnitL, N1UnitL, N1UnitV, N1UnitV]>;
def N1WriteSTRHro : SchedWriteVariant<[
                      SchedVar<N1IsMemScaled, [N1Write_2c_1I_1L_1V]>,
                      SchedVar<NoSchedPred,   [N1Write_2c_1L_1V]>]>;
def N1WriteSTRQro : SchedWriteVariant<[
                      SchedVar<N1IsMemScaled, [N1Write_2c_1I_2L_2V]>,
                      SchedVar<NoSchedPred,   [N1Write_2c_2L_2V]>]>;
def : InstRW<[N1WriteSTRHro, ReadST, ReadAdrBase], (instregex "^STRHro[WX]$")>;
def : InstRW<[N1WriteSTRQro, ReadST, ReadAdrBase], (instregex "^STRQro[WX]$")>;

// Store vector pair, immed offset, S-form
def : InstRW<[N1Write_2c_1L_1V, ReadST, ReadST, ReadAdrBase], (instrs STPSi, STNPSi)>;

// Store vector pair, immed offset, D-form
def : InstRW<[N1Write_2c_2L_2V, ReadST, ReadST, ReadAdrBase], (instrs STPDi, STNPDi)>;

// Store vector pair, immed offset, Q-form
def : InstRW<[N1Write_3c_4L_2V, ReadST, ReadST, ReadAdrBase], (instrs STPQi, STNPQi)>;

// Store vector pair, immed post-index, S-form
// Store vector pair, immed pre-index, S-form
// [a76_store_rows] "2 1 L, V" (l.1138-1139, l.1144-1145), against "2 2" for the offset form
// (l.1132-1133): one L uop held two cycles (it used to be one cycle, throughput 2).
def : InstRW<[WriteAdr, N1WriteLS<2, 2, 2, 1>, ReadST, ReadST, ReadAdrBase], (instrs STPSpre, STPSpost)>;

// Store vector pair, immed post-index, D-form
// Store vector pair, immed pre-index, D-form
def : InstRW<[WriteAdr, N1Write_2c_2L_2V, ReadST, ReadST, ReadAdrBase], (instrs STPDpre, STPDpost)>;

// Store vector pair, immed post-index, Q-form
// Store vector pair, immed pre-index, Q-form
// [a76_stpq_post] docs/a76-sog.md C-16, I-5: the post-index row prints throughput 1 (l.1142-1143),
// the offset and pre-index rows 1/2 (l.1136-1137, l.1148-1149).  Each row is taken as printed: the
// post-index form holds the L pipes 2 cycles in total (N1WriteLS), the pre-index form 4.
def : InstRW<[WriteAdr, N1Write_3c_4L_2V, ReadST, ReadST, ReadAdrBase], (instrs STPQpre)>;
def : InstRW<[WriteAdr, N1WriteLS<3, 6, 2, 2>, ReadST, ReadST, ReadAdrBase], (instrs STPQpost)>;


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
def : SchedAlias<WriteVd, N1Write_2c_1V>;
def : SchedAlias<WriteVq, N1Write_2c_1V>;

// ASIMD absolute diff accum
// ASIMD absolute diff accum long
def : InstRW<[N1Wr_VA, N1Rd_VA], (instregex "^[SU]ABAL?v")>;

// ASIMD arith, reduce, 4H/4S
def : InstRW<[N1Write_3c_1V1], (instregex "^(ADDV|[SU]ADDLV)v4(i16|i32)v$")>;

// ASIMD arith, reduce, 8B/8H
def : InstRW<[N1Write_5c_1V1_1V], (instregex "^(ADDV|[SU]ADDLV)v8(i8|i16)v$")>;

// ASIMD arith, reduce, 16B
def : InstRW<[N1Write_6c_2V1], (instregex "^(ADDV|[SU]ADDLV)v16i8v$")>;

// ASIMD max/min, reduce, 4H/4S
def : InstRW<[N1Write_3c_1V1], (instregex "^[SU](MAX|MIN)Vv4(i16|i32)v$")>;

// ASIMD max/min, reduce, 8B/8H
def : InstRW<[N1Write_5c_1V1_1V], (instregex "^[SU](MAX|MIN)Vv8(i8|i16)v$")>;

// ASIMD max/min, reduce, 16B
def : InstRW<[N1Write_6c_2V1], (instregex "[SU](MAX|MIN)Vv16i8v$")>;

// [a76_asimd_forms] docs/a76-sog.md C-17: the by-element forms ("..._indexed") of MUL,
// SQDMULH, SQRDMULH, MLA, MLS, SQRDMLAH, SQRDMLSH are the same mnemonics as the rows below
// (l.1215-1237); they used to fall to the default (2, V).  The scalar SQDMLAL/SQDMLSL/SQDMULL
// (i16/i32) likewise (l.1240-1247, [a76_asimd_scalar], docs/a76-sog.md D-5).
// ASIMD multiply, D-form
def : InstRW<[N1Write_4c_1V0], (instregex "^MUL(v[14]i16|v[12]i32)(_indexed)?$",
                                          "^SQ(R)?DMULH(v[14]i16|v[12]i32)(_indexed)?$")>;

// ASIMD multiply long
def : InstRW<[N1Write_4c_1V0], (instregex "^([SU]|SQD)MULLv", "^SQDMULLi(16|32)$")>;

// ASIMD multiply accumulate, D-form
def : InstRW<[N1Wr_VMA, N1Rd_VMA], (instregex "^ML[AS](v[14]i16|v[12]i32)(_indexed)?$")>;

// ASIMD multiply accumulate high, D-form
def : InstRW<[N1Write_4c_1V0], (instregex "^SQRDML[AS]H(v[14]i16|v[12]i32)(_indexed)?$")>;

// ASIMD multiply accumulate long
def : InstRW<[N1Wr_VMAL, N1Rd_VMAL], (instregex "^[SU]ML[AS]Lv")>;

// ASIMD multiply accumulate saturating long
def : InstRW<[N1Write_4c_1V0], (instregex "^SQDML[AS]Lv", "^SQDML[AS]Li(16|32)$")>;

// ASIMD multiply, Q-form
def : InstRW<[N1Write_5c_2V0], (instregex "^MUL(v8i16|v4i32)(_indexed)?$",
                                          "^SQ(R)?DMULH(v8i16|v4i32)(_indexed)?$")>;

// ASIMD multiply accumulate, Q-form
def : InstRW<[N1Wr_VMAQ, N1Rd_VMA], (instregex "^ML[AS](v8i16|v4i32)(_indexed)?$")>;

// ASIMD multiply accumulate high, Q-form
def : InstRW<[N1Write_5c_2V0], (instregex "^SQRDML[AS]H(v8i16|v4i32)(_indexed)?$")>;

// ASIMD multiply/multiply long (8x8) polynomial, D-form
def : InstRW<[N1Write_3c_1V0], (instrs PMULv8i8, PMULLv8i8)>;

// ASIMD multiply/multiply long (8x8) polynomial, Q-form
def : InstRW<[N1Write_4c_2V0], (instrs PMULv16i8, PMULLv16i8)>;

// ASIMD pairwise add and accumulate long
def : InstRW<[N1Wr_VPA, N1Rd_VPA], (instregex "^[SU]ADALPv")>;

// ASIMD shift accumulate
// [a76_asimd_scalar] the scalar forms (SSRA Dd, ... "d") as well.
def : InstRW<[N1Wr_VSA, N1Rd_VSA], (instregex "^[SU]R?SRAv", "^[SU]R?SRAd$")>;

// ASIMD shift by immed, basic
// ASIMD shift by immed and insert, basic
// ASIMD shift by register, basic
// [a76_asimd_scalar] the scalar forms SHL, SSHR, USHR, SLI, SRI (Dd, "d") as well.
def : InstRW<[N1Write_2c_1V1], (instregex "^SHLL?v", "^SHRNv", "^[SU]SHLLv",
                                          "^[SU]SHRv", "^S[LR]Iv", "^[SU]SHLv",
                                          "^(SHL|[SU]SHR|S[LR]I)d$")>;

// ASIMD shift by immed, complex
// ASIMD shift by register, complex
// [a76_asimd_forms] "^SQSHU?RNv" cannot match SQSHRUN (l.1258), which used to fall to the default
// (2, V); "^SQSHRUNv" added.  [a76_asimd_scalar] the scalar SRSHR/URSHR (d) and the scalar narrowing
// shifts (b, h, s) as well.
def : InstRW<[N1Write_4c_1V1],
             (instregex "^RSHRNv", "^SQRSHRU?Nv", "^(SQSHLU?|UQSHL)[bhsd]$",
                        "^(SQSHLU?|UQSHL)(v8i8|v16i8|v4i16|v8i16|v2i32|v4i32|v2i64)_shift$",
                        "^SQSHU?RNv", "^SQSHRUNv", "^[SU]RSHRv", "^UQR?SHRNv",
                        "^[SU]Q?RSHLv", "^[SU]QSHLv", "^[SU]RSHRd$",
                        "^(SQR?SHRU?N|UQR?SHRN)[bhs]$")>;


// ASIMD FP instructions
// -----------------------------------------------------------------------------

// ASIMD FP absolute value/difference
// ASIMD FP arith, normal
// ASIMD FP compare
// ASIMD FP max/min, normal
// ASIMD FP max/min, pairwise
// ASIMD FP negate
// Covered by "SchedAlias (WriteV[dq]...)" above

// ASIMD FP convert, long (F16 to F32)
def : InstRW<[N1Write_4c_2V0], (instregex "^FCVTL(v4|v8)i16$")>;

// ASIMD FP convert, long (F32 to F64)
def : InstRW<[N1Write_3c_1V0], (instregex "^FCVTL(v2|v4)i32$")>;

// ASIMD FP convert, narrow (F32 to F16)
def : InstRW<[N1Write_4c_2V0], (instregex "^FCVTN(v4|v8)i16$")>;

// ASIMD FP convert, narrow (F64 to F32)
def : InstRW<[N1Write_3c_1V0], (instregex "^FCVTN(v2|v4)i32$",
                                          "^FCVTXN(v2|v4)f32$")>;

// [a76_fp_cvt] docs/a76-sog.md C-14, C-18: the fixed-point vector forms (FCVTZS/FCVTZU/SCVTF/UCVTF
// Vd.T, Vn.T, #fbits, "..._shift") are the same mnemonics and arrangements as the rows below
// (l.1377-1402); they used to fall to the default (2, V).
// ASIMD FP convert, other, D-form F32 and Q-form F64
def : InstRW<[N1Write_3c_1V0], (instregex "^[FSU]CVT[AMNPZ][SU]v2f(32|64)$",
                                          "^[SU]CVTFv2f(32|64)$",
                                          "^(FCVTZ[SU]|[SU]CVTF)v2i(32|64)_shift$")>;

// ASIMD FP convert, other, D-form F16 and Q-form F32
def : InstRW<[N1Write_4c_2V0], (instregex "^[FSU]CVT[AMNPZ][SU]v4f(16|32)$",
                                          "^[SU]CVTFv4f(16|32)$",
                                          "^(FCVTZ[SU]|[SU]CVTF)v4i(16|32)_shift$")>;

// ASIMD FP convert, other, Q-form F16
def : InstRW<[N1Write_6c_4V0], (instregex "^[FSU]CVT[AMNPZ][SU]v8f16$",
                                          "^[SU]CVTFv8f16$",
                                          "^(FCVTZ[SU]|[SU]CVTF)v8i16_shift$")>;

// ASIMD FP divide, D-form, F16
// ASIMD FP square root, D-form, F16
def : InstRW<[N1WriteFDS_7c7], (instrs FDIVv4f16, FSQRTv4f16)>;

// ASIMD FP divide, D-form, F32
// ASIMD FP square root, D-form, F32
def : InstRW<[N1WriteFDS_7c4], (instrs FDIVv2f32, FSQRTv2f32)>;

// ASIMD FP divide, Q-form, F16
// ASIMD FP square root, Q-form, F16
def : InstRW<[N1WriteFDS_10c10], (instrs FDIVv8f16)>;
def : InstRW<[N1WriteFDS_11c11], (instrs FSQRTv8f16)>;

// ASIMD FP divide, Q-form, F32
// ASIMD FP square root, Q-form, F32
def : InstRW<[N1WriteFDS_7c7], (instrs FDIVv4f32, FSQRTv4f32)>;

// ASIMD FP divide, Q-form, F64
def : InstRW<[N1WriteFDS_7c7], (instrs FDIVv2f64)>;

// ASIMD FP square root, Q-form, F64
def : InstRW<[N1WriteFDS_7c7], (instrs FSQRTv2f64)>;

// ASIMD FP max/min, reduce, F32 and D-form F16
def : InstRW<[N1Write_5c_1V], (instregex "^F(MAX|MIN)(NM)?Vv4(i16|i32)v$")>;

// ASIMD FP max/min, reduce, Q-form F16
def : InstRW<[N1Write_8c_3V], (instregex "^F(MAX|MIN)(NM)?Vv8i16v$")>;

// ASIMD FP multiply
def : InstRW<[N1Wr_VFM], (instregex "^FMULX?v")>;

// ASIMD FP multiply accumulate
def : InstRW<[N1Wr_VFMA, N1Rd_VFMA], (instregex "^FML[AS]v")>;

// ASIMD FP multiply accumulate long
def : InstRW<[N1Wr_VFMAL, N1Rd_VFMAL], (instregex "^FML[AS]L2?v")>;

// ASIMD FP round, D-form F32 and Q-form F64
def : InstRW<[N1Write_3c_1V0], (instregex "^FRINT[AIMNPXZ]v2f(32|64)$")>;

// ASIMD FP round, D-form F16 and Q-form F32
def : InstRW<[N1Write_4c_2V0], (instregex "^FRINT[AIMNPXZ]v4f(16|32)$")>;

// ASIMD FP round, Q-form F16
def : InstRW<[N1Write_6c_4V0], (instregex "^FRINT[AIMNPXZ]v8f16$")>;


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
def : InstRW<[N1Write_3c_1M],
             (instregex "^DUP((v16|v8)i8|(v8|v4)i16|(v4|v2)i32|v2i64)gpr$")>;

// ASIMD extract narrow, saturating
def : InstRW<[N1Write_4c_1V1], (instregex "^[SU]QXTNv", "^SQXTUNv")>;

// ASIMD reciprocal and square root estimate, D-form F32 and F64
def : InstRW<[N1Write_3c_1V0], (instrs FRECPEv1i32, FRECPEv2f32, FRECPEv1i64,
                                       FRECPXv1i32, FRECPXv1i64,
                                       URECPEv2i32,
                                       FRSQRTEv1i32, FRSQRTEv2f32, FRSQRTEv1i64,
                                       URSQRTEv2i32)>;

// ASIMD reciprocal and square root estimate, D-form F16 and Q-form F32
def : InstRW<[N1Write_4c_2V0], (instrs FRECPEv1f16, FRECPEv4f16, FRECPEv4f32,
                                       FRECPXv1f16,
                                       URECPEv4i32,
                                       FRSQRTEv1f16, FRSQRTEv4f16, FRSQRTEv4f32,
                                       URSQRTEv4i32)>;

// ASIMD reciprocal and square root estimate, Q-form F16
def : InstRW<[N1Write_6c_4V0], (instrs FRECPEv8f16,
                                       FRSQRTEv8f16)>;

// ASIMD reciprocal step
def : InstRW<[N1Write_4c_1V], (instregex "^FRECPS(16|32|64)$", "^FRECPSv",
                                         "^FRSQRTS(16|32|64)$", "^FRSQRTSv")>;

// ASIMD table lookup, 3 table regs
// ASIMD table lookup extension, 2 table reg
def : InstRW<[N1Write_4c_4V], (instrs TBLv8i8Three, TBLv16i8Three,
                                      TBXv8i8Two, TBXv16i8Two)>;

// ASIMD table lookup, 4 table regs
def : InstRW<[N1Write_4c_3V], (instrs TBLv8i8Four, TBLv16i8Four)>;

// ASIMD table lookup extension, 3 table reg
def : InstRW<[N1Write_6c_3V], (instrs TBXv8i8Three, TBXv16i8Three)>;

// ASIMD table lookup extension, 4 table reg
def : InstRW<[N1Write_6c_5V], (instrs TBXv8i8Four, TBXv16i8Four)>;

// ASIMD transfer, element to gen reg
def : InstRW<[N1Write_2c_1V1], (instregex "^SMOVvi(((8|16)to(32|64))|32to64)$",
                                          "^UMOVvi(8|16|32|64)$")>;

// ASIMD transfer, gen reg to element
def : InstRW<[N1Write_5c_1M_1V], (instregex "^INSvi(8|16|32|64)gpr$")>;


// ASIMD load instructions
// -----------------------------------------------------------------------------
// [a76_ls_throughput] docs/a76-sog.md C-20, C-21: LD3 (multiple, one lane, all lanes, 1/2,
// l.1694-1727) and LD4 multiple (D 2/7, Q 1/5, l.1728-1733) use N1WriteLS (the L pipes held 2/T
// cycles); with one cycle per uop they reached 2/3 (LD3) and 1/2 (LD4 multiple).

// ASIMD load, 1 element, multiple, 1 reg
def : InstRW<[N1Write_5c_1L],
             (instregex "^LD1Onev(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, N1Write_5c_1L],
             (instregex "^LD1Onev(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 2 reg
def : InstRW<[N1Write_5c_2L],
             (instregex "^LD1Twov(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, N1Write_5c_2L],
             (instregex "^LD1Twov(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 3 reg
def : InstRW<[N1Write_6c_3L],
             (instregex "^LD1Threev(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, N1Write_6c_3L],
             (instregex "^LD1Threev(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 4 reg
def : InstRW<[N1Write_6c_4L],
             (instregex "^LD1Fourv(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, N1Write_6c_4L],
             (instregex "^LD1Fourv(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 1 element, one lane
// ASIMD load, 1 element, all lanes
def : InstRW<[N1Write_7c_1L_1V],
             (instregex "LD1(i|Rv)(8|16|32|64)$",
                        "LD1Rv(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, N1Write_7c_1L_1V],
             (instregex "LD1i(8|16|32|64)_POST$",
                        "LD1Rv(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 2 element, multiple
// ASIMD load, 2 element, one lane
// ASIMD load, 2 element, all lanes
def : InstRW<[N1Write_7c_2L_2V],
             (instregex "LD2Twov(8b|16b|4h|8h|2s|4s|2d)$",
                        "LD2i(8|16|32|64)$",
                        "LD2Rv(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, N1Write_7c_2L_2V],
             (instregex "LD2Twov(8b|16b|4h|8h|2s|4s|2d)_POST$",
                        "LD2i(8|16|32|64)_POST$",
                        "LD2Rv(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 3 element, multiple
def : InstRW<[N1WriteLS<8, 6, 4, 3>],
             (instregex "LD3Threev(8b|16b|4h|8h|2s|4s|2d)$")>;
def : InstRW<[WriteAdr, N1WriteLS<8, 6, 4, 3>],
             (instregex "LD3Threev(8b|16b|4h|8h|2s|4s|2d)_POST$")>;

// ASIMD load, 3 element, one lane
// ASIMD load, 3 element, all lanes
def : InstRW<[N1WriteLS<7, 5, 4, 3>],
             (instregex "LD3i(8|16|32|64)$",
                        "LD3Rv(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, N1WriteLS<7, 5, 4, 3>],
             (instregex "LD3i(8|16|32|64)_POST$",
                        "LD3Rv(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;

// ASIMD load, 4 element, multiple, D-form
def : InstRW<[N1WriteLS<8, 7, 7, 4>],
             (instregex "LD4Fourv(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, N1WriteLS<8, 7, 7, 4>],
             (instregex "LD4Fourv(8b|4h|2s)_POST$")>;

// ASIMD load, 4 element, multiple, Q-form
def : InstRW<[N1WriteLS<10, 8, 10, 4>],
             (instregex "LD4Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N1WriteLS<10, 8, 10, 4>],
             (instregex "LD4Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 4 element, one lane
// ASIMD load, 4 element, all lanes
def : InstRW<[N1Write_8c_4L_4V],
             (instregex "LD4i(8|16|32|64)$",
                        "LD4Rv(8b|16b|4h|8h|2s|4s|1d|2d)$")>;
def : InstRW<[WriteAdr, N1Write_8c_4L_4V],
             (instregex "LD4i(8|16|32|64)_POST$",
                        "LD4Rv(8b|16b|4h|8h|2s|4s|1d|2d)_POST$")>;


// ASIMD store instructions
// -----------------------------------------------------------------------------
// [a76_ls_throughput] docs/a76-sog.md C-20, C-21: the rows below with N1WriteLS reach the
// SOG throughput (l.1833-1892) by holding the L pipes 2/T cycles; with one cycle per uop they were
// up to twice as fast as the SOG (e.g. ST1 one register Q-form 2 per cycle, SOG 1).  The SOG's
// figures amount to one 128-bit store per cycle (sec. 4.4, l.2069-2070).  ST4 one lane keeps its
// write: its latency and throughput are partly blank in the SOG (l.1893-1898).
// [a76_from_a55_store_data] docs/a76-sog.md B-1: the data Vt reads ReadST, the base Rn
// ReadAdrBase; for the one-lane forms the lane index takes the read in between (ReadDefault).

// ASIMD store, 1 element, multiple, 1 reg, D-form
def : InstRW<[N1Write_2c_1L_1V, ReadST, ReadAdrBase],
             (instregex "ST1Onev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N1Write_2c_1L_1V, ReadST, ReadAdrBase],
             (instregex "ST1Onev(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 1 reg, Q-form
def : InstRW<[N1WriteLS<2, 2, 2, 1>, ReadST, ReadAdrBase],
             (instregex "ST1Onev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N1WriteLS<2, 2, 2, 1>, ReadST, ReadAdrBase],
             (instregex "ST1Onev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 2 reg, D-form
def : InstRW<[N1Write_2c_1L_2V, ReadST, ReadAdrBase],
             (instregex "ST1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N1Write_2c_1L_2V, ReadST, ReadAdrBase],
             (instregex "ST1Twov(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 2 reg, Q-form
def : InstRW<[N1WriteLS<3, 4, 4, 2>, ReadST, ReadAdrBase],
             (instregex "ST1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N1WriteLS<3, 4, 4, 2>, ReadST, ReadAdrBase],
             (instregex "ST1Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 3 reg, D-form
def : InstRW<[N1Write_3c_2L_3V, ReadST, ReadAdrBase],
             (instregex "ST1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N1Write_3c_2L_3V, ReadST, ReadAdrBase],
             (instregex "ST1Threev(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 3 reg, Q-form
def : InstRW<[N1WriteLS<4, 6, 6, 3>, ReadST, ReadAdrBase],
             (instregex "ST1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N1WriteLS<4, 6, 6, 3>, ReadST, ReadAdrBase],
             (instregex "ST1Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 4 reg, D-form
def : InstRW<[N1WriteLS<3, 4, 4, 2>, ReadST, ReadAdrBase],
             (instregex "ST1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N1WriteLS<3, 4, 4, 2>, ReadST, ReadAdrBase],
             (instregex "ST1Fourv(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 4 reg, Q-form
def : InstRW<[N1WriteLS<5, 8, 8, 4>, ReadST, ReadAdrBase],
             (instregex "ST1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N1WriteLS<5, 8, 8, 4>, ReadST, ReadAdrBase],
             (instregex "ST1Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, one lane
def : InstRW<[N1WriteLS<4, 2, 2, 1>, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST1i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, N1WriteLS<4, 2, 2, 1>, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST1i(8|16|32|64)_POST$")>;

// ASIMD store, 2 element, multiple, D-form, B/H/S
def : InstRW<[N1WriteLS<4, 2, 2, 1>, ReadST, ReadAdrBase],
             (instregex "ST2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, N1WriteLS<4, 2, 2, 1>, ReadST, ReadAdrBase],
             (instregex "ST2Twov(8b|4h|2s)_POST$")>;

// ASIMD store, 2 element, multiple, Q-form
def : InstRW<[N1WriteLS<5, 4, 4, 2>, ReadST, ReadAdrBase],
             (instregex "ST2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N1WriteLS<5, 4, 4, 2>, ReadST, ReadAdrBase],
             (instregex "ST2Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 2 element, one lane
def : InstRW<[N1WriteLS<4, 2, 2, 1>, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST2i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, N1WriteLS<4, 2, 2, 1>, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST2i(8|16|32|64)_POST$")>;

// ASIMD store, 3 element, multiple, D-form, B/H/S
def : InstRW<[N1WriteLS<5, 4, 4, 2>, ReadST, ReadAdrBase],
             (instregex "ST3Threev(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, N1WriteLS<5, 4, 4, 2>, ReadST, ReadAdrBase],
             (instregex "ST3Threev(8b|4h|2s)_POST$")>;

// ASIMD store, 3 element, multiple, Q-form
def : InstRW<[N1WriteLS<6, 6, 6, 3>, ReadST, ReadAdrBase],
             (instregex "ST3Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N1WriteLS<6, 6, 6, 3>, ReadST, ReadAdrBase],
             (instregex "ST3Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 3 element, one lane, B/H/S
def : InstRW<[N1WriteLS<4, 6, 4, 3>, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST3i(8|16|32)$")>;
def : InstRW<[WriteAdr, N1WriteLS<4, 6, 4, 3>, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST3i(8|16|32)_POST$")>;

// ASIMD store, 3 element, one lane, D
def : InstRW<[N1WriteLS<5, 6, 4, 3>, ReadST, ReadDefault, ReadAdrBase],
             (instrs ST3i64)>;
def : InstRW<[WriteAdr, N1WriteLS<5, 6, 4, 3>, ReadST, ReadDefault, ReadAdrBase],
             (instrs ST3i64_POST)>;

// ASIMD store, 4 element, multiple, D-form, B/H/S
def : InstRW<[N1WriteLS<7, 6, 6, 3>, ReadST, ReadAdrBase],
             (instregex "ST4Fourv(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, N1WriteLS<7, 6, 6, 3>, ReadST, ReadAdrBase],
             (instregex "ST4Fourv(8b|4h|2s)_POST$")>;

// ASIMD store, 4 element, multiple, Q-form, B/H/S
def : InstRW<[N1WriteLS<9, 12, 12, 6>, ReadST, ReadAdrBase],
             (instregex "ST4Fourv(16b|8h|4s)$")>;
def : InstRW<[WriteAdr, N1WriteLS<9, 12, 12, 6>, ReadST, ReadAdrBase],
             (instregex "ST4Fourv(16b|8h|4s)_POST$")>;

// ASIMD store, 4 element, multiple, Q-form, D
def : InstRW<[N1WriteLS<6, 8, 8, 4>, ReadST, ReadAdrBase],
             (instrs ST4Fourv2d)>;
def : InstRW<[WriteAdr, N1WriteLS<6, 8, 8, 4>, ReadST, ReadAdrBase],
             (instrs ST4Fourv2d_POST)>;

// ASIMD store, 4 element, one lane, B/H/S
def : InstRW<[N1Write_5c_3L_3V, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST4i(8|16|32)$")>;
def : InstRW<[WriteAdr, N1Write_5c_3L_3V, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST4i(8|16|32)_POST$")>;

// ASIMD store, 4 element, one lane, D
def : InstRW<[N1Write_4c_3L_3V, ReadST, ReadDefault, ReadAdrBase],
             (instrs ST4i64)>;
def : InstRW<[WriteAdr, N1Write_4c_3L_3V, ReadST, ReadDefault, ReadAdrBase],
             (instrs ST4i64_POST)>;


// Cryptography extensions
// -----------------------------------------------------------------------------

// Crypto AES ops
def N1WriteVC : WriteSequence<[N1Write_2c_1V0]>;
def N1ReadVC  : SchedReadAdvance<2, [N1WriteVC]>;
def           : InstRW<[N1WriteVC], (instrs AESDrr, AESErr)>;
def           : InstRW<[N1Write_2c_1V0, N1ReadVC], (instrs AESMCrr, AESIMCrr)>;

// Crypto polynomial (64x64) multiply long
// Crypto SHA1 hash acceleration op
// Crypto SHA1 schedule acceleration ops
// Crypto SHA256 schedule acceleration ops
def : InstRW<[N1Write_2c_1V0], (instregex "^PMULLv[12]i64$",
                                          "^SHA1(H|SU0|SU1)rr",
                                          "^SHA256SU[01]rr")>;

// Crypto SHA1 hash acceleration ops
// Crypto SHA256 hash acceleration ops
def : InstRW<[N1Write_4c_1V0], (instregex "^SHA1[CMP]rrr$",
                                          "^SHA256H2?rrr$")>;


// CRC
// -----------------------------------------------------------------------------

// CRC checksum ops
def : InstRW<[N1Wr_CRC, N1Rd_CRC], (instregex "^CRC32C?[BHWX]rr$")>;


}
