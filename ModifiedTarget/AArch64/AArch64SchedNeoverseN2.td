//=- AArch64SchedNeoverseN2.td - NeoverseN2 Scheduling Defs --*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the scheduling model for the Arm Neoverse N2 processors.
//
//===----------------------------------------------------------------------===//

def NeoverseN2Model : SchedMachineModel {
  let IssueWidth            =   5; // Micro-ops dispatched at a time.
  let MicroOpBufferSize     = 160; // Entries in micro-op re-order buffer.
  let LoadLatency           =   4; // Optimistic load latency.
  let MispredictPenalty     =  10; // Extra cycles for mispredicted branch.
  let LoopMicroOpBufferSize =  16; // NOTE: Copied from Cortex-A57.
  let CompleteModel         =   1;

  list<Predicate> UnsupportedFeatures = !listconcat(SMEUnsupported.F,
    [HasSVE2p1, HasSVEB16B16, HasPAuthLR, HasCPA, HasCSSC, HasMOPS, HasMOPS_GO]);
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available on Neoverse N2.
// Instructions are first fetched and then decoded into internal macro-ops
// (MOPs). From there, the MOPs proceed through register renaming and dispatch
// stages. A MOP can be split into two micro-ops further down the pipeline
// after the decode stage. Once dispatched, micro-ops wait for their operands
// and issue out-of-order to one of thirteen issue pipelines. Each issue
// pipeline can accept one micro-op per cycle.

let SchedModel = NeoverseN2Model in {

// Define the (13) issue ports.
def N2UnitB   : ProcResource<2>;  // Branch 0/1
def N2UnitS   : ProcResource<2>;  // Integer single Cycle 0/1
def N2UnitM0  : ProcResource<1>;  // Integer multicycle 0
def N2UnitM1  : ProcResource<1>;  // Integer multicycle 1
def N2UnitL01 : ProcResource<2>;  // Load/Store 0/1
def N2UnitL2  : ProcResource<1>;  // Load 2
def N2UnitD   : ProcResource<2>;  // Store data 0/1
def N2UnitV0  : ProcResource<1>;  // FP/ASIMD 0
def N2UnitV1  : ProcResource<1>;  // FP/ASIMD 1
def N2UnitFlg : ProcResource<3>;  // Flags

def N2UnitV : ProcResGroup<[N2UnitV0, N2UnitV1]>;  // FP/ASIMD 0/1
def N2UnitM : ProcResGroup<[N2UnitM0, N2UnitM1]>;  // Integer single/multicycle 0/1
def N2UnitL : ProcResGroup<[N2UnitL01, N2UnitL2]>; // Load/Store 0/1 and Load 2
def N2UnitI : ProcResGroup<[N2UnitS, N2UnitM0, N2UnitM1]>; // Integer single cycle 0/1 and single/multicycle 0/1

// This local copy of the Neoverse N2 model is installed only for cortex-a78 / -a78ae / -a78c
// (src/cpu/cpu_traits.cpp, CpuModel::A78); neoverse-n2, cortex-a710 and cortex-a715 keep libLLVM's
// stock model.  The edits below follow the Cortex-A78 Software Optimization Guide
// (PJDOC-466751330-9691, Issue 4.0); docs/a78-sog.md lists them item by item, and the tags
// [a78_...] name the items (docs/a78-sog.md section 2.4).
//
// [a78_dividers] docs/a78-sog.md D-3: the iterative dividers.  An integer divide "block[s] any
// subsequent divide operations until complete" (l.787, note 1 of sec. 3.6) and an FP divide /
// square root blocks "subsequent similar operations to the same pipeline until complete"
// (l.1286-1287, l.1919-1920); every issue pipeline "can accept one µOP per cycle" (l.389-390).  So a
// divide takes its pipeline (M0, V0) for one cycle and holds the divider, not the pipeline, for
// its occupancy (as for the Cortex-A76, docs/a76-sog.md D-3).
def A78UnitIDiv : ProcResource<1>;  // integer divider, in the M0 pipeline
def A78UnitFDiv : ProcResource<1>;  // FP divide / square root unit, in the V0 pipeline

// Define commonly used read types.

// No forwarding is provided for these types.
def : ReadAdvance<ReadI,       0>;
def : ReadAdvance<ReadISReg,   0>;
def : ReadAdvance<ReadIEReg,   0>;
def : ReadAdvance<ReadIM,      0>;
def : ReadAdvance<ReadIMA,     0>;
def : ReadAdvance<ReadID,      0>;
def : ReadAdvance<ReadExtrHi,  0>;
def : ReadAdvance<ReadAdrBase, 0>;
// [a78_from_a55_store_data] docs/a78-sog.md B-1: the data operand of a store is read one cycle
// late (advance 1 from every producer).  The A78 SOG says only that stores are split into address
// and data µOPs (l.1102-1103, l.1455, l.2245); it gives no operand timing.  This is the project's
// decision for the in-order cores (Cortex-A55, SOG-based there; Cortex-A520, borrowed) that was
// applied to the Cortex-A76 for consistency (docs/a76-sog.md B-1), and is applied here likewise:
// ReadST is attached to the data operands of every store form (STXR/STXP and the atomics
// excepted, as there).  NOT SOG-based.  It acts only on Facile's store -> load memory edges.
def : ReadAdvance<ReadST,      1>;
def : ReadAdvance<ReadVLD,     0>;

def : WriteRes<WriteAtomic,  []> { let Unsupported = 1; }
def : WriteRes<WriteBarrier, []> { let Latency = 1; }
def : WriteRes<WriteHint,    []> { let Latency = 1; }
// [a78_ldp_hi] docs/a78-sog.md D-10: the second destination of a load pair is loaded by the same
// access as the first one ("the maximum latency to load all the registers written by the
// instruction", l.957-958) and takes no µOP of its own (the row's pipelines are those of the
// first write).  WriteLDHi is the second write of the integer LDP W/X (latency 4, l.994-997).
// LDPSW and the FP pairs use A78Write_5c_0Z.
def : WriteRes<WriteLDHi,    []> { let Latency = 4; let NumMicroOps = 0; }

//===----------------------------------------------------------------------===//
// Define customized scheduler read/write types specific to the Neoverse N2.

//===----------------------------------------------------------------------===//

// Define generic 0 micro-op types
def N2Write_0c : SchedWriteRes<[]> { let Latency = 0; }

// Define generic 1 micro-op types

def N2Write_1c_1B   : SchedWriteRes<[N2UnitB]>   { let Latency = 1; }
def N2Write_1c_1I   : SchedWriteRes<[N2UnitI]>   { let Latency = 1; }
def N2Write_1c_1I_1Flg : SchedWriteRes<[N2UnitI, N2UnitFlg]>   { let Latency = 1; }
def N2Write_1c_1M   : SchedWriteRes<[N2UnitM]>   { let Latency = 1; }
def N2Write_1c_1M0  : SchedWriteRes<[N2UnitM0]>  { let Latency = 1; }
def N2Write_1c_1L01 : SchedWriteRes<[N2UnitL01]> { let Latency = 1; }
def N2Write_2c_1M   : SchedWriteRes<[N2UnitM]>   { let Latency = 2; }
def N2Write_2c_1M_1Flg : SchedWriteRes<[N2UnitM, N2UnitFlg]>   { let Latency = 2; }
def N2Write_3c_1M   : SchedWriteRes<[N2UnitM]>   { let Latency = 3; }
def N2Write_2c_1M0  : SchedWriteRes<[N2UnitM0]>  { let Latency = 2;
                                                   let ReleaseAtCycles = [2]; }
def N2Write_3c_1M0  : SchedWriteRes<[N2UnitM0]>  { let Latency = 3;
                                                   let ReleaseAtCycles = [3]; }
def N2Write_5c_1M0  : SchedWriteRes<[N2UnitM0]>  { let Latency = 5;
                                                   let ReleaseAtCycles = [5]; }
def N2Write_4c_1L   : SchedWriteRes<[N2UnitL]>   { let Latency = 4; }
def N2Write_6c_1L   : SchedWriteRes<[N2UnitL]>   { let Latency = 6; }
def N2Write_2c_1V   : SchedWriteRes<[N2UnitV]>   { let Latency = 2; }
def N2Write_3c_1V   : SchedWriteRes<[N2UnitV]>   { let Latency = 3; }
def N2Write_4c_1V   : SchedWriteRes<[N2UnitV]>   { let Latency = 4; }
def N2Write_2c_1V0  : SchedWriteRes<[N2UnitV0]>  { let Latency = 2; }
def N2Write_3c_1V0  : SchedWriteRes<[N2UnitV0]>  { let Latency = 3; }
def N2Write_4c_1V0  : SchedWriteRes<[N2UnitV0]>  { let Latency = 4; }
def N2Write_10c_1V0 : SchedWriteRes<[N2UnitV0]>  { let Latency = 10; }
def N2Write_12c_1V0 : SchedWriteRes<[N2UnitV0]>  { let Latency = 12; }
def N2Write_13c_1V0 : SchedWriteRes<[N2UnitV0]>  { let Latency = 13; }
def N2Write_15c_1V0 : SchedWriteRes<[N2UnitV0]>  { let Latency = 15; }
def N2Write_16c_1V0 : SchedWriteRes<[N2UnitV0]>  { let Latency = 16; }
def N2Write_20c_1V0 : SchedWriteRes<[N2UnitV0]>  { let Latency = 20; }
def N2Write_2c_1V1  : SchedWriteRes<[N2UnitV1]>  { let Latency = 2; }
def N2Write_3c_1V1  : SchedWriteRes<[N2UnitV1]>  { let Latency = 3; }
def N2Write_4c_1V1  : SchedWriteRes<[N2UnitV1]>  { let Latency = 4; }
def N2Write_6c_1V1  : SchedWriteRes<[N2UnitV1]>  { let Latency = 6; }
def N2Write_10c_1V1 : SchedWriteRes<[N2UnitV1]>  { let Latency = 10; }
def N2Write_6c_1L01 : SchedWriteRes<[N2UnitL01]> { let Latency = 6; }

//===----------------------------------------------------------------------===//
// Define generic 2 micro-op types

def N2Write_1c_1B_1S : SchedWriteRes<[N2UnitB, N2UnitS]> {
  let Latency     = 1;
  let NumMicroOps = 2;
}

def N2Write_6c_1M0_1B : SchedWriteRes<[N2UnitM0, N2UnitB]> {
  let Latency     = 6;
  let NumMicroOps = 2;
}

def N2Write_9c_1M0_1L : SchedWriteRes<[N2UnitM0, N2UnitL]> {
  let Latency     = 9;
  let NumMicroOps = 2;
}

def N2Write_3c_1I_1M : SchedWriteRes<[N2UnitI, N2UnitM]> {
  let Latency     = 3;
  let NumMicroOps = 2;
}





def N2Write_1c_1L01_1D : SchedWriteRes<[N2UnitL01, N2UnitD]> {
  let Latency     = 1;
  let NumMicroOps = 2;
}

def N2Write_5c_1M0_1V : SchedWriteRes<[N2UnitM0, N2UnitV]> {
  let Latency     = 5;
  let NumMicroOps = 2;
}

def N2Write_2c_1L01_1V : SchedWriteRes<[N2UnitL01, N2UnitV]> {
  let Latency     = 2;
  let NumMicroOps = 2;
}

def N2Write_4c_1V1_1V : SchedWriteRes<[N2UnitV1, N2UnitV]> {
  let Latency     = 4;
  let NumMicroOps = 2;
}

def N2Write_4c_2V0 : SchedWriteRes<[N2UnitV0, N2UnitV0]> {
  let Latency     = 4;
  let NumMicroOps = 2;
}

def N2Write_4c_2V : SchedWriteRes<[N2UnitV, N2UnitV]> {
  let Latency     = 4;
  let NumMicroOps = 2;
}

def N2Write_6c_2V : SchedWriteRes<[N2UnitV, N2UnitV]> {
  let Latency     = 6;
  let NumMicroOps = 2;
}

def N2Write_4c_2L : SchedWriteRes<[N2UnitL, N2UnitL]> {
  let Latency     = 4;
  let NumMicroOps = 2;
}


def N2Write_8c_1L_1V : SchedWriteRes<[N2UnitL, N2UnitV]> {
  let Latency     = 8;
  let NumMicroOps = 2;
}

def N2Write_4c_1L01_1V : SchedWriteRes<[N2UnitL01, N2UnitV]> {
  let Latency     = 4;
  let NumMicroOps = 2;
}

def N2Write_3c_1M0_1M  : SchedWriteRes<[N2UnitM0, N2UnitM]> {
  let Latency     = 3;
  let NumMicroOps = 2;
}

def N2Write_2c_1M0_1M  : SchedWriteRes<[N2UnitM0, N2UnitM]> {
  let Latency     = 2;
  let NumMicroOps = 2;
}

def N2Write_6c_2V1 : SchedWriteRes<[N2UnitV1, N2UnitV1]> {
  let Latency     = 6;
  let NumMicroOps = 2;
}

def N2Write_4c_1V0_1M : SchedWriteRes<[N2UnitV0, N2UnitM]> {
  let Latency     = 4;
  let NumMicroOps = 2;
}

def N2Write_5c_2V0 : SchedWriteRes<[N2UnitV0, N2UnitV0]> {
  let Latency     = 5;
  let NumMicroOps = 2;
}

def N2Write_5c_1V1_1M0 : SchedWriteRes<[N2UnitV1, N2UnitM0]> {
  let Latency     = 5;
  let NumMicroOps = 2;
}

def N2Write_7c_1M0_1V0 : SchedWriteRes<[N2UnitM0, N2UnitV0]> {
  let Latency     = 7;
  let NumMicroOps = 2;
}

def N2Write_2c_1V0_1M : SchedWriteRes<[N2UnitV0, N2UnitM]> {
  let Latency     = 2;
  let NumMicroOps = 2;
}

def N2Write_6c_1V_1V1 : SchedWriteRes<[N2UnitV, N2UnitV1]> {
  let Latency     = 6;
  let NumMicroOps = 2;
}

def N2Write_6c_1L_1M : SchedWriteRes<[N2UnitL, N2UnitM]> {
  let Latency     = 6;
  let NumMicroOps = 2;
}

def N2Write_6c_1L_1S : SchedWriteRes<[N2UnitL, N2UnitS]> {
  let Latency     = 6;
  let NumMicroOps = 2;
}

def N2Write_9c_1L_1V : SchedWriteRes<[N2UnitL, N2UnitV]> {
  let Latency     = 9;
  let NumMicroOps = 2;
}

def N2Write_4c_2V1 : SchedWriteRes<[N2UnitV1, N2UnitV1]> {
  let Latency     = 4;
  let NumMicroOps = 2;
}

//===----------------------------------------------------------------------===//
// Define generic 3 micro-op types

def N2Write_1c_1L01_1D_1I : SchedWriteRes<[N2UnitL01, N2UnitD, N2UnitI]> {
  let Latency     = 1;
  let NumMicroOps = 3;
}



def N2Write_7c_1M_1M0_1V : SchedWriteRes<[N2UnitM, N2UnitM0, N2UnitV]> {
  let Latency     = 7;
  let NumMicroOps = 3;
}

def N2Write_8c_1M0_1V1_1V : SchedWriteRes<[N2UnitM0, N2UnitV1, N2UnitV]> {
  let Latency     = 8;
  let NumMicroOps = 3;
}

def N2Write_10c_1V_1L_1S : SchedWriteRes<[N2UnitV, N2UnitL, N2UnitL]> {
  let Latency     = 10;
  let NumMicroOps = 3;
}

def N2Write_2c_1L01_1S_1V : SchedWriteRes<[N2UnitL01, N2UnitS, N2UnitV]> {
  let Latency     = 2;
  let NumMicroOps = 3;
}

def N2Write_4c_1L01_1S_1V : SchedWriteRes<[N2UnitL01, N2UnitS, N2UnitV]> {
  let Latency     = 4;
  let NumMicroOps = 3;
}



//===----------------------------------------------------------------------===//
// Define generic 4 micro-op types


def N2Write_6c_4V0 : SchedWriteRes<[N2UnitV0, N2UnitV0, N2UnitV0, N2UnitV0]> {
  let Latency     = 6;
  let NumMicroOps = 4;
}




def N2Write_9c_2L_2V : SchedWriteRes<[N2UnitL, N2UnitL, N2UnitV, N2UnitV]> {
  let Latency     = 9;
  let NumMicroOps = 4;
}

def N2Write_2c_2L01_2V : SchedWriteRes<[N2UnitL01, N2UnitL01, N2UnitV,
                                        N2UnitV]> {
  let Latency     = 2;
  let NumMicroOps = 4;
}

def N2Write_4c_2L01_2V : SchedWriteRes<[N2UnitL01, N2UnitL01, N2UnitV,
                                        N2UnitV]> {
  let Latency     = 4;
  let NumMicroOps = 4;
}


def N2Write_8c_2M0_2V0 : SchedWriteRes<[N2UnitM0, N2UnitM0, N2UnitV0,
                                        N2UnitV0]> {
  let Latency     = 8;
  let NumMicroOps = 4;
}

def N2Write_11c_2V_2V1 : SchedWriteRes<[N2UnitV, N2UnitV, N2UnitV1,
                                        N2UnitV1]> {
  let Latency     = 11;
  let NumMicroOps = 4;
}

def N2Write_9c_2V_2V1 : SchedWriteRes<[N2UnitV, N2UnitV, N2UnitV1,
                                       N2UnitV1]> {
  let Latency     = 9;
  let NumMicroOps = 4;
}

def N2Write_8c_2V_2V1 : SchedWriteRes<[N2UnitV, N2UnitV, N2UnitV1,
                                       N2UnitV1]> {
  let Latency     = 8;
  let NumMicroOps = 4;
}

def N2Write_10c_2L_2V1 : SchedWriteRes<[N2UnitV, N2UnitV, N2UnitV1,
                                        N2UnitV1]> {
  let Latency     = 10;
  let NumMicroOps = 4;
}

def N2Write_10c_2L_2V : SchedWriteRes<[N2UnitL, N2UnitL, N2UnitV, N2UnitV]> {
  let Latency     = 10;
  let NumMicroOps = 4;
}

def N2Write_4c_2M0_2M : SchedWriteRes<[N2UnitM0, N2UnitM0, N2UnitM,
                                       N2UnitM]> {
  let Latency     = 4;
  let NumMicroOps = 4;
}



//===----------------------------------------------------------------------===//
// Define generic 5 micro-op types


def N2Write_8c_2L_3V : SchedWriteRes<[N2UnitL, N2UnitL, N2UnitV, N2UnitV,
                                      N2UnitV]> {
  let Latency     = 8;
  let NumMicroOps = 5;
}

//===----------------------------------------------------------------------===//
// Define generic 6 micro-op types

def N2Write_8c_3L_3V : SchedWriteRes<[N2UnitL, N2UnitL, N2UnitL,
                                      N2UnitV, N2UnitV, N2UnitV]> {
  let Latency     = 8;
  let NumMicroOps = 6;
}

def N2Write_2c_3L01_3V : SchedWriteRes<[N2UnitL01, N2UnitL01, N2UnitL01,
                                        N2UnitV, N2UnitV, N2UnitV]> {
  let Latency     = 2;
  let NumMicroOps = 6;
}

def N2Write_6c_3L01_3V : SchedWriteRes<[N2UnitL01, N2UnitL01, N2UnitL01,
                                        N2UnitV, N2UnitV, N2UnitV]> {
  let Latency     = 6;
  let NumMicroOps = 6;
}


def N2Write_10c_2L_2V_2S : SchedWriteRes<[N2UnitL, N2UnitL, N2UnitV, N2UnitV,
                                          N2UnitS, N2UnitS]> {
  let Latency     = 10;
  let NumMicroOps = 6;
}

//===----------------------------------------------------------------------===//
// Define generic 7 micro-op types

def N2Write_8c_3L_4V : SchedWriteRes<[N2UnitL, N2UnitL, N2UnitL,
                                      N2UnitV, N2UnitV, N2UnitV, N2UnitV]> {
  let Latency     = 8;
  let NumMicroOps = 7;
}

def N2Write_7c_7V0 : SchedWriteRes<[N2UnitV0]> {
  let Latency     = 7;
  let NumMicroOps = 7;
  let ReleaseAtCycles = [7];
}

//===----------------------------------------------------------------------===//
// Define generic 8 micro-op types


def N2Write_2c_4L01_4V : SchedWriteRes<[N2UnitL01, N2UnitL01, N2UnitL01,
                                        N2UnitL01, N2UnitV, N2UnitV, N2UnitV,
                                        N2UnitV]> {
  let Latency     = 2;
  let NumMicroOps = 8;
}


def N2Write_8c_4L_4V : SchedWriteRes<[N2UnitL, N2UnitL, N2UnitL, N2UnitL,
                                      N2UnitV, N2UnitV, N2UnitV, N2UnitV]> {
  let Latency     = 8;
  let NumMicroOps = 8;
}


//===----------------------------------------------------------------------===//
// Define generic 10 micro-op types

def N2Write_7c_5L01_5V : SchedWriteRes<[N2UnitL01, N2UnitL01, N2UnitL01,
                                        N2UnitL01, N2UnitL01, N2UnitV,
                                        N2UnitV, N2UnitV, N2UnitV, N2UnitV]> {
  let Latency     = 7;
  let NumMicroOps = 10;
}

//===----------------------------------------------------------------------===//
// Define generic 15 micro-op types

def N2Write_7c_5L01_5S_5V : SchedWriteRes<[N2UnitL01, N2UnitL01, N2UnitL01,
                                           N2UnitL01, N2UnitL01, N2UnitS,
                                           N2UnitS, N2UnitS, N2UnitS,
                                           N2UnitS, N2UnitV, N2UnitV,
                                           N2UnitV, N2UnitV, N2UnitV]> {
  let Latency     = 7;
  let NumMicroOps = 15;
}

//===----------------------------------------------------------------------===//
// Define generic 18 micro-op types

def N2Write_11c_9L01_9V : SchedWriteRes<[N2UnitL01, N2UnitL01, N2UnitL01,
                                         N2UnitL01, N2UnitL01, N2UnitL01,
                                         N2UnitL01, N2UnitL01, N2UnitL01,
                                         N2UnitV, N2UnitV, N2UnitV,
                                         N2UnitV, N2UnitV, N2UnitV,
                                         N2UnitV, N2UnitV, N2UnitV]> {
  let Latency     = 11;
  let NumMicroOps = 18;
}

//===----------------------------------------------------------------------===//
// Define generic 27 micro-op types

def N2Write_11c_9L01_9S_9V : SchedWriteRes<[N2UnitL01, N2UnitL01, N2UnitL01,
                                            N2UnitL01, N2UnitL01, N2UnitL01,
                                            N2UnitL01, N2UnitL01, N2UnitL01,
                                            N2UnitS, N2UnitS, N2UnitS,
                                            N2UnitS, N2UnitS, N2UnitS,
                                            N2UnitS, N2UnitS, N2UnitS,
                                            N2UnitV, N2UnitV, N2UnitV,
                                            N2UnitV, N2UnitV, N2UnitV,
                                            N2UnitV, N2UnitV, N2UnitV]> {
  let Latency     = 11;
  let NumMicroOps = 27;
}

//===----------------------------------------------------------------------===//
// Cortex-A78 writes (docs/a78-sog.md).  Each is used by the rows below that name its tag.

// [a78_div_best] C-10, D-1, D-3: integer divide, the best value of "5 to 12, 1/12 to 1/5" (W) and
// "5 to 20, 1/20 to 1/5" (X) (l.699-700): latency 5, the divider held 5 cycles, M0 one cycle.
def A78Write_5c_1M0_IDiv : SchedWriteRes<[N2UnitM0, A78UnitIDiv]> { let Latency = 5;
                                                                   let ReleaseAtCycles = [1, 5]; }
// [a78_m0_pipelined] C-15, C-20: SCVTF/UCVTF from a GPR, FMOV from a GPR to the low half and DUP
// from a GPR are "3 1 M0" (l.1302, l.1310, l.1937): one cycle of M0, not three.
def A78Write_3c_1M0 : SchedWriteRes<[N2UnitM0]> { let Latency = 3; }
// [a78_ldp_hi] D-10: second destination of LDPSW and of the FP load pairs (latency 5).
def A78Write_5c_0Z : SchedWriteRes<[]> { let Latency = 5; let NumMicroOps = 0; }
// [a78_ldpsw] C-12: "Load pair, signed immed offset, signed words LDPSW 5 3/2 I, L" (l.998-999,
// l.1019-1020): one I µOP and the load pipes held two cycles (3/2 per cycle on three L pipes).
def A78Write_5c_1I_2L : SchedWriteRes<[N2UnitI, N2UnitL]> { let Latency = 5; let NumMicroOps = 2;
                                                          let ReleaseAtCycles = [1, 2]; }
// [a78_reg_offset_h] C-12, C-16: register-offset loads with an I µOP.
def A78Write_5c_1I_1L : SchedWriteRes<[N2UnitI, N2UnitL]> { let Latency = 5; let NumMicroOps = 2; }
def A78Write_6c_1I_1L : SchedWriteRes<[N2UnitI, N2UnitL]> { let Latency = 6; let NumMicroOps = 2; }
// [a78_fp_ld] C-16: FP loads, latency 5 (l.1366-1419).
def A78Write_5c_1L    : SchedWriteRes<[N2UnitL]> { let Latency = 5; }
def A78Write_5c_2L    : SchedWriteRes<[N2UnitL, N2UnitL]> { let Latency = 5; let NumMicroOps = 2; }
// [a78_store_rows] C-13, C-17: register-offset stores with an I µOP.
def A78Write_2c_1I_1L01_1D : SchedWriteRes<[N2UnitI, N2UnitL01, N2UnitD]> { let Latency = 2;
                                                                         let NumMicroOps = 3; }
def A78Write_2c_1I_1L01_1V : SchedWriteRes<[N2UnitI, N2UnitL01, N2UnitV]> { let Latency = 2;
                                                                         let NumMicroOps = 3; }
// [a78_stpq_wb] C-17, D-6: STP Q pre/post-index, "2 1 I, L01, V" (l.1516-1517, l.1522-1523): one per cycle
// is expressed on the two L01 pipelines (two L01 slots), as the A76 and the X1 do and as D-6 says, and not
// with a second V µOP; WriteAdr + 3 µOPs.
def A78Write_2c_2L01_1V : SchedWriteRes<[N2UnitL01, N2UnitL01, N2UnitV]> { let Latency = 2;
                                                                         let NumMicroOps = 3; }
// [a78_asimd_ld] C-21: ASIMD loads (l.2060-2172).
def A78Write_5c_3L      : SchedWriteRes<[N2UnitL, N2UnitL, N2UnitL]> { let Latency = 5;
                                                                       let NumMicroOps = 3; }
def A78Write_6c_4L      : SchedWriteRes<[N2UnitL, N2UnitL, N2UnitL, N2UnitL]> { let Latency = 6;
                                                                                let NumMicroOps = 4; }
def A78Write_7c_1L_1V   : SchedWriteRes<[N2UnitL, N2UnitV]> { let Latency = 7; let NumMicroOps = 2; }
def A78Write_7c_1L_2V   : SchedWriteRes<[N2UnitL, N2UnitV, N2UnitV]> { let Latency = 7;
                                                                       let NumMicroOps = 3; }
def A78Write_7c_2L_2V   : SchedWriteRes<[N2UnitL, N2UnitL, N2UnitV, N2UnitV]> { let Latency = 7;
                                                                                let NumMicroOps = 4; }
// [a78_ls_throughput] C-21, C-22, D-6: a load or store whose throughput T is below (or, for
// LD1 four registers D-form, above) what its µOPs give on the three load pipes L (two store
// pipes L01) and the two V pipes.  The µOP count and the V cycles are those of the write it
// replaces; the L (L01) pipes are held 3/T (2/T) cycles in total, so that the row's throughput is
// reached.  Which pipeline limits is not stated; L is taken, as for the Cortex-A76.
class A78WriteLDV<int lat, int uops, int lcyc, int vcyc> : SchedWriteRes<[N2UnitL, N2UnitV]> {
  let Latency = lat;
  let NumMicroOps = uops;
  let ReleaseAtCycles = [lcyc, vcyc];
}
class A78WriteSTV<int lat, int uops, int lcyc, int vcyc> : SchedWriteRes<[N2UnitL01, N2UnitV]> {
  let Latency = lat;
  let NumMicroOps = uops;
  let ReleaseAtCycles = [lcyc, vcyc];
}
def A78Write_5c_4L_2cyc : SchedWriteRes<[N2UnitL]> { let Latency = 5; let NumMicroOps = 4;
                                                     let ReleaseAtCycles = [2]; }
// [a78_v_throughput] C-18, C-19, C-20: a V-pipe row whose throughput is below what its µOPs give:
// the pipe(s) held 1/T (V1, V0) or 2/T (V) cycles in total; the µOP count is the replaced write's.
class A78WriteV<int lat, int uops, int vcyc> : SchedWriteRes<[N2UnitV]> {
  let Latency = lat;
  let NumMicroOps = uops;
  let ReleaseAtCycles = [vcyc];
}
def A78Write_4c_1V1_2cyc : SchedWriteRes<[N2UnitV1]> { let Latency = 4; let ReleaseAtCycles = [2]; }

// [a78_extr] C-11: "Bitfield extract, one reg EXTR 1 4 I", note 1: "One reg form is when Rn==Rm
// or imm==0, all other forms are considered two regs" (l.886-887, l.899).  The stock
// IsRORImmIdiomPred checks only Rn == Rm.
def A78IsExtrOneRegPred : MCSchedPredicate<
                            CheckAll<[CheckOpcode<[EXTRWrri, EXTRXrri]>,
                                      CheckAny<[CheckSameRegOperand<1, 2>,
                                                CheckImmOperand<3, 0>]>]>>;
// [a78_reg_offset_h] "Scaled" = the S bit of the register-offset addressing mode
// (AArch64_AM::getMemDoShift), as for the Cortex-A76.
def A78IsMemScaled : MCSchedPredicate<CheckMemScaled>;

//===----------------------------------------------------------------------===//
// Define predicate-controlled types

def N2Write_0or1c_1I : SchedWriteVariant<[
                      SchedVar<NeoverseZeroMove, [N2Write_0c]>,
                      SchedVar<NoSchedPred,      [N2Write_1c_1I]>]>;



def N2Write_Extr : SchedWriteVariant<[
                      SchedVar<A78IsExtrOneRegPred, [N2Write_1c_1I]>,  // [a78_extr]
                      SchedVar<NoSchedPred,       [N2Write_3c_1I_1M]>]>;

//===----------------------------------------------------------------------===//
// Define types for arithmetic and logical ops with short shifts
def N2Write_Arith : SchedWriteVariant<[
                      SchedVar<IsCheapLSL,  [N2Write_1c_1I]>,
                      SchedVar<NoSchedPred, [N2Write_2c_1M]>]>;

def N2Write_ArithI : SchedWriteVariant<[
                      SchedVar<IsCheapLSL,  [N2Write_1c_1I_1Flg]>,
                      SchedVar<NoSchedPred, [N2Write_2c_1M_1Flg]>]>;

def N2Write_Logical: SchedWriteVariant<[
                      SchedVar<NeoverseNoLSL, [N2Write_1c_1I_1Flg]>,
                      SchedVar<NoSchedPred,   [N2Write_2c_1M_1Flg]>]>;

//===----------------------------------------------------------------------===//
// Define forwarded types

def N2Wr_IM   : SchedWriteRes<[N2UnitM]>  { let Latency = 2; }
def N2Wr_IMA  : SchedWriteRes<[N2UnitM0]> { let Latency = 2; }
def N2Wr_IMUL : SchedWriteVariant<[
                  SchedVar<IsReg3ZeroPred, [N2Wr_IM]>,
                  SchedVar<NoSchedPred,    [N2Wr_IMA]>]>;
// [a78_maddl] docs/a78-sog.md C-10, U-5: "Multiply accumulate long SMADDL, SMSUBL, UMADDL, UMSUBL
// 2(1) 2 M" (l.704-705): on M (both pipes), not M0 like MADD/MSUB (l.702-703).  The accumulator
// forwarding (note 2, l.789-791) is taken to work from either kind to either kind.
def A78Wr_IMAL  : SchedWriteRes<[N2UnitM]> { let Latency = 2; }
def A78Wr_IMULL : SchedWriteVariant<[
                    SchedVar<IsReg3ZeroPred, [N2Wr_IM]>,
                    SchedVar<NoSchedPred,    [A78Wr_IMAL]>]>;
// [a78_mul_acc_fwd] docs/a78-sog.md U-5: "Multiply-accumulate pipelines support late-forwarding of
// accumulate operands from similar µOPs" (note 2, l.789-791), the wording under which the
// Cortex-A55, A520, A76 and A720 models forward the result of a zero-accumulator multiply (MUL,
// MNEG, SMULL, UMULL, ... = N2Wr_IM) to the accumulator of a following multiply-accumulate.  N2Wr_IM
// is added; before, a MUL result reached the accumulator after its full latency.
def N2Rd_IMA  : SchedReadAdvance<1, [N2Wr_IMA, A78Wr_IMAL, N2Wr_IM]>;

def N2Wr_FMA : SchedWriteRes<[N2UnitV]> { let Latency = 4; }
def N2Rd_FMA : SchedReadAdvance<2, [WriteFMul, N2Wr_FMA]>;

// [a78_asimd_acc] docs/a78-sog.md C-18: "ASIMD absolute diff accum SABA, UABA" and "... long
// SABAL(2), UABAL(2)" are 3(1) 1 V1 (l.1559-1561).
def N2Wr_VA : SchedWriteRes<[N2UnitV1]> { let Latency = 3; }
def N2Rd_VA : SchedReadAdvance<2, [N2Wr_VA]>;

// [a78_asimd_acc] "ASIMD dot product SDOT, UDOT 2(1) 2 V" (l.1597).
def N2Wr_VDOT : SchedWriteRes<[N2UnitV]> { let Latency = 2; }
def N2Rd_VDOT : SchedReadAdvance<1, [N2Wr_VDOT]>;

def N2Wr_VMMA : SchedWriteRes<[N2UnitV]> { let Latency = 3; }
def N2Rd_VMMA : SchedReadAdvance<2, [N2Wr_VMMA]>;

def N2Wr_VMA : SchedWriteRes<[N2UnitV0]> { let Latency = 4; }
// [a78_mul_acc_fwd] docs/a78-sog.md U-5: ASIMD "Multiply-accumulate pipelines support late-forwarding
// of accumulate operands from similar µOPs" (sec. 3.15 note 1, l.1752-1754).  The ASIMD MUL is taken
// as similar to MLA/MLS and SMULL/UMULL(2) to SMLAL/SMLSL/UMLAL/UMLSL(2), as on the Cortex-A55,
// A520, A76, A720 and X1, and MLA and MLAL forward to each other, as the integer MADD and SMADDL
// forms do (U-5, the lower estimate): every one of them reaches the accumulator of MLA/MLS and of the
// SMLAL family with the advance 3 of the rows' "4(1)".  SQDMULH, SQRDMULH and SQDMULL keep their
// writes: their accumulating rows (SQRDMLAH/SQRDMLSH, SQDMLAL/SQDMLSL "4 1 V0", l.1626-1633) print no
// accumulate latency.
def A78Wr_VMUL  : SchedWriteRes<[N2UnitV0]> { let Latency = 4; }
def A78Wr_VMULL : SchedWriteRes<[N2UnitV0]> { let Latency = 3; }
def N2Wr_VMAL : SchedWriteRes<[N2UnitV0]> { let Latency = 4; }
def N2Rd_VMA : SchedReadAdvance<3, [N2Wr_VMA, A78Wr_VMUL, N2Wr_VMAL, A78Wr_VMULL]>;

// SQRDMLAH/SQRDMLSH have no accumulator forwarding: [a78_asimd_acc] docs/a78-sog.md C-18.

def N2Rd_VMAL : SchedReadAdvance<3, [N2Wr_VMA, A78Wr_VMUL, N2Wr_VMAL, A78Wr_VMULL]>;   // [a78_mul_acc_fwd]

def N2Wr_VPA : SchedWriteRes<[N2UnitV1]> { let Latency = 4; }
def N2Rd_VPA : SchedReadAdvance<3, [N2Wr_VPA]>;

def N2Wr_VSA : SchedWriteRes<[N2UnitV1]> { let Latency = 4; }
def N2Rd_VSA : SchedReadAdvance<3, [N2Wr_VSA]>;

def N2Wr_VFCMA : SchedWriteRes<[N2UnitV]> { let Latency = 4; }
def N2Rd_VFCMA : SchedReadAdvance<2, [N2Wr_VFCMA]>;

def N2Wr_VFM  : SchedWriteRes<[N2UnitV]> { let Latency = 3; }
def N2Wr_VFMA : SchedWriteRes<[N2UnitV]> { let Latency = 4; }
def N2Rd_VFMA : SchedReadAdvance<2, [N2Wr_VFM, N2Wr_VFMA]>;

def N2Wr_VFMAL : SchedWriteRes<[N2UnitV]> { let Latency = 5; }
def N2Rd_VFMAL : SchedReadAdvance<2, [N2Wr_VFMAL]>;

def N2Wr_VBFDOT : SchedWriteRes<[N2UnitV]> { let Latency = 4; }
def N2Rd_VBFDOT : SchedReadAdvance<2, [N2Wr_VBFDOT]>;
def N2Wr_VBFMMA : SchedWriteRes<[N2UnitV]> { let Latency = 5; }
def N2Rd_VBFMMA : SchedReadAdvance<2, [N2Wr_VBFMMA]>;
def N2Wr_VBFMAL : SchedWriteRes<[N2UnitV]> { let Latency = 4; }
def N2Rd_VBFMAL : SchedReadAdvance<3, [N2Wr_VBFMAL]>;

def N2Wr_CRC : SchedWriteRes<[N2UnitM0]> { let Latency = 2; }
def N2Rd_CRC : SchedReadAdvance<1, [N2Wr_CRC]>;

def N2Wr_ZA  : SchedWriteRes<[N2UnitV1]> { let Latency = 4; }
def N2Rd_ZA  : SchedReadAdvance<3, [N2Wr_ZA]>;
def N2Wr_ZPA : SchedWriteRes<[N2UnitV1]> { let Latency = 4; }
def N2Rd_ZPA : SchedReadAdvance<3, [N2Wr_ZPA]>;
def N2Wr_ZSA : SchedWriteRes<[N2UnitV1]> { let Latency = 4; }
def N2Rd_ZSA : SchedReadAdvance<3, [N2Wr_ZSA]>;

def N2Wr_ZDOTB : SchedWriteRes<[N2UnitV]>   { let Latency = 3; }
def N2Rd_ZDOTB : SchedReadAdvance<2, [N2Wr_ZDOTB]>;
def N2Wr_ZDOTH : SchedWriteRes<[N2UnitV0]> { let Latency = 4; }
def N2Rd_ZDOTH : SchedReadAdvance<3, [N2Wr_ZDOTH]>;

def N2Wr_ZCMABHS : SchedWriteRes<[N2UnitV0]> { let Latency = 4; }
def N2Rd_ZCMABHS : SchedReadAdvance<3, [N2Wr_ZCMABHS]>;
def N2Wr_ZCMAD   : SchedWriteRes<[N2UnitV0, N2UnitV0]> { let Latency = 5; }
def N2Rd_ZCMAD   : SchedReadAdvance<2, [N2Wr_ZCMAD]>;

def N2Wr_ZMMA : SchedWriteRes<[N2UnitV]> { let Latency = 3; }
def N2Rd_ZMMA : SchedReadAdvance<2, [N2Wr_ZMMA]>;

def N2Wr_ZMABHS : SchedWriteRes<[N2UnitV0]> { let Latency = 4; }
def N2Rd_ZMABHS : SchedReadAdvance<3, [N2Wr_ZMABHS]>;
def N2Wr_ZMAD  : SchedWriteRes<[N2UnitV0, N2UnitV0]> { let Latency = 5; }
def N2Rd_ZMAD  : SchedReadAdvance<2, [N2Wr_ZMAD]>;

def N2Wr_ZMAL : SchedWriteRes<[N2UnitV0]> { let Latency = 4; }
def N2Rd_ZMAL : SchedReadAdvance<3, [N2Wr_ZMAL]>;

def N2Wr_ZMASQL   : SchedWriteRes<[N2UnitV0]>           { let Latency = 4; }
def N2Wr_ZMASQBHS : SchedWriteRes<[N2UnitV0]>           { let Latency = 4; }
def N2Wr_ZMASQD   : SchedWriteRes<[N2UnitV0, N2UnitV0]> { let Latency = 5; }
def N2Rd_ZMASQ    : SchedReadAdvance<2, [N2Wr_ZMASQL, N2Wr_ZMASQBHS,
                                         N2Wr_ZMASQD]>;

def N2Wr_ZFCMA : SchedWriteRes<[N2UnitV]> { let Latency = 5; }
def N2Rd_ZFCMA : SchedReadAdvance<3, [N2Wr_ZFCMA]>;

def N2Wr_ZFMA : SchedWriteRes<[N2UnitV]> { let Latency = 4; }
def N2Rd_ZFMA : SchedReadAdvance<2, [N2Wr_ZFMA]>;

def N2Wr_ZFMAL : SchedWriteRes<[N2UnitV]> { let Latency = 4; }
def N2Rd_ZFMAL : SchedReadAdvance<2, [N2Wr_ZFMAL]>;

def N2Wr_ZBFDOT : SchedWriteRes<[N2UnitV]> { let Latency = 4; }
def N2Rd_ZBFDOT : SchedReadAdvance<2, [N2Wr_ZBFDOT]>;
def N2Wr_ZBFMMA : SchedWriteRes<[N2UnitV]> { let Latency = 5; }
def N2Rd_ZBFMMA : SchedReadAdvance<2, [N2Wr_ZBFMMA]>;
def N2Wr_ZBFMAL : SchedWriteRes<[N2UnitV]> { let Latency = 4; }
def N2Rd_ZBFMAL : SchedReadAdvance<3, [N2Wr_ZBFMAL]>;

// Miscellaneous
// -----------------------------------------------------------------------------

def : InstRW<[WriteI], (instrs COPY)>;

// Branch Instructions
// -----------------------------------------------------------------------------

// Branch, immed
// Compare and branch
def : SchedAlias<WriteBr,    N2Write_1c_1B>;

// Branch, register
def : SchedAlias<WriteBrReg, N2Write_1c_1B>;

// Branch and link, immed
// Branch and link, register
def : InstRW<[N2Write_1c_1B_1S], (instrs BL, BLR)>;

// Arithmetic and Logical Instructions
// -----------------------------------------------------------------------------

// ALU, basic
def : SchedAlias<WriteI,     N2Write_1c_1I>;
def : InstRW<[N2Write_0or1c_1I], (instregex "^MOVZ[WX]i$")>;

// ALU, basic, flagset
def : InstRW<[N2Write_1c_1I_1Flg],
             (instregex "^(ADD|SUB)S[WX]r[ir]$",
                        "^(ADC|SBC)S[WX]r$",
                        "^ANDS[WX]ri$",
                        "^(AND|BIC)S[WX]rr$")>;

// ALU, extend and shift
def : SchedAlias<WriteIEReg, N2Write_2c_1M>;

// Arithmetic, LSL shift, shift <= 4
// Arithmetic, flagset, LSL shift, shift <= 4
// Arithmetic, LSR/ASR/ROR shift or LSL shift > 4
def : SchedAlias<WriteISReg, N2Write_Arith>;
def : InstRW<[N2Write_ArithI],
             (instregex "^(ADD|SUB)S[WX]rs$")>;

// Arithmetic, flagset, extend
def : InstRW<[N2Write_2c_1M_1Flg],
             (instregex "^(ADD|SUB)S[WX]rx(64)?$")>;

// Logical, shift, no flagset
def : InstRW<[N2Write_1c_1I],
             (instregex "^(AND|BIC|EON|EOR|ORN)[WX]rs$")>;
def : InstRW<[N2Write_0or1c_1I], (instregex "^ORR[WX]rs$")>;

// Logical, shift, flagset
def : InstRW<[N2Write_Logical], (instregex "^(AND|BIC)S[WX]rs$")>;

// Arithmetic, immediate to logical address tag
def : InstRW<[N2Write_2c_1M], (instrs ADDG, SUBG)>;

// Conditional compare
def : InstRW<[N2Write_1c_1I_1Flg], (instregex "^CCM[NP][WX][ir]")>;

// Convert floating-point condition flags
// Flag manipulation instructions
def : WriteRes<WriteSys, []> { let Latency = 1; }

// Insert Random Tags
def : InstRW<[N2Write_2c_1M], (instrs IRG, IRGstack)>;

// Insert Tag Mask
// Subtract Pointer
// Subtract Pointer, flagset
def : InstRW<[N2Write_1c_1I], (instrs GMI, SUBP)>;
def : InstRW<[N2Write_1c_1I_1Flg], (instrs SUBPS)>;

// Move and shift instructions
// -----------------------------------------------------------------------------

def : SchedAlias<WriteImm, N2Write_1c_1I>;

// Divide and Multiply Instructions
// -----------------------------------------------------------------------------

// SDIV, UDIV
// [a78_div_best] docs/a78-sog.md C-10, D-1, D-3: latency 5 and the divider held 5 cycles (the
// best value of each range).
def : SchedAlias<WriteID32,  A78Write_5c_1M0_IDiv>;
def : SchedAlias<WriteID64,  A78Write_5c_1M0_IDiv>;

def : WriteRes<WriteIM32, [N2UnitM]> { let Latency = 2; }
def : WriteRes<WriteIM64, [N2UnitM]> { let Latency = 2; }

// Multiply accumulate, W-form
// Multiply accumulate, X-form
def : InstRW<[N2Wr_IMUL, ReadIM, ReadIM, N2Rd_IMA],
             (instregex "^M(ADD|SUB)[WX]rrr$")>;

// Multiply accumulate long
// [a78_maddl] on M, 2 per cycle; the zero-accumulator forms (SMULL, UMULL, SMNEGL, UMNEGL:
// "Multiply long 2 2 M", l.707-708) stay N2Wr_IM.
def : InstRW<[A78Wr_IMULL, ReadIM, ReadIM, N2Rd_IMA],
             (instregex "^(S|U)M(ADD|SUB)Lrrr$")>;

// Multiply high
def : InstRW<[N2Write_3c_1M], (instrs SMULHrr, UMULHrr)>;

// Pointer Authentication Instructions (v8.3 PAC)
// -----------------------------------------------------------------------------

// Authenticate data address
// Authenticate instruction address
// Compute pointer authentication code for data address
// Compute pointer authentication code, using generic key
// Compute pointer authentication code for instruction address
def : InstRW<[N2Write_5c_1M0],
             (instregex "^AUT(D[AB]|DZ[AB]|I[AB]|IZ[AB]|I[AB]171615|I[AB]SPPC[ir])$",
                        "^PAC(D[AB]|DZ[AB]|GA|I[AB]|IZ[AB]|I[AB]171615|I[AB]SPPC|NBI[AB]SPPC|M)$",
                        "^(PAC|AUTPAC|AUTx16x17|AUTxMxN)$")>;
// [a78_pac_hint] docs/a78-sog.md D-12: the pointer-authentication instructions of the HINT space
// are NOPs on a core without FEAT_PAuth.  The Cortex-A78 implements Armv8.2-A without pointer
// authentication, and its SOG has no PAC row; LLVM gives cortex-a78 and cortex-a78ae no PAuth
// feature, cortex-a78c has it.  Without the feature they take the NOP treatment (WriteHint: no
// pipeline, C-3); with it (cortex-a78c, no A78C SOG consulted, U-12) the stock Neoverse N2 rows.
def A78HasPAuth : FeatureSchedPredicate<FeaturePAuth>;
def A78Write_Hint : SchedWriteRes<[]> { let Latency = 1; }
def A78Write_PACHint : SchedWriteVariant<[
                         SchedVar<A78HasPAuth, [N2Write_5c_1M0]>,
                         SchedVar<NoSchedPred, [A78Write_Hint]>]>;
def A78Write_XPACHint : SchedWriteVariant<[
                          SchedVar<A78HasPAuth, [N2Write_2c_1M0]>,
                          SchedVar<NoSchedPred, [A78Write_Hint]>]>;
def : InstRW<[A78Write_PACHint], (instrs PACIAZ, PACIBZ, PACIASP, PACIBSP, PACIA1716, PACIB1716,
                                         AUTIAZ, AUTIBZ, AUTIASP, AUTIBSP, AUTIA1716, AUTIB1716)>;

// Branch and link, register, with pointer authentication
// Branch, register, with pointer authentication
// Branch, return, with pointer authentication
def : InstRW<[N2Write_6c_1M0_1B], (instrs BLRAA, BLRAAZ, BLRAB, BLRABZ, BRAA,
                                          BRAAZ, BRAB, BRABZ, RETAA, RETAB,
                                          ERETAA, ERETAB)>;


// Load register, with pointer authentication
def : InstRW<[N2Write_9c_1M0_1L], (instregex "^LDRA[AB](indexed|writeback)")>;

// Strip pointer authentication code
def : InstRW<[N2Write_2c_1M0], (instrs XPACD, XPACI)>;
def : InstRW<[A78Write_XPACHint], (instrs XPACLRI)>;  // [a78_pac_hint]

// Miscellaneous data-processing instructions
// -----------------------------------------------------------------------------

// Bitfield extract, one reg
// Bitfield extract, two regs
def : SchedAlias<WriteExtr, N2Write_Extr>;

// Bitfield move, basic
def : SchedAlias<WriteIS, N2Write_1c_1I>;

// Bitfield move, insert
def : InstRW<[N2Write_2c_1M], (instregex "^BFM[WX]ri$")>;

// Load instructions
// -----------------------------------------------------------------------------

def : SchedAlias<WriteLD,    N2Write_4c_1L>;
// [a78_reg_offset] docs/a78-sog.md C-12: "Load register, register offset, basic / scale by 4/8 /
// extend / extend, scale by 4/8 ... 4 3 L" (l.980-991): no I µOP (the stock N2 class has one).
def : SchedAlias<WriteLDIdx, N2Write_4c_1L>;

// [a78_reg_offset_h] docs/a78-sog.md C-12: "Load register, register offset, scale by 2" and
// "..., extend, scale by 2" LDRH, LDRSH: "5 3 I, L" (l.985-986, l.992-993); without the scale
// they are "basic" / "extend" (4 3 L).
def A78WriteLDRH : SchedWriteVariant<[
                     SchedVar<A78IsMemScaled, [A78Write_5c_1I_1L]>,
                     SchedVar<NoSchedPred,    [N2Write_4c_1L]>]>;
def : InstRW<[A78WriteLDRH, ReadAdrBase], (instregex "^LDRHHro[WX]$", "^LDRSH[WX]ro[WX]$")>;

// Load pair, signed immed offset, normal, X-form
def : InstRW<[N2Write_4c_2L, WriteLDHi],
             (instrs LDPXi, LDNPXi)>;

// Load pair, immed post-index, normal, X-form
// Load pair, immed pre-index, normal, X-form
def : InstRW<[WriteAdr, N2Write_4c_2L, WriteLDHi],
             (instregex "^LDPX(post|pre)$")>;

// Load pair, signed immed offset, signed words
// [a78_ldpsw] docs/a78-sog.md C-12: 5 3/2 I, L (l.998-999, l.1019-1020), both registers 5.
def : InstRW<[A78Write_5c_1I_2L, A78Write_5c_0Z], (instrs LDPSWi)>;
// Load pair, immed post-index or immed pre-index, signed words
def : InstRW<[WriteAdr, A78Write_5c_1I_2L, A78Write_5c_0Z],
             (instregex "^LDPSW(post|pre)$")>;

// Store instructions
// -----------------------------------------------------------------------------

def : SchedAlias<WriteST,    N2Write_1c_1L01_1D>;
// [a78_reg_offset] docs/a78-sog.md C-13: "Store register, register offset, basic / scaled by 4/8 /
// extend / extend, scale by 4/8 ... 1 2 L01, D" (l.1127-1136): no I µOP.
def : SchedAlias<WriteSTIdx, N2Write_1c_1L01_1D>;
def : SchedAlias<WriteSTP,   N2Write_1c_1L01_1D>;
def : SchedAlias<WriteAdr,   N2Write_1c_1I>; // copied from A57.

// [a78_store_rows] docs/a78-sog.md C-13, I-3: "Store register, register offset, scaled by 2" and
// "..., extend, scale by 1" (sic) STRH: "2 2 I, L01, D" (l.1131-1132, l.1137-1138).
def A78WriteSTRH : SchedWriteVariant<[
                     SchedVar<A78IsMemScaled, [A78Write_2c_1I_1L01_1D]>,
                     SchedVar<NoSchedPred,    [N2Write_1c_1L01_1D]>]>;
def : InstRW<[A78WriteSTRH, ReadST, ReadAdrBase], (instregex "^STRHHro[WX]$")>;

// [a78_from_a55_store_data] docs/a78-sog.md B-1: the integer store forms whose stock Sched lists
// name no reads: same writes, plus ReadST on the data.
def : InstRW<[WriteST, ReadST, ReadAdrBase], (instregex "^STR(BB|HH|W|X)ui$", "^STUR(BB|HH|W|X)i$",
                                                        "^STTR(B|H|W|X)i$", "^STL?LR(B|H|W|X)$")>;
def : InstRW<[WriteSTP, ReadST, ReadST, ReadAdrBase], (instregex "^STN?P(W|X)i$")>;
def : InstRW<[WriteSTX, ReadST, ReadAdrBase], (instregex "^STLXR(B|H|W|X)$")>;
def : InstRW<[WriteSTX, ReadST, ReadST, ReadAdrBase], (instregex "^STLXP(W|X)$")>;
def : InstRW<[WriteAdr, WriteST, ReadST, ReadAdrBase], (instregex "^STR(BB|HH|W|X)(pre|post)$")>;
def : InstRW<[WriteAdr, WriteSTP, ReadST, ReadST, ReadAdrBase], (instregex "^STP(W|X)(pre|post)$")>;

// Tag load instructions
// -----------------------------------------------------------------------------

// Load allocation tag
// Load multiple allocation tags
def : InstRW<[N2Write_4c_1L], (instrs LDG, LDGM)>;

// Tag store instructions
// -----------------------------------------------------------------------------

// Store allocation tags to one or two granules, post-index
// Store allocation tags to one or two granules, pre-index
// Store allocation tag to one or two granules, zeroing, post-index
// Store Allocation Tag to one or two granules, zeroing, pre-index
// Store allocation tag and reg pair to memory, post-Index
// Store allocation tag and reg pair to memory, pre-Index
def : InstRW<[N2Write_1c_1L01_1D_1I], (instrs STGPreIndex, STGPostIndex,
                                              ST2GPreIndex, ST2GPostIndex,
                                              STZGPreIndex, STZGPostIndex,
                                              STZ2GPreIndex, STZ2GPostIndex,
                                              STGPpre, STGPpost)>;

// Store allocation tags to one or two granules, signed offset
// Store allocation tag to two granules, zeroing, signed offset
// Store allocation tag and reg pair to memory, signed offset
// Store multiple allocation tags
def : InstRW<[N2Write_1c_1L01_1D], (instrs STGi, ST2Gi, STZGi,
                                           STZ2Gi, STGPi, STGM, STZGM)>;

// FP data processing instructions
// -----------------------------------------------------------------------------

// FP absolute value
// FP arithmetic
// FP min/max
// FP negate
// FP select
def : SchedAlias<WriteF,     N2Write_2c_1V>;

// FP compare
def : SchedAlias<WriteFCmp,  N2Write_2c_1V0>;

// FP divide and square root operations are performed using an iterative
// algorithm and block subsequent similar operations to the same pipeline
// until complete (Arm Neoverse N2 Software Optimization Guide, 3.14).

// FP divide, square root
def : SchedAlias<WriteFDiv,  N2Write_7c_7V0>;

// [a78_fdiv_best] docs/a78-sog.md C-14, C-19, D-1, D-3: FP divide / square root, one write per
// (best latency, occupancy, uOPs).  FP and ASIMD divide / square root tables (l.1213-1215,
// l.1239-1241, l.1809-1813, l.1852-1861): the BEST latency of each range, and the divide unit is
// occupied for ceil(1 / best throughput) cycles:
//   scalar H 4/7 -> 2, S 4/7 -> 2, D 2/7 -> 4 (FSQRT: H 4/7, S 4/7, D 2/7);
//   ASIMD D-form F16 1/7 -> 7, D-form F32 2/7 -> 4, Q-form F16 1/10 (FDIV) / 1/11 (FSQRT)
//   -> 10 / 11, Q-form F32 1/7 -> 7, Q-form F64 1/7 -> 7.
// Best latency: 7, except Q-form F16 (FDIV 10, FSQRT 11).  [a78_dividers] the occupancy is on the
// FP divide unit A78UnitFDiv; V0 is taken for one cycle.
// One uOP each, as the Cortex-A76 and the X1 (the SOG gives no uOP count, docs/a78-sog.md U-2); the
// divide's iterations are expressed by the occupancy of the divide unit, not by extra uOPs.
def N2WriteFDS_7c2   : SchedWriteRes<[N2UnitV0, A78UnitFDiv]> { let Latency = 7;  let ReleaseAtCycles = [1, 2]; }
def N2WriteFDS_7c4   : SchedWriteRes<[N2UnitV0, A78UnitFDiv]> { let Latency = 7;  let ReleaseAtCycles = [1, 4]; }
def N2WriteFDS_7c7   : SchedWriteRes<[N2UnitV0, A78UnitFDiv]> { let Latency = 7;  let ReleaseAtCycles = [1, 7]; }
def N2WriteFDS_10c10 : SchedWriteRes<[N2UnitV0, A78UnitFDiv]> { let Latency = 10; let ReleaseAtCycles = [1, 10]; }
def N2WriteFDS_11c11 : SchedWriteRes<[N2UnitV0, A78UnitFDiv]> { let Latency = 11; let ReleaseAtCycles = [1, 11]; }

// FP divide, H-form
def : InstRW<[N2WriteFDS_7c2],  (instrs FDIVHrr)>;
// FP divide, S-form
def : InstRW<[N2WriteFDS_7c2], (instrs FDIVSrr)>;
// FP divide, D-form
def : InstRW<[N2WriteFDS_7c4], (instrs FDIVDrr)>;

// FP square root, H-form
def : InstRW<[N2WriteFDS_7c2],  (instrs FSQRTHr)>;
// FP square root, S-form
def : InstRW<[N2WriteFDS_7c2],  (instrs FSQRTSr)>;
// FP square root, D-form
def : InstRW<[N2WriteFDS_7c4], (instrs FSQRTDr)>;

// FP multiply
def : WriteRes<WriteFMul, [N2UnitV]> { let Latency = 3; }

// FP multiply accumulate
def : InstRW<[N2Wr_FMA, ReadDefault, ReadDefault, N2Rd_FMA], (instregex "^FN?M(ADD|SUB)[HSD]rrr$")>;

// FP round to integral
// [a78_frint] docs/a78-sog.md C-14: "FP round to integral FRINTA, ... FRINTZ 2 1 V0" (l.1234-1237).
def : InstRW<[N2Write_2c_1V0], (instregex "^FRINT[AIMNPXZ][HSD]r$",
                                          "^FRINT(32|64)[XZ][SD]r$")>;

// FP miscellaneous instructions
// -----------------------------------------------------------------------------

// FP convert, from gen to vec reg
// [a78_m0_pipelined] docs/a78-sog.md C-15: "3 1 M0" (l.1302): M0 one cycle.
def : InstRW<[A78Write_3c_1M0], (instregex "^[SU]CVTF[SU][WX][HSD]ri$")>;

// FP convert, from vec to gen reg
// [a78_fp_cvt] docs/a78-sog.md C-15: "FP convert, from vec to gen reg FCVTAS, ... FCVTZU 3 1 V0"
// (l.1303-1307).  The fixed-point forms
// (FCVTZS/FCVTZU Rd, Fn, #fbits) are WriteFCvt below, 3 1 V0 as well.
def : InstRW<[N2Write_3c_1V0], (instregex "^FCVT[AMNPZ][SU][SU][WX][HSD]r$")>;

// FP convert, Javascript from vec to gen reg
// FP convert, from vec to vec reg
def : SchedAlias<WriteFCvt, N2Write_3c_1V0>;
// [a78_fp_cvt] "FP convert, from vec to vec reg FCVT, FCVTXN 2 1 V0" (l.1301): the precision
// conversions and the scalar FCVTXN (not WriteFCvt above).
def : InstRW<[N2Write_2c_1V0], (instregex "^FCVT[HSD][HSD]r$", "^FCVTXNv1i64$")>;
// [a78_fp_cvt_scalar] docs/a78-sog.md D-4: FCVT[AMNPZ][SU], SCVTF, UCVTF with the operand and the
// result in FP/ASIMD registers (scalar forms, integer and fixed-point) have no row of their own; the
// rows of the same mnemonics give 3 1 (V0 from a vector register, M0 from a GPR, l.1302-1307) and
// the ASIMD rows 3 1 V0 for two lanes, more for four and eight (l.1791-1808).  One lane is taken as
// the two-lane row, as for the Cortex-A76.
def : InstRW<[N2Write_3c_1V0], (instregex "^FCVT[AMNPZ][SU]v1(i32|i64|f16)$",
                                          "^[SU]CVTFv1(i16|i32|i64)$",
                                          "^FCVTZ[SU][hsd]$", "^[SU]CVTF[hsd]$")>;

// FP move, immed
def : SchedAlias<WriteFImm, N2Write_2c_1V>;

// FP move, register
def : InstRW<[N2Write_2c_1V], (instrs FMOVHr, FMOVSr, FMOVDr)>;

// FP transfer, from gen to low half of vec reg
// [a78_zero_mov] docs/a78-sog.md C-4: "FMOV Hd/Sd/Dd, WZR/XZR" is not among the zero-latency MOVs
// of the A78 (sec. 4.15, l.2829-2848); the stock NeoverseZeroMove variant made it latency 0 with no
// pipeline.  [a78_m0_pipelined] C-15: "3 1 M0" (l.1310): M0 one cycle.
def : InstRW<[A78Write_3c_1M0], (instrs FMOVWHr, FMOVXHr, FMOVWSr, FMOVXDr)>;

// FP transfer, from gen to high half of vec reg
def : InstRW<[N2Write_5c_1M0_1V], (instrs FMOVXDHighr)>;

// FP transfer, from vec to gen reg
// [a78_fp_cvt] docs/a78-sog.md C-15: "FP transfer, from vec to gen reg FMOV 2 1 V1" (l.1314).
// WriteFCopy is left with FMOV Wd/Xd, Hn/Sn/Dn and FMOV Xd, Vn.D[1] (the other direction has
// the rows above).
def : SchedAlias<WriteFCopy, N2Write_2c_1V1>;

// FP load instructions
// -----------------------------------------------------------------------------
// [a78_fp_ld] docs/a78-sog.md C-16: Table 3-20 (l.1366-1419) gives latency 5 for the FP loads and 6
// for the register-offset scaled H/Q forms; the Neoverse N2 rows have 6 and 7.

// Load vector reg, literal, S/D/Q forms (5, L) / unscaled immed (5, L)
def : InstRW<[A78Write_5c_1L], (instregex "^LDR[SDQ]l$",
                                          "^LDUR[BHSDQ]i$")>;
// Load vector reg, immed post-index / pre-index (5, L, I)
// [a78_fp_ld] the I of "L, I" is the base update (WriteAdr); the load itself takes no I µOP.
def : InstRW<[WriteAdr, A78Write_5c_1L], (instregex "^LDR[BHSDQ](post|pre)$")>;
// Load vector reg, unsigned immed (5, L)
def : InstRW<[A78Write_5c_1L], (instregex "^LDR[BHSDQ]ui$")>;
// [a78_reg_offset_h] docs/a78-sog.md C-16, I-4: register offset (l.1384-1407): "basic" 5 L, I;
// "scale, S/D-form" 5 L; "scale, H/Q-form" 6 I, L; "extend" 5 L; "extend, scale, S/D-form" 5 L;
// "extend, scale, H/Q-form" 6 I, L.  The I of "basic" (the X index without the scale, which the
// "extend" and "scale, S/D" rows, and the integer "basic" row, do not have) is taken as a misprint
// and not modelled (I-4).  B has no scaled row: it is "basic" with an X index,
// "extend" with a W index.
def A78WriteLDRHQroX : SchedWriteVariant<[
                         SchedVar<A78IsMemScaled, [A78Write_6c_1I_1L]>,
                         SchedVar<NoSchedPred,    [A78Write_5c_1L]>]>;
def A78WriteLDRHQroW : SchedWriteVariant<[
                         SchedVar<A78IsMemScaled, [A78Write_6c_1I_1L]>,
                         SchedVar<NoSchedPred,    [A78Write_5c_1L]>]>;
def : InstRW<[A78Write_5c_1L, ReadAdrBase], (instregex "^LDR[BSD]ro[WX]$")>;
def : InstRW<[A78WriteLDRHQroX, ReadAdrBase], (instregex "^LDR[HQ]roX$")>;
def : InstRW<[A78WriteLDRHQroW, ReadAdrBase], (instregex "^LDR[HQ]roW$")>;
// [a78_ldp_hi] docs/a78-sog.md D-10: the second register of every FP load pair has the row's
// latency 5 and no µOP (A78Write_5c_0Z, not WriteLDHi).
// Load vector pair, immed offset, S/D-form (5, 3, L)
def : InstRW<[A78Write_5c_1L, A78Write_5c_0Z], (instregex "^LDN?P[SD]i$")>;
// Load vector pair, immed offset, Q-form (5, 3/2, L)
def : InstRW<[A78Write_5c_2L, A78Write_5c_0Z], (instrs LDPQi, LDNPQi)>;
// Load vector pair, immed post/pre-index, S/D-form (5, 3, I, L)
// [a78_fp_ld] one I µOP, the base update.
def : InstRW<[WriteAdr, A78Write_5c_1L, A78Write_5c_0Z],
             (instregex "^LDP[SD](pre|post)$")>;
// Load vector pair, immed post/pre-index, Q-form (5, 3/2, L, I)
def : InstRW<[WriteAdr, A78Write_5c_2L, A78Write_5c_0Z], (instrs LDPQpost, LDPQpre)>;

// FP store instructions
// -----------------------------------------------------------------------------
// [a78_from_a55_store_data] docs/a78-sog.md B-1: the data operands read ReadST, the base
// ReadAdrBase.

// Store vector reg, unscaled immed, B/H/S/D-form
// Store vector reg, unscaled immed, Q-form
def : InstRW<[N2Write_2c_1L01_1V, ReadST, ReadAdrBase], (instregex "^STUR[BHSDQ]i$")>;

// Store vector reg, immed post-index, B/H/S/D-form
// Store vector reg, immed post-index, Q-form
// Store vector reg, immed pre-index, B/H/S/D-form
// Store vector reg, immed pre-index, Q-form
// [a78_fp_st] docs/a78-sog.md C-17: "2 2 L01, V, I" (l.1465-1472): the I is the base update
// (WriteAdr); the store itself takes no I µOP.
def : InstRW<[WriteAdr, N2Write_2c_1L01_1V, ReadST, ReadAdrBase],
             (instregex "^STR[BHSDQ](pre|post)$")>;

// Store vector reg, unsigned immed, B/H/S/D-form
// Store vector reg, unsigned immed, Q-form
def : InstRW<[N2Write_2c_1L01_1V, ReadST, ReadAdrBase], (instregex "^STR[BHSDQ]ui$")>;

// Store vector reg, register offset, basic, B/H/S/D-form
// Store vector reg, register offset, basic, Q-form
// Store vector reg, register offset, scale, S/D-form
// Store vector reg, register offset, extend, B/H/S/D-form
// Store vector reg, register offset, extend, Q-form
// Store vector reg, register offset, extend, scale, S/D-form
def : InstRW<[N2Write_2c_1L01_1V, ReadST, ReadAdrBase],
             (instregex "^STR[BSD]ro[WX]$", "^STRQroX$")>;

// [a78_store_rows] docs/a78-sog.md C-17, I-5: "Store vector reg, register offset, scale, H-form"
// and "..., extend, scale, H-form" 2 2 I, L01, V (l.1481-1482, l.1491-1492); "..., extend, scale,
// Q-form" 2 2 I, L01, V (l.1495-1496), while "..., scale, Q-form" is L01, V (l.1485-1486).
def A78WriteSTRHro : SchedWriteVariant<[
                       SchedVar<A78IsMemScaled, [A78Write_2c_1I_1L01_1V]>,
                       SchedVar<NoSchedPred,    [N2Write_2c_1L01_1V]>]>;
def : InstRW<[A78WriteSTRHro, ReadST, ReadAdrBase], (instregex "^STRHro[WX]$", "^STRQroW$")>;

// Store vector pair, immed offset, S-form
// Store vector pair, immed offset, D-form
def : InstRW<[N2Write_2c_1L01_1V, ReadST, ReadST, ReadAdrBase], (instregex "^STN?P[SD]i$")>;

// Store vector pair, immed offset, Q-form
// [a78_fp_st] docs/a78-sog.md C-17, I-6: "2 2 L01, V" (l.1510-1511): one V µOP, 2 per cycle.
def : InstRW<[N2Write_2c_1L01_1V, ReadST, ReadST, ReadAdrBase], (instrs STPQi, STNPQi)>;

// Store vector pair, immed post-index, S-form
// Store vector pair, immed post-index, D-form
// Store vector pair, immed pre-index, S-form
// Store vector pair, immed pre-index, D-form
// [a78_fp_st] "2 2 I, L01, V" (l.1512-1515, l.1518-1521): one I µOP, the base update.
def : InstRW<[WriteAdr, N2Write_2c_1L01_1V, ReadST, ReadST, ReadAdrBase],
             (instregex "^STP[SD](pre|post)$")>;

// Store vector pair, immed post-index, Q-form
// Store vector pair, immed pre-index, Q-form
// [a78_fp_st] "2 1 I, L01, V" (l.1516-1517, l.1522-1523): the base update is WriteAdr (latency 1,
// the "(1)" of the writeback notes, l.1537-1542), for both forms.  [a78_stpq_wb]: one per cycle on L01.
def : InstRW<[WriteAdr, A78Write_2c_2L01_1V, ReadST, ReadST, ReadAdrBase], (instrs STPQpost, STPQpre)>;

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
def : SchedAlias<WriteVd, N2Write_2c_1V>;
def : SchedAlias<WriteVq, N2Write_2c_1V>;

// ASIMD absolute diff accum
// ASIMD absolute diff accum long
def : InstRW<[N2Wr_VA, N2Rd_VA],
             (instregex "^SABAv", "^UABAv", "^SABALv", "^UABALv")>;

// ASIMD arith, reduce, 4H/4S
def : InstRW<[N2Write_2c_1V1], (instregex "^(ADDV|[SU]ADDLV)v4(i16|i32)v$")>;

// ASIMD arith, reduce, 8B/8H
def : InstRW<[N2Write_4c_1V1_1V],
             (instregex "^(ADDV|[SU]ADDLV)v8(i8|i16)v$")>;

// ASIMD arith, reduce, 16B
// [a78_v_throughput] docs/a78-sog.md C-18: "4 1/2 V1" (l.1591-1592): V1 held 2 cycles.
def : InstRW<[A78Write_4c_1V1_2cyc], (instrs ADDVv16i8v, SADDLVv16i8v,
                                             UADDLVv16i8v)>;

// ASIMD dot product
// ASIMD dot product using signed and unsigned integers
def : InstRW<[N2Wr_VDOT, N2Rd_VDOT],
             (instregex "^([SU]|SU|US)DOT(lane)?(v8|v16)i8$")>;

// ASIMD matrix multiply-accumulate
def : InstRW<[N2Wr_VMMA, N2Rd_VMMA], (instrs SMMLA, UMMLA, USMMLA)>;

// ASIMD max/min, reduce, 4H/4S
def : InstRW<[N2Write_2c_1V1], (instregex "^[SU](MAX|MIN)Vv4i16v$",
                                          "^[SU](MAX|MIN)Vv4i32v$")>;

// ASIMD max/min, reduce, 8B/8H
def : InstRW<[N2Write_4c_1V1_1V], (instregex "^[SU](MAX|MIN)Vv8i8v$",
                                             "^[SU](MAX|MIN)Vv8i16v$")>;

// ASIMD max/min, reduce, 16B
def : InstRW<[N2Write_4c_2V1], (instregex "[SU](MAX|MIN)Vv16i8v$")>;

// ASIMD multiply
// [a78_mul_acc_fwd] MUL: A78Wr_VMUL (4, V0; forwards to the MLA/MLS accumulator).
def : InstRW<[A78Wr_VMUL], (instregex "^MULv")>;
def : InstRW<[N2Write_4c_1V0], (instregex "^SQ(R)?DMULHv")>;

// ASIMD multiply accumulate
def : InstRW<[N2Wr_VMA, N2Rd_VMA], (instregex "^MLAv", "^MLSv")>;

// ASIMD multiply accumulate high
// [a78_asimd_acc] docs/a78-sog.md C-18: "SQRDMLAH, SQRDMLSH 4 1 V0" (l.1626-1627), with no
// accumulate latency in parentheses and no note; the N2 row forwarded the accumulator (advance 2).
def : InstRW<[N2Write_4c_1V0], (instregex "^SQRDMLAHv", "^SQRDMLSHv")>;

// ASIMD multiply accumulate long
def : InstRW<[N2Wr_VMAL, N2Rd_VMAL], (instregex "^[SU]MLALv", "^[SU]MLSLv")>;

// ASIMD multiply accumulate saturating long
// [a78_asimd_scalar] docs/a78-sog.md D-5: the scalar SQDMLAL/SQDMLSL (i16/i32) as well.
def : InstRW<[N2Write_4c_1V0], (instregex "^SQDMLALv", "^SQDMLSLv", "^SQDML[AS]Li(16|32)$")>;

// ASIMD multiply/multiply long (8x8) polynomial, D-form
// ASIMD multiply/multiply long (8x8) polynomial, Q-form
def : InstRW<[N2Write_3c_1V0], (instregex "^PMULL?(v8i8|v16i8)$")>;

// ASIMD multiply long
// [a78_asimd_mull] docs/a78-sog.md C-18: "SMULL(2), UMULL(2), SQDMULL(2) 3 1 V0" (l.1638-1640).
// [a78_asimd_scalar] D-5: the scalar SQDMULL (i16/i32) as well.
// [a78_mul_acc_fwd] SMULL/UMULL(2): A78Wr_VMULL (3, V0; forwards to the SMLAL-family
// accumulator).
def : InstRW<[A78Wr_VMULL], (instregex "^[SU]MULLv")>;
def : InstRW<[N2Write_3c_1V0], (instregex "^SQDMULLv", "^SQDMULLi(16|32)$")>;

// ASIMD pairwise add and accumulate long
def : InstRW<[N2Wr_VPA, N2Rd_VPA], (instregex "^[SU]ADALPv")>;

// ASIMD shift accumulate
// [a78_asimd_scalar] docs/a78-sog.md D-5: the scalar forms (SSRA Dd, ... "d") as well.
def : InstRW<[N2Wr_VSA, N2Rd_VSA], (instregex "^[SU]SRAv", "^[SU]RSRAv", "^[SU]R?SRAd$")>;

// ASIMD shift by immed, basic
// [a78_asimd_scalar] the scalar SHL, SSHR, USHR (Dd, "d") as well.
def : InstRW<[N2Write_2c_1V1], (instregex "^SHLv", "^SHLLv", "^SHRNv",
                                          "^SSHLLv", "^SSHRv", "^USHLLv",
                                          "^USHRv", "^(SHL|[SU]SHR)d$")>;

// ASIMD shift by immed and insert, basic
// [a78_asimd_scalar] the scalar SLI, SRI (Dd, "d") as well.
def : InstRW<[N2Write_2c_1V1], (instregex "^SLIv", "^SRIv", "^S[LR]Id$")>;

// ASIMD shift by immed, complex
// [a78_asimd_scalar] the scalar SRSHR/URSHR (d) and the scalar narrowing shifts (b, h, s) as well.
def : InstRW<[N2Write_4c_1V1],
             (instregex "^RSHRNv", "^SQRSHRNv", "^SQRSHRUNv",
                        "^(SQSHLU?|UQSHL)[bhsd]$",
                        "^(SQSHLU?|UQSHL)(v8i8|v16i8|v4i16|v8i16|v2i32|v4i32|v2i64)_shift$",
                        "^SQSHRNv", "^SQSHRUNv", "^SRSHRv", "^UQRSHRNv",
                        "^UQSHRNv", "^URSHRv", "^[SU]RSHRd$",
                        "^(SQR?SHRU?N|UQR?SHRN)[bhs]$")>;

// ASIMD shift by register, basic
def : InstRW<[N2Write_2c_1V1], (instregex "^[SU]SHLv")>;

// ASIMD shift by register, complex
def : InstRW<[N2Write_4c_1V1],
             (instregex "^[SU]RSHLv", "^[SU]QRSHLv",
                        "^[SU]QSHL(v1i8|v1i16|v1i32|v1i64|v8i8|v16i8|v4i16|v8i16|v2i32|v4i32|v2i64)$")>;

// ASIMD floating-point instructions
// -----------------------------------------------------------------------------

// ASIMD FP absolute value/difference
// ASIMD FP arith, normal
// ASIMD FP compare
// ASIMD FP complex add
// ASIMD FP max/min, normal
// ASIMD FP max/min, pairwise
// ASIMD FP negate
// Handled by SchedAlias<WriteV[dq], ...>

// ASIMD FP complex multiply add
def : InstRW<[N2Wr_VFCMA, N2Rd_VFCMA], (instregex "^FCMLAv")>;

// ASIMD FP convert, long (F16 to F32)
def : InstRW<[N2Write_4c_2V0], (instregex "^FCVTL(v4|v8)i16")>;

// ASIMD FP convert, long (F32 to F64)
def : InstRW<[N2Write_3c_1V0], (instregex "^FCVTL(v2|v4)i32")>;

// ASIMD FP convert, narrow (F32 to F16)
def : InstRW<[N2Write_4c_2V0], (instregex "^FCVTN(v4|v8)i16")>;

// ASIMD FP convert, narrow (F64 to F32)
def : InstRW<[N2Write_3c_1V0], (instregex "^FCVTN(v2|v4)i32",
                                          "^FCVTXN(v2|v4)f32")>;

// ASIMD FP convert, other, D-form F32 and Q-form F64
// [a78_fp_cvt] docs/a78-sog.md C-19: the fixed-point vector forms (FCVTZS/FCVTZU/SCVTF/UCVTF
// Vd.T, Vn.T, #fbits, "..._shift") are the same mnemonics and arrangements as these rows
// (l.1791-1808), not the default (2, V).
def : InstRW<[N2Write_3c_1V0], (instregex "^[FSU]CVT[AMNPZ][SU]v2f(32|64)$",
                                          "^[SU]CVTFv2f(32|64)$",
                                          "^(FCVTZ[SU]|[SU]CVTF)v2i(32|64)_shift$")>;

// ASIMD FP convert, other, D-form F16 and Q-form F32
def : InstRW<[N2Write_4c_2V0], (instregex "^[FSU]CVT[AMNPZ][SU]v4f(16|32)$",
                                          "^[SU]CVTFv4f(16|32)$",
                                          "^(FCVTZ[SU]|[SU]CVTF)v4i(16|32)_shift$")>;

// ASIMD FP convert, other, Q-form F16
def : InstRW<[N2Write_6c_4V0], (instregex "^[FSU]CVT[AMNPZ][SU]v8f16$",
                                          "^[SU]CVTFv8f16$",
                                          "^(FCVTZ[SU]|[SU]CVTF)v8i16_shift$")>;

// ASIMD FP divide, D-form, F16
def : InstRW<[N2WriteFDS_7c7], (instrs FDIVv4f16)>;

// ASIMD FP divide, D-form, F32
def : InstRW<[N2WriteFDS_7c4], (instrs FDIVv2f32)>;

// ASIMD FP divide, Q-form, F16
def : InstRW<[N2WriteFDS_10c10], (instrs FDIVv8f16)>;

// ASIMD FP divide, Q-form, F32
def : InstRW<[N2WriteFDS_7c7], (instrs FDIVv4f32)>;

// ASIMD FP divide, Q-form, F64
def : InstRW<[N2WriteFDS_7c7], (instrs FDIVv2f64)>;

// ASIMD FP max/min, reduce, F32 and D-form F16
// [a78_v_throughput] docs/a78-sog.md C-19: "4 1 V" (l.1831-1833): V held 2 cycles in total (one µOP
// of one cycle would give 2 per cycle).
def : InstRW<[A78WriteV<4, 1, 2>], (instregex "^(FMAX|FMIN)(NM)?Vv4(i16|i32)v$")>;

// ASIMD FP max/min, reduce, Q-form F16
// [a78_v_throughput] "6 2/3 V" (l.1834-1836): V held 3 cycles in total.
def : InstRW<[A78WriteV<6, 2, 3>], (instregex "^(FMAX|FMIN)(NM)?Vv8i16v$")>;

// ASIMD FP multiply
def : InstRW<[N2Wr_VFM], (instregex "^FMULv", "^FMULXv")>;
// [a78_asimd_scalar] docs/a78-sog.md C-19, D-5: the scalar FMULX (Hd/Sd/Dd, FMULX16/32/64), missed by
// "^FMULXv", takes the row of its mnemonic, "ASIMD FP multiply FMUL, FMULX 3 2 V" (l.1837), with
// the write of the scalar FMUL (WriteFMul, forwarded to the FMADD accumulator).
def : InstRW<[WriteFMul], (instregex "^FMULX(16|32|64)$")>;

// ASIMD FP multiply accumulate
def : InstRW<[N2Wr_VFMA, N2Rd_VFMA], (instregex "^FMLAv", "^FMLSv")>;

// ASIMD FP multiply accumulate long
def : InstRW<[N2Wr_VFMAL, N2Rd_VFMAL], (instregex "^FMLALv", "^FMLSLv")>;

// ASIMD FP round, D-form F32 and Q-form F64
def : InstRW<[N2Write_3c_1V0],
             (instregex "^FRINT[AIMNPXZ]v2f(32|64)$",
                        "^FRINT(32|64)[XZ]v2f(32|64)$")>;

// ASIMD FP round, D-form F16 and Q-form F32
def : InstRW<[N2Write_4c_2V0],
             (instregex "^FRINT[AIMNPXZ]v4f(16|32)$",
                        "^FRINT(32|64)[XZ]v4f32$")>;


// ASIMD FP round, Q-form F16
def : InstRW<[N2Write_6c_4V0], (instregex "^FRINT[AIMNPXZ]v8f16$")>;

// ASIMD FP square root, D-form, F16
def : InstRW<[N2WriteFDS_7c7], (instrs FSQRTv4f16)>;

// ASIMD FP square root, D-form, F32
def : InstRW<[N2WriteFDS_7c4], (instrs FSQRTv2f32)>;

// ASIMD FP square root, Q-form, F16
def : InstRW<[N2WriteFDS_11c11], (instrs FSQRTv8f16)>;

// ASIMD FP square root, Q-form, F32
def : InstRW<[N2WriteFDS_7c7], (instrs FSQRTv4f32)>;

// ASIMD FP square root, Q-form, F64
def : InstRW<[N2WriteFDS_7c7], (instrs FSQRTv2f64)>;

// ASIMD BFloat16 (BF16) instructions
// -----------------------------------------------------------------------------

// ASIMD convert, F32 to BF16
def : InstRW<[N2Write_4c_1V0], (instrs BFCVTN, BFCVTN2)>;

// ASIMD dot product
def : InstRW<[N2Wr_VBFDOT, N2Rd_VBFDOT], (instrs BFDOTv4bf16, BFDOTv8bf16)>;

// ASIMD matrix multiply accumulate
def : InstRW<[N2Wr_VBFMMA, N2Rd_VBFMMA], (instrs BFMMLA)>;

// ASIMD multiply accumulate long
def : InstRW<[N2Wr_VBFMAL, N2Rd_VBFMAL], (instrs BFMLALB, BFMLALBIdx, BFMLALT,
                                                 BFMLALTIdx)>;

// Scalar convert, F32 to BF16
def : InstRW<[N2Write_3c_1V0], (instrs BFCVT)>;

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
// Handled by SchedAlias<WriteV[dq], ...>

// [a78_zero_mov] docs/a78-sog.md C-4: "MOVI Dd, #0" and "MOVI Vd.2D, #0" are not among the
// zero-latency MOVs of the A78 (sec. 4.15, l.2829-2848); the stock NeoverseZeroMove variant made
// them latency 0 with no pipeline.  "ASIMD move, integer immed MOVI 2 2 V" (l.1947).
def : InstRW<[N2Write_2c_1V], (instrs MOVID, MOVIv2d_ns)>;

// ASIMD duplicate, gen reg
// [a78_m0_pipelined] docs/a78-sog.md C-20: "3 1 M0" (l.1937): M0 one cycle.
def : InstRW<[A78Write_3c_1M0], (instregex "^DUPv.+gpr")>;

// ASIMD transfer, element to gen reg
// [a78_fp_cvt] docs/a78-sog.md C-20: "UMOV, SMOV 2 1 V1" (l.1988-1989).
def : InstRW<[N2Write_2c_1V1], (instregex "^[SU]MOVvi")>;

// ASIMD extract narrow, saturating
def : InstRW<[N2Write_4c_1V1], (instregex "^[SU]QXTNv", "^SQXTUNv")>;

// ASIMD reciprocal and square root estimate, D-form U32
def : InstRW<[N2Write_3c_1V0], (instrs URECPEv2i32, URSQRTEv2i32)>;

// ASIMD reciprocal and square root estimate, Q-form U32
def : InstRW<[N2Write_4c_2V0], (instrs URECPEv4i32, URSQRTEv4i32)>;

// ASIMD reciprocal and square root estimate, D-form F32 and scalar forms
def : InstRW<[N2Write_3c_1V0], (instrs FRECPEv1f16, FRECPEv1i32,
                                       FRECPEv1i64, FRECPEv2f32,
                                       FRSQRTEv1f16, FRSQRTEv1i32,
                                       FRSQRTEv1i64, FRSQRTEv2f32)>;

// ASIMD reciprocal and square root estimate, D-form F16 and Q-form F32
def : InstRW<[N2Write_4c_2V0], (instrs FRECPEv4f16, FRECPEv4f32,
                                         FRSQRTEv4f16, FRSQRTEv4f32)>;

// ASIMD reciprocal and square root estimate, Q-form F16
def : InstRW<[N2Write_6c_4V0], (instrs FRECPEv8f16, FRSQRTEv8f16)>;

// ASIMD reciprocal exponent
def : InstRW<[N2Write_3c_1V0], (instregex "^FRECPXv")>;

// ASIMD reciprocal step
// [a78_asimd_scalar] docs/a78-sog.md C-20, D-5: the scalar FRECPS/FRSQRTS (16/32/64), missed by the
// "v" patterns, take "ASIMD reciprocal step FRECPS, FRSQRTS 4 2 V" (l.1961).
def : InstRW<[N2Write_4c_1V], (instregex "^FRECPSv", "^FRSQRTSv",
                                         "^FRECPS(16|32|64)$", "^FRSQRTS(16|32|64)$")>;

// ASIMD table lookup, 3 table regs
def : InstRW<[N2Write_4c_2V], (instrs TBLv8i8Three, TBLv16i8Three)>;

// ASIMD table lookup, 4 table regs
// [a78_v_throughput] docs/a78-sog.md C-20: "4 2/3 V" (l.1968-1969): V held 3 cycles in total (four
// µOPs of one cycle would give 1/2).
def : InstRW<[A78WriteV<4, 4, 3>], (instrs TBLv8i8Four, TBLv16i8Four)>;

// ASIMD table lookup extension, 2 table reg
def : InstRW<[N2Write_4c_2V], (instrs TBXv8i8Two, TBXv16i8Two)>;

// ASIMD table lookup extension, 3 table reg
// [a78_v_throughput] "6 2/3 V" (l.1984-1985): V held 3 cycles.
def : InstRW<[A78WriteV<6, 4, 3>], (instrs TBXv8i8Three, TBXv16i8Three)>;

// ASIMD table lookup extension, 4 table reg
// [a78_v_throughput] "6 2/5 V" (l.1986-1987): V held 5 cycles.
def : InstRW<[A78WriteV<6, 8, 5>], (instrs TBXv8i8Four, TBXv16i8Four)>;

// ASIMD transfer, gen reg to element
def : InstRW<[N2Write_5c_1M0_1V], (instregex "^INSvi(8|16|32|64)gpr$")>;

// ASIMD load instructions
// -----------------------------------------------------------------------------
// [a78_asimd_ld] docs/a78-sog.md C-21: Table 3-30 (l.2065-2172); the A78 latencies differ from the
// stock N2 rows for LD1, LD1 one lane / LD1R, LD2 and LD4 multiple Q-form.  LD1 four registers D-form ("5 3/2") and LD4
// multiple Q-form ("10 1/4") also have [a78_ls_throughput] (D-6).  LD3 and the other LD4 rows
// match.  The writeback forms add WriteAdr ("(1) +I", l.2172).

// ASIMD load, 1 element, multiple, 1 reg, D-form
def : InstRW<[A78Write_5c_1L], (instregex "^LD1Onev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, A78Write_5c_1L],
             (instregex "^LD1Onev(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, multiple, 1 reg, Q-form
def : InstRW<[A78Write_5c_1L], (instregex "^LD1Onev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A78Write_5c_1L],
             (instregex "^LD1Onev(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 2 reg, D-form
def : InstRW<[A78Write_5c_2L], (instregex "^LD1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, A78Write_5c_2L],
             (instregex "^LD1Twov(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, multiple, 2 reg, Q-form
def : InstRW<[A78Write_5c_2L], (instregex "^LD1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A78Write_5c_2L],
             (instregex "^LD1Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 3 reg, D-form
def : InstRW<[A78Write_5c_3L], (instregex "^LD1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, A78Write_5c_3L],
             (instregex "^LD1Threev(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, multiple, 3 reg, Q-form
def : InstRW<[A78Write_5c_3L], (instregex "^LD1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A78Write_5c_3L],
             (instregex "^LD1Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, multiple, 4 reg, D-form
def : InstRW<[A78Write_5c_4L_2cyc], (instregex "^LD1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, A78Write_5c_4L_2cyc],
             (instregex "^LD1Fourv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, multiple, 4 reg, Q-form
def : InstRW<[A78Write_6c_4L], (instregex "^LD1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A78Write_6c_4L],
             (instregex "^LD1Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 1 element, one lane, B/H/S
// ASIMD load, 1 element, one lane, D
def : InstRW<[A78Write_7c_1L_1V],           (instregex "LD1i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, A78Write_7c_1L_1V], (instregex "LD1i(8|16|32|64)_POST$")>;

// ASIMD load, 1 element, all lanes, D-form, B/H/S
// ASIMD load, 1 element, all lanes, D-form, D
def : InstRW<[A78Write_7c_1L_1V],           (instregex "LD1Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, A78Write_7c_1L_1V], (instregex "LD1Rv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 1 element, all lanes, Q-form
def : InstRW<[A78Write_7c_1L_1V],           (instregex "LD1Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A78Write_7c_1L_1V], (instregex "LD1Rv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 2 element, multiple, D-form, B/H/S
def : InstRW<[A78Write_7c_1L_2V],           (instregex "LD2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, A78Write_7c_1L_2V], (instregex "LD2Twov(8b|4h|2s)_POST$")>;

// ASIMD load, 2 element, multiple, Q-form, B/H/S
// ASIMD load, 2 element, multiple, Q-form, D
def : InstRW<[A78Write_7c_2L_2V],           (instregex "LD2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A78Write_7c_2L_2V], (instregex "LD2Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 2 element, one lane, B/H
// ASIMD load, 2 element, one lane, S
// ASIMD load, 2 element, one lane, D
def : InstRW<[A78Write_7c_1L_2V],           (instregex "LD2i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, A78Write_7c_1L_2V], (instregex "LD2i(8|16|32|64)_POST$")>;

// ASIMD load, 2 element, all lanes, D-form, B/H/S
// ASIMD load, 2 element, all lanes, D-form, D
def : InstRW<[A78Write_7c_1L_2V],            (instregex "LD2Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, A78Write_7c_1L_2V],  (instregex "LD2Rv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 2 element, all lanes, Q-form
def : InstRW<[A78Write_7c_1L_2V],           (instregex "LD2Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A78Write_7c_1L_2V], (instregex "LD2Rv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 3 element, multiple, D-form, B/H/S
def : InstRW<[N2Write_8c_2L_3V],           (instregex "LD3Threev(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, N2Write_8c_2L_3V], (instregex "LD3Threev(8b|4h|2s)_POST$")>;

// ASIMD load, 3 element, multiple, Q-form, B/H/S
def : InstRW<[N2Write_8c_3L_3V],           (instregex "LD3Threev(16b|8h|4s)$")>;
def : InstRW<[WriteAdr, N2Write_8c_3L_3V], (instregex "LD3Threev(16b|8h|4s)_POST$")>;

// ASIMD load, 3 element, multiple, Q-form, D
def : InstRW<[N2Write_8c_3L_3V],           (instregex "LD3Threev(2d)$")>;
def : InstRW<[WriteAdr, N2Write_8c_3L_3V], (instregex "LD3Threev(2d)_POST$")>;

// ASIMD load, 3 element, one lane, B/H
// ASIMD load, 3 element, one lane, S
// ASIMD load, 3 element, one lane, D
def : InstRW<[N2Write_8c_2L_3V],           (instregex "LD3i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, N2Write_8c_2L_3V], (instregex "LD3i(8|16|32|64)_POST$")>;

// ASIMD load, 3 element, all lanes, D-form, B/H/S
// ASIMD load, 3 element, all lanes, D-form, D
def : InstRW<[N2Write_8c_2L_3V],           (instregex "LD3Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N2Write_8c_2L_3V], (instregex "LD3Rv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 3 element, all lanes, Q-form, B/H/S
// ASIMD load, 3 element, all lanes, Q-form, D
def : InstRW<[N2Write_8c_3L_3V],           (instregex "LD3Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N2Write_8c_3L_3V], (instregex "LD3Rv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 4 element, multiple, D-form, B/H/S
def : InstRW<[N2Write_8c_3L_4V],           (instregex "LD4Fourv(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, N2Write_8c_3L_4V], (instregex "LD4Fourv(8b|4h|2s)_POST$")>;

// ASIMD load, 4 element, multiple, Q-form, B/H/S
// ASIMD load, 4 element, multiple, Q-form, D
def : InstRW<[A78WriteLDV<10, 8, 12, 4>],           (instregex "LD4Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A78WriteLDV<10, 8, 12, 4>], (instregex "LD4Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD load, 4 element, one lane, B/H
// ASIMD load, 4 element, one lane, S
// ASIMD load, 4 element, one lane, D
def : InstRW<[N2Write_8c_3L_4V],           (instregex "LD4i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, N2Write_8c_3L_4V], (instregex "LD4i(8|16|32|64)_POST$")>;

// ASIMD load, 4 element, all lanes, D-form, B/H/S
// ASIMD load, 4 element, all lanes, D-form, D
def : InstRW<[N2Write_8c_3L_4V],              (instregex "LD4Rv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N2Write_8c_3L_4V],    (instregex "LD4Rv(8b|4h|2s|1d)_POST$")>;

// ASIMD load, 4 element, all lanes, Q-form, B/H/S
// ASIMD load, 4 element, all lanes, Q-form, D
def : InstRW<[N2Write_8c_4L_4V],            (instregex "LD4Rv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N2Write_8c_4L_4V],  (instregex "LD4Rv(16b|8h|4s|2d)_POST$")>;

// ASIMD store instructions
// -----------------------------------------------------------------------------
// [a78_asimd_st] docs/a78-sog.md C-22: Table 3-32 (l.2248-2314).  [a78_ls_throughput] D-6: the rows
// with A78WriteSTV reach the SOG throughput by holding the L01 pipes 2/T cycles (one cycle per µOP
// would be up to twice as fast as the SOG, e.g. ST2 multiple Q-form 1 per cycle, SOG 1/2).
// [a78_from_a55_store_data] B-1: the data Vt reads ReadST, the base Rn ReadAdrBase; for the
// one-lane forms the lane index takes the read in between (ReadDefault).

// ASIMD store, 1 element, multiple, 1 reg, D-form
def : InstRW<[N2Write_2c_1L01_1V, ReadST, ReadAdrBase],
             (instregex "ST1Onev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N2Write_2c_1L01_1V, ReadST, ReadAdrBase],
             (instregex "ST1Onev(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 1 reg, Q-form
def : InstRW<[N2Write_2c_1L01_1V, ReadST, ReadAdrBase],
             (instregex "ST1Onev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N2Write_2c_1L01_1V, ReadST, ReadAdrBase],
             (instregex "ST1Onev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 2 reg, D-form
def : InstRW<[N2Write_2c_1L01_1V, ReadST, ReadAdrBase],
             (instregex "ST1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N2Write_2c_1L01_1V, ReadST, ReadAdrBase],
             (instregex "ST1Twov(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 2 reg, Q-form
def : InstRW<[N2Write_2c_2L01_2V, ReadST, ReadAdrBase],
             (instregex "ST1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N2Write_2c_2L01_2V, ReadST, ReadAdrBase],
             (instregex "ST1Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 3 reg, D-form
def : InstRW<[N2Write_2c_2L01_2V, ReadST, ReadAdrBase],
             (instregex "ST1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N2Write_2c_2L01_2V, ReadST, ReadAdrBase],
             (instregex "ST1Threev(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 3 reg, Q-form
def : InstRW<[N2Write_2c_3L01_3V, ReadST, ReadAdrBase],
             (instregex "ST1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N2Write_2c_3L01_3V, ReadST, ReadAdrBase],
             (instregex "ST1Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, multiple, 4 reg, D-form
def : InstRW<[N2Write_2c_2L01_2V, ReadST, ReadAdrBase],
             (instregex "ST1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[WriteAdr, N2Write_2c_2L01_2V, ReadST, ReadAdrBase],
             (instregex "ST1Fourv(8b|4h|2s|1d)_POST$")>;

// ASIMD store, 1 element, multiple, 4 reg, Q-form
def : InstRW<[N2Write_2c_4L01_4V, ReadST, ReadAdrBase],
             (instregex "ST1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, N2Write_2c_4L01_4V, ReadST, ReadAdrBase],
             (instregex "ST1Fourv(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 1 element, one lane, B/H/S
// ASIMD store, 1 element, one lane, D
def : InstRW<[A78WriteSTV<4, 2, 2, 1>, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST1i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, A78WriteSTV<4, 2, 2, 1>, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST1i(8|16|32|64)_POST$")>;

// ASIMD store, 2 element, multiple, D-form, B/H/S
def : InstRW<[A78WriteSTV<4, 2, 2, 1>, ReadST, ReadAdrBase],
             (instregex "ST2Twov(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, A78WriteSTV<4, 2, 2, 1>, ReadST, ReadAdrBase],
             (instregex "ST2Twov(8b|4h|2s)_POST$")>;

// ASIMD store, 2 element, multiple, Q-form, B/H/S
// ASIMD store, 2 element, multiple, Q-form, D
def : InstRW<[A78WriteSTV<5, 4, 4, 2>, ReadST, ReadAdrBase],
             (instregex "ST2Twov(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A78WriteSTV<5, 4, 4, 2>, ReadST, ReadAdrBase],
             (instregex "ST2Twov(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 2 element, one lane, B/H/S
// ASIMD store, 2 element, one lane, D
def : InstRW<[A78WriteSTV<4, 2, 2, 1>, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST2i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, A78WriteSTV<4, 2, 2, 1>, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST2i(8|16|32|64)_POST$")>;

// ASIMD store, 3 element, multiple, D-form, B/H/S
def : InstRW<[A78WriteSTV<5, 4, 4, 2>, ReadST, ReadAdrBase],
             (instregex "ST3Threev(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, A78WriteSTV<5, 4, 4, 2>, ReadST, ReadAdrBase],
             (instregex "ST3Threev(8b|4h|2s)_POST$")>;

// ASIMD store, 3 element, multiple, Q-form, B/H/S
// ASIMD store, 3 element, multiple, Q-form, D
def : InstRW<[A78WriteSTV<6, 6, 6, 3>, ReadST, ReadAdrBase],
             (instregex "ST3Threev(16b|8h|4s|2d)$")>;
def : InstRW<[WriteAdr, A78WriteSTV<6, 6, 6, 3>, ReadST, ReadAdrBase],
             (instregex "ST3Threev(16b|8h|4s|2d)_POST$")>;

// ASIMD store, 3 element, one lane, B/H
// ASIMD store, 3 element, one lane, S
// ASIMD store, 3 element, one lane, D
def : InstRW<[A78WriteSTV<5, 6, 4, 3>, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST3i(8|16|32|64)$")>;
def : InstRW<[WriteAdr, A78WriteSTV<5, 6, 4, 3>, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST3i(8|16|32|64)_POST$")>;

// ASIMD store, 4 element, multiple, D-form, B/H/S
def : InstRW<[A78WriteSTV<6, 6, 6, 3>, ReadST, ReadAdrBase],
             (instregex "ST4Fourv(8b|4h|2s)$")>;
def : InstRW<[WriteAdr, A78WriteSTV<6, 6, 6, 3>, ReadST, ReadAdrBase],
             (instregex "ST4Fourv(8b|4h|2s)_POST$")>;

// ASIMD store, 4 element, multiple, Q-form, B/H/S
def : InstRW<[A78WriteSTV<7, 12, 12, 6>, ReadST, ReadAdrBase],
             (instregex "ST4Fourv(16b|8h|4s)$")>;
def : InstRW<[WriteAdr, A78WriteSTV<7, 12, 12, 6>, ReadST, ReadAdrBase],
             (instregex "ST4Fourv(16b|8h|4s)_POST$")>;

// ASIMD store, 4 element, multiple, Q-form, D
def : InstRW<[A78WriteSTV<5, 8, 8, 4>, ReadST, ReadAdrBase],
             (instregex "ST4Fourv(2d)$")>;
def : InstRW<[WriteAdr, A78WriteSTV<5, 8, 8, 4>, ReadST, ReadAdrBase],
             (instregex "ST4Fourv(2d)_POST$")>;

// ASIMD store, 4 element, one lane, B/H/S
def : InstRW<[N2Write_6c_3L01_3V, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST4i(8|16|32)$")>;
def : InstRW<[WriteAdr, N2Write_6c_3L01_3V, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST4i(8|16|32)_POST$")>;

// ASIMD store, 4 element, one lane, D
def : InstRW<[A78WriteSTV<4, 6, 4, 3>, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST4i(64)$")>;
def : InstRW<[WriteAdr, A78WriteSTV<4, 6, 4, 3>, ReadST, ReadDefault, ReadAdrBase],
             (instregex "ST4i(64)_POST$")>;

// Cryptography extensions
// -----------------------------------------------------------------------------

// Crypto AES ops
def : InstRW<[N2Write_2c_1V], (instregex "^AES[DE]rr$", "^AESI?MCrr")>;

// Crypto polynomial (64x64) multiply long
def : InstRW<[N2Write_2c_1V0], (instrs PMULLv1i64, PMULLv2i64)>;

// Crypto SHA1 hash acceleration op
// Crypto SHA1 schedule acceleration ops
def : InstRW<[N2Write_2c_1V0], (instregex "^SHA1(H|SU0|SU1)")>;

// Crypto SHA1 hash acceleration ops
// Crypto SHA256 hash acceleration ops
def : InstRW<[N2Write_4c_1V0], (instregex "^SHA1[CMP]", "^SHA256H2?")>;

// Crypto SHA256 schedule acceleration ops
def : InstRW<[N2Write_2c_1V0], (instregex "^SHA256SU[01]")>;

// Crypto SHA512 hash acceleration ops
def : InstRW<[N2Write_2c_1V0], (instregex "^SHA512(H|H2|SU0|SU1)")>;

// Crypto SHA3 ops
def : InstRW<[N2Write_2c_1V0], (instrs BCAX, EOR3, RAX1, XAR)>;

// Crypto SM3 ops
def : InstRW<[N2Write_2c_1V0], (instregex "^SM3PARTW[12]$", "^SM3SS1$",
                                          "^SM3TT[12][AB]$")>;

// Crypto SM4 ops
def : InstRW<[N2Write_4c_1V0], (instrs SM4E, SM4ENCKEY)>;

// CRC
// -----------------------------------------------------------------------------

def : InstRW<[N2Wr_CRC, N2Rd_CRC], (instregex "^CRC32")>;

// SVE Predicate instructions
// -----------------------------------------------------------------------------

// Loop control, based on predicate
def : InstRW<[N2Write_2c_1M], (instrs BRKA_PPmP, BRKA_PPzP,
                                      BRKB_PPmP, BRKB_PPzP)>;

// Loop control, based on predicate and flag setting
def : InstRW<[N2Write_3c_1M], (instrs BRKAS_PPzP, BRKBS_PPzP)>;

// Loop control, propagating
def : InstRW<[N2Write_2c_1M0], (instrs BRKN_PPzP, BRKPA_PPzPP, BRKPB_PPzPP)>;

// Loop control, propagating and flag setting
def : InstRW<[N2Write_3c_1M0_1M], (instrs BRKNS_PPzP, BRKPAS_PPzPP,
                                          BRKPBS_PPzPP)>;

// Loop control, based on GPR
def : InstRW<[N2Write_3c_1M],
             (instregex "^WHILE(GE|GT|HI|HS|LE|LO|LS|LT)_P(WW|XX)_[BHSD]$")>;

def : InstRW<[N2Write_3c_1M], (instregex "^WHILE(RW|WR)_PXX_[BHSD]$")>;

// Loop terminate
def : InstRW<[N2Write_1c_1M], (instregex "^CTERM(EQ|NE)_(WW|XX)$")>;

// Predicate counting scalar
def : InstRW<[N2Write_2c_1M], (instrs ADDPL_XXI, ADDVL_XXI, RDVLI_XI)>;
def : InstRW<[N2Write_2c_1M],
             (instregex "^(CNT|DEC|INC|SQDEC|SQINC|UQDEC|UQINC)[BHWD]_XPiI$",
                        "^SQ(DEC|INC)[BHWD]_XPiWdI$",
                        "^(UQDEC|UQINC)[BHWD]_WPiI$")>;

// Predicate counting scalar, active predicate
def : InstRW<[N2Write_2c_1M],
             (instregex "^CNTP_XPP_[BHSD]$",
                        "^(DEC|INC|SQDEC|SQINC|UQDEC|UQINC)P_XP_[BHSD]$",
                        "^(UQDEC|UQINC)P_WP_[BHSD]$",
                        "^(SQDEC|SQINC|UQDEC|UQINC)P_XPWd_[BHSD]$")>;

// Predicate counting vector, active predicate
def : InstRW<[N2Write_7c_1M_1M0_1V],
             (instregex "^(DEC|INC|SQDEC|SQINC|UQDEC|UQINC)P_ZP_[HSD]$")>;

// Predicate logical
def : InstRW<[N2Write_1c_1M0],
             (instregex "^(AND|BIC|EOR|NAND|NOR|ORN|ORR)_PPzPP$")>;

// Predicate logical, flag setting
def : InstRW<[N2Write_2c_1M0_1M],
             (instregex "^(ANDS|BICS|EORS|NANDS|NORS|ORNS|ORRS)_PPzPP$")>;

// Predicate reverse
def : InstRW<[N2Write_2c_1M], (instregex "^REV_PP_[BHSD]$")>;

// Predicate select
def : InstRW<[N2Write_1c_1M0], (instrs SEL_PPPP)>;

// Predicate set
def : InstRW<[N2Write_2c_1M], (instregex "^PFALSE$", "^PTRUE_[BHSD]$")>;

// Predicate set/initialize, set flags
def : InstRW<[N2Write_3c_1M], (instregex "^PTRUES_[BHSD]$")>;

// Predicate find first/next
def : InstRW<[N2Write_3c_1M], (instregex "^PFIRST_B$", "^PNEXT_[BHSD]$")>;

// Predicate test
def : InstRW<[N2Write_1c_1M], (instrs PTEST_PP, PTEST_PP_ANY, PTEST_PP_FIRST)>;

// Predicate transpose
def : InstRW<[N2Write_2c_1M], (instregex "^TRN[12]_PPP_[BHSDQ]$")>;

// Predicate unpack and widen
def : InstRW<[N2Write_2c_1M], (instrs PUNPKHI_PP, PUNPKLO_PP)>;

// Predicate zip/unzip
def : InstRW<[N2Write_2c_1M], (instregex "^(ZIP|UZP)[12]_PPP_[BHSDQ]$")>;

// SVE integer instructions
// -----------------------------------------------------------------------------

// Arithmetic, absolute diff
def : InstRW<[N2Write_2c_1V], (instregex "^[SU]ABD_ZPmZ_[BHSD]",
                                         "^[SU]ABD_ZPZZ_[BHSD]")>;

// Arithmetic, absolute diff accum
def : InstRW<[N2Wr_ZA, N2Rd_ZA], (instregex "^[SU]ABA_ZZZ_[BHSD]$")>;

// Arithmetic, absolute diff accum long
def : InstRW<[N2Wr_ZA, N2Rd_ZA], (instregex "^[SU]ABAL[TB]_ZZZ_[HSD]$")>;

// Arithmetic, absolute diff long
def : InstRW<[N2Write_2c_1V], (instregex "^[SU]ABDL[TB]_ZZZ_[HSD]$")>;

// Arithmetic, basic
def : InstRW<[N2Write_2c_1V],
             (instregex "^(ABS|ADD|CNOT|NEG|SUB|SUBR)_ZPmZ_[BHSD]",
                        "^(ADD|SUB)_ZZZ_[BHSD]",
                        "^(ADD|SUB|SUBR)_ZPZZ_[BHSD]",
                        "^(ADD|SUB|SUBR)_ZI_[BHSD]",
                        "^ADR_[SU]XTW_ZZZ_D_[0123]",
                        "^ADR_LSL_ZZZ_[SD]_[0123]",
                        "^[SU](ADD|SUB)[LW][BT]_ZZZ_[HSD]",
                        "^SADDLBT_ZZZ_[HSD]",
                        "^[SU]H(ADD|SUB|SUBR)_(ZPmZ|ZPZZ)_[BHSD]",
                        "^SSUBL(BT|TB)_ZZZ_[HSD]")>;

// Arithmetic, complex
def : InstRW<[N2Write_2c_1V],
             (instregex "^R?(ADD|SUB)HN[BT]_ZZZ_[BHS]",
                        "^SQ(ABS|ADD|NEG|SUB|SUBR)_ZPmZ_[BHSD]",
                        "^[SU]Q(ADD|SUB)_ZZZ_[BHSD]",
                        "^[SU]Q(ADD|SUB)_ZI_[BHSD]",
                        "^(SRH|SUQ|UQ|USQ|URH)ADD_ZPmZ_[BHSD]",
                        "^(UQSUB|UQSUBR)_ZPmZ_[BHSD]")>;

// Arithmetic, large integer
def : InstRW<[N2Write_2c_1V], (instregex "^(AD|SB)CL[BT]_ZZZ_[SD]$")>;

// Arithmetic, pairwise add
def : InstRW<[N2Write_2c_1V], (instregex "^ADDP_ZPmZ_[BHSD]$")>;

// Arithmetic, pairwise add and accum long
def : InstRW<[N2Wr_ZPA, ReadDefault, N2Rd_ZPA], (instregex "^[SU]ADALP_ZPmZ_[HSD]$")>;

// Arithmetic, shift
def : InstRW<[N2Write_2c_1V1],
             (instregex "^(ASR|LSL|LSR)_WIDE_ZPmZ_[BHS]",
                        "^(ASR|LSL|LSR)_WIDE_ZZZ_[BHS]",
                        "^(ASR|LSL|LSR)_ZPmI_[BHSD]",
                        "^(ASR|LSL|LSR)_ZPmZ_[BHSD]",
                        "^(ASR|LSL|LSR)_ZZI_[BHSD]",
                        "^(ASR|LSL|LSR)_ZPZ[IZ]_[BHSD]",
                        "^(ASRR|LSLR|LSRR)_ZPmZ_[BHSD]")>;

// Arithmetic, shift and accumulate
def : InstRW<[N2Wr_ZSA, N2Rd_ZSA],
             (instregex "^(SRSRA|SSRA|URSRA|USRA)_ZZI_[BHSD]$")>;

// Arithmetic, shift by immediate
// Arithmetic, shift by immediate and insert
def : InstRW<[N2Write_2c_1V1],
             (instregex "^(SHRNB|SHRNT|SSHLLB|SSHLLT|USHLLB|USHLLT|SLI|SRI)_ZZI_[BHSD]$")>;

// Arithmetic, shift complex
def : InstRW<[N2Write_4c_1V1],
             (instregex "^(SQ)?RSHRU?N[BT]_ZZI_[BHS]",
                        "^(SQRSHL|SQRSHLR|SQSHL|SQSHLR|UQRSHL|UQRSHLR|UQSHL|UQSHLR)_ZPmZ_[BHSD]",
                        "^[SU]QR?SHL_ZPZZ_[BHSD]",
                        "^(SQSHL|SQSHLU|UQSHL)_(ZPmI|ZPZI)_[BHSD]",
                        "^SQSHRU?N[BT]_ZZI_[BHS]",
                        "^UQR?SHRN[BT]_ZZI_[BHS]")>;

// Arithmetic, shift right for divide
def : InstRW<[N2Write_4c_1V1], (instregex "^ASRD_(ZPmI|ZPZI)_[BHSD]")>;

// Arithmetic, shift rounding
def : InstRW<[N2Write_4c_1V1], (instregex "^[SU]RSHLR?_ZPmZ_[BHSD]",
                                          "^[SU]RSHL_ZPZZ_[BHSD]",
                                          "^[SU]RSHR_(ZPmI|ZPZI)_[BHSD]")>;

// Bit manipulation
def : InstRW<[N2Write_6c_2V1], (instregex "^(BDEP|BEXT|BGRP)_ZZZ_[BHSD]")>;

def : InstRW<[N2Write_2c_1V], (instregex "^(BSL|BSL1N|BSL2N|NBSL)_ZZZZ$")>;

// Count/reverse bits
def : InstRW<[N2Write_2c_1V], (instregex "^(CLS|CLZ|CNT|RBIT)_ZPmZ_[BHSD]")>;

// Broadcast logical bitmask immediate to vector
def : InstRW<[N2Write_2c_1V], (instrs DUPM_ZI)>;

// Compare and set flags
def : InstRW<[N2Write_4c_1V0_1M],
             (instregex "^CMP(EQ|GE|GT|HI|HS|LE|LO|LS|LT|NE)_PPzZ[IZ]_[BHSD]$",
                        "^CMP(EQ|GE|GT|HI|HS|LE|LO|LS|LT|NE)_WIDE_PPzZZ_[BHS]$")>;

// Complex add
def : InstRW<[N2Write_2c_1V], (instregex "^(SQ)?CADD_ZZI_[BHSD]$")>;

// Complex dot product 8-bit element
def : InstRW<[N2Wr_ZDOTB, N2Rd_ZDOTB], (instrs CDOT_ZZZ_S, CDOT_ZZZI_S)>;

// Complex dot product 16-bit element
def : InstRW<[N2Wr_ZDOTH, N2Rd_ZDOTH], (instrs CDOT_ZZZ_D, CDOT_ZZZI_D)>;

// Complex multiply-add B, H, S element size
def : InstRW<[N2Wr_ZCMABHS, N2Rd_ZCMABHS], (instregex "^CMLA_ZZZ_[BHS]$",
                                                      "^CMLA_ZZZI_[HS]$")>;

// Complex multiply-add D element size
def : InstRW<[N2Wr_ZCMAD, N2Rd_ZCMAD], (instrs CMLA_ZZZ_D)>;

// Conditional extract operations, scalar form
def : InstRW<[N2Write_8c_1M0_1V1_1V], (instregex "^CLAST[AB]_RPZ_[BHSD]$")>;

// Conditional extract operations, SIMD&FP scalar and vector forms
def : InstRW<[N2Write_3c_1V1], (instregex "^CLAST[AB]_[VZ]PZ_[BHSD]$",
                                          "^COMPACT_ZPZ_[SD]$",
                                          "^SPLICE_ZPZZ?_[BHSD]$")>;

// Convert to floating point, 64b to float or convert to double
def : InstRW<[N2Write_3c_1V0], (instregex "^[SU]CVTF_ZPmZ_Dto[HSD]",
                                          "^[SU]CVTF_ZPmZ_StoD")>;

// Convert to floating point, 32b to single or half
def : InstRW<[N2Write_4c_2V0], (instregex "^[SU]CVTF_ZPmZ_Sto[HS]")>;

// Convert to floating point, 16b to half
def : InstRW<[N2Write_6c_4V0], (instregex "^[SU]CVTF_ZPmZ_HtoH")>;

// Copy, scalar
def : InstRW<[N2Write_5c_1M0_1V], (instregex "^CPY_ZPmR_[BHSD]$")>;

// Copy, scalar SIMD&FP or imm
def : InstRW<[N2Write_2c_1V], (instregex "^CPY_ZPm[IV]_[BHSD]$",
                                         "^CPY_ZPzI_[BHSD]$")>;

// Divides, 32 bit
def : InstRW<[N2Write_12c_1V0], (instregex "^[SU]DIVR?_ZPmZ_S",
                                           "^[SU]DIV_ZPZZ_S")>;

// Divides, 64 bit
def : InstRW<[N2Write_20c_1V0], (instregex "^[SU]DIVR?_ZPmZ_D",
                                           "^[SU]DIV_ZPZZ_D")>;

// Dot product, 8 bit
def : InstRW<[N2Wr_ZDOTB, N2Rd_ZDOTB], (instregex "^[SU]DOT_ZZZI?_BtoS")>;

// Dot product, 8 bit, using signed and unsigned integers
def : InstRW<[N2Wr_ZDOTB, N2Rd_ZDOTB], (instrs SUDOT_ZZZI, USDOT_ZZZI, USDOT_ZZZ)>;

// Dot product, 16 bit
def : InstRW<[N2Wr_ZDOTH, N2Rd_ZDOTH], (instregex "^[SU]DOT_ZZZI?_HtoD$")>;

// Duplicate, immediate and indexed form
def : InstRW<[N2Write_2c_1V], (instregex "^DUP_ZI_[BHSD]$",
                                         "^DUP_ZZI_[BHSDQ]$")>;

// Duplicate, scalar form
def : InstRW<[N2Write_3c_1M0], (instregex "^DUP_ZR_[BHSD]$")>;

// Extend, sign or zero
def : InstRW<[N2Write_2c_1V1], (instregex "^[SU]XTB_ZPmZ_[HSD]",
                                          "^[SU]XTH_ZPmZ_[SD]",
                                          "^[SU]XTW_ZPmZ_[D]")>;

// Extract
def : InstRW<[N2Write_2c_1V], (instrs EXT_ZZI, EXT_ZZI_CONSTRUCTIVE, EXT_ZZI_B)>;

// Extract narrow saturating
def : InstRW<[N2Write_4c_1V1], (instregex "^[SU]QXTN[BT]_ZZ_[BHS]$",
                                          "^SQXTUN[BT]_ZZ_[BHS]$")>;

// Extract/insert operation, SIMD and FP scalar form
def : InstRW<[N2Write_3c_1V1], (instregex "^LAST[AB]_VPZ_[BHSD]$",
                                          "^INSR_ZV_[BHSD]$")>;

// Extract/insert operation, scalar
def : InstRW<[N2Write_5c_1V1_1M0], (instregex "^LAST[AB]_RPZ_[BHSD]$",
                                              "^INSR_ZR_[BHSD]$")>;

// Histogram operations
def : InstRW<[N2Write_2c_1V], (instregex "^HISTCNT_ZPzZZ_[SD]$",
                                         "^HISTSEG_ZZZ$")>;

// Horizontal operations, B, H, S form, immediate operands only
def : InstRW<[N2Write_4c_1V0], (instregex "^INDEX_II_[BHS]$")>;

// Horizontal operations, B, H, S form, scalar, immediate operands/ scalar
// operands only / immediate, scalar operands
def : InstRW<[N2Write_7c_1M0_1V0], (instregex "^INDEX_(IR|RI|RR)_[BHS]$")>;

// Horizontal operations, D form, immediate operands only
def : InstRW<[N2Write_5c_2V0], (instrs INDEX_II_D)>;

// Horizontal operations, D form, scalar, immediate operands)/ scalar operands
// only / immediate, scalar operands
def : InstRW<[N2Write_8c_2M0_2V0], (instregex "^INDEX_(IR|RI|RR)_D$")>;

// Logical
def : InstRW<[N2Write_2c_1V],
             (instregex "^(AND|EOR|ORR)_ZI",
                        "^(AND|BIC|EOR|ORR)_ZZZ",
                        "^EOR(BT|TB)_ZZZ_[BHSD]",
                        "^(AND|BIC|EOR|NOT|ORR)_(ZPmZ|ZPZZ)_[BHSD]",
                        "^NOT_ZPmZ_[BHSD]")>;

// Max/min, basic and pairwise
def : InstRW<[N2Write_2c_1V], (instregex "^[SU](MAX|MIN)_ZI_[BHSD]",
                                         "^[SU](MAX|MIN)P?_ZPmZ_[BHSD]",
                                         "^[SU](MAX|MIN)_ZPZZ_[BHSD]")>;

// Matching operations
def : InstRW<[N2Write_2c_1V0_1M], (instregex "^N?MATCH_PPzZZ_[BH]$")>;

// Matrix multiply-accumulate
def : InstRW<[N2Wr_ZMMA, N2Rd_ZMMA], (instrs SMMLA_ZZZ, UMMLA_ZZZ, USMMLA_ZZZ)>;

// Move prefix
def : InstRW<[N2Write_2c_1V], (instregex "^MOVPRFX_ZP[mz]Z_[BHSD]$",
                                         "^MOVPRFX_ZZ$")>;

// Multiply, B, H, S element size
def : InstRW<[N2Write_4c_1V0], (instregex "^MUL_(ZI|ZPmZ|ZZZI|ZZZ)_[BHS]",
                                          "^MUL_ZPZZ_[BHS]",
                                          "^[SU]MULH_(ZPmZ|ZZZ)_[BHS]",
                                          "^[SU]MULH_ZPZZ_[BHS]")>;

// Multiply, D element size
def : InstRW<[N2Write_5c_2V0], (instregex "^MUL_(ZI|ZPmZ|ZZZI|ZZZ)_D",
                                          "^MUL_ZPZZ_D",
                                          "^[SU]MULH_(ZPmZ|ZZZ)_D",
                                          "^[SU]MULH_ZPZZ_D")>;

// Multiply long
def : InstRW<[N2Write_4c_1V0], (instregex "^[SU]MULL[BT]_ZZZI_[SD]$",
                                          "^[SU]MULL[BT]_ZZZ_[HSD]$")>;

// Multiply accumulate, B, H, S element size
def : InstRW<[N2Wr_ZMABHS, N2Rd_ZMABHS], (instregex "^ML[AS]_ZZZI_[HS]$",
                                                    "^ML[AS]_ZPZZZ_[BHS]")>;
def : InstRW<[N2Wr_ZMABHS, ReadDefault, N2Rd_ZMABHS], (instregex "^(ML[AS]|MAD|MSB)_ZPmZZ_[BHS]$")>;

// Multiply accumulate, D element size
def : InstRW<[N2Wr_ZMAD, N2Rd_ZMAD], (instregex "^ML[AS]_ZZZI_D$",
                                                "^ML[AS]_ZPZZZ_D")>;
def : InstRW<[N2Wr_ZMAD, ReadDefault, N2Rd_ZMAD],
             (instregex "^(ML[AS]|MAD|MSB)_ZPmZZ_D$")>;

// Multiply accumulate long
def : InstRW<[N2Wr_ZMAL, N2Rd_ZMAL], (instregex "^[SU]ML[AS]L[BT]_ZZZ_[HSD]$",
                                                "^[SU]ML[AS]L[BT]_ZZZI_[SD]$")>;

// Multiply accumulate saturating doubling long regular
def : InstRW<[N2Wr_ZMASQL, N2Rd_ZMASQ], (instregex "^SQDML[AS](LB|LT|LBT)_ZZZ_[HSD]$",
                                                   "^SQDML[AS](LB|LT)_ZZZI_[SD]$")>;

// Multiply saturating doubling high, B, H, S element size
def : InstRW<[N2Write_4c_1V0], (instregex "^SQDMULH_ZZZ_[BHS]$",
                                          "^SQDMULH_ZZZI_[HS]$")>;

// Multiply saturating doubling high, D element size
def : InstRW<[N2Write_5c_2V0], (instrs SQDMULH_ZZZ_D, SQDMULH_ZZZI_D)>;

// Multiply saturating doubling long
def : InstRW<[N2Write_4c_1V0], (instregex "^SQDMULL[BT]_ZZZ_[HSD]$",
                                          "^SQDMULL[BT]_ZZZI_[SD]$")>;

// Multiply saturating rounding doubling regular/complex accumulate, B, H, S
// element size
def : InstRW<[N2Wr_ZMASQBHS, N2Rd_ZMASQ], (instregex "^SQRDML[AS]H_ZZZ_[BHS]$",
                                                     "^SQRDCMLAH_ZZZ_[BHS]$",
                                                     "^SQRDML[AS]H_ZZZI_[HS]$",
                                                     "^SQRDCMLAH_ZZZI_[HS]$")>;

// Multiply saturating rounding doubling regular/complex accumulate, D element
// size
def : InstRW<[N2Wr_ZMASQD, N2Rd_ZMASQ], (instregex "^SQRDML[AS]H_ZZZI?_D$",
                                                   "^SQRDCMLAH_ZZZ_D$")>;

// Multiply saturating rounding doubling regular/complex, B, H, S element size
def : InstRW<[N2Write_4c_1V0], (instregex "^SQRDMULH_ZZZ_[BHS]$",
                                          "^SQRDMULH_ZZZI_[HS]$")>;

// Multiply saturating rounding doubling regular/complex, D element size
def : InstRW<[N2Write_5c_2V0], (instregex "^SQRDMULH_ZZZI?_D$")>;

// Multiply/multiply long, (8x8) polynomial
def : InstRW<[N2Write_2c_1V0], (instregex "^PMUL_ZZZ_B$",
                                          "^PMULL[BT]_ZZZ_[HDQ]$")>;

// Predicate counting vector
def : InstRW<[N2Write_2c_1V0],
             (instregex "^(DEC|INC|SQDEC|SQINC|UQDEC|UQINC)[HWD]_ZPiI$")>;

// Reciprocal estimate
def : InstRW<[N2Write_4c_2V0], (instregex "^URECPE_ZPmZ_S", "^URSQRTE_ZPmZ_S")>;

// Reduction, arithmetic, B form
def : InstRW<[N2Write_11c_2V_2V1], (instregex "^[SU](ADD|MAX|MIN)V_VPZ_B")>;

// Reduction, arithmetic, H form
def : InstRW<[N2Write_9c_2V_2V1], (instregex "^[SU](ADD|MAX|MIN)V_VPZ_H")>;

// Reduction, arithmetic, S form
def : InstRW<[N2Write_8c_2V_2V1], (instregex "^[SU](ADD|MAX|MIN)V_VPZ_S")>;

// Reduction, arithmetic, D form
def : InstRW<[N2Write_8c_2V_2V1], (instregex "^[SU](ADD|MAX|MIN)V_VPZ_D")>;

// Reduction, logical
def : InstRW<[N2Write_6c_1V_1V1], (instregex "^(ANDV|EORV|ORV)_VPZ_[BHSD]$")>;

// Reverse, vector
def : InstRW<[N2Write_2c_1V], (instregex "^REV_ZZ_[BHSD]$",
                                         "^REVB_ZPmZ_[HSD]$",
                                         "^REVH_ZPmZ_[SD]$",
                                         "^REVW_ZPmZ_D$")>;

// Select, vector form
def : InstRW<[N2Write_2c_1V], (instregex "^SEL_ZPZZ_[BHSD]$")>;

// Table lookup
def : InstRW<[N2Write_2c_1V], (instregex "^TBL_ZZZZ?_[BHSD]$")>;

// Table lookup extension
def : InstRW<[N2Write_2c_1V], (instregex "^TBX_ZZZ_[BHSD]$")>;

// Transpose, vector form
def : InstRW<[N2Write_2c_1V], (instregex "^TRN[12]_ZZZ_[BHSDQ]$")>;

// Unpack and extend
def : InstRW<[N2Write_2c_1V], (instregex "^[SU]UNPK(HI|LO)_ZZ_[HSD]$")>;

// Zip/unzip
def : InstRW<[N2Write_2c_1V], (instregex "^(UZP|ZIP)[12]_ZZZ_[BHSDQ]$")>;

// SVE floating-point instructions
// -----------------------------------------------------------------------------

// Floating point absolute value/difference
def : InstRW<[N2Write_2c_1V], (instregex "^FAB[SD]_ZPmZ_[HSD]",
                                         "^FABD_ZPZZ_[HSD]",
                                         "^FABS_ZPmZ_[HSD]")>;

// Floating point arithmetic
def : InstRW<[N2Write_2c_1V], (instregex "^F(ADD|SUB)_(ZPm[IZ]|ZZZ)_[HSD]",
                                         "^F(ADD|SUB)_ZPZ[IZ]_[HSD]",
                                         "^FADDP_ZPmZZ_[HSD]",
                                         "^FNEG_ZPmZ_[HSD]",
                                         "^FSUBR_ZPm[IZ]_[HSD]",
                                         "^FSUBR_(ZPZI|ZPZZ)_[HSD]")>;

// Floating point associative add, F16
def : InstRW<[N2Write_10c_1V1], (instrs FADDA_VPZ_H)>;

// Floating point associative add, F32
def : InstRW<[N2Write_6c_1V1], (instrs FADDA_VPZ_S)>;

// Floating point associative add, F64
def : InstRW<[N2Write_4c_1V], (instrs FADDA_VPZ_D)>;

// Floating point compare
def : InstRW<[N2Write_2c_1V0], (instregex "^FACG[ET]_PPzZZ_[HSD]$",
                                          "^FCM(EQ|GE|GT|NE)_PPzZ[0Z]_[HSD]$",
                                          "^FCM(LE|LT)_PPzZ0_[HSD]$",
                                          "^FCMUO_PPzZZ_[HSD]$")>;

// Floating point complex add
def : InstRW<[N2Write_3c_1V], (instregex "^FCADD_ZPmZ_[HSD]$")>;

// Floating point complex multiply add
def : InstRW<[N2Wr_ZFCMA, ReadDefault, N2Rd_ZFCMA], (instregex "^FCMLA_ZPmZZ_[HSD]$")>;
def : InstRW<[N2Wr_ZFCMA, N2Rd_ZFCMA], (instregex "^FCMLA_ZZZI_[HS]$")>;

// Floating point convert, long or narrow (F16 to F32 or F32 to F16)
def : InstRW<[N2Write_4c_2V0], (instregex "^FCVT_ZPmZ_(HtoS|StoH)",
                                          "^FCVTLT_ZPmZ_HtoS",
                                          "^FCVTNT_ZPmZ_StoH")>;

// Floating point convert, long or narrow (F16 to F64, F32 to F64, F64 to F32
// or F64 to F16)
def : InstRW<[N2Write_3c_1V0], (instregex "^FCVT_ZPmZ_(HtoD|StoD|DtoS|DtoH)",
                                          "^FCVTLT_ZPmZ_StoD",
                                          "^FCVTNT_ZPmZ_DtoS")>;

// Floating point convert, round to odd
def : InstRW<[N2Write_3c_1V0], (instrs FCVTX_ZPmZ_DtoS, FCVTXNT_ZPmZ_DtoS)>;

// Floating point base2 log, F16
def : InstRW<[N2Write_6c_4V0], (instregex "^FLOGB_(ZPmZ|ZPZZ)_H")>;

// Floating point base2 log, F32
def : InstRW<[N2Write_4c_2V0], (instregex "^FLOGB_(ZPmZ|ZPZZ)_S")>;

// Floating point base2 log, F64
def : InstRW<[N2Write_3c_1V0], (instregex "^FLOGB_(ZPmZ|ZPZZ)_D")>;

// Floating point convert to integer, F16
def : InstRW<[N2Write_6c_4V0], (instregex "^FCVTZ[SU]_ZPmZ_HtoH")>;

// Floating point convert to integer, F32
def : InstRW<[N2Write_4c_2V0], (instregex "^FCVTZ[SU]_ZPmZ_(HtoS|StoS)")>;

// Floating point convert to integer, F64
def : InstRW<[N2Write_3c_1V0],
             (instregex "^FCVTZ[SU]_ZPmZ_(HtoD|StoD|DtoS|DtoD)")>;

// Floating point copy
def : InstRW<[N2Write_2c_1V], (instregex "^FCPY_ZPmI_[HSD]$",
                                         "^FDUP_ZI_[HSD]$")>;

// Floating point divide, F16
def : InstRW<[N2Write_13c_1V0], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_H")>;

// Floating point divide, F32
def : InstRW<[N2Write_10c_1V0], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_S")>;

// Floating point divide, F64
def : InstRW<[N2Write_15c_1V0], (instregex "^FDIVR?_(ZPmZ|ZPZZ)_D")>;

// Floating point min/max pairwise
def : InstRW<[N2Write_2c_1V], (instregex "^F(MAX|MIN)(NM)?P_ZPmZZ_[HSD]")>;

// Floating point min/max
def : InstRW<[N2Write_2c_1V], (instregex "^F(MAX|MIN)(NM)?_ZPm[IZ]_[HSD]",
                                         "^F(MAX|MIN)(NM)?_ZPZ[IZ]_[HSD]")>;

// Floating point multiply
def : InstRW<[N2Write_3c_1V], (instregex "^(FSCALE|FMULX)_ZPmZ_[HSD]",
                                         "^FMULX_ZPZZ_[HSD]",
                                         "^FMUL_(ZPm[IZ]|ZZZI?)_[HSD]",
                                         "^FMUL_ZPZ[IZ]_[HSD]")>;

// Floating point multiply accumulate
def : InstRW<[N2Wr_ZFMA, ReadDefault, N2Rd_ZFMA], (instregex "^FN?ML[AS]_ZPmZZ_[HSD]",
                                                             "^FN?(MAD|MSB)_ZPmZZ_[HSD]")>;
def : InstRW<[N2Wr_ZFMA, N2Rd_ZFMA], (instregex "^FML[AS]_ZZZI_[HSD]$",
                                                "^FN?ML[AS]_ZPZZZ_[HSD]")>;

// Floating point multiply add/sub accumulate long
def : InstRW<[N2Wr_ZFMAL, N2Rd_ZFMAL], (instregex "^FML[AS]L[BT]_ZZZI?_SHH$")>;

// Floating point reciprocal estimate, F16
def : InstRW<[N2Write_6c_4V0], (instregex "^FR(ECP|SQRT)E_ZZ_H", "^FRECPX_ZPmZ_H")>;

// Floating point reciprocal estimate, F32
def : InstRW<[N2Write_4c_2V0], (instregex "^FR(ECP|SQRT)E_ZZ_S", "^FRECPX_ZPmZ_S")>;

// Floating point reciprocal estimate, F64
def : InstRW<[N2Write_3c_1V0], (instregex "^FR(ECP|SQRT)E_ZZ_D", "^FRECPX_ZPmZ_D")>;

// Floating point reciprocal step
def : InstRW<[N2Write_4c_1V0], (instregex "^F(RECPS|RSQRTS)_ZZZ_[HSD]$")>;

// Floating point reduction, F16
def : InstRW<[N2Write_6c_2V],
             (instregex "^(FADDV|FMAXNMV|FMAXV|FMINNMV|FMINV)_VPZ_H$")>;

// Floating point reduction, F32
def : InstRW<[N2Write_4c_1V],
             (instregex "^(FADDV|FMAXNMV|FMAXV|FMINNMV|FMINV)_VPZ_S$")>;

// Floating point reduction, F64
def : InstRW<[N2Write_2c_1V],
             (instregex "^(FADDV|FMAXNMV|FMAXV|FMINNMV|FMINV)_VPZ_D$")>;

// Floating point round to integral, F16
def : InstRW<[N2Write_6c_4V0], (instregex "^FRINT[AIMNPXZ]_ZPmZ_H")>;

// Floating point round to integral, F32
def : InstRW<[N2Write_4c_2V0], (instregex "^FRINT[AIMNPXZ]_ZPmZ_S")>;

// Floating point round to integral, F64
def : InstRW<[N2Write_3c_1V0], (instregex "^FRINT[AIMNPXZ]_ZPmZ_D")>;

// Floating point square root, F16
def : InstRW<[N2Write_13c_1V0], (instregex "^FSQRT_ZPmZ_H")>;

// Floating point square root, F32
def : InstRW<[N2Write_10c_1V0], (instregex "^FSQRT_ZPmZ_S")>;

// Floating point square root, F64
def : InstRW<[N2Write_16c_1V0], (instregex "^FSQRT_ZPmZ_D")>;

// Floating point trigonometric exponentiation
def : InstRW<[N2Write_3c_1V1], (instregex "^FEXPA_ZZ_[HSD]$")>;

// Floating point trigonometric multiply add
def : InstRW<[N2Write_4c_1V], (instregex "^FTMAD_ZZI_[HSD]$")>;

// Floating point trigonometric, miscellaneous
def : InstRW<[N2Write_3c_1V], (instregex "^FTS(MUL|SEL)_ZZZ_[HSD]$")>;

// SVE BFloat16 (BF16) instructions
// -----------------------------------------------------------------------------

// Convert, F32 to BF16
def : InstRW<[N2Write_3c_1V0], (instrs BFCVT_ZPmZ, BFCVTNT_ZPmZ)>;

// Dot product
def : InstRW<[N2Wr_ZBFDOT, N2Rd_ZBFDOT], (instrs BFDOT_ZZI, BFDOT_ZZZ)>;

// Matrix multiply accumulate
def : InstRW<[N2Wr_ZBFMMA, N2Rd_ZBFMMA], (instrs BFMMLA_ZZZ_HtoS)>;

// Multiply accumulate long
def : InstRW<[N2Wr_ZBFMAL, N2Rd_ZBFMAL], (instregex "^BFMLAL[BT]_ZZZ(I)?$")>;

// SVE Load instructions
// -----------------------------------------------------------------------------

// Load vector
def : InstRW<[N2Write_6c_1L], (instrs LDR_ZXI)>;

// Load predicate
def : InstRW<[N2Write_6c_1L_1M], (instrs LDR_PXI)>;

// Contiguous load, scalar + imm
def : InstRW<[N2Write_6c_1L], (instregex "^LD1[BHWD]_IMM$",
                                         "^LD1S?B_[HSD]_IMM$",
                                         "^LD1S?H_[SD]_IMM$",
                                         "^LD1S?W_D_IMM$" )>;
// Contiguous load, scalar + scalar
def : InstRW<[N2Write_6c_1L01], (instregex "^LD1[BHWD]$",
                                           "^LD1S?B_[HSD]$",
                                           "^LD1S?H_[SD]$",
                                           "^LD1S?W_D$" )>;

// Contiguous load broadcast, scalar + imm
def : InstRW<[N2Write_6c_1L], (instregex "^LD1R[BHWD]_IMM$",
                                         "^LD1RSW_IMM$",
                                         "^LD1RS?B_[HSD]_IMM$",
                                         "^LD1RS?H_[SD]_IMM$",
                                         "^LD1RS?W_D_IMM$",
                                         "^LD1RQ_[BHWD]_IMM$")>;

// Contiguous load broadcast, scalar + scalar
def : InstRW<[N2Write_6c_1L], (instregex "^LD1RQ_[BHWD]$")>;

// Non temporal load, scalar + imm
def : InstRW<[N2Write_6c_1L], (instregex "^LDNT1[BHWD]_ZRI$")>;

// Non temporal load, scalar + scalar
def : InstRW<[N2Write_6c_1L_1S], (instregex "^LDNT1[BHWD]_ZRR$")>;

// Non temporal gather load, vector + scalar 32-bit element size
def : InstRW<[N2Write_9c_1L_1V], (instregex "^LDNT1[BHW]_ZZR_S$",
                                            "^LDNT1S[BH]_ZZR_S$")>;

// Non temporal gather load, vector + scalar 64-bit element size
def : InstRW<[N2Write_10c_2L_2V1], (instregex "^LDNT1S?[BHW]_ZZR_D$")>;
def : InstRW<[N2Write_10c_2L_2V1], (instrs LDNT1D_ZZR_D)>;

// Contiguous first faulting load, scalar + scalar
def : InstRW<[N2Write_6c_1L_1S], (instregex "^LDFF1[BHWD]$",
                                            "^LDFF1S?B_[HSD]$",
                                            "^LDFF1S?H_[SD]$",
                                            "^LDFF1S?W_D$")>;

// Contiguous non faulting load, scalar + imm
def : InstRW<[N2Write_6c_1L], (instregex "^LDNF1[BHWD]_IMM$",
                                         "^LDNF1S?B_[HSD]_IMM$",
                                         "^LDNF1S?H_[SD]_IMM$",
                                         "^LDNF1S?W_D_IMM$")>;

// Contiguous Load two structures to two vectors, scalar + imm
def : InstRW<[N2Write_8c_1L_1V], (instregex "^LD2[BHWD]_IMM$")>;

// Contiguous Load two structures to two vectors, scalar + scalar
def : InstRW<[N2Write_9c_1L_1V], (instregex "^LD2[BHWD]$")>;

// Contiguous Load three structures to three vectors, scalar + imm
def : InstRW<[N2Write_9c_1L_1V], (instregex "^LD3[BHWD]_IMM$")>;

// Contiguous Load three structures to three vectors, scalar + scalar
def : InstRW<[N2Write_10c_1V_1L_1S], (instregex "^LD3[BHWD]$")>;

// Contiguous Load four structures to four vectors, scalar + imm
def : InstRW<[N2Write_9c_2L_2V], (instregex "^LD4[BHWD]_IMM$")>;

// Contiguous Load four structures to four vectors, scalar + scalar
def : InstRW<[N2Write_10c_2L_2V_2S], (instregex "^LD4[BHWD]$")>;

// Gather load, vector + imm, 32-bit element size
def : InstRW<[N2Write_9c_1L_1V], (instregex "^GLD(FF)?1S?[BH]_S_IMM$",
                                            "^GLD(FF)?1W_IMM$")>;

// Gather load, vector + imm, 64-bit element size
def : InstRW<[N2Write_9c_2L_2V], (instregex "^GLD(FF)?1S?[BHW]_D_IMM$",
                                            "^GLD(FF)?1D_IMM$")>;

// Gather load, 64-bit element size
def : InstRW<[N2Write_9c_2L_2V],
             (instregex "^GLD(FF)?1S?[BHW]_D_[SU]XTW(_SCALED)?$",
                        "^GLD(FF)?1S?[BHW]_D(_SCALED)?$",
                        "^GLD(FF)?1D_[SU]XTW(_SCALED)?$",
                        "^GLD(FF)?1D(_SCALED)?$")>;

// Gather load, 32-bit scaled offset
def : InstRW<[N2Write_10c_2L_2V],
             (instregex "^GLD(FF)?1S?[HW]_S_[SU]XTW_SCALED$",
                        "^GLD(FF)?1W_[SU]XTW_SCALED")>;

// Gather load, 32-bit unpacked unscaled offset
def : InstRW<[N2Write_9c_1L_1V], (instregex "^GLD(FF)?1S?[BH]_S_[SU]XTW$",
                                            "^GLD(FF)?1W_[SU]XTW$")>;

// SVE Store instructions
// -----------------------------------------------------------------------------

// Store from predicate reg
def : InstRW<[N2Write_1c_1L01], (instrs STR_PXI)>;

// Store from vector reg
def : InstRW<[N2Write_2c_1L01_1V], (instrs STR_ZXI)>;

// Contiguous store, scalar + imm
def : InstRW<[N2Write_2c_1L01_1V], (instregex "^ST1[BHWD]_IMM$",
                                              "^ST1B_[HSD]_IMM$",
                                              "^ST1H_[SD]_IMM$",
                                              "^ST1W_D_IMM$")>;

// Contiguous store, scalar + scalar
def : InstRW<[N2Write_2c_1L01_1S_1V], (instregex "^ST1H(_[SD])?$")>;
def : InstRW<[N2Write_2c_1L01_1V], (instregex "^ST1[BWD]$",
                                              "^ST1B_[HSD]$",
                                              "^ST1W_D$")>;

// Contiguous store two structures from two vectors, scalar + imm
def : InstRW<[N2Write_4c_1L01_1V], (instregex "^ST2[BHWD]_IMM$")>;

// Contiguous store two structures from two vectors, scalar + scalar
def : InstRW<[N2Write_4c_1L01_1S_1V], (instrs ST2H)>;

// Contiguous store two structures from two vectors, scalar + scalar
def : InstRW<[N2Write_4c_1L01_1V], (instregex "^ST2[BWD]$")>;

// Contiguous store three structures from three vectors, scalar + imm
def : InstRW<[N2Write_7c_5L01_5V], (instregex "^ST3[BHWD]_IMM$")>;

// Contiguous store three structures from three vectors, scalar + scalar
def : InstRW<[N2Write_7c_5L01_5S_5V], (instrs ST3H)>;

// Contiguous store three structures from three vectors, scalar + scalar
def : InstRW<[N2Write_7c_5L01_5S_5V], (instregex "^ST3[BWD]$")>;

// Contiguous store four structures from four vectors, scalar + imm
def : InstRW<[N2Write_11c_9L01_9V], (instregex "^ST4[BHWD]_IMM$")>;

// Contiguous store four structures from four vectors, scalar + scalar
def : InstRW<[N2Write_11c_9L01_9S_9V], (instrs ST4H)>;

// Contiguous store four structures from four vectors, scalar + scalar
def : InstRW<[N2Write_11c_9L01_9S_9V], (instregex "^ST4[BWD]$")>;

// Non temporal store, scalar + imm
def : InstRW<[N2Write_2c_1L01_1V], (instregex "^STNT1[BHWD]_ZRI$")>;

// Non temporal store, scalar + scalar
def : InstRW<[N2Write_2c_1L01_1S_1V], (instrs STNT1H_ZRR)>;
def : InstRW<[N2Write_2c_1L01_1V], (instregex "^STNT1[BWD]_ZRR$")>;

// Scatter non temporal store, vector + scalar 32-bit element size
def : InstRW<[N2Write_4c_2L01_2V], (instregex "^STNT1[BHW]_ZZR_S")>;

// Scatter non temporal store, vector + scalar 64-bit element size
def : InstRW<[N2Write_2c_1L01_1V], (instregex "^STNT1[BHWD]_ZZR_D")>;

// Scatter store vector + imm 32-bit element size
def : InstRW<[N2Write_4c_2L01_2V], (instregex "^SST1[BH]_S_IMM$",
                                              "^SST1W_IMM$")>;

// Scatter store vector + imm 64-bit element size
def : InstRW<[N2Write_2c_1L01_1V], (instregex "^SST1[BHW]_D_IMM$",
                                              "^SST1D_IMM$")>;

// Scatter store, 32-bit scaled offset
def : InstRW<[N2Write_4c_2L01_2V],
             (instregex "^SST1(H_S|W)_[SU]XTW_SCALED$")>;

// Scatter store, 32-bit unpacked unscaled offset
def : InstRW<[N2Write_2c_1L01_1V], (instregex "^SST1[BHW]_D_[SU]XTW$",
                                              "^SST1D_[SU]XTW$")>;

// Scatter store, 32-bit unpacked scaled offset
def : InstRW<[N2Write_2c_1L01_1V], (instregex "^SST1[HW]_D_[SU]XTW_SCALED$",
                                              "^SST1D_[SU]XTW_SCALED$")>;

// Scatter store, 32-bit unscaled offset
def : InstRW<[N2Write_4c_2L01_2V], (instregex "^SST1[BH]_S_[SU]XTW$",
                                              "^SST1W_[SU]XTW$")>;

// Scatter store, 64-bit scaled offset
def : InstRW<[N2Write_2c_1L01_1V], (instregex "^SST1[HW]_D_SCALED$",
                                              "^SST1D_SCALED$")>;

// Scatter store, 64-bit unscaled offset
def : InstRW<[N2Write_2c_1L01_1V], (instregex "^SST1[BHW]_D$",
                                              "^SST1D$")>;

// SVE Miscellaneous instructions
// -----------------------------------------------------------------------------

// Read first fault register, unpredicated
def : InstRW<[N2Write_2c_1M0], (instrs RDFFR_P)>;

// Read first fault register, predicated
def : InstRW<[N2Write_3c_1M0_1M], (instrs RDFFR_PPz)>;

// Read first fault register and set flags
def : InstRW<[N2Write_4c_2M0_2M], (instrs RDFFRS_PPz)>;

// Set first fault register
// Write to first fault register
def : InstRW<[N2Write_2c_1M0], (instrs SETFFR, WRFFR)>;

// Prefetch
def : InstRW<[N2Write_4c_1L], (instregex "^PRF[BHWD]")>;

// SVE Cryptographic instructions
// -----------------------------------------------------------------------------

// Crypto AES ops
def : InstRW<[N2Write_2c_1V], (instregex "^AES[DE]_ZZZ_B$",
                                         "^AESI?MC_ZZ_B$")>;

// Crypto SHA3 ops
def : InstRW<[N2Write_2c_1V0], (instregex "^(BCAX|EOR3)_ZZZZ$",
                                          "^RAX1_ZZZ_D$",
                                          "^XAR_ZZZI_[BHSD]$")>;

// Crypto SM4 ops
def : InstRW<[N2Write_4c_1V0], (instregex "^SM4E(KEY)?_ZZZ_S$")>;

}
