//=- AArch64SchedFirestorm.td - Apple Firestorm Scheduling Definitions -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Apple Firestorm (M1 P-Core).
//
// Reference & Source Attribution:
//   Apple Microarchitecture Research by Dougall Johnson
//   https://dougallj.github.io/applecpu/firestorm.html
//   https://dougallj.github.io/applecpu/firestorm-int.html
//   https://dougallj.github.io/applecpu/firestorm-simd.html
//
//===----------------------------------------------------------------------===//

def FirestormModel : SchedMachineModel {
  let IssueWidth = 8;           // 8 MOPs per cycle decode / dispatch
  let MicroOpBufferSize = 330;  // Coalesced Retire Queue (ROB) = 330 entries
  let LoadLatency = 3;          // L1 hit latency (3 cycles measured by Dougall Johnson)
  let MispredictPenalty = 16;
  let CompleteModel = 0;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC]);
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available on Firestorm.

// 6 INT Schedulers (BufferSizes: 24, 26, 16, 12, 28, 28)
def FireUnitI0 : ProcResource<1> { let BufferSize = 24; } // u1: ALU, Branch
def FireUnitI1 : ProcResource<1> { let BufferSize = 26; } // u2: ALU, Branch, IndBranch, PtAuth
def FireUnitI2 : ProcResource<1> { let BufferSize = 16; } // u3: ALU, FP->GPR
def FireUnitI3 : ProcResource<1> { let BufferSize = 12; } // u4: ALU, FP->GPR
def FireUnitI4 : ProcResource<1> { let BufferSize = 28; } // u5: ALU, MUL, DIV
def FireUnitI5 : ProcResource<1> { let BufferSize = 28; } // u6: ALU, MUL, MADD, CRC
def FireUnitI  : ProcResource<6> { let BufferSize = 134; }

// Multiply execution ports (u5, u6 = FireUnitI4, FireUnitI5)
def FireUnitIMul : ProcResource<2> { let BufferSize = 56; }

// Branch execution ports (u1, u2 = FireUnitI0, FireUnitI1)
def FireUnitBranch : ProcResource<2> { let BufferSize = 50; }

// 4 FP/Vector Schedulers (u11-u14, each 36 entries)
def FireUnitFP0 : ProcResource<1> { let BufferSize = 36; } // u11: FP/SIMD
def FireUnitFP1 : ProcResource<1> { let BufferSize = 36; } // u12: FP/SIMD
def FireUnitFP2 : ProcResource<1> { let BufferSize = 36; } // u13: FP/SIMD, FCSEL, to-GPR
def FireUnitFP3 : ProcResource<1> { let BufferSize = 36; } // u14: FP/SIMD, FCSEL, FDIV, FSQRT
def FireUnitFP  : ProcResource<4> { let BufferSize = 144; }

// MEM Scheduler (u7-u10, 28 entries)
def FireUnitMem : ProcResource<7> { let BufferSize = 28; } // u7-u10: Store x2 (u7, u8), Load x3 (u8, u9, u10)

def FireLoadBuffer  : ProcResource<3> { let BufferSize = 130; } // 3 loads per cycle (u8, u9, u10)
def FireStoreBuffer : ProcResource<2> { let BufferSize = 60; }  // 2 stores per cycle (u7, u8)

//===----------------------------------------------------------------------===//
// Target-wide SchedWrites mappings for Firestorm Model

let SchedModel = FirestormModel in {

// Basic GPR ALU: 1 cycle latency, 6 per cycle (u1-u6)
def : WriteRes<WriteImm, []> { let Latency = 0; }
def : WriteRes<WriteI, [FireUnitI]> { let Latency = 1; }
// Shifted/Extended GPR ALU: 2 cycles latency, 2 issue ops across u1-u6
def : WriteRes<WriteISReg, [FireUnitI, FireUnitI]> { let Latency = 2; }
def : WriteRes<WriteIEReg, [FireUnitI, FireUnitI]> { let Latency = 2; }
def : WriteRes<WriteExtr, [FireUnitI]> { let Latency = 1; }
def : WriteRes<WriteIS, [FireUnitI]> { let Latency = 1; }

// Integer Multiply: 3 cycles latency (u5, u6)
def : WriteRes<WriteIM32, [FireUnitIMul]> { let Latency = 3; }
def : WriteRes<WriteIM64, [FireUnitIMul]> { let Latency = 3; }

// Integer Divide (Empirical data from Dougall Johnson: u5, Reciprocal Throughput = 2 cycles)
// 32-bit SDIV/UDIV: 7-8 cycles latency (slow form 8c)
def : WriteRes<WriteID32, [FireUnitI4]> { let Latency = 8; }
// 64-bit SDIV/UDIV: 7-9 cycles latency (slow form 9c)
def : WriteRes<WriteID64, [FireUnitI4]> { let Latency = 9; }

// Branch: 1 cycle latency (u1, u2)
def : WriteRes<WriteBr, [FireUnitBranch]> { let Latency = 1; }
def : WriteRes<WriteBrReg, [FireUnitBranch]> { let Latency = 1; }

// GPR Loads: 3 cycles latency (L1 hit measured by Dougall Johnson)
def : WriteRes<WriteLD, [FireUnitMem, FireLoadBuffer]> { let Latency = 3; }
def : WriteRes<WriteLDIdx, [FireUnitMem, FireLoadBuffer]> { let Latency = 3; }
def : WriteRes<WriteLDHi, [FireUnitMem, FireLoadBuffer]> { let Latency = 3; }

// GPR Stores: 1 cycle latency (u7, u8)
def : WriteRes<WriteST, [FireUnitMem, FireStoreBuffer]> { let Latency = 1; }
def : WriteRes<WriteSTP, [FireUnitMem, FireStoreBuffer]> { let Latency = 1; }
def : WriteRes<WriteSTIdx, [FireUnitMem, FireStoreBuffer]> { let Latency = 1; }

def : WriteRes<WriteAdr, [FireUnitI]> { let Latency = 1; }
def : WriteRes<WriteSys, [FireUnitI0]> { let Latency = 10; }
def : WriteRes<WriteBarrier, [FireUnitMem]> { let Latency = 10; }
def : WriteRes<WriteHint, [FireUnitI0]> { let Latency = 1; }

// Floating-Point Operations (4 ports u11-u14)
def : WriteRes<WriteF, [FireUnitFP]> { let Latency = 2; }
def : WriteRes<WriteFCmp, [FireUnitFP2]> { let Latency = 1; }
def : WriteRes<WriteFCvt, [FireUnitFP]> { let Latency = 3; }
def : WriteRes<WriteFCopy, [FireUnitFP]> { let Latency = 1; }
def : WriteRes<WriteFImm, [FireUnitFP]> { let Latency = 1; }
def : WriteRes<WriteFMul, [FireUnitFP]> { let Latency = 3; }

// Floating-Point Divide (Empirical data from Dougall Johnson: u14 port)
// Single precision FDIVSrr: 8 cycles latency
def : WriteRes<WriteFDiv, [FireUnitFP3]> { let Latency = 8; }

// Vector Operations (4 ports u11-u14)
def : WriteRes<WriteVd, [FireUnitFP]> { let Latency = 2; }
def : WriteRes<WriteVq, [FireUnitFP]> { let Latency = 2; }
def : WriteRes<WriteVLD, [FireUnitMem, FireLoadBuffer]> { let Latency = 3; }
def : WriteRes<WriteVST, [FireUnitMem, FireStoreBuffer]> { let Latency = 1; }

def : WriteRes<WriteAtomic, []> { let Unsupported = 1; }

// Zero-cycle execution (Zero-latency, no ProcResource consumed) for Register Renaming elimination
def WriteZeroFire : SchedWriteRes<[]> { let Latency = 0; }

// InstRW overrides for Firestorm Processor Model:
// 1. Zero-cycle Register Renaming Elimination (MOVs, FMOVs, NOP)
def : InstRW<[WriteZeroFire], (instrs MOVZWi, MOVZXi, MOVNWi, MOVNXi, MOVKWi, MOVKXi, FMOVDr, FMOVSr, NOP)>;

// 2. Specific double-precision FP divide & sqrt overrides (Empirical data from Dougall Johnson)
def FireWriteFDivDouble : SchedWriteRes<[FireUnitFP3]> { let Latency = 10; }
def FireWriteFSqrtDouble : SchedWriteRes<[FireUnitFP3]> { let Latency = 13; }

def : InstRW<[FireWriteFDivDouble], (instrs FDIVDrr)>;
def : InstRW<[FireWriteFSqrtDouble], (instrs FSQRTDr)>;

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

} // SchedModel = FirestormModel
