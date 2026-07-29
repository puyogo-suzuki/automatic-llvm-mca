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
def FireUnitI1 : ProcResource<1> { let BufferSize = 26; } // u2: ALU, Branch, IndBranch
def FireUnitI2 : ProcResource<1> { let BufferSize = 16; } // u3: ALU, FP->GPR
def FireUnitI3 : ProcResource<1> { let BufferSize = 12; } // u4: ALU, FP->GPR
def FireUnitI4 : ProcResource<1> { let BufferSize = 28; } // u5: ALU, MUL, DIV
def FireUnitI5 : ProcResource<1> { let BufferSize = 28; } // u6: ALU, MUL, MADD, CRC
def FireUnitI  : ProcResource<6> { let BufferSize = 134; }

// Multiply execution ports (u5, u6 = FireUnitI4, FireUnitI5)
def FireUnitIMul : ProcResource<2> { let BufferSize = 56; }

// 4 FP/Vector Schedulers (each 36 entries)
def FireUnitFP0 : ProcResource<1> { let BufferSize = 36; }
def FireUnitFP1 : ProcResource<1> { let BufferSize = 36; }
def FireUnitFP2 : ProcResource<1> { let BufferSize = 36; }
def FireUnitFP3 : ProcResource<1> { let BufferSize = 36; }
def FireUnitFP  : ProcResource<4> { let BufferSize = 144; }

// 1 MEM Scheduler (28 entries)
def FireUnitMem : ProcResource<7> { let BufferSize = 28; } // u7-u10: Store x2, Load x3

def FireLoadBuffer  : ProcResource<1> { let BufferSize = 130; }
def FireStoreBuffer : ProcResource<1> { let BufferSize = 60; }

//===----------------------------------------------------------------------===//
// Target-wide SchedWrites mappings for Firestorm Model

let SchedModel = FirestormModel in {

def : WriteRes<WriteImm, [FireUnitI]> { let Latency = 1; }
def : WriteRes<WriteI, [FireUnitI]> { let Latency = 1; }
def : WriteRes<WriteISReg, [FireUnitI, FireUnitI]> { let Latency = 2; }
def : WriteRes<WriteIEReg, [FireUnitI, FireUnitI]> { let Latency = 2; }
def : WriteRes<WriteExtr, [FireUnitI]> { let Latency = 1; }
def : WriteRes<WriteIS, [FireUnitI]> { let Latency = 1; }
def : WriteRes<WriteIM32, [FireUnitIMul]> { let Latency = 3; }
def : WriteRes<WriteIM64, [FireUnitIMul]> { let Latency = 3; }
def : WriteRes<WriteID32, [FireUnitI4]> { let Latency = 9; }
def : WriteRes<WriteID64, [FireUnitI4]> { let Latency = 9; }
def : WriteRes<WriteBr, [FireUnitI0]> { let Latency = 1; }
def : WriteRes<WriteBrReg, [FireUnitI1]> { let Latency = 1; }
def : WriteRes<WriteLD, [FireUnitMem, FireLoadBuffer]> { let Latency = 3; }
def : WriteRes<WriteST, [FireUnitMem, FireStoreBuffer]> { let Latency = 1; }
def : WriteRes<WriteSTP, [FireUnitMem, FireStoreBuffer]> { let Latency = 1; }
def : WriteRes<WriteAdr, [FireUnitI]> { let Latency = 1; }
def : WriteRes<WriteLDIdx, [FireUnitMem, FireLoadBuffer]> { let Latency = 3; }
def : WriteRes<WriteSTIdx, [FireUnitMem, FireStoreBuffer]> { let Latency = 1; }

def : WriteRes<WriteLDHi, [FireUnitMem, FireLoadBuffer]> { let Latency = 3; }
def : WriteRes<WriteSys, [FireUnitI0]> { let Latency = 10; }
def : WriteRes<WriteBarrier, [FireUnitMem]> { let Latency = 10; }
def : WriteRes<WriteHint, [FireUnitI0]> { let Latency = 1; }

def : WriteRes<WriteF, [FireUnitFP]> { let Latency = 3; }
def : WriteRes<WriteFCmp, [FireUnitFP0]> { let Latency = 1; }
def : WriteRes<WriteFCvt, [FireUnitFP]> { let Latency = 3; }
def : WriteRes<WriteFCopy, [FireUnitFP]> { let Latency = 1; }
def : WriteRes<WriteFImm, [FireUnitFP]> { let Latency = 1; }
def : WriteRes<WriteFMul, [FireUnitFP]> { let Latency = 3; }
def : WriteRes<WriteFDiv, [FireUnitFP0]> { let Latency = 9; }

def : WriteRes<WriteVd, [FireUnitFP]> { let Latency = 2; }
def : WriteRes<WriteVq, [FireUnitFP]> { let Latency = 2; }
def : WriteRes<WriteVLD, [FireUnitMem, FireLoadBuffer]> { let Latency = 3; }
def : WriteRes<WriteVST, [FireUnitMem, FireStoreBuffer]> { let Latency = 1; }

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

} // SchedModel = FirestormModel
