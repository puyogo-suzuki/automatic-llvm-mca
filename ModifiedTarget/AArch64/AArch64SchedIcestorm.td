//=- AArch64SchedIcestorm.td - Apple Icestorm Scheduling Definitions -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Apple Icestorm (M1 E-Core).
//
// Reference & Source Attribution:
//   Apple Microarchitecture Research by Dougall Johnson
//   https://dougallj.github.io/applecpu/icestorm.html
//   https://dougallj.github.io/applecpu/icestorm-int.html
//   https://dougallj.github.io/applecpu/icestorm-simd.html
//
//===----------------------------------------------------------------------===//

def IcestormModel : SchedMachineModel {
  let IssueWidth = 4;           // 4 MOPs per cycle decode / dispatch
  let MicroOpBufferSize = 60;   // Coalesced Retire Queue (ROB) = 60 entries
  let LoadLatency = 3;          // L1 hit latency (3 cycles measured by Dougall Johnson)
  let MispredictPenalty = 16;
  let CompleteModel = 0;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC]);
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available on Icestorm.

// 3 INT Schedulers (u1, u2, u3 - each 9 entries)
def IceUnitI0 : ProcResource<1> { let BufferSize = 9; }
def IceUnitI1 : ProcResource<1> { let BufferSize = 9; }
def IceUnitI2 : ProcResource<1> { let BufferSize = 9; }
def IceUnitI  : ProcResource<3> { let BufferSize = 27; }

// 2 FP/Vector Schedulers (u6, u7 - each 13 entries)
def IceUnitFP0 : ProcResource<1> { let BufferSize = 13; }
def IceUnitFP1 : ProcResource<1> { let BufferSize = 13; }
def IceUnitFP  : ProcResource<2> { let BufferSize = 26; }

// MEM Scheduler (u4, u5 - 18 entries)
def IceUnitMem : ProcResource<4> { let BufferSize = 18; }

def IceLoadBuffer  : ProcResource<2> { let BufferSize = 30; } // 2 loads per cycle (u4, u5)
def IceStoreBuffer : ProcResource<1> { let BufferSize = 18; } // 1 store per cycle (u4)

//===----------------------------------------------------------------------===//
// Target-wide SchedWrites mappings for Icestorm Model

let SchedModel = IcestormModel in {

// Basic GPR ALU: 1 cycle latency, 3 per cycle (u1-u3)
def : WriteRes<WriteImm, []> { let Latency = 0; }
def : WriteRes<WriteI, [IceUnitI]> { let Latency = 1; }
// Shifted/Extended GPR ALU: 2 cycles latency, 2 issue ops across u1-u3
def : WriteRes<WriteISReg, [IceUnitI, IceUnitI]> { let Latency = 2; }
def : WriteRes<WriteIEReg, [IceUnitI, IceUnitI]> { let Latency = 2; }
def : WriteRes<WriteExtr, [IceUnitI]> { let Latency = 1; }
def : WriteRes<WriteIS, [IceUnitI]> { let Latency = 1; }

// Integer Multiply: 3 cycles latency (u3)
def : WriteRes<WriteIM32, [IceUnitI2]> { let Latency = 3; }
def : WriteRes<WriteIM64, [IceUnitI2]> { let Latency = 3; }

// Integer Divide (Empirical data from Dougall Johnson: u2 port)
// 32-bit SDIV/UDIV: 7-13 cycles latency (slow form 13c)
def : WriteRes<WriteID32, [IceUnitI1]> { let Latency = 13; }
// 64-bit SDIV/UDIV: 7-21 cycles latency (slow form 21c)
def : WriteRes<WriteID64, [IceUnitI1]> { let Latency = 21; }

// Branch: 1 cycle latency (u1)
def : WriteRes<WriteBr, [IceUnitI0]> { let Latency = 1; }
def : WriteRes<WriteBrReg, [IceUnitI0]> { let Latency = 1; }

// GPR Loads: 3 cycles latency (L1 hit measured by Dougall Johnson)
def : WriteRes<WriteLD, [IceUnitMem, IceLoadBuffer]> { let Latency = 3; }
def : WriteRes<WriteLDIdx, [IceUnitMem, IceLoadBuffer]> { let Latency = 3; }
def : WriteRes<WriteLDHi, [IceUnitMem, IceLoadBuffer]> { let Latency = 3; }

// GPR Stores: 1 cycle latency
def : WriteRes<WriteST, [IceUnitMem, IceStoreBuffer]> { let Latency = 1; }
def : WriteRes<WriteSTP, [IceUnitMem, IceStoreBuffer]> { let Latency = 1; }
def : WriteRes<WriteSTIdx, [IceUnitMem, IceStoreBuffer]> { let Latency = 1; }

def : WriteRes<WriteAdr, [IceUnitI]> { let Latency = 1; }
def : WriteRes<WriteSys, [IceUnitI0]> { let Latency = 10; }
def : WriteRes<WriteBarrier, [IceUnitMem]> { let Latency = 10; }
def : WriteRes<WriteHint, [IceUnitI0]> { let Latency = 1; }

// Floating-Point Operations (2 ports u6, u7)
def : WriteRes<WriteF, [IceUnitFP]> { let Latency = 2; }
def : WriteRes<WriteFCmp, [IceUnitFP0]> { let Latency = 1; }
def : WriteRes<WriteFCvt, [IceUnitFP]> { let Latency = 3; }
def : WriteRes<WriteFCopy, [IceUnitFP]> { let Latency = 1; }
def : WriteRes<WriteFImm, [IceUnitFP]> { let Latency = 1; }
def : WriteRes<WriteFMul, [IceUnitFP]> { let Latency = 3; }

// Floating-Point Divide (Empirical data from Dougall Johnson: u6 port)
// Single precision FDIVSrr: 8 cycles latency
def : WriteRes<WriteFDiv, [IceUnitFP0]> { let Latency = 8; }

// Vector Operations (2 ports u6, u7)
def : WriteRes<WriteVd, [IceUnitFP]> { let Latency = 2; }
def : WriteRes<WriteVq, [IceUnitFP]> { let Latency = 2; }
def : WriteRes<WriteVLD, [IceUnitMem, IceLoadBuffer]> { let Latency = 3; }
def : WriteRes<WriteVST, [IceUnitMem, IceStoreBuffer]> { let Latency = 1; }

def : WriteRes<WriteAtomic, []> { let Unsupported = 1; }

// Zero-cycle execution (Zero-latency, no ProcResource consumed) for Register Renaming elimination
def WriteZeroIce : SchedWriteRes<[]> { let Latency = 0; }

// InstRW overrides for Icestorm Processor Model:
// 1. Zero-cycle Register Renaming Elimination (MOVs, FMOVs, NOP)
def : InstRW<[WriteZeroIce], (instrs MOVZWi, MOVZXi, MOVNWi, MOVNXi, MOVKWi, MOVKXi, FMOVDr, FMOVSr, NOP)>;

// 2. Specific double-precision FP divide & sqrt overrides (Empirical data from Dougall Johnson)
def IceWriteFDivDouble : SchedWriteRes<[IceUnitFP0]> { let Latency = 10; }
def IceWriteFSqrtDouble : SchedWriteRes<[IceUnitFP0]> { let Latency = 13; }

def : InstRW<[IceWriteFDivDouble], (instrs FDIVDrr)>;
def : InstRW<[IceWriteFSqrtDouble], (instrs FSQRTDr)>;

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

} // SchedModel = IcestormModel
