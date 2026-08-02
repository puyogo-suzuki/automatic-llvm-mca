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

// 3 INT Schedulers (each 9 entries)
def IceUnitI0 : ProcResource<1> { let BufferSize = 9; }
def IceUnitI1 : ProcResource<1> { let BufferSize = 9; }
def IceUnitI2 : ProcResource<1> { let BufferSize = 9; }
def IceUnitI  : ProcResource<3> { let BufferSize = 27; }

// 2 FP/Vector Schedulers (each 13 entries)
def IceUnitFP0 : ProcResource<1> { let BufferSize = 13; }
def IceUnitFP1 : ProcResource<1> { let BufferSize = 13; }
def IceUnitFP  : ProcResource<2> { let BufferSize = 26; }

// 1 MEM Scheduler (18 entries)
def IceUnitMem : ProcResource<4> { let BufferSize = 18; }

def IceLoadBuffer  : ProcResource<2> { let BufferSize = 30; } // 2 loads per cycle
def IceStoreBuffer : ProcResource<1> { let BufferSize = 18; } // 1 store per cycle

//===----------------------------------------------------------------------===//
// Target-wide SchedWrites mappings for Icestorm Model

let SchedModel = IcestormModel in {

def : WriteRes<WriteImm, []> { let Latency = 0; }
def : WriteRes<WriteI, [IceUnitI]> { let Latency = 1; }
def : WriteRes<WriteISReg, [IceUnitI]> { let Latency = 2; }
def : WriteRes<WriteIEReg, [IceUnitI]> { let Latency = 2; }
def : WriteRes<WriteExtr, [IceUnitI]> { let Latency = 1; }
def : WriteRes<WriteIS, [IceUnitI]> { let Latency = 1; }
def : WriteRes<WriteIM32, [IceUnitI2]> { let Latency = 3; }
def : WriteRes<WriteIM64, [IceUnitI2]> { let Latency = 3; }
def : WriteRes<WriteID32, [IceUnitI1]> { let Latency = 10; }
def : WriteRes<WriteID64, [IceUnitI1]> { let Latency = 10; }
def : WriteRes<WriteBr, [IceUnitI0]> { let Latency = 1; }
def : WriteRes<WriteBrReg, [IceUnitI0]> { let Latency = 1; }
def : WriteRes<WriteLD, [IceUnitMem, IceLoadBuffer]> { let Latency = 3; }
def : WriteRes<WriteST, [IceUnitMem, IceStoreBuffer]> { let Latency = 1; }
def : WriteRes<WriteSTP, [IceUnitMem, IceStoreBuffer]> { let Latency = 1; }
def : WriteRes<WriteAdr, [IceUnitI]> { let Latency = 1; }
def : WriteRes<WriteLDIdx, [IceUnitMem, IceLoadBuffer]> { let Latency = 3; }
def : WriteRes<WriteSTIdx, [IceUnitMem, IceStoreBuffer]> { let Latency = 1; }

def : WriteRes<WriteLDHi, [IceUnitMem, IceLoadBuffer]> { let Latency = 3; }
def : WriteRes<WriteSys, [IceUnitI0]> { let Latency = 10; }
def : WriteRes<WriteBarrier, [IceUnitMem]> { let Latency = 10; }
def : WriteRes<WriteHint, [IceUnitI0]> { let Latency = 1; }

def : WriteRes<WriteF, [IceUnitFP]> { let Latency = 3; }
def : WriteRes<WriteFCmp, [IceUnitFP0]> { let Latency = 1; }
def : WriteRes<WriteFCvt, [IceUnitFP]> { let Latency = 3; }
def : WriteRes<WriteFCopy, [IceUnitFP]> { let Latency = 1; }
def : WriteRes<WriteFImm, [IceUnitFP]> { let Latency = 1; }
def : WriteRes<WriteFMul, [IceUnitFP]> { let Latency = 3; }
def : WriteRes<WriteFDiv, [IceUnitFP0]> { let Latency = 10; }

def : WriteRes<WriteVd, [IceUnitFP]> { let Latency = 2; }
def : WriteRes<WriteVq, [IceUnitFP]> { let Latency = 2; }
def : WriteRes<WriteVLD, [IceUnitMem, IceLoadBuffer]> { let Latency = 3; }
def : WriteRes<WriteVST, [IceUnitMem, IceStoreBuffer]> { let Latency = 1; }

def : WriteRes<WriteAtomic, []> { let Unsupported = 1; }

// Zero-cycle execution (Zero-latency, no ProcResource consumed) for Register Renaming elimination (Hardware Copy-on-Write)
def WriteZeroIce : SchedWriteRes<[]> { let Latency = 0; }

// InstRW overrides for IceStorm Processor Model:
// 1. Zero-cycle Register Renaming Elimination (MOVs, FMOVs, NOP)
def : InstRW<[WriteZeroIce], (instrs MOVZWi, MOVZXi, MOVNWi, MOVNXi, MOVKWi, MOVKXi, FMOVDr, FMOVSr, NOP)>;
// 2. PC-relative Address Calculation
def : InstRW<[WriteAdr], (instrs ADR, ADRP)>;
// 3. Scalar Loads - 3 cycle latency
def : InstRW<[WriteLD], (instrs LDRBui, LDRHui, LDRWui, LDRXui, LDRBroX, LDRHroX, LDRWroX, LDRXroX, LDPWi, LDPXi)>;
// 4. Scalar Stores - 1 cycle latency
def : InstRW<[WriteST], (instrs STRBui, STRHui, STRWui, STRXui, STRBroX, STRHroX, STRWroX, STRXroX, STPWi, STPXi)>;

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
