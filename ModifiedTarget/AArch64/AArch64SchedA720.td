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
  let MicroOpBufferSize = 160;  // Reorder Buffer (ROB)
  let LoadLatency = 3;          // L1 hit latency (3 cycles from Cortex-A720 SOG)
  let MispredictPenalty = 11;
  let CompleteModel = 0;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC]);
}

//===----------------------------------------------------------------------===//
// Define Cortex-A720 execution pipelines matching SOG Section 2.1 (13 Issue Pipelines)

// 2 Branch pipelines (Branch 0, Branch 1)
def A720UnitB0 : ProcResource<1> { let BufferSize = 16; }
def A720UnitB1 : ProcResource<1> { let BufferSize = 16; }
def A720UnitB  : ProcResource<2> { let BufferSize = 32; }

// 4 Integer pipelines (2 Single-Cycle, 2 Single/Multi-Cycle ALU/MUL/DIV)
def A720UnitI0 : ProcResource<1> { let BufferSize = 20; }
def A720UnitI1 : ProcResource<1> { let BufferSize = 20; }
def A720UnitM0 : ProcResource<1> { let BufferSize = 20; } // MUL/DIV
def A720UnitM1 : ProcResource<1> { let BufferSize = 20; } // MUL
def A720UnitI  : ProcResource<4> { let BufferSize = 80; }

def A720UnitMul : ProcResource<2> { let BufferSize = 40; } // M0 + M1

// 3 AGU Load/Store pipelines (2 Load/Store, 1 Load-only)
def A720UnitL0 : ProcResource<1> { let BufferSize = 24; }
def A720UnitL1 : ProcResource<1> { let BufferSize = 24; }
def A720UnitL2 : ProcResource<1> { let BufferSize = 24; }
def A720UnitLS : ProcResource<3> { let BufferSize = 72; }

// 2 Store Data pipelines
def A720UnitSD : ProcResource<2> { let BufferSize = 32; }

// 2 FP/ASIMD Vector pipelines
def A720UnitV0 : ProcResource<1> { let BufferSize = 24; }
def A720UnitV1 : ProcResource<1> { let BufferSize = 24; }
def A720UnitV  : ProcResource<2> { let BufferSize = 48; }

//===----------------------------------------------------------------------===//
// Target-wide SchedWrites mappings for Cortex-A720 Model

let SchedModel = CortexA720Model in {

def : WriteRes<WriteImm, [A720UnitI]> { let Latency = 1; }
def : WriteRes<WriteI, [A720UnitI]> { let Latency = 1; }
def : WriteRes<WriteISReg, [A720UnitI]> { let Latency = 1; }
def : WriteRes<WriteIEReg, [A720UnitI]> { let Latency = 1; }
def : WriteRes<WriteExtr, [A720UnitI]> { let Latency = 1; }
def : WriteRes<WriteIS, [A720UnitI]> { let Latency = 1; }

def : WriteRes<WriteIM32, [A720UnitMul]> { let Latency = 2; }
def : WriteRes<WriteIM64, [A720UnitMul]> { let Latency = 3; }

// Integer Divide (SOG Table 3-4 variable latencies: 32-bit 5-12c [median 8], 64-bit 5-20c [median 12])
def : WriteRes<WriteID32, [A720UnitM0]> { let Latency = 8; }
def : WriteRes<WriteID64, [A720UnitM0]> { let Latency = 12; }

def : WriteRes<WriteBr, [A720UnitB]> { let Latency = 1; }
def : WriteRes<WriteBrReg, [A720UnitB]> { let Latency = 1; }

def : WriteRes<WriteLD, [A720UnitLS]> { let Latency = 3; }
def : WriteRes<WriteST, [A720UnitLS, A720UnitSD]> { let Latency = 1; }
def : WriteRes<WriteSTP, [A720UnitLS, A720UnitSD]> { let Latency = 1; }
def : WriteRes<WriteAdr, [A720UnitI]> { let Latency = 1; }
def : WriteRes<WriteLDIdx, [A720UnitLS]> { let Latency = 3; }
def : WriteRes<WriteSTIdx, [A720UnitLS, A720UnitSD]> { let Latency = 1; }

def : WriteRes<WriteLDHi, [A720UnitLS]> { let Latency = 3; }
def : WriteRes<WriteSys, [A720UnitI0]> { let Latency = 10; }
def : WriteRes<WriteBarrier, [A720UnitLS]> { let Latency = 10; }
def : WriteRes<WriteHint, [A720UnitI0]> { let Latency = 1; }

def : WriteRes<WriteF, [A720UnitV]> { let Latency = 3; }
def : WriteRes<WriteFCmp, [A720UnitV0]> { let Latency = 2; }
def : WriteRes<WriteFCvt, [A720UnitV]> { let Latency = 3; }
def : WriteRes<WriteFCopy, [A720UnitV]> { let Latency = 1; }
def : WriteRes<WriteFImm, [A720UnitV]> { let Latency = 1; }
def : WriteRes<WriteFMul, [A720UnitV]> { let Latency = 3; }

// Floating-Point Divide (SOG Table 3-12 variable latencies: 32-bit 7-10c [median 8], 64-bit 7-15c [median 11])
def : WriteRes<WriteFDiv, [A720UnitV0]> { let Latency = 10; }

def : WriteRes<WriteVd, [A720UnitV]> { let Latency = 2; }
def : WriteRes<WriteVq, [A720UnitV]> { let Latency = 2; }
def : WriteRes<WriteVLD, [A720UnitLS]> { let Latency = 3; }
def : WriteRes<WriteVST, [A720UnitLS, A720UnitV]> { let Latency = 1; }

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

} // SchedModel = CortexA720Model
