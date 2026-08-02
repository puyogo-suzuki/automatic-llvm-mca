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
//   Arm Cortex-A520 Core Software Optimization Guide (PJDOC-1505342170-671342)
//   https://developer.arm.com/documentation/102553/latest/
//
//===----------------------------------------------------------------------===//

def CortexA520Model : SchedMachineModel {
  let IssueWidth = 2;           // Dual-issue in-order execution
  let MicroOpBufferSize = 0;    // In-order core
  let LoadLatency = 2;          // L1 hit latency (2 cycles from Cortex-A520 SOG)
  let MispredictPenalty = 10;
  let CompleteModel = 0;

  list<Predicate> UnsupportedFeatures = !listconcat(SVEUnsupported.F,
                                                    PAUnsupported.F,
                                                    SMEUnsupported.F,
                                                    [HasMTE, HasCSSC]);
}

//===----------------------------------------------------------------------===//
// Define Cortex-A520 execution pipelines matching SOG Section 2.1

def A520UnitALU  : ProcResource<2>; // 2 ALU pipelines (ALU0, ALU1)
def A520UnitMAC  : ProcResource<1>; // MAC pipeline
def A520UnitDIV  : ProcResource<1>; // DIV pipeline
def A520UnitLS   : ProcResource<2>; // 2 Load/Store pipelines
def A520UnitVALU : ProcResource<2>; // 2 Vector/FP ALU pipelines
def A520UnitVMAC : ProcResource<1>; // Vector MAC pipeline

//===----------------------------------------------------------------------===//
// Target-wide SchedWrites mappings for Cortex-A520 Model (A76-style SOG mapping)

let SchedModel = CortexA520Model in {

def : WriteRes<WriteImm, [A520UnitALU]> { let Latency = 1; }
def : WriteRes<WriteI, [A520UnitALU]> { let Latency = 1; }
def : WriteRes<WriteISReg, [A520UnitALU]> { let Latency = 1; }
def : WriteRes<WriteIEReg, [A520UnitALU]> { let Latency = 1; }
def : WriteRes<WriteExtr, [A520UnitALU]> { let Latency = 1; }
def : WriteRes<WriteIS, [A520UnitALU]> { let Latency = 1; }

// Integer Multiply: 3 cycles latency (SOG Section 3.3)
def : WriteRes<WriteIM32, [A520UnitMAC]> { let Latency = 3; }
def : WriteRes<WriteIM64, [A520UnitMAC]> { let Latency = 3; }

// Integer Divide: W-form 12c, X-form 20c (SOG Table 3-4, A76-style mapping)
def : WriteRes<WriteID32, [A520UnitDIV]> { let Latency = 12; }
def : WriteRes<WriteID64, [A520UnitDIV]> { let Latency = 20; }

def : WriteRes<WriteBr, [A520UnitALU]> { let Latency = 1; }
def : WriteRes<WriteBrReg, [A520UnitALU]> { let Latency = 1; }

// Memory Operations: L1 load latency = 2 cycles (SOG Section 3.7)
def : WriteRes<WriteLD, [A520UnitLS]> { let Latency = 2; }
def : WriteRes<WriteST, [A520UnitLS]> { let Latency = 1; }
def : WriteRes<WriteSTP, [A520UnitLS]> { let Latency = 1; }
def : WriteRes<WriteAdr, [A520UnitALU]> { let Latency = 1; }
def : WriteRes<WriteLDIdx, [A520UnitLS]> { let Latency = 2; }
def : WriteRes<WriteSTIdx, [A520UnitLS]> { let Latency = 1; }

def : WriteRes<WriteLDHi, [A520UnitLS]> { let Latency = 2; }
def : WriteRes<WriteSys, [A520UnitALU]> { let Latency = 10; }
def : WriteRes<WriteBarrier, [A520UnitLS]> { let Latency = 10; }
def : WriteRes<WriteHint, [A520UnitALU]> { let Latency = 1; }

// FP Operations (SOG Table 3-12)
def : WriteRes<WriteF, [A520UnitVALU]> { let Latency = 3; }
def : WriteRes<WriteFCmp, [A520UnitVALU]> { let Latency = 2; }
def : WriteRes<WriteFCvt, [A520UnitVALU]> { let Latency = 3; }
def : WriteRes<WriteFCopy, [A520UnitVALU]> { let Latency = 1; }
def : WriteRes<WriteFImm, [A520UnitVALU]> { let Latency = 1; }
def : WriteRes<WriteFMul, [A520UnitVMAC]> { let Latency = 4; }

// FP Divide: S-form 13c, D-form 22c (SOG Table 3-12)
def : WriteRes<WriteFDiv, [A520UnitVMAC]> { let Latency = 13; }

def : WriteRes<WriteVd, [A520UnitVALU]> { let Latency = 2; }
def : WriteRes<WriteVq, [A520UnitVALU]> { let Latency = 2; }
def : WriteRes<WriteVLD, [A520UnitLS]> { let Latency = 2; }
def : WriteRes<WriteVST, [A520UnitLS]> { let Latency = 1; }

def : WriteRes<WriteAtomic, []> { let Unsupported = 1; }

// Specific double-precision FP divide override (SOG Table 3-12)
def A520WriteFDivDouble : SchedWriteRes<[A520UnitVMAC]> { let Latency = 22; }
def : InstRW<[A520WriteFDivDouble], (instrs FDIVDrr)>;

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

} // SchedModel = CortexA520Model
