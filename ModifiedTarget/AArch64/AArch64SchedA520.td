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

// Cortex-A520 SOG (PJDOC-1505342170-671342, Issue 1.2) sec. 4.1 "Issue
// constraints", p.72: "The issue queue has space for three instructions ...
// A maximum of three issue queue entries can be co-issued per cycle (ignoring
// hazards) consisting of at most: Two ALU instructions / Two load instructions
// / One store instruction / Two VPU data processing instructions."
//
// WHY THIS WAS WRONG.  IssueWidth was 2, carrying the comment "Dual-issue
// in-order execution" -- a value and a justification inherited verbatim from
// CortexA55Model (AArch64SchedA55.td: "It dual-issues under most
// circumstances") when this model was written by analogy to A55.  A520 is
// A55's in-order LITTLE successor, but it is NOT dual-issue: it is 3-wide.
// The identical "three issue queue entries can be co-issued per cycle" wording
// already appears in the Cortex-A510 SOG sec. 4.1, so 3-wide co-issue is a
// stable property of the A510/A520 generation, not an A520-specific quirk.
//
// MECHANISM OF THE BUG.  Unlike the Cortex-A55 dual-issue bug (which was a
// MISSING ENFORCEMENT guard in the project's own a55_issue_stage.cpp), the
// A520 enforcement path is stock and correct: A520 has MicroOpBufferSize = 0,
// so getSchedModel().isOutOfOrder() is false, and mca.cpp's pipeline selector
// gates the custom A55 stage on the exact string "cortex-a55" -- so A520 falls
// through to Context::createInOrderPipeline().  llvm-mca's InOrderIssueStage
// caps issue with its `Bandwidth` field, which it seeds from getIssueWidth(),
// i.e. straight from this constant.  The width was therefore enforced
// faithfully -- at the wrong number.  Every A520 simulation was capped at IPC
// 2 by an issue limit that does not exist on the real core, so any region
// whose true bottleneck was between 2 and 3 instructions/cycle was reported as
// issue-bound when the hardware would not be.  Correcting the constant is the
// whole fix; no C++ change is needed.
//
// NOT CHANGED HERE: the per-class co-issue sub-limits above (at most 1 store,
// at most 2 loads) are finer-grained than a single IssueWidth and are only
// approximated by the ProcResource counts below (A520UnitLS is ProcResource<2>
// and so admits two same-cycle stores). That is a separate modelling question
// from the issue width and is left alone deliberately.
def CortexA520Model : SchedMachineModel {
  let IssueWidth = 3;           // SOG sec. 4.1: max three entries co-issued per cycle
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
