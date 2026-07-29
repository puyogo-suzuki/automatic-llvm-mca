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
//===----------------------------------------------------------------------===//

def FirestormModel : SchedMachineModel {
  let IssueWidth = 8;           // 8 MOPs per cycle decode / dispatch
  let MicroOpBufferSize = 330;  // Coalesced Retire Queue (ROB) = 330 entries
  let LoadLatency = 4;          // Standard L1 hit latency
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
def FireUnitI0 : ProcResource<1> { let BufferSize = 24; }
def FireUnitI1 : ProcResource<1> { let BufferSize = 26; }
def FireUnitI2 : ProcResource<1> { let BufferSize = 16; }
def FireUnitI3 : ProcResource<1> { let BufferSize = 12; }
def FireUnitI4 : ProcResource<1> { let BufferSize = 28; }
def FireUnitI5 : ProcResource<1> { let BufferSize = 28; }
def FireUnitI  : ProcResource<6> { let BufferSize = 134; }

// 4 FP/Vector Schedulers (each 36 entries)
def FireUnitFP0 : ProcResource<1> { let BufferSize = 36; }
def FireUnitFP1 : ProcResource<1> { let BufferSize = 36; }
def FireUnitFP2 : ProcResource<1> { let BufferSize = 36; }
def FireUnitFP3 : ProcResource<1> { let BufferSize = 36; }
def FireUnitFP  : ProcResource<4> { let BufferSize = 144; }

// 1 MEM Scheduler (28 entries)
def FireUnitMem : ProcResource<7> { let BufferSize = 28; }

def FireLoadBuffer  : ProcResource<1> { let BufferSize = 130; }
def FireStoreBuffer : ProcResource<1> { let BufferSize = 60; }
