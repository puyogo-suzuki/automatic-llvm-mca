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
//===----------------------------------------------------------------------===//

def IcestormModel : SchedMachineModel {
  let IssueWidth = 4;           // 4 MOPs per cycle decode / dispatch
  let MicroOpBufferSize = 60;   // Coalesced Retire Queue (ROB) = 60 entries
  let LoadLatency = 4;          // Standard L1 hit latency
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

def IceLoadBuffer  : ProcResource<1> { let BufferSize = 30; }
def IceStoreBuffer : ProcResource<1> { let BufferSize = 18; }
