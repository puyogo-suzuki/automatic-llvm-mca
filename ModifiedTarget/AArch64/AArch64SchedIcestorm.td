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
  let IssueWidth = 4;           // 4 MOPs per cycle decode / dispatch / retire
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
// Processor Resources & Units (Ports) on IceStorm
//
// Integer units (3 units):
//   u1: alu + br + mrs
//   u2: alu + br + div + ptrauth
//   u3: alu + mul + bfm + crc
//
// Load & Store units (2 units):
//   u4: load/store + amx
//   u5: load
//
// FP / SIMD units (2 units):
//   u6: fp/simd
//   u7: fp/simd + fdiv + to-int + div + recp + sqrt + sha + jcvtzs
//===----------------------------------------------------------------------===//

def IceUnitI0 : ProcResource<1> { let BufferSize = 9; } // u1
def IceUnitI1 : ProcResource<1> { let BufferSize = 9; } // u2
def IceUnitI2 : ProcResource<1> { let BufferSize = 9; } // u3
def IceUnitI  : ProcResource<3> { let BufferSize = 27; }

def IceUnitFP0 : ProcResource<1> { let BufferSize = 13; } // u6
def IceUnitFP1 : ProcResource<1> { let BufferSize = 13; } // u7
def IceUnitFP  : ProcResource<2> { let BufferSize = 26; }  // Dual 128-bit FP/SIMD pipes

def IceUnitMem     : ProcResource<2> { let BufferSize = 24; } // Dual memory units (u4, u5)
def IceLoadBuffer  : ProcResource<2> { let BufferSize = 30; } // 2 loads/cyc (128-bit loads supported)
def IceStoreBuffer : ProcResource<1> { let BufferSize = 18; } // 1 store/cyc

//===----------------------------------------------------------------------===//
// Target-wide SchedWrites mappings for Icestorm Model
//===----------------------------------------------------------------------===//

let SchedModel = IcestormModel in {

// 1. Basic Integer Writes
def : WriteRes<WriteImm, []> { let Latency = 0; }
def : WriteRes<WriteI, [IceUnitI]> { let Latency = 1; }
def : WriteRes<WriteISReg, [IceUnitI]> { let Latency = 2; }
def : WriteRes<WriteIEReg, [IceUnitI]> { let Latency = 2; }
def : WriteRes<WriteExtr, [IceUnitI2]> { let Latency = 1; }
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

// 2. Floating Point & SIMD Writes (Dual 128-bit pipelines, using worst-case measured latencies)
def : WriteRes<WriteF, [IceUnitFP]> { let Latency = 3; }
def : WriteRes<WriteFCmp, [IceUnitFP1]> { let Latency = 1; }
def : WriteRes<WriteFCvt, [IceUnitFP]> { let Latency = 4; } // Worst-case FP convert (3-4c)
def : WriteRes<WriteFCopy, [IceUnitFP]> { let Latency = 1; }
def : WriteRes<WriteFImm, [IceUnitFP]> { let Latency = 1; }
def : WriteRes<WriteFMul, [IceUnitFP]> { let Latency = 4; } // Worst-case FP multiply / FMA (3-4c)
def : WriteRes<WriteFDiv, [IceUnitFP1]> { let Latency = 17; } // Worst-case FP divide / sqrt (7-17c)

def : WriteRes<WriteVd, [IceUnitFP]> { let Latency = 2; }
def : WriteRes<WriteVq, [IceUnitFP]> { let Latency = 2; }
def : WriteRes<WriteVLD, [IceUnitMem, IceLoadBuffer]> { let Latency = 3; }
def : WriteRes<WriteVST, [IceUnitMem, IceStoreBuffer]> { let Latency = 1; }
def : WriteRes<WriteAtomic, []> { let Unsupported = 1; }

// Zero-cycle execution (Zero-latency, Register Renaming Elimination)
def WriteZeroIce : SchedWriteRes<[]> { let Latency = 0; }

//===----------------------------------------------------------------------===//
// Instruction-Specific Overrides (InstRW) for IceStorm (Dougall Johnson Research)
//===----------------------------------------------------------------------===//

// 1. Zero-cycle Elimination (0 latency, no execution resource)
def : InstRW<[WriteZeroIce], (instrs MOVZWi, MOVZXi, MOVNWi, MOVNXi, MOVKWi, MOVKXi, FMOVDr, FMOVSr, NOP)>;
def : InstRW<[WriteAdr], (instrs ADR, ADRP)>;

// 2. Scalar Loads & Stores (3c load / 1c store)
def : InstRW<[WriteLD], (instregex "^LDR[BHSDQ]?(ui|l|ro[WX])?$")>;
def : InstRW<[WriteLD], (instregex "^LDP[WXSDQ]i$")>;
def : InstRW<[WriteST], (instregex "^STR[BHSDQ]?(ui|ro[WX])?$")>;
def : InstRW<[WriteST], (instregex "^STP[WXSDQ]i$")>;

// 3. Vector Loads & Stores (128-bit & 64-bit SIMD - 3c load / 1c store, TP=0.5 on Dual IceLoadBuffer)
def : InstRW<[WriteVLD], (instregex "^LD[1234](One|Two|Three|Four)?v.*")>;
def : InstRW<[WriteVLD], (instregex "^LD[1234]r?_.*")>;
def : InstRW<[WriteVST], (instregex "^ST[1234](One|Two|Three|Four)?v.*")>;
def : InstRW<[WriteVST], (instregex "^ST[1234]r?_.*")>;

// 4. Vector Integer Arithmetic, Logic, Negation & Absolute Value (Latency=2c, TP=0.5 on Dual IceUnitFP)
def : InstRW<[WriteVd], (instregex "^(ADD|SUB|AND|BIC|ORR|EOR|ORN|NOT|NEG|ABS)v.*")>;
def : InstRW<[WriteVd], (instregex "^(ADDHN|SUBHN|RADDHN|RSUBHN)v.*")>;

// 5. Vector Min / Max Operations (Latency=2c, TP=0.5 on Dual IceUnitFP)
def : InstRW<[WriteVd], (instregex "^[SU]?(MAX|MIN)P?v.*")>;
def : InstRW<[WriteVd], (instregex "^F(MAX|MIN)P?N?M?v.*")>;

// 6. Vector Shifts (Latency=2c, TP=0.5 on Dual IceUnitFP)
def : InstRW<[WriteVd], (instregex "^[SU]?(SHL|SHRN|SHLL|SHR|SRA)v.*")>;

// 7. Vector Multiply & Multiply-Accumulate (Latency=3-4c, TP=0.5 on Dual IceUnitFP)
def : InstRW<[WriteFMul], (instregex "^(MUL|MLA|MLS|SQDMULH|SQRDMLAH|SQRDMLSH)[v_].*")>;
def : InstRW<[WriteFMul], (instregex "^F(MUL|MLA|MLS|NMUL|NMLA|NMLS)[v_].*")>;

// 8. Floating Point Add/Sub/Cmp/Convert (Latency=2-3c, TP=0.5 on Dual IceUnitFP)
def : InstRW<[WriteF], (instregex "^F(ADD|SUB|NEG|ABS)v.*")>;
def : InstRW<[WriteFCmp], (instregex "^FCMP.*")>;
def : InstRW<[WriteFCvt], (instregex "^FCVT.*")>;
def : InstRW<[WriteFCvt], (instregex "^[SU]CVTF.*")>;

// 9. Floating Point Divide & Square Root (Latency=10-15c on IceUnitFP1)
def : InstRW<[WriteFDiv], (instregex "^FDIV.*")>;
def : InstRW<[WriteFDiv], (instregex "^FSQRT.*")>;

// 10. Vector Permute, Extract, Table & Duplicate (Latency=2c, TP=0.5 on Dual IceUnitFP)
def : InstRW<[WriteVd], (instregex "^(EXT|TBL|TBX|DUP|INS|UMOV|SMOV|REV|UZP|TRN|ZIP)[12]?v.*")>;

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
