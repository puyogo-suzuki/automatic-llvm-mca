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
  let LoadLatency = 4;
  // L1 load-to-use latency.  Dougall Johnson's Firestorm/Icestorm pages
  // (https://dougallj.github.io/applecpu/firestorm.html, "Complex Latencies")
  // state verbatim: "Loads may be passed to the base address of other loads
  // with 3c latency (which is nice for linked lists), but chaining with ALU
  // operations gives a latency of 4c. (Although the second destination
  // register in an LDP always has 4c latency.)"  The 3c figure is therefore
  // the SPECIAL CASE of a load feeding another load's base address, not the
  // general case: general load->consumer latency is 4c on BOTH cores, measured
  // 4.0047-4.0057 for every GPR addressing mode (base, base+imm, base+reg,
  // base+reg lsl #N, wreg sxtw).  This model previously used 3 for all loads,
  // i.e. it applied the pointer-chase special case universally and so
  // understated every load->ALU dependence by 1c.
  // NOTE (2026-09-10): correcting this is metric-neutral on all four datasets
  // (eval_estimation.py byte-identical for data / 0818_a78 / 0907_a78 /
  // macbook).  Almost every hot block in SPEC is bound by IssueWidth or by an
  // execution resource rather than by the precedence bound, so the load
  // latency does not reach the answer - the same masking that the
  // --no-facile-memory-deps note describes.  Kept as a source-fidelity fix.
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

// Branch execution ports (u1, u2 = FireUnitI0, FireUnitI1)
def FireUnitBranch : ProcResource<2> { let BufferSize = 50; }

// 4 FP/Vector Schedulers (each 36 entries)
def FireUnitFP0 : ProcResource<1> { let BufferSize = 36; }
def FireUnitFP1 : ProcResource<1> { let BufferSize = 36; }
def FireUnitFP2 : ProcResource<1> { let BufferSize = 36; }
def FireUnitFP3 : ProcResource<1> { let BufferSize = 36; }
def FireUnitFP  : ProcResource<4> { let BufferSize = 144; }

// 1 MEM Scheduler (28 entries)
def FireUnitMem : ProcResource<7> { let BufferSize = 28; } // u7-u10: Store x2, Load x3

def FireLoadBuffer  : ProcResource<3> { let BufferSize = 130; } // 3 loads per cycle
def FireStoreBuffer : ProcResource<2> { let BufferSize = 60; }  // 2 stores per cycle

//===----------------------------------------------------------------------===//
// Target-wide SchedWrites mappings for Firestorm Model

let SchedModel = FirestormModel in {

def : WriteRes<WriteImm, []> { let Latency = 0; }
def : WriteRes<WriteI, [FireUnitI]> { let Latency = 1; }
def : WriteRes<WriteISReg, [FireUnitI]> { let Latency = 2; }
def : WriteRes<WriteIEReg, [FireUnitI]> { let Latency = 2; }
def : WriteRes<WriteExtr, [FireUnitI]> { let Latency = 1; }
def : WriteRes<WriteIS, [FireUnitI]> { let Latency = 1; }
def : WriteRes<WriteIM32, [FireUnitIMul]> { let Latency = 3; }
def : WriteRes<WriteIM64, [FireUnitIMul]> { let Latency = 3; }
def : WriteRes<WriteID32, [FireUnitI4]> { let Latency = 11; } // Worst-case integer divide (7-11c)
def : WriteRes<WriteID64, [FireUnitI4]> { let Latency = 11; } // Worst-case integer divide (7-11c)
def : WriteRes<WriteBr, [FireUnitBranch]> { let Latency = 1; }
def : WriteRes<WriteBrReg, [FireUnitBranch]> { let Latency = 1; }
def : WriteRes<WriteLD, [FireUnitMem, FireLoadBuffer]> { let Latency = 4; }
def : WriteRes<WriteST, [FireUnitMem, FireStoreBuffer]> { let Latency = 1; }
def : WriteRes<WriteSTP, [FireUnitMem, FireStoreBuffer]> { let Latency = 1; }
def : WriteRes<WriteAdr, [FireUnitI]> { let Latency = 1; }
def : WriteRes<WriteLDIdx, [FireUnitMem, FireLoadBuffer]> { let Latency = 4; }
def : WriteRes<WriteSTIdx, [FireUnitMem, FireStoreBuffer]> { let Latency = 1; }

def : WriteRes<WriteLDHi, [FireUnitMem, FireLoadBuffer]> { let Latency = 4; }
def : WriteRes<WriteSys, [FireUnitI0]> { let Latency = 10; }
def : WriteRes<WriteBarrier, [FireUnitMem]> { let Latency = 10; }
def : WriteRes<WriteHint, [FireUnitI0]> { let Latency = 1; }

def : WriteRes<WriteF, [FireUnitFP]> { let Latency = 3; }
def : WriteRes<WriteFCmp, [FireUnitFP0]> { let Latency = 1; }
def : WriteRes<WriteFCvt, [FireUnitFP]> { let Latency = 4; } // Worst-case FP convert (3-4c)
def : WriteRes<WriteFCopy, [FireUnitFP]> { let Latency = 1; }
def : WriteRes<WriteFImm, [FireUnitFP]> { let Latency = 1; }
def : WriteRes<WriteFMul, [FireUnitFP]> { let Latency = 4; } // Worst-case FP multiply / FMA (4c)
def : WriteRes<WriteFDiv, [FireUnitFP0]> { let Latency = 13; } // Worst-case FP divide / sqrt (7-13c)

def : WriteRes<WriteVd, [FireUnitFP]> { let Latency = 2; }
def : WriteRes<WriteVq, [FireUnitFP]> { let Latency = 2; }
def : WriteRes<WriteVLD, [FireUnitMem, FireLoadBuffer]> { let Latency = 3; }
def : WriteRes<WriteVST, [FireUnitMem, FireStoreBuffer]> { let Latency = 1; }

def : WriteRes<WriteAtomic, []> { let Unsupported = 1; }

// Zero-cycle execution (Zero-latency, no ProcResource consumed) for Register Renaming elimination (Hardware Copy-on-Write)
def WriteZeroFire : SchedWriteRes<[]> { let Latency = 0; }

// InstRW overrides for Firestorm Processor Model:
// 1. Zero-cycle Register Renaming Elimination (MOVs, FMOVs, NOP)
def : InstRW<[WriteZeroFire], (instrs MOVZWi, MOVZXi, MOVNWi, MOVNXi, MOVKWi, MOVKXi, FMOVDr, FMOVSr, NOP)>;
// 2. PC-relative Address Calculation (u1, u2)
def : InstRW<[WriteAdr], (instrs ADR, ADRP)>;
// 3. Scalar Loads (u8-u10) - 3 cycle latency
def : InstRW<[WriteLD], (instrs LDRBui, LDRHui, LDRWui, LDRXui, LDRBroX, LDRHroX, LDRWroX, LDRXroX, LDPWi, LDPXi)>;
// 4. Scalar Stores (u7-u8) - 1 cycle latency
def : InstRW<[WriteST], (instrs STRBui, STRHui, STRWui, STRXui, STRBroX, STRHroX, STRWroX, STRXroX, STPWi, STPXi)>;
// 5. Vector floating-point multiply / multiply-accumulate, and add/sub.
// AArch64 gives the SIMD FP data-processing instructions Sched<[WriteVq]> /
// Sched<[WriteVd]>, i.e. the generic 2c vector-ALU write, NOT WriteFMul /
// WriteF.  Without these two rules every vector FP op on Firestorm simulated
// at 2c: sched class #809 (FMLAv2f64_FMLSv2f64) resolved to latency 2 here
// while Icestorm gave it 4, #1640 (FMULv2f64) 2 vs 4, #1619 (FADDv2f64) 2 vs 3.
// Icestorm already carries the equivalent overrides (AArch64SchedIcestorm.td
// rules 7-8); Firestorm was simply missing them, so the P-core - the one that
// actually runs vectorised SPEC FP code - was the less faithful of the two.
// Dougall Johnson's firestorm-simd.html gives, for EVERY element size and for
// both the vector and by-element forms:
//   FMUL / FNMUL / FMLA / FMLS   4c, throughput 0.25 (u11-14)
//   FADD / FSUB                  3c, throughput 0.25 (u11-14)
// (icestorm-simd.html: identical latencies, throughput 0.5.)  So FMUL-family
// -> WriteFMul (4c) and FADD/FSUB -> WriteF (3c) reproduce the source exactly.
// The SCALAR forms (FMULDrr, FMADDDrrr, ...) already resolve correctly through
// their own Sched<[WriteFMul]>; these rules fix the vector / by-element forms.
// NOTE (2026-09-10): metric-neutral on all four datasets - eval_estimation.py
// is identical to 3 decimal places on all 20 CPI/SF numbers with and without
// these two lines.  Reason: these SPEC binaries are gcc-4.6 -O2 base builds
// with essentially no autovectorisation, so vector-form FP mul/mac/add/sub is
// only 2875 of 6744507 static instructions (0.043%), against 48859 SCALAR
// fmadd/fmsub/fnmadd/fnmsub.  Kept as a source-fidelity fix and for symmetry
// with Icestorm; it will start to matter for any -march=armv8-a+simd build.
def : InstRW<[WriteFMul], (instregex "^F(MUL|MLA|MLS|NMUL|NMLA|NMLS)[v_].*")>;
def : InstRW<[WriteF], (instregex "^F(ADD|SUB)v.*")>;

def : ReadAdvance<ReadI, 0>;
def : ReadAdvance<ReadISReg, 0>;
def : ReadAdvance<ReadIEReg, 0>;
def : ReadAdvance<ReadIM, 0>;
// MADD / MSUB accumulator (addend) forwarding.  Dougall Johnson's
// "Complex Latencies" note, identical on firestorm.html and icestorm.html:
// "MADD's output can be passed to its third operand (the addend) with 1c
// latency, but if it's chained with other instructions it has 3c latency."
// AArch64 gives MADD/MSUB Sched<[WriteIM32/64, ReadIM, ReadIM, ReadIMA]>, so
// the addend is exactly the ReadIMA slot and the 1c figure is expressible as
// a 2-cycle ReadAdvance off the 3c WriteIM32/WriteIM64 latency -- RESTRICTED
// to a multiply producer, because the quoted 1c applies only when a MADD
// feeds a MADD's addend; anything else chained off MADD still sees 3c.
def : ReadAdvance<ReadIMA, 2, [WriteIM32, WriteIM64]>;
def : ReadAdvance<ReadID, 0>;
def : ReadAdvance<ReadST, 0>;
def : ReadAdvance<ReadExtrHi, 0>;
def : ReadAdvance<ReadAdrBase, 0>;
def : ReadAdvance<ReadVLD, 0>;

} // SchedModel = FirestormModel
