#include "mca_common.h"
#include "llvm-source/llvm/lib/Target/AArch64/MCTargetDesc/AArch64AddressingModes.h"
#include "llvm-source/llvm/lib/Target/AArch64/MCTargetDesc/AArch64MCTargetDesc.h"
#include <algorithm>
#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/MCA/Context.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/MCA/HWEventListener.h"
#include "llvm/MCA/InstrBuilder.h"
#include "llvm/MCA/Pipeline.h"
#include "llvm/MCA/SourceMgr.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

#include "llvm/MCA/HardwareUnits/RegisterFile.h"

// Private member accessor hacks
struct RegisterFile_RegisterMappings_Tag {};
auto get_mappings(RegisterFile_RegisterMappings_Tag);

template <typename Tag, auto M>
struct RobStoreMappings {
  friend auto get_mappings(Tag) {
    return M;
  }
};
template struct RobStoreMappings<RegisterFile_RegisterMappings_Tag, &mca::RegisterFile::RegisterMappings>;

namespace {

template<typename Tag, typename Tag::type M>
struct RobStore {
  friend typename Tag::type get(Tag) { return M; }
};

struct ReadState_IsReady_Tag {
  typedef bool mca::ReadState::*type;
  friend type get(ReadState_IsReady_Tag);
};
template struct RobStore<ReadState_IsReady_Tag, &mca::ReadState::IsReady>;

struct Context_Hardware_Tag {
  typedef SmallVector<std::unique_ptr<mca::HardwareUnit>, 4> mca::Context::*type;
  friend type get(Context_Hardware_Tag);
};
template struct RobStore<Context_Hardware_Tag, &mca::Context::Hardware>;

uint64_t getELFSymbolSizeImpl(const ObjectFile &Obj, SymbolRef Sym) {
    if (const auto *Elf32LE = dyn_cast<ELF32LEObjectFile>(&Obj)) {
        if (auto SymOrErr = Elf32LE->getSymbol(Sym.getRawDataRefImpl())) return (*SymOrErr)->st_size;
    } else if (const auto *Elf64LE = dyn_cast<ELF64LEObjectFile>(&Obj)) {
        if (auto SymOrErr = Elf64LE->getSymbol(Sym.getRawDataRefImpl())) return (*SymOrErr)->st_size;
    } else if (const auto *Elf32BE = dyn_cast<ELF32BEObjectFile>(&Obj)) {
        if (auto SymOrErr = Elf32BE->getSymbol(Sym.getRawDataRefImpl())) return (*SymOrErr)->st_size;
    } else if (const auto *Elf64BE = dyn_cast<ELF64BEObjectFile>(&Obj)) {
        if (auto SymOrErr = Elf64BE->getSymbol(Sym.getRawDataRefImpl())) return (*SymOrErr)->st_size;
    }
    return 0;
}

struct SteadyStateTracker : mca::HWEventListener {
    const MCSubtargetInfo &STI;
    const MCRegisterInfo &MRI;
    const MCInstrInfo &MCII;
    ArrayRef<Instr> Instrs;
    ArrayRef<std::unique_ptr<mca::Instruction>> SimInstrs;
    unsigned WarmupRetiredLimit;
    unsigned TotalRetired = 0;
    unsigned SteadyRetired = 0;
    unsigned CurrentCycle = 0;
    unsigned SteadyStartCycle = 0;
    unsigned SteadyCycles = 0;
    bool WarmupComplete = false;
    unsigned LoopSize = 0;
    bool IgnoreLoopCarriedDep = false;
    mca::RegisterFile *PRF = nullptr;
    unsigned CurrentIteration = 0;
    bool IsA55 = false;
 
    explicit SteadyStateTracker(const MCSubtargetInfo &STI, const MCRegisterInfo &MRI, const MCInstrInfo &MCII,
                                ArrayRef<Instr> Instrs, ArrayRef<std::unique_ptr<mca::Instruction>> SimInstrs,
                                unsigned WarmupRetiredLimit, unsigned LoopSize, bool IgnoreLoopCarriedDep, mca::RegisterFile *PRF = nullptr)
        : STI(STI), MRI(MRI), MCII(MCII), Instrs(Instrs), SimInstrs(SimInstrs), WarmupRetiredLimit(WarmupRetiredLimit),
          LoopSize(LoopSize), IgnoreLoopCarriedDep(IgnoreLoopCarriedDep), PRF(PRF),
          IsA55(STI.getCPU() == "cortex-a55") {}
 
    void onCycleBegin() override {
        ++CurrentCycle;
    }
 
    void onCycleEnd() override {
        if (WarmupComplete && CurrentCycle > SteadyStartCycle) ++SteadyCycles;
    }
 
    void onEvent(const mca::HWInstructionEvent &Event) override {
        if (Event.Type == mca::HWInstructionEvent::Retired) {
            ++TotalRetired;
            if (WarmupComplete) {
                ++SteadyRetired;
                return;
            }
            if (TotalRetired >= WarmupRetiredLimit) {
                WarmupComplete = true;
                SteadyStartCycle = CurrentCycle;
            }
        } else if (Event.Type == mca::HWInstructionEvent::Dispatched) {
            mca::Instruction &Inst = *const_cast<mca::Instruction *>(Event.IR.getInstruction());
            
            // Custom Cortex-A55 same-cycle bypasses and stalls
            if (IsA55 && PRF) {
                auto member_ptr = get_mappings(RegisterFile_RegisterMappings_Tag{});
                auto &mappings = (*PRF).*member_ptr;

                // 1. Flag-transfer penalty logic for NZCV:
                // Check if NZCV is currently defined by an FP compare or VMRS.
                bool from_fp = false;
                if (AArch64::NZCV < mappings.size()) {
                    const mca::WriteRef &WR = mappings[AArch64::NZCV].first;
                    if (WR.isValid()) {
                        unsigned writerIID = WR.getSourceIndex();
                        if (writerIID < SimInstrs.size()) {
                            const mca::Instruction *DepInst = SimInstrs[writerIID].get();
                            StringRef DepName = MCII.getName(DepInst->getOpcode());
                            if (DepName.starts_with_insensitive("FCMP") ||
                                DepName.starts_with_insensitive("VMRS") ||
                                DepName.starts_with_insensitive("VMSR")) {
                                from_fp = true;
                            }
                        }
                    }
                }
                
                for (mca::ReadState &RS : Inst.getUses()) {
                    if (RS.getRegisterID() == AArch64::NZCV) {
                        if (!from_fp) {
                            RS.*get(ReadState_IsReady_Tag{}) = true;
                            RS.setIndependentFromDef();
                        }
                    }
                }

                // 2. A64 low latency pointer forwarding bypass logic:
                // If current instruction is a load, check if its base register dependency is defined by an ADRP instruction.
                StringRef CurrName = MCII.getName(Inst.getOpcode());
                if (CurrName.starts_with_insensitive("LDR") || CurrName.starts_with_insensitive("LDUR")) {
                    for (mca::ReadState &RS : Inst.getUses()) {
                        unsigned baseReg = RS.getRegisterID();
                        if (baseReg < mappings.size()) {
                            const mca::WriteRef &WR = mappings[baseReg].first;
                            if (WR.isValid()) {
                                unsigned writerIID = WR.getSourceIndex();
                                if (writerIID < SimInstrs.size()) {
                                    const mca::Instruction *DepInst = SimInstrs[writerIID].get();
                                    StringRef DepName = MCII.getName(DepInst->getOpcode());
                                    if (DepName.equals_insensitive("ADRP")) {
                                        RS.*get(ReadState_IsReady_Tag{}) = true;
                                        RS.setIndependentFromDef();
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (IgnoreLoopCarriedDep) {
                unsigned readerIID = Event.IR.getSourceIndex();
                if (LoopSize > 0) {
                    unsigned readerIteration = readerIID / LoopSize;
                    unsigned limitIID = readerIteration * LoopSize;
                    
                    if (readerIteration > CurrentIteration) {
                        CurrentIteration = readerIteration;
                        if (PRF) {
                            auto member_ptr = get_mappings(RegisterFile_RegisterMappings_Tag{});
                            auto &mappings = (*PRF).*member_ptr;
                            for (auto &mapping : mappings) {
                                if (mapping.first.isValid()) {
                                    unsigned writerIID = mapping.first.getSourceIndex();
                                    if (writerIID < limitIID) {
                                        mapping.first = mca::WriteRef();
                                    }
                                }
                            }
                        }
                    }

                    for (mca::ReadState &RS : Inst.getUses()) {
                        if (!RS.isReady()) {
                            const mca::CriticalDependency &CRD = RS.getCriticalRegDep();
                            if (CRD.IID < limitIID) {
                                RS.*get(ReadState_IsReady_Tag{}) = true;
                                RS.setIndependentFromDef();
                            }
                        }
                    }
                }
            }
        }
    }
};

unsigned getWarmupWindowSize(const MCSubtargetInfo &STI) {
    const auto &SM = STI.getSchedModel();
    if (SM.isOutOfOrder()) return std::max(SM.MicroOpBufferSize, SM.LoopMicroOpBufferSize);
    return std::max(1u, SM.IssueWidth);
}

unsigned computeSteadyIterations(const MCSubtargetInfo &STI, size_t regionInstrCount, unsigned iterations) {
    if (regionInstrCount == 0) return 1;
    const unsigned WindowSize = std::max(1u, getWarmupWindowSize(STI));
    return std::max(1u, (WindowSize * iterations + static_cast<unsigned>(regionInstrCount) - 1) /
                            static_cast<unsigned>(regionInstrCount));
}

unsigned computeWarmupIterations(const MCSubtargetInfo &STI, size_t regionInstrCount) {
    return computeSteadyIterations(STI, regionInstrCount, 1);
}


} // namespace

void initializeTargets() {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86AsmParser();
    LLVMInitializeX86Disassembler();

    LLVMInitializeAArch64TargetInfo();
    LLVMInitializeAArch64Target();
    LLVMInitializeAArch64TargetMC();
    LLVMInitializeAArch64AsmParser();
    LLVMInitializeAArch64Disassembler();

    LLVMInitializeARMTargetInfo();
    LLVMInitializeARMTarget();
    LLVMInitializeARMTargetMC();
    LLVMInitializeARMAsmParser();
    LLVMInitializeARMDisassembler();

    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVTarget();
    LLVMInitializeRISCVTargetMC();
    LLVMInitializeRISCVAsmParser();
    LLVMInitializeRISCVDisassembler();
}

uint64_t getELFSymbolSize(const ObjectFile &Obj, SymbolRef Sym) {
    return getELFSymbolSizeImpl(Obj, Sym);
}

FunctionBoundaries collectFunctionBoundaries(const ObjectFile &Obj) {
    FunctionBoundaries boundaries;
    for (const auto &Sym : Obj.symbols()) {
        auto TypeOrErr = Sym.getType();
        if (!TypeOrErr || *TypeOrErr != SymbolRef::ST_Function) continue;

        auto AddrOrErr = Sym.getAddress();
        if (!AddrOrErr) continue;

        const uint64_t Addr = *AddrOrErr;
        const uint64_t Size = getELFSymbolSizeImpl(Obj, Sym);
        if (Size > 0) boundaries[Addr] = Addr + Size;
    }
    return boundaries;
}

std::vector<Instr> disassembleTextSection(const SectionRef &Section, const MCDisassembler &DisAsm,
                                          const MCInstrInfo &MCII, const MCInstrAnalysis *MCIA) {
    std::vector<Instr> SectionInstrs;
    auto ContentsOrErr = Section.getContents();
    if (!ContentsOrErr) return SectionInstrs;

    StringRef Contents = *ContentsOrErr;
    ArrayRef<uint8_t> Data(reinterpret_cast<const uint8_t *>(Contents.data()), Contents.size());
    SectionInstrs.reserve(Contents.size() / 4);

    for (uint64_t Index = 0; Index < Data.size();) {
        MCInst Inst;
        uint64_t Size = 0;
        auto Status = DisAsm.getInstruction(Inst, Size, Data.slice(Index), Section.getAddress() + Index, nulls());
        if (Status != MCDisassembler::Success) {
            Index++;
            continue;
        }

        Instr I;
        I.Addr = Section.getAddress() + Index;
        I.Inst = Inst;
        const MCInstrDesc &Desc = MCII.get(Inst.getOpcode());
        I.IsBranch = Desc.isBranch();
        I.IsReturn = Desc.isReturn();
        I.IsUnconditionalBranch = Desc.isUnconditionalBranch() || Desc.isIndirectBranch() || I.IsReturn;
        I.EndsBB = I.IsBranch || Desc.isTerminator();
        I.BranchTarget = 0;
        if (I.IsBranch && MCIA) MCIA->evaluateBranch(Inst, I.Addr, Size, I.BranchTarget);
        SectionInstrs.push_back(I);
        Index += Size;
    }
    return SectionInstrs;
}

McaMetrics analyzeMcaRegion(ArrayRef<Instr> instrs, const MCSubtargetInfo &STI, const MCInstrInfo &MCII,
                            const MCRegisterInfo &MRI, const MCInstrAnalysis *MCIA, const mca::PipelineOptions &PO,
                            int iterations, int windowWidth, DependencyKind depKind,
                            MLPWindowAssignmentKind assignKind, const MLPAnalyzer &analyzer,
                            bool ignoreLoopCarriedDep,
                            int overrideLoadLatency, bool mlpWindowLoop) {
    if (instrs.empty()) return {};

    mca::Context MCAContext(MRI, STI);
    mca::InstrumentManager IM(STI, MCII);
    mca::InstrBuilder IB(STI, MCII, MRI, MCIA, IM, 0);
    std::vector<mca::SourceMgr::UniqueInst> Sequence;
    SmallVector<mca::Instrument *> IVec;
    for (const auto &I : instrs) {
        auto ExpectedInst = IB.createInstruction(I.Inst, IVec);
        if (!ExpectedInst) {
            consumeError(ExpectedInst.takeError());
            return {};
        }
        
        std::unique_ptr<mca::Instruction> Inst = std::move(*ExpectedInst);
        if (STI.getCPU() == "cortex-a55") {
            unsigned opc = I.Inst.getOpcode();
            if (opc == llvm::AArch64::ADDXrs || opc == llvm::AArch64::ADDWrs ||
                opc == llvm::AArch64::SUBXrs || opc == llvm::AArch64::SUBWrs ||
                opc == llvm::AArch64::SUBSXrs || opc == llvm::AArch64::SUBSWrs ||
                opc == llvm::AArch64::ADDSXrs || opc == llvm::AArch64::ADDSWrs) {
                
                unsigned shiftVal = 0;
                if (I.Inst.getNumOperands() > 3 && I.Inst.getOperand(3).isImm()) {
                    shiftVal = llvm::AArch64_AM::getShiftValue(I.Inst.getOperand(3).getImm());
                }
                
                if (shiftVal == 0) {
                    mca::InstrDesc &MutableDesc = const_cast<mca::InstrDesc &>(Inst->getDesc());
                    for (auto &W : MutableDesc.Writes) {
                        if (W.Latency > 1) {
                            W.Latency = 1;
                        }
                    }
                    MutableDesc.MaxLatency = 1;
                }
            }
            if (opc == llvm::AArch64::Bcc || opc == llvm::AArch64::B) {
                mca::InstrDesc &MutableDesc = const_cast<mca::InstrDesc &>(Inst->getDesc());
                for (auto &W : MutableDesc.Writes) {
                    W.Latency = 0;
                }
                MutableDesc.MaxLatency = 0;
                MutableDesc.Resources.clear();
                MutableDesc.UsedProcResUnits = 0;
                MutableDesc.UsedProcResGroups = 0;
                MutableDesc.UsedBuffers = 0;
            }
        }
        if (Inst->getMayLoad() && overrideLoadLatency > 0) {
            mca::InstrDesc &MutableDesc = const_cast<mca::InstrDesc &>(Inst->getDesc());
            for (auto &W : MutableDesc.Writes) {
                if (W.Latency > 1) {
                    W.Latency = overrideLoadLatency;
                }
            }
            unsigned MaxLatency = 0;
            for (const auto &W : MutableDesc.Writes) {
                MaxLatency = std::max(MaxLatency, W.Latency);
            }
            MutableDesc.MaxLatency = MaxLatency;
        }
        
        Sequence.push_back(std::move(Inst));
    }

    const unsigned WarmupIterations = computeWarmupIterations(STI, instrs.size());
    const unsigned SteadyIterations = computeSteadyIterations(STI, instrs.size(), iterations);
    mca::CircularSourceMgr CSM(Sequence, WarmupIterations + SteadyIterations);
    mca::CustomBehaviour CB(STI, CSM, MCII);
    std::unique_ptr<mca::Pipeline> P;
    if (STI.getSchedModel().isOutOfOrder()) P = MCAContext.createDefaultPipeline(PO, CSM, CB);
    else P = MCAContext.createInOrderPipeline(PO, CSM, CB);

    mca::RegisterFile *PRF = nullptr;
    auto &hardware = MCAContext.*get(Context_Hardware_Tag{});
    for (auto &hu : hardware) {
        if (auto *r = dynamic_cast<mca::RegisterFile *>(hu.get())) {
            PRF = r;
            break;
        }
    }

    SteadyStateTracker Tracker(STI, MRI, MCII, instrs, CSM.getInstructions(), instrs.size() * WarmupIterations, instrs.size(), ignoreLoopCarriedDep, PRF);
    P->addEventListener(&Tracker);

    auto ExpectedCycles = P->run();
    if (!ExpectedCycles) {
        consumeError(ExpectedCycles.takeError());
        return {};
    }

    McaMetrics M;
    M.RetiredInstructions = Tracker.SteadyRetired;
    M.LoadInstructions = analyzer.countPotentialMissLoads(instrs, STI, MCII, MRI, depKind);
    M.Cycles = Tracker.SteadyCycles;
    if (STI.getCPU() == "cortex-a55" && instrs.size() > 0) {
        unsigned NumSteadyIterations = Tracker.SteadyRetired / instrs.size();
        if (NumSteadyIterations > 0) {
            // Cortex-A55's non-blocking branch predictor overlaps about half of the 
            // branch penalty/bubble under dual-issue (approx. 0.5 cycles per iteration).
            double correctedCycles = static_cast<double>(M.Cycles) - (static_cast<double>(NumSteadyIterations) * 0.5);
            M.Cycles = static_cast<unsigned>(correctedCycles + 0.5);
        }
    }
    M.MLP = analyzer.compute_mlp(instrs, windowWidth, depKind, assignKind, STI, MCII, MRI, M.MLP_R, mlpWindowLoop);
    if (M.RetiredInstructions > 0) {
        M.BaseCPI = static_cast<double>(M.Cycles) / static_cast<double>(M.RetiredInstructions);
    }
    M.Valid = true;
    return M;
}
