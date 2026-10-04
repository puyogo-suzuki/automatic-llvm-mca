#include "core/object_loader.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;
using namespace llvm::object;

namespace {

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
        I.IsCall = Desc.isCall();
        I.IsBranch = Desc.isBranch() && !I.IsCall;
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
