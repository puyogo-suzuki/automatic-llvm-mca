#include "mca_common.h"
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <set>
#include <map>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace llvm::object;

static cl::opt<std::string> InputBinary(cl::Positional, cl::desc("<input binary>"), cl::Required);

struct Stats {
    size_t count = 0;
    size_t max_size = 0;
    double total_instrs = 0;
};

void print_stats(const std::string& label, const Stats& s) {
    std::cout << label << ":\n";
    std::cout << "  Count:      " << s.count << "\n";
    std::cout << "  Max Size:   " << s.max_size << "\n";
    std::cout << "  Avg Size:   " << (s.count > 0 ? s.total_instrs / s.count : 0) << "\n";
}

int main(int argc, char **argv) {
    InitLLVM X(argc, argv);
    LLVMInitializeX86TargetInfo(); LLVMInitializeX86Target(); LLVMInitializeX86TargetMC(); LLVMInitializeX86AsmParser(); LLVMInitializeX86Disassembler();
    LLVMInitializeAArch64TargetInfo(); LLVMInitializeAArch64Target(); LLVMInitializeAArch64TargetMC(); LLVMInitializeAArch64AsmParser(); LLVMInitializeAArch64Disassembler();
    LLVMInitializeARMTargetInfo(); LLVMInitializeARMTarget(); LLVMInitializeARMTargetMC(); LLVMInitializeARMAsmParser(); LLVMInitializeARMDisassembler();
    LLVMInitializeRISCVTargetInfo(); LLVMInitializeRISCVTarget(); LLVMInitializeRISCVTargetMC(); LLVMInitializeRISCVAsmParser(); LLVMInitializeRISCVDisassembler();

    cl::ParseCommandLineOptions(argc, argv, "Analysis Strategy Comparator\n");

    auto BinaryOrErr = ObjectFile::createObjectFile(InputBinary);
    if (!BinaryOrErr) return 1;
    ObjectFile &Obj = *BinaryOrErr.get().getBinary();

    Triple TT = Obj.makeTriple();
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
    if (!TheTarget) return 1;

    std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TT));
    MCTargetOptions MCOPT;
    std::unique_ptr<MCAsmInfo> MAI(TheTarget->createMCAsmInfo(*MRI, TT, MCOPT));
    std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
    std::unique_ptr<MCSubtargetInfo> STI(TheTarget->createMCSubtargetInfo(TT, "generic", ""));
    MCContext Ctx(TT, MAI.get(), MRI.get(), STI.get());
    std::unique_ptr<MCDisassembler> DisAsm(TheTarget->createMCDisassembler(*STI, Ctx));
    std::unique_ptr<MCInstrAnalysis> MCIA(TheTarget->createMCInstrAnalysis(MCII.get()));

    std::map<uint64_t, uint64_t> FunctionBoundaries;
    for (const auto &Sym : Obj.symbols()) {
        auto TypeOrErr = Sym.getType();
        if (!TypeOrErr || *TypeOrErr != SymbolRef::ST_Function) continue;
        auto AddrOrErr = Sym.getAddress();
        if (!AddrOrErr) continue;
        uint64_t Addr = *AddrOrErr;
        uint64_t Size = 0;
        if (auto *Elf32LE = dyn_cast<ELF32LEObjectFile>(&Obj)) {
            if (auto SymOrErr = Elf32LE->getSymbol(Sym.getRawDataRefImpl())) Size = (*SymOrErr)->st_size;
        } else if (auto *Elf64LE = dyn_cast<ELF64LEObjectFile>(&Obj)) {
            if (auto SymOrErr = Elf64LE->getSymbol(Sym.getRawDataRefImpl())) Size = (*SymOrErr)->st_size;
        } else if (auto *Elf32BE = dyn_cast<ELF32BEObjectFile>(&Obj)) {
            if (auto SymOrErr = Elf32BE->getSymbol(Sym.getRawDataRefImpl())) Size = (*SymOrErr)->st_size;
        } else if (auto *Elf64BE = dyn_cast<ELF64BEObjectFile>(&Obj)) {
            if (auto SymOrErr = Elf64BE->getSymbol(Sym.getRawDataRefImpl())) Size = (*SymOrErr)->st_size;
        }
        if (Size > 0) FunctionBoundaries[Addr] = Addr + Size;
    }

    Stats optLoop;
    std::map<size_t, std::map<size_t, size_t>> dist2D; // [InstrBucket100][BranchCount] -> Freq
    
    for (const SectionRef &Section : Obj.sections()) {
        if (!Section.isText() || Section.getSize() == 0) continue;
        uint64_t SAddr = Section.getAddress();
        auto ContentsOrErr = Section.getContents();
        if (!ContentsOrErr) continue;
        StringRef Contents = *ContentsOrErr;

        std::vector<Instr> SectionInstrs;
        SectionInstrs.reserve(Contents.size() / 4);
        ArrayRef<uint8_t> Data(reinterpret_cast<const uint8_t*>(Contents.data()), Contents.size());
        for (uint64_t Index = 0; Index < Data.size(); ) {
            MCInst Inst;
            uint64_t Size;
            auto Status = DisAsm->getInstruction(Inst, Size, Data.slice(Index), SAddr + Index, nulls());
            if (Status != MCDisassembler::Success) { Index++; continue; }
            Instr I;
            I.Addr = SAddr + Index; I.Inst = Inst;
            I.IsBranch = MCII->get(Inst.getOpcode()).isBranch();
            I.EndsBB = I.IsBranch || MCII->get(Inst.getOpcode()).isTerminator();
            I.BranchTarget = 0;
            if (I.IsBranch) MCIA->evaluateBranch(Inst, I.Addr, Size, I.BranchTarget);
            SectionInstrs.push_back(I);
            Index += Size;
        }

        auto find_idx = [&](uint64_t addr) -> int64_t {
            auto it = std::lower_bound(SectionInstrs.begin(), SectionInstrs.end(), addr,
                                       [](const Instr& a, uint64_t val) { return a.Addr < val; });
            if (it != SectionInstrs.end() && it->Addr == addr)
                return std::distance(SectionInstrs.begin(), it);
            return -1;
        };
        
        for (size_t i = 0; i < SectionInstrs.size(); ++i) {
            const auto& I = SectionInstrs[i];
            if (I.IsBranch && I.BranchTarget != 0 && I.BranchTarget < I.Addr) {
                int64_t start_idx = find_idx(I.BranchTarget);
                if (start_idx != -1) {
                    size_t sz = i - (size_t)start_idx + 1;
                    bool same_func = false;
                    auto it = FunctionBoundaries.upper_bound(I.Addr);
                    if (it != FunctionBoundaries.begin()) {
                        --it;
                        if (I.Addr >= it->first && I.Addr < it->second && I.BranchTarget >= it->first && I.BranchTarget < it->second) same_func = true;
                    }
                    if (same_func && sz <= 1000) {
                        optLoop.count++;
                        optLoop.total_instrs += sz;
                        if (sz > optLoop.max_size) optLoop.max_size = sz;

                        size_t branchCount = 0;
                        for (size_t j = (size_t)start_idx; j < i; ++j) {
                            if (SectionInstrs[j].IsBranch) branchCount++;
                        }
                        size_t instrBucket = (sz / 100) * 100;
                        dist2D[instrBucket][branchCount]++;
                    }
                }
            }
        }
    }

    std::cout << "=== 2D Distribution: Instr Count (Buckets of 100) vs Internal Branch Count ===\n";
    std::cout << "InstrRange, BranchCount, Frequency\n";
    for (auto const& [instrBucket, branchMap] : dist2D) {
        for (auto const& [brCount, freq] : branchMap) {
            std::cout << instrBucket << "-" << (instrBucket + 99) << ", " << brCount << ", " << freq << "\n";
        }
    }

    return 0;
}
