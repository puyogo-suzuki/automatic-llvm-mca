#ifndef CORE_OBJECT_LOADER_H
#define CORE_OBJECT_LOADER_H

// Target registration, ELF symbol/section access and linear disassembly.

#include <vector>
#include "core/types.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Object/ObjectFile.h"

void initializeTargets();
uint64_t getELFSymbolSize(const llvm::object::ObjectFile &Obj, llvm::object::SymbolRef Sym);
FunctionBoundaries collectFunctionBoundaries(const llvm::object::ObjectFile &Obj);
std::vector<Instr> disassembleTextSection(const llvm::object::SectionRef &Section,
                                          const llvm::MCDisassembler &DisAsm,
                                          const llvm::MCInstrInfo &MCII,
                                          const llvm::MCInstrAnalysis *MCIA);

#endif // CORE_OBJECT_LOADER_H
