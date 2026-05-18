#ifndef MCA_COMMON_H
#define MCA_COMMON_H

#include <vector>
#include <string>
#include <bitset>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"

enum class DependencyKind { None, IO, OOO };
enum class MLPWindowAssignmentKind { Forward, MaxContaining };

struct Instr {
    uint64_t Addr;
    llvm::MCInst Inst;
    bool IsBranch;
    bool EndsBB;
    uint64_t BranchTarget; 
};

typedef std::bitset<1024> RegSet;

float compute_mlp(llvm::ArrayRef<Instr> instrs, int width, 
                  DependencyKind DepKind, 
                  MLPWindowAssignmentKind AssignKind, 
                  const llvm::MCInstrInfo& MCII);

#endif
