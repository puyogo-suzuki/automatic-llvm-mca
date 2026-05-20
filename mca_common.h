#ifndef MCA_COMMON_H
#define MCA_COMMON_H

#include <vector>
#include <string>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"

enum class DependencyKind { None, IO, OOO };
enum class MLPWindowAssignmentKind { Forward, MaxContaining };

struct Instr {
    uint64_t Addr;
    llvm::MCInst Inst;
    bool IsBranch;
    bool EndsBB;
    uint64_t BranchTarget; 
};

using RegSet = llvm::BitVector;

float compute_mlp(llvm::ArrayRef<Instr> instrs, int width, 
                  DependencyKind DepKind, 
                  MLPWindowAssignmentKind AssignKind, 
                  const llvm::MCInstrInfo& MCII,
                  const llvm::MCRegisterInfo& MRI);

#endif
