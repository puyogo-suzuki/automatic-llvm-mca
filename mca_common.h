#ifndef MCA_COMMON_H
#define MCA_COMMON_H

#include <functional>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MCA/Context.h"
#include "llvm/Object/ObjectFile.h"

enum class DependencyKind { None, IO, OOO, Dependency };
enum class MLPWindowAssignmentKind { Forward, MaxContaining };
enum class IgnoreLoopCarriedMode { Default, Force, Disable };
enum class MlpWindowLoopMode { Default, Force, Disable };

struct Instr {
    uint64_t Addr;
    llvm::MCInst Inst;
    bool IsBranch;
    bool EndsBB;
    uint64_t BranchTarget; 
};

using RegSet = llvm::BitVector;
using FunctionBoundaries = std::map<uint64_t, uint64_t>;

struct RegionSpan {
    size_t Start;
    size_t Size;
};

struct McaMetrics {
    uint64_t RetiredInstructions = 0;
    uint64_t LoadInstructions = 0;
    uint64_t Cycles = 0;
    float MLP = 0.0f;
    float MLP_R = 0.0f;
    double BaseCPI = 0.0;
    bool Valid = false;
};

void initializeTargets();
uint64_t getELFSymbolSize(const llvm::object::ObjectFile &Obj, llvm::object::SymbolRef Sym);
FunctionBoundaries collectFunctionBoundaries(const llvm::object::ObjectFile &Obj);
std::vector<Instr> disassembleTextSection(const llvm::object::SectionRef &Section,
                                          const llvm::MCDisassembler &DisAsm,
                                          const llvm::MCInstrInfo &MCII,
                                          const llvm::MCInstrAnalysis *MCIA);
McaMetrics analyzeMcaRegion(llvm::ArrayRef<Instr> instrs, const llvm::MCSubtargetInfo &STI,
                            const llvm::MCInstrInfo &MCII, const llvm::MCRegisterInfo &MRI,
                            const llvm::MCInstrAnalysis *MCIA, const llvm::mca::PipelineOptions &PO,
                            int iterations, int windowWidth, DependencyKind depKind,
                            MLPWindowAssignmentKind assignKind, bool ignoreLoopCarriedDep = false,
                            int overrideLoadLatency = -1, bool mlpWindowLoop = false);
void walkRegions(llvm::ArrayRef<Instr> instrs, const FunctionBoundaries &boundaries, int loopMaxInstrs,
                 int bbMaxInstrs, const std::function<void(const RegionSpan &)> &onLoop,
                 const std::function<void(const RegionSpan &)> &onBasicBlock);

float compute_mlp(llvm::ArrayRef<Instr> instrs, int width, 
                  DependencyKind DepKind, 
                  MLPWindowAssignmentKind AssignKind, 
                  const llvm::MCSubtargetInfo& STI,
                  const llvm::MCInstrInfo& MCII,
                  const llvm::MCRegisterInfo& MRI,
                  float &mlp_r,
                  bool mlpWindowLoop = false);

size_t countNonStackLoads(llvm::ArrayRef<Instr> instrs,
                          const llvm::MCSubtargetInfo& STI,
                          const llvm::MCInstrInfo& MCII,
                          const llvm::MCRegisterInfo& MRI);

#endif
