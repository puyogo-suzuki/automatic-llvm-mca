#ifndef MCA_COMMON_H
#define MCA_COMMON_H

#include <functional>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <bitset>
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

struct MemAccessInfo {
    std::bitset<6> flags;
    unsigned base_reg = 0;
    int64_t offset = 0;

    bool valid() const { return flags.test(0); }
    void set_valid(bool val) { flags.set(0, val); }

    bool is_pc_relative() const { return flags.test(1); }
    void set_is_pc_relative(bool val) { flags.set(1, val); }

    bool is_stack_access() const { return flags.test(2); }
    void set_is_stack_access(bool val) { flags.set(2, val); }

    bool is_load() const { return flags.test(3); }
    void set_is_load(bool val) { flags.set(3, val); }

    bool is_store() const { return flags.test(4); }
    void set_is_store(bool val) { flags.set(4, val); }

    bool is_writeback() const { return flags.test(5); }
    void set_is_writeback(bool val) { flags.set(5, val); }
};

class MLPAnalyzer {
public:
    virtual ~MLPAnalyzer() = default;

    static std::unique_ptr<MLPAnalyzer> create(const llvm::MCSubtargetInfo &STI);

    virtual MemAccessInfo getMemAccessInfo(const llvm::MCInst &Inst,
                                           const llvm::MCInstrDesc &MCID,
                                           const llvm::MCRegisterInfo &MRI,
                                           const llvm::MCInstrInfo &MCII) const = 0;

    virtual bool isZeroRegister(unsigned reg, const llvm::MCRegisterInfo &MRI) const;

    virtual float compute_mlp(llvm::ArrayRef<Instr> instrs, int width,
                              DependencyKind DepKind,
                              MLPWindowAssignmentKind AssignKind,
                              const llvm::MCSubtargetInfo& STI,
                              const llvm::MCInstrInfo& MCII,
                              const llvm::MCRegisterInfo& MRI,
                              float &mlp_r,
                              bool mlpWindowLoop = false) const;

    virtual size_t countNonStackLoads(llvm::ArrayRef<Instr> instrs,
                                      const llvm::MCSubtargetInfo& STI,
                                      const llvm::MCInstrInfo& MCII,
                                      const llvm::MCRegisterInfo& MRI) const;
};

class RISCVMLPAnalyzer : public MLPAnalyzer {
public:
    MemAccessInfo getMemAccessInfo(const llvm::MCInst &Inst,
                                   const llvm::MCInstrDesc &MCID,
                                   const llvm::MCRegisterInfo &MRI,
                                   const llvm::MCInstrInfo &MCII) const override;
    bool isZeroRegister(unsigned reg, const llvm::MCRegisterInfo &MRI) const override;
};

class X86MLPAnalyzer : public MLPAnalyzer {
public:
    MemAccessInfo getMemAccessInfo(const llvm::MCInst &Inst,
                                   const llvm::MCInstrDesc &MCID,
                                   const llvm::MCRegisterInfo &MRI,
                                   const llvm::MCInstrInfo &MCII) const override;
    bool isZeroRegister(unsigned reg, const llvm::MCRegisterInfo &MRI) const override;
};

class AArch64MLPAnalyzer : public MLPAnalyzer {
public:
    MemAccessInfo getMemAccessInfo(const llvm::MCInst &Inst,
                                   const llvm::MCInstrDesc &MCID,
                                   const llvm::MCRegisterInfo &MRI,
                                   const llvm::MCInstrInfo &MCII) const override;
    bool isZeroRegister(unsigned reg, const llvm::MCRegisterInfo &MRI) const override;
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
                            MLPWindowAssignmentKind assignKind, const MLPAnalyzer &analyzer,
                            bool ignoreLoopCarriedDep = false,
                            int overrideLoadLatency = -1, bool mlpWindowLoop = false);
void walkRegions(llvm::ArrayRef<Instr> instrs, const FunctionBoundaries &boundaries, int loopMaxInstrs,
                 int bbMaxInstrs, const std::function<void(const RegionSpan &)> &onLoop,
                 const std::function<void(const RegionSpan &)> &onBasicBlock);

#endif
