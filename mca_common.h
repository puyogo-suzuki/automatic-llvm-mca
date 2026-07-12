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
    bool IsUnconditionalBranch; // unconditional branch or indirect jump (cannot fall through)
    bool IsReturn;              // return instruction
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
    std::bitset<7> flags;
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

    bool is_constant_offset() const { return flags.test(6); }
    void set_is_constant_offset(bool val) { flags.set(6, val); }
};

struct RegDeps {
    llvm::SmallVector<unsigned, 4> inputs;
    llvm::SmallVector<unsigned, 4> outputs;
};

struct MLPInstInfo {
    std::bitset<3> flags; // bit 0: is_load, bit 1: is_store, bit 2: is_call
    short num_uops = 1;
    RegDeps io_regs;
    MemAccessInfo mem_info;

    bool is_load() const { return flags.test(0); }
    bool is_store() const { return flags.test(1); }
    bool is_call() const { return flags.test(2); }
    void set_is_load(bool val) { flags.set(0, val); }
    void set_is_store(bool val) { flags.set(1, val); }
    void set_is_call(bool val) { flags.set(2, val); }
};

#include "llvm/ADT/SmallVector.h"

struct SeenBaseRegs {
    struct Element {
        unsigned base_reg;
        int64_t cache_line;
        bool is_completed;
        llvm::SmallVector<unsigned, 4> pending_dest_regs;
    };
    llvm::SmallVector<Element, 16> data;
    bool test(unsigned reg, int64_t cache_line) const;
    void set(unsigned reg, int64_t cache_line, const llvm::MCRegisterInfo &MRI);
    void reset(unsigned reg, const llvm::MCRegisterInfo &MRI);
    void add_load(unsigned reg, int64_t cache_line, const llvm::SmallVectorImpl<unsigned> &dests, const llvm::MCRegisterInfo &MRI);
    void check_uses(const llvm::SmallVectorImpl<unsigned> &uses, const llvm::MCRegisterInfo &MRI);
};

class MLPAnalyzer;

std::vector<unsigned> getReturnRegisters(const llvm::MCRegisterInfo &MRI, const std::string &ArchName);
std::vector<MLPInstInfo> buildInstInfos(llvm::ArrayRef<Instr> instrs,
                                        const llvm::MCSubtargetInfo& STI,
                                        const llvm::MCInstrInfo& MCII,
                                        const llvm::MCRegisterInfo& MRI,
                                        const MLPAnalyzer* Analyzer);
void updateSeenBaseRegs(const MLPInstInfo &inst_info, SeenBaseRegs &seen_base_regs, const llvm::MCRegisterInfo &MRI);

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

    virtual size_t countPotentialMissLoads(llvm::ArrayRef<Instr> instrs,
                                           const llvm::MCSubtargetInfo& STI,
                                           const llvm::MCInstrInfo& MCII,
                                           const llvm::MCRegisterInfo& MRI,
                                           DependencyKind depKind = DependencyKind::None) const;
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

struct McaRegion {
    size_t Start;
    size_t Size;
    size_t SimulatedSize;
    bool IsLoop;
    McaMetrics Metrics;
    bool Valid = false;
    uint64_t StartAddr;
    uint64_t EndAddr;
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
                 int bbMaxInstrs, int nestLimitOuter, int nestLimitInner,
                 const std::function<void(const RegionSpan &)> &onLoop,
                 const std::function<void(const RegionSpan &)> &onBasicBlock);
void computePostDominatorOverwrites(llvm::ArrayRef<Instr> SectionInstrs,
                                    const FunctionBoundaries &Boundaries,
                                    std::vector<McaRegion> &regions,
                                    std::map<size_t, size_t> &overwrite_map);

bool isNopInstruction(const llvm::MCInst &Inst, const llvm::MCInstrInfo &MCII);
bool isAllNopRegion(llvm::ArrayRef<Instr> instrs, const llvm::MCInstrInfo &MCII);

#endif
