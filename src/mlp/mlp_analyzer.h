#ifndef MLP_MLP_ANALYZER_H
#define MLP_MLP_ANALYZER_H

// MLP (memory-level parallelism) analysis, one analyzer per ISA.

#include <memory>
#include <string>
#include "mlp/mlp_types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"

class MLPAnalyzer;

std::vector<unsigned> getReturnRegisters(const llvm::MCRegisterInfo &MRI, const std::string &ArchName);
std::vector<MLPInstInfo> buildInstInfos(llvm::ArrayRef<Instr> instrs,
                                        const llvm::MCSubtargetInfo& STI,
                                        const llvm::MCInstrInfo& MCII,
                                        const llvm::MCRegisterInfo& MRI,
                                        const MLPAnalyzer* Analyzer);
void updateSeenBaseRegs(const MLPInstInfo &inst_info, SeenBaseRegs &seen_base_regs, const llvm::MCRegisterInfo &MRI);
// Records this instruction's store (if any) into seen_store_addrs, then
// invalidates any entries whose base register is redefined by this
// instruction's outputs (any instruction, not just stores, since a register
// redefinition changes the address a base_reg denotes).
void updateSeenStoreAddrs(const MLPInstInfo &inst_info, SeenStoreAddrs &seen_store_addrs, const llvm::MCRegisterInfo &MRI);

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
                                           DependencyKind depKind = DependencyKind::None,
                                           bool mlpWindowLoop = false) const;
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

#endif // MLP_MLP_ANALYZER_H
