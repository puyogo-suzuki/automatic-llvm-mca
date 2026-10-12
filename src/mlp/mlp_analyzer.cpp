#include "mlp/mlp_analyzer.h"
#include <algorithm>
#include <bitset>
#include <cmath>
#include <iostream>
#include <string>
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

namespace {

static int getNumMicroOps(const llvm::MCInst &Inst, const llvm::MCSubtargetInfo &STI, const llvm::MCInstrInfo &MCII) {
    const llvm::MCSchedModel &SM = STI.getSchedModel();
    const llvm::MCInstrDesc &Desc = MCII.get(Inst.getOpcode());
    unsigned SchedClass = Desc.getSchedClass();
    unsigned ResolvedSchedClass = SchedClass;

    if (SchedClass < SM.NumSchedClasses) {
        const llvm::MCSchedClassDesc *SCDesc = SM.getSchedClassDesc(ResolvedSchedClass);
        unsigned PrevSchedClass = ResolvedSchedClass;
        while (SCDesc && SCDesc->isVariant()) {
            ResolvedSchedClass = STI.resolveVariantSchedClass(ResolvedSchedClass, &Inst, &MCII, SM.getProcessorID());
            if (ResolvedSchedClass == PrevSchedClass) {
                break;
            }
            PrevSchedClass = ResolvedSchedClass;
            SCDesc = SM.getSchedClassDesc(ResolvedSchedClass);
        }
        if (SCDesc && SCDesc->isValid()) {
            return SCDesc->NumMicroOps;
        }
    }
    return 1;
}

} // namespace

std::vector<unsigned> getReturnRegisters(const llvm::MCRegisterInfo &MRI, const std::string &ArchName) {
    std::vector<unsigned> reg_ids;
    StringRef arch = ArchName;
    if (arch.contains_insensitive("aarch64")) {
        unsigned num_regs = MRI.getNumRegs();
        for (unsigned r = 1; r < num_regs; ++r) {
            StringRef name = MRI.getName(r);
            if (name.empty()) continue;
            char first = name[0];
            if (first >= 'A' && first <= 'Z') {
                first = first - 'A' + 'a';
            }
            if (first == 'x' || first == 'w' || first == 'v' || first == 's' || first == 'd' || first == 'h' || first == 'b' || first == 'q') {
                StringRef num_part = name.drop_front(1);
                unsigned val;
                if (!num_part.getAsInteger(10, val) && val <= 7) {
                    reg_ids.push_back(r);
                }
            }
        }
    } else if (arch.contains_insensitive("x86")) {
        unsigned num_regs = MRI.getNumRegs();
        for (unsigned r = 1; r < num_regs; ++r) {
            StringRef name = MRI.getName(r);
            if (name.empty()) continue;
            if (name.equals_insensitive("rax") || name.equals_insensitive("eax") || name.equals_insensitive("ax") || name.equals_insensitive("al") || name.equals_insensitive("ah") ||
                name.equals_insensitive("rdx") || name.equals_insensitive("edx") || name.equals_insensitive("dx") || name.equals_insensitive("dl") || name.equals_insensitive("dh") ||
                name.equals_insensitive("rcx") || name.equals_insensitive("ecx") || name.equals_insensitive("cx") || name.equals_insensitive("cl") || name.equals_insensitive("ch")) {
                reg_ids.push_back(r);
            } else if (name.starts_with_insensitive("xmm") || name.starts_with_insensitive("ymm") || name.starts_with_insensitive("zmm")) {
                StringRef num_part = name.drop_front(3);
                unsigned val;
                if (!num_part.getAsInteger(10, val) && val <= 7) {
                    reg_ids.push_back(r);
                }
            }
        }
    } else if (arch.contains_insensitive("riscv")) {
        unsigned num_regs = MRI.getNumRegs();
        for (unsigned r = 1; r < num_regs; ++r) {
            StringRef name = MRI.getName(r);
            if (name.empty()) continue;
            if (name.equals_insensitive("a0") || name.equals_insensitive("x10") ||
                name.equals_insensitive("a1") || name.equals_insensitive("x11") ||
                name.equals_insensitive("fa0") || name.equals_insensitive("f10") ||
                name.equals_insensitive("fa1") || name.equals_insensitive("f11")) {
                reg_ids.push_back(r);
            }
        }
    }
    return reg_ids;
}

std::vector<MLPInstInfo> buildInstInfos(
    llvm::ArrayRef<Instr> instrs,
    const llvm::MCSubtargetInfo& STI,
    const llvm::MCInstrInfo& MCII,
    const llvm::MCRegisterInfo& MRI,
    const MLPAnalyzer* Analyzer) {
    int n = instrs.size();
    std::vector<MLPInstInfo> inst_infos(n);
    for (int i = 0; i < n; ++i) {
        const MCInst& Inst = instrs[i].Inst;
        const MCInstrDesc& MCID = MCII.get(Inst.getOpcode());
        MemAccessInfo mem_info = Analyzer->getMemAccessInfo(Inst, MCID, MRI, MCII);
        inst_infos[i].set_is_load(mem_info.is_load());
        inst_infos[i].set_is_store(mem_info.is_store());
        bool is_call = MCID.isCall();
        if (!is_call) {
            StringRef name = MCII.getName(Inst.getOpcode());
            if (name.equals_insensitive("JAL") || name.equals_insensitive("JALR")) {
                if (Inst.getNumOperands() > 0 && Inst.getOperand(0).isReg()) {
                    unsigned rd = Inst.getOperand(0).getReg();
                    if (const char* rd_name = MRI.getName(rd)) {
                        StringRef rdn(rd_name);
                        if (rdn.equals_insensitive("x1") || rdn.equals_insensitive("x5") ||
                            rdn.equals_insensitive("ra") || rdn.equals_insensitive("t0")) {
                            is_call = true;
                        }
                    }
                }
            }
        }
        inst_infos[i].set_is_call(is_call);

        inst_infos[i].num_uops = static_cast<short>(getNumMicroOps(Inst, STI, MCII));
        inst_infos[i].mem_info = mem_info;
        {
            // AArch64 pair / exclusive-pair forms transfer two registers.
            std::string nm = MCII.getName(Inst.getOpcode()).upper();
            auto starts = [&](const char *pfx) { return StringRef(nm).starts_with(pfx); };
            bool pair_load = starts("LDP") || starts("LDNP") || starts("LDXP") || starts("LDAXP");
            bool pair_store = starts("STP") || starts("STNP") || starts("STXP") || starts("STLXP");
            inst_infos[i].load_slots = mem_info.is_load() ? (pair_load ? 2 : 1) : 0;
            inst_infos[i].store_slots = mem_info.is_store() ? (pair_store ? 2 : 1) : 0;
            inst_infos[i].branch_slots = (MCID.isBranch() || MCID.isCall() || MCID.isReturn()) ? 1 : 0;
        }
        for (unsigned j = 0; j < Inst.getNumOperands(); ++j) {
            const MCOperand& Op = Inst.getOperand(j);
            if (Op.isReg() && Op.getReg() != 0) {
                unsigned reg = Op.getReg();
                if (Analyzer->isZeroRegister(reg, MRI)) continue;
                if (j < MCID.getNumDefs()) inst_infos[i].io_regs.outputs.push_back(reg);
                else inst_infos[i].io_regs.inputs.push_back(reg);
            }
        }
        for (MCPhysReg R : MCID.implicit_defs()) {
            if (!Analyzer->isZeroRegister(R, MRI)) inst_infos[i].io_regs.outputs.push_back(R);
        }
        for (MCPhysReg R : MCID.implicit_uses()) {
            if (!Analyzer->isZeroRegister(R, MRI)) inst_infos[i].io_regs.inputs.push_back(R);
        }
        inst_infos[i].rename_entries = static_cast<short>(inst_infos[i].io_regs.outputs.size());
    }
    return inst_infos;
}

bool MLPAnalyzer::isZeroRegister(unsigned reg, const llvm::MCRegisterInfo &MRI) const {
    if (reg == 0) return true;
    if (const char* name = MRI.getName(reg)) {
        std::string s(name);
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return (s == "zero");
    }
    return false;
}

std::unique_ptr<MLPAnalyzer> MLPAnalyzer::create(const llvm::MCSubtargetInfo &STI) {
    std::string Arch = STI.getTargetTriple().getArchName().str();
    std::transform(Arch.begin(), Arch.end(), Arch.begin(), ::tolower);
    if (Arch.rfind("riscv", 0) == 0) {
        return std::make_unique<RISCVMLPAnalyzer>();
    } else if (Arch.rfind("x86", 0) == 0 || Arch == "i386") {
        return std::make_unique<X86MLPAnalyzer>();
    } else {
        return std::make_unique<AArch64MLPAnalyzer>();
    }
}
