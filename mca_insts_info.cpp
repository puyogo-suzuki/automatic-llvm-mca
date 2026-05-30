#include "mca_common.h"
#include "custom_a55_sched.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

static cl::opt<std::string> MTriple("mtriple", cl::desc("Target triple"), cl::Required);
static cl::opt<std::string> MCPU("mcpu", cl::desc("Target CPU"), cl::init("generic"));
static cl::opt<std::string> Format("format", cl::desc("Output format (csv or tsv)"), cl::init("csv"));

static std::string getBypassedMnemonic(const std::string &Name) {
    if (Name.rfind("MSR", 0) == 0) return "msr";
    if (Name.rfind("MRS", 0) == 0) return "mrs";
    if (Name.rfind("SYS", 0) == 0) return "sys";
    if (Name.rfind("DMB", 0) == 0) return "dmb";
    if (Name.rfind("DSB", 0) == 0) return "dsb";
    if (Name.rfind("ISB", 0) == 0) return "isb";
    if (Name.rfind("HINT", 0) == 0) return "hint";
    if (Name.rfind("TLBI", 0) == 0) return "tlbi";
    if (Name.rfind("DC", 0) == 0) return "dc";
    if (Name.rfind("IC", 0) == 0) return "ic";
    if (Name.rfind("AT", 0) == 0) return "at";
    if (Name.rfind("CLREX", 0) == 0) return "clrex";
    if (Name.rfind("PRFM", 0) == 0) return "prfm";
    if (Name.rfind("BTI", 0) == 0) return "bti";
    if (Name.rfind("PSB", 0) == 0) return "psb";
    if (Name.rfind("TSB", 0) == 0) return "tsb";
    if (Name.rfind("ESB", 0) == 0) return "esb";
    if (Name.rfind("HLT", 0) == 0) return "hlt";
    if (Name.rfind("BRK", 0) == 0) return "brk";
    return "";
}

static std::string getMnemonic(const MCInst &Inst, const std::string &Name, MCInstPrinter &IP, const MCSubtargetInfo &STI) {
    std::string Bypass = getBypassedMnemonic(Name);
    if (!Bypass.empty()) {
        return Bypass;
    }

    std::string AsmStr;
    raw_string_ostream OS(AsmStr);
    IP.printInst(&Inst, 0, "", STI, OS);
    OS.flush();
    
    // Trim leading whitespace
    size_t start = AsmStr.find_first_not_of(" \t");
    if (start == std::string::npos) return "";
    
    // The mnemonic is the first word (before space, tab, or newline)
    size_t end = AsmStr.find_first_of(" \t\r\n", start);
    if (end == std::string::npos) return AsmStr.substr(start);
    return AsmStr.substr(start, end - start);
}

static MCInst createDummyInst(unsigned Opcode, const MCInstrDesc &Desc, const MCRegisterInfo &MRI) {
    MCInst Inst;
    Inst.setOpcode(Opcode);
    auto Operands = Desc.operands();
    for (unsigned i = 0; i < Desc.getNumOperands(); ++i) {
        if (i < Operands.size() && Operands[i].RegClass != -1) {
            const MCRegisterClass &RC = MRI.getRegClass(Operands[i].RegClass);
            unsigned Reg = 0;
            if (RC.begin() != RC.end()) {
                Reg = *RC.begin();
            }
            Inst.addOperand(MCOperand::createReg(Reg));
        } else {
            Inst.addOperand(MCOperand::createImm(0));
        }
    }
    return Inst;
}

int main(int argc, char **argv) {
    InitLLVM X(argc, argv);
    initializeTargets();

    cl::ParseCommandLineOptions(argc, argv, "LLVM Instruction MCA Cost Model Info dumper\n");

    Triple TT(MTriple);
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
    if (!TheTarget) {
        std::cerr << "Error: No target for " << TT.str() << ": " << Error << std::endl;
        return 1;
    }

    std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TT));
    MCTargetOptions MCOPT;
    std::unique_ptr<MCAsmInfo> MAI(TheTarget->createMCAsmInfo(*MRI, TT, MCOPT));
    std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
    std::unique_ptr<MCSubtargetInfo> STI(TheTarget->createMCSubtargetInfo(TT, MCPU, ""));
    if (STI) {
        llvm::overrideCortexA55SchedModel(*STI);
    }
    std::unique_ptr<MCInstPrinter> IP(TheTarget->createMCInstPrinter(TT, 0, *MAI, *MCII, *MRI));

    if (!MRI || !MAI || !MCII || !STI || !IP) {
        std::cerr << "Error: Failed to initialize LLVM components" << std::endl;
        return 1;
    }

    const MCSchedModel &SM = STI->getSchedModel();
    char delimiter = (Format == "tsv") ? '\t' : ',';

    // Print header
    std::cout << "OpcodeID" << delimiter
              << "OpcodeName" << delimiter
              << "Mnemonic" << delimiter
              << "SchedClass" << delimiter
              << "Latency" << delimiter
              << "RThroughput" << delimiter
              << "Resources" << "\n";

    for (unsigned Opcode = 1; Opcode < MCII->getNumOpcodes(); ++Opcode) {
        const MCInstrDesc &Desc = MCII->get(Opcode);
        if (Desc.isPseudo()) {
            continue; // Skip compiler pseudo-instructions
        }

        std::string Name = std::string(MCII->getName(Opcode));
        MCInst Inst = createDummyInst(Opcode, Desc, *MRI);
        std::string Mnemonic = getMnemonic(Inst, Name, *IP, *STI);
        if (Mnemonic.empty()) {
            Mnemonic = "-";
        }

        unsigned SchedClass = Desc.getSchedClass();
        unsigned ResolvedSchedClass = SchedClass;

        if (SchedClass < SM.NumSchedClasses) {
            const MCSchedClassDesc *SCDesc = SM.getSchedClassDesc(ResolvedSchedClass);
            unsigned PrevSchedClass = ResolvedSchedClass;
            while (SCDesc && SCDesc->isVariant()) {
                ResolvedSchedClass = STI->resolveVariantSchedClass(ResolvedSchedClass, &Inst, MCII.get(), SM.getProcessorID());
                if (ResolvedSchedClass == PrevSchedClass) {
                    break;
                }
                PrevSchedClass = ResolvedSchedClass;
                SCDesc = SM.getSchedClassDesc(ResolvedSchedClass);
            }
        }

        std::string SchedClassName = std::to_string(ResolvedSchedClass);
        std::string LatencyStr = "-";
        std::string RThroughputStr = "-";
        std::string ResourcesStr = "";

        if (ResolvedSchedClass < SM.NumSchedClasses) {
            const MCSchedClassDesc *SCDesc = SM.getSchedClassDesc(ResolvedSchedClass);
            if (SCDesc) {
                if (SCDesc->isVariant()) {
                    LatencyStr = "0";
                    RThroughputStr = "0.000";
                    ResourcesStr = "-";
                } else {
                    // Compute Latency
                    int MaxLatency = 0;
                    if (SCDesc->NumWriteLatencyEntries > 0) {
                        for (unsigned DefIdx = 0; DefIdx < SCDesc->NumWriteLatencyEntries; ++DefIdx) {
                            const MCWriteLatencyEntry *WLE = STI->getWriteLatencyEntry(SCDesc, DefIdx);
                            if (WLE) {
                                MaxLatency = std::max<int>(MaxLatency, WLE->Cycles);
                            }
                        }
                        LatencyStr = std::to_string(MaxLatency);
                    } else {
                        LatencyStr = "0";
                    }

                    // Compute Reciprocal Throughput and resources
                    double RThroughput = 0.0;
                    std::vector<std::string> ResourceList;
                    unsigned NumResources = 0;

                    for (const MCWriteProcResEntry *WRE = STI->getWriteProcResBegin(SCDesc),
                                                   *WREEnd = STI->getWriteProcResEnd(SCDesc);
                         WRE != WREEnd; ++WRE) {
                        unsigned ResourceIdx = WRE->ProcResourceIdx;
                        const MCProcResourceDesc *PRD = SM.getProcResource(ResourceIdx);
                        if (PRD) {
                            unsigned Cycles = WRE->ReleaseAtCycle - WRE->AcquireAtCycle;
                            if (PRD->NumUnits > 0) {
                                double ResourceThroughput = static_cast<double>(Cycles) / PRD->NumUnits;
                                RThroughput = std::max(RThroughput, ResourceThroughput);
                            }
                            std::string resName = std::string(PRD->Name);
                            ResourceList.push_back(resName + ":" + std::to_string(Cycles));
                            NumResources++;
                        }
                    }

                    if (NumResources > 0) {
                        char buf[32];
                        std::snprintf(buf, sizeof(buf), "%.3f", RThroughput);
                        RThroughputStr = buf;
                    } else {
                        RThroughputStr = "0.000";
                    }

                    if (!ResourceList.empty()) {
                        ResourcesStr = "[";
                        for (size_t i = 0; i < ResourceList.size(); ++i) {
                            ResourcesStr += ResourceList[i];
                            if (i + 1 < ResourceList.size()) {
                                ResourcesStr += " ";
                            }
                        }
                        ResourcesStr += "]";
                    } else {
                        ResourcesStr = "-";
                    }
                }
            }
        }

        std::cout << Opcode << delimiter
                  << Name << delimiter
                  << Mnemonic << delimiter
                  << SchedClassName << delimiter
                  << LatencyStr << delimiter
                  << RThroughputStr << delimiter
                  << ResourcesStr << "\n";
    }

    return 0;
}
