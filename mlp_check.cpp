#include "mca_common.h"
#include "custom_a55_sched.h"
#include <cstdio>
#include <algorithm>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <unistd.h>
#include <iomanip>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace llvm::object;

static cl::opt<std::string> InputBinary(cl::Positional, cl::desc("<input binary>"), cl::Required);
static cl::opt<std::string> TargetAddressStr("target-address", cl::desc("Only analyze region containing this target address"), cl::Required);
static cl::opt<std::string> MTriple("mtriple", cl::desc("Target triple"));
static cl::opt<std::string> MCPU("mcpu", cl::desc("Target CPU"));
static cl::opt<int> WindowWidth("window-width", cl::desc("MLP window width"), cl::init(4));
static cl::opt<DependencyKind> DepKind("dependency", cl::desc("Dependency mode"),
    cl::values(
        clEnumValN(DependencyKind::None, "none", "No dependency tracking"),
        clEnumValN(DependencyKind::IO, "io", "In-order dependency"),
        clEnumValN(DependencyKind::OOO, "ooo", "Out-of-order dependency"),
        clEnumValN(DependencyKind::Dependency, "dependency", "Load-use dependency distance")
    ), cl::init(DependencyKind::OOO));
static cl::opt<MLPWindowAssignmentKind> AssignKind("mlp-window-assignment", cl::desc("Per-load MLP assignment mode"),
    cl::values(
        clEnumValN(MLPWindowAssignmentKind::Forward, "forward", "Forward window"),
        clEnumValN(MLPWindowAssignmentKind::MaxContaining, "max-containing", "Max MLP of containing windows")
    ), cl::init(MLPWindowAssignmentKind::MaxContaining));
static cl::opt<int> LoopMaxInstrs("loop-max-instrs", cl::desc("Maximum instructions in a loop to analyze"), cl::init(100));
static cl::opt<int> BBMaxInstrs("bb-max-instrs", cl::desc("Maximum instructions in a basic block to analyze"), cl::init(100));
static cl::opt<int> NestLimitOuter("nest-limit-outer", cl::desc("Maximum nesting depth of loops to analyze from outer to inner"), cl::init(2));
static cl::opt<int> NestLimitInner("nest-limit-inner", cl::desc("Maximum nesting depth of loops to analyze from inner to outer"), cl::init(2));

static bool has_intersection(const llvm::SmallVectorImpl<unsigned>& regs, const RegSet& mask) {
    for (unsigned r : regs) {
        if (r < mask.size() && mask.test(r)) return true;
    }
    return false;
}

static void set_reg(RegSet &mask, unsigned reg, const llvm::MCRegisterInfo &MRI) {
    if (reg == 0) return;
    for (llvm::MCRegAliasIterator AI(reg, &MRI, true); AI.isValid(); ++AI) {
        unsigned alias = *AI;
        if (alias < mask.size()) mask.set(alias);
    }
}

static void reset_reg(RegSet &mask, unsigned reg, const llvm::MCRegisterInfo &MRI) {
    if (reg == 0) return;
    for (llvm::MCRegAliasIterator AI(reg, &MRI, true); AI.isValid(); ++AI) {
        unsigned alias = *AI;
        if (alias < mask.size()) mask.reset(alias);
    }
}

static void printInstruction(MCInstPrinter &IP, const MCInst &Inst, uint64_t Addr, const MCSubtargetInfo &STI) {
    std::string InstText;
    raw_string_ostream OS(InstText);
    IP.printInst(&Inst, Addr, "", STI, OS);
    OS.flush();
    std::cout << InstText;
}

int main(int argc, char **argv) {
    InitLLVM X(argc, argv);
    initializeTargets();

    cl::ParseCommandLineOptions(argc, argv, "mlp-check debug tool\n");

    uint64_t TargetAddress = 0;
    if (TargetAddressStr.empty() || StringRef(TargetAddressStr).getAsInteger(0, TargetAddress)) {
        WithColor::error() << "Invalid target address specified.\n";
        return 1;
    }

    auto BinaryOrErr = ObjectFile::createObjectFile(InputBinary);
    if (!BinaryOrErr) {
        WithColor::error() << "Failed to open binary: " << toString(BinaryOrErr.takeError()) << "\n";
        return 1;
    }
    ObjectFile &Obj = *BinaryOrErr.get().getBinary();

    Triple TT = Obj.makeTriple();
    if (!MTriple.empty()) TT = Triple(MTriple);
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
    if (!TheTarget) {
        WithColor::error() << "No target for " << TT.str() << ": " << Error << "\n";
        return 1;
    }

    std::string CPU = MCPU.empty() ? "generic" : std::string(MCPU);
    std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TT));
    MCTargetOptions MCOPT;
    std::unique_ptr<MCAsmInfo> MAI(TheTarget->createMCAsmInfo(*MRI, TT, MCOPT));
    std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
    std::unique_ptr<MCSubtargetInfo> STI(TheTarget->createMCSubtargetInfo(TT, CPU, ""));
    if (STI) {
        llvm::overrideCortexA55SchedModel(*STI);
    }
    MCContext Ctx(TT, MAI.get(), MRI.get(), STI.get());
    std::unique_ptr<MCDisassembler> DisAsm(TheTarget->createMCDisassembler(*STI, Ctx));
    std::unique_ptr<MCInstrAnalysis> MCIA(TheTarget->createMCInstrAnalysis(MCII.get()));
    std::unique_ptr<MCInstPrinter> IP(TheTarget->createMCInstPrinter(TT, 0, *MAI, *MCII, *MRI));

    if (!MRI || !MAI || !MCII || !STI || !DisAsm || !MCIA || !IP) {
        WithColor::error() << "Failed to initialize LLVM components\n";
        return 1;
    }

    IP->setPrintImmHex(true);
    IP->setPrintBranchImmAsAddress(true);

    std::unique_ptr<MLPAnalyzer> Analyzer = MLPAnalyzer::create(*STI);
    const MCSchedModel &SM = STI->getSchedModel();
    int windowWidth = WindowWidth;
    if (WindowWidth.getNumOccurrences() == 0 && !MCPU.empty()) {
        if (SM.MicroOpBufferSize > 0) {
            windowWidth = SM.MicroOpBufferSize;
        } else {
            windowWidth = SM.IssueWidth * SM.MispredictPenalty;
        }
    }
    FunctionBoundaries FunctionRanges = collectFunctionBoundaries(Obj);

    bool Found = false;
    for (const SectionRef &Section : Obj.sections()) {
        if (!Section.isText() || Section.getSize() == 0) continue;

        auto SectionInstrs = disassembleTextSection(Section, *DisAsm, *MCII, MCIA.get());
        if (SectionInstrs.empty()) continue;

        std::vector<McaRegion> regions;
        walkRegions(SectionInstrs, FunctionRanges, LoopMaxInstrs, BBMaxInstrs, NestLimitOuter, NestLimitInner,
                    [&](const RegionSpan &Span) {
                        McaRegion r;
                        r.Start = Span.Start;
                        r.Size = Span.Size;
                        r.IsLoop = true;
                        r.StartAddr = SectionInstrs[Span.Start].Addr;
                        r.EndAddr = SectionInstrs[Span.Start + Span.Size - 1].Addr + 4;
                        regions.push_back(r);
                    },
                    [&](const RegionSpan &Span) {
                        McaRegion r;
                        r.Start = Span.Start;
                        r.Size = Span.Size;
                        r.IsLoop = false;
                        r.StartAddr = SectionInstrs[Span.Start].Addr;
                        r.EndAddr = SectionInstrs[Span.Start + Span.Size - 1].Addr + 4;
                        regions.push_back(r);
                    });

        for (auto &r : regions) {
            if (TargetAddress >= r.StartAddr && TargetAddress < r.EndAddr) {
                Found = true;
                std::cout << "\n=======================================================\n";
                std::cout << "FOUND TARGET ADDRESS 0x" << std::hex << TargetAddress 
                          << " IN REGION [0x" << r.StartAddr << " - 0x" << r.EndAddr << "]\n";
                std::cout << "Region Type: " << (r.IsLoop ? "Loop" : "Basic Block") << "\n";
                std::cout << "=======================================================\n\n";

                auto region_instrs = ArrayRef<Instr>(SectionInstrs).slice(r.Start, r.Size);
                int n = region_instrs.size();
                
                // Disassemble region with indexes
                std::cout << "--- Disassembly ---\n";
                for (int i = 0; i < n; ++i) {
                    const auto &I = region_instrs[i];
                    std::cout << "  [" << std::setw(2) << std::dec << i << "]  0x" << std::hex << I.Addr << ":  ";
                    printInstruction(*IP, I.Inst, I.Addr, *STI);
                    std::cout << "\n";
                }
                std::cout << "\n";

                // Evaluate pre-loads and dependencies
                const unsigned reg_count = MRI->getNumRegs() + 1;
                std::vector<unsigned> return_regs = getReturnRegisters(*MRI, STI->getTargetTriple().getArchName().str());
                
                // Reconstruct MLPInstInfo using common helper
                std::vector<MLPInstInfo> inst_infos = buildInstInfos(region_instrs, *STI, *MCII, *MRI, Analyzer.get());

                // Filter load indices
                std::vector<int> load_indices;
                SeenBaseRegs global_seen_base_regs;
                std::cout << "--- Base Register Cache Match Scan ---\n";
                for (int i = 0; i < n; ++i) {
                    bool is_hit = false;
                    if (inst_infos[i].is_load()) {
                        if (inst_infos[i].mem_info.is_stack_access()) {
                            std::cout << "  [" << i << "] Load is STACK access (ignored from MLP)\n";
                            continue;
                        }
                        if (DepKind == DependencyKind::OOO) {
                            unsigned base_reg = inst_infos[i].mem_info.valid() ? inst_infos[i].mem_info.base_reg : 0;
                            if (base_reg != 0 && inst_infos[i].mem_info.is_constant_offset()) {
                                int64_t cache_line = inst_infos[i].mem_info.offset / 64;
                                is_hit = global_seen_base_regs.test(base_reg, cache_line);
                                std::cout << "  [" << i << "] Load base reg: " << MRI->getName(base_reg) 
                                          << ", offset: " << inst_infos[i].mem_info.offset
                                          << " (line " << cache_line << ") -> "
                                          << (is_hit ? "CACHE MATCH (excl)" : "NO MATCH") << "\n";
                            } else {
                                std::cout << "  [" << i << "] Load has non-constant offset or invalid base reg -> NO MATCH\n";
                            }
                        }
                        if (!is_hit) {
                            load_indices.push_back(i);
                        }
                    }
                    // update seen base
                    updateSeenBaseRegs(inst_infos[i], global_seen_base_regs, *MRI);
                    if (inst_infos[i].is_call()) {
                        for (unsigned ret_reg : return_regs) {
                            global_seen_base_regs.reset(ret_reg, *MRI);
                        }
                    }
                }
                std::cout << "\n";

                // Evaluate windows for each load
                std::vector<float> mlp_vals(n, 1.0f);
                std::vector<float> mlp_r_vals(n, -1.0f);
                RegSet load_dep_regs(reg_count);
                
                std::cout << "--- Trace for each load instruction ---\n";
                for (int i : load_indices) {
                    std::cout << "Evaluating load at index [" << i << "]: ";
                    printInstruction(*IP, region_instrs[i].Inst, region_instrs[i].Addr, *STI);
                    std::cout << "\n";

                    load_dep_regs.reset();
                    SeenBaseRegs window_seen_base_regs;

                    int count_indep = 0;
                    int count_dep = 0;
                    int uops_sum = 0;

                    std::cout << "  Window trace:\n";
                    for (int step = 0;; ++step) {
                        if (!r.IsLoop && (i + step) >= n) break;
                        int j = (i + step) % n;
                        uops_sum += std::max(1, static_cast<int>(inst_infos[j].num_uops));
                        if (uops_sum > windowWidth && step > 0) {
                            std::cout << "    [Step " << step << "] uops sum " << uops_sum 
                                      << " exceeds window width " << windowWidth << ". Stopping.\n";
                            break;
                        }

                        bool is_dep = has_intersection(inst_infos[j].io_regs.inputs, load_dep_regs);
                        bool is_load = inst_infos[j].is_load() && inst_infos[j].mem_info.valid();
                        unsigned base_reg = is_load ? inst_infos[j].mem_info.base_reg : 0;
                        
                        bool is_hit = false;
                        if (is_load) {
                            if (base_reg != 0 && inst_infos[j].mem_info.is_constant_offset()) {
                                int64_t cache_line = inst_infos[j].mem_info.offset / 64;
                                is_hit = window_seen_base_regs.test(base_reg, cache_line);
                            }
                        }

                        std::cout << "    - instr [" << j << "] 0x" << std::hex << region_instrs[j].Addr << ": ";
                        printInstruction(*IP, region_instrs[j].Inst, region_instrs[j].Addr, *STI);
                        std::cout << std::dec << " | uops: " << std::max(1, (int)inst_infos[j].num_uops);
                        
                        if (is_load) {
                            std::cout << " | Load";
                            if (is_hit) {
                                std::cout << " (Window Cache Match)";
                            } else {
                                if (is_dep) {
                                    std::cout << " (DEPENDENT)";
                                    ++count_dep;
                                } else {
                                    std::cout << " (INDEPENDENT)";
                                    ++count_indep;
                                }
                            }
                        } else {
                            if (is_dep) std::cout << " (Propagates Dependency)";
                        }
                        std::cout << "\n";

                        // update window dependency registers
                        updateSeenBaseRegs(inst_infos[j], window_seen_base_regs, *MRI);
                        for (unsigned reg : inst_infos[j].io_regs.outputs) {
                            if (is_dep) {
                                set_reg(load_dep_regs, reg, *MRI);
                            } else if (is_load) {
                                if (reg == base_reg && inst_infos[j].mem_info.is_writeback())
                                    reset_reg(load_dep_regs, reg, *MRI);
                                else
                                    set_reg(load_dep_regs, reg, *MRI);
                            } else {
                                reset_reg(load_dep_regs, reg, *MRI);
                            }
                        }
                        if (inst_infos[j].is_call()) {
                            for (unsigned ret_reg : return_regs) {
                                window_seen_base_regs.reset(ret_reg, *MRI);
                                reset_reg(load_dep_regs, ret_reg, *MRI);
                            }
                        }
                    }

                    int total = count_indep + count_dep;
                    float score = static_cast<float>(count_indep);
                    float score_r = (total == 0) ? -1.0f : static_cast<float>(count_indep) / total;
                    std::cout << "  => Load Score: " << score << ", ratio (score_r): " << score_r << "\n\n";

                    // Assign score
                    if (AssignKind == MLPWindowAssignmentKind::Forward) {
                        mlp_vals[i] = score;
                        mlp_r_vals[i] = score_r;
                    } else {
                        // MaxContaining
                        int u_sum = 0;
                        int s_limit = r.IsLoop ? windowWidth : (n - i);
                        for (int step = 0; step < s_limit; ++step) {
                            int j = (i + step) % n;
                            int inst_uops = std::max(1, static_cast<int>(inst_infos[j].num_uops));
                            if (u_sum + inst_uops > windowWidth && step > 0) break;
                            u_sum += inst_uops;
                            if (inst_infos[j].is_load()) {
                                mlp_vals[j] = std::max(mlp_vals[j], score);
                                mlp_r_vals[j] = std::max(mlp_r_vals[j], score_r);
                            }
                        }
                    }
                }

                std::cout << "--- Final Assigned MLP per Load (Window Assignment: " 
                          << (AssignKind == MLPWindowAssignmentKind::Forward ? "Forward" : "MaxContaining") << ") ---\n";
                double total_mlp = 0;
                double total_mlp_r = 0;
                int count_r = 0;
                for (int i : load_indices) {
                    std::cout << "  [" << i << "] 0x" << std::hex << region_instrs[i].Addr << ":  ";
                    printInstruction(*IP, region_instrs[i].Inst, region_instrs[i].Addr, *STI);
                    std::cout << std::fixed << std::setprecision(2) << "  => Assigned MLP: " << mlp_vals[i] 
                              << ", Ratio: " << mlp_r_vals[i] << "\n";
                    total_mlp += mlp_vals[i];
                    if (mlp_r_vals[i] >= 0.0f) {
                        total_mlp_r += mlp_r_vals[i];
                        count_r++;
                    }
                }
                
                float final_mlp = total_mlp / load_indices.size();
                float final_mlp_r = (count_r == 0) ? 0.0f : total_mlp_r / count_r;
                std::cout << "\n=======================================================\n";
                std::cout << "REGION MLP   : " << final_mlp << "\n";
                std::cout << "REGION MLP_R : " << final_mlp_r << "\n";
                std::cout << "=======================================================\n";
                break;
            }
        }
        if (Found) break;
    }

    if (!Found) {
        WithColor::error() << "Address 0x" << Twine::utohexstr(TargetAddress) << " not found in any text section.\n";
        return 1;
    }

    return 0;
}
