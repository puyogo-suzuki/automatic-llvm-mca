#include "mca_common.h"
#include "frontend.h"
#include "custom_a55_sched.h"
#include <cstdio>
#include <algorithm>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <unistd.h>
#include <iomanip>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

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
    std::unique_ptr<ObjectFile> Obj;
    TargetInfo TI;
    if (!initializeFrontend(argc, argv, "mlp-check debug tool\n", Obj, TI)) {
        return 1;
    }

    if (opts::TargetAddressStr.empty()) {
        WithColor::error() << "Target address must be specified.\n";
        return 1;
    }

    std::unique_ptr<MCInstPrinter> IP(TI.TheTarget->createMCInstPrinter(Triple(TI.TripleName), 0, *TI.MAI, *TI.MCII, *TI.MRI));
    if (!IP) {
        WithColor::error() << "Failed to create MCInstPrinter\n";
        return 1;
    }

    IP->setPrintImmHex(true);
    IP->setPrintBranchImmAsAddress(true);

    FunctionBoundaries FunctionRanges = collectFunctionBoundaries(*Obj);

    bool Found = false;
    for (const SectionRef &Section : Obj->sections()) {
        if (!Section.isText() || Section.getSize() == 0) continue;

        auto SectionInstrs = disassembleTextSection(Section, *TI.DisAsm, *TI.MCII, TI.MCIA.get());
        if (SectionInstrs.empty()) continue;

        std::vector<McaRegion> regions;
        walkRegions(SectionInstrs, FunctionRanges, opts::LoopMaxInstrs, opts::BBMaxInstrs, opts::NestLimitOuter, opts::NestLimitInner,
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
            if (TI.TargetAddress >= r.StartAddr && TI.TargetAddress < r.EndAddr) {
                Found = true;
                std::cout << "\n=======================================================\n";
                std::cout << "FOUND TARGET ADDRESS 0x" << std::hex << TI.TargetAddress 
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
                    printInstruction(*IP, I.Inst, I.Addr, *TI.STI);
                    std::cout << "\n";
                }
                std::cout << "\n";

                // Evaluate pre-loads and dependencies
                const unsigned reg_count = TI.MRI->getNumRegs() + 1;
                std::vector<unsigned> return_regs = getReturnRegisters(*TI.MRI, TI.STI->getTargetTriple().getArchName().str());
                
                // Reconstruct MLPInstInfo using common helper
                std::vector<MLPInstInfo> inst_infos = buildInstInfos(region_instrs, *TI.STI, *TI.MCII, *TI.MRI, TI.Analyzer.get());

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
                        if (opts::DepKind == DependencyKind::OOO) {
                            unsigned base_reg = inst_infos[i].mem_info.valid() ? inst_infos[i].mem_info.base_reg : 0;
                            if (base_reg != 0 && inst_infos[i].mem_info.is_constant_offset()) {
                                int64_t cache_line = inst_infos[i].mem_info.offset / 64;
                                is_hit = global_seen_base_regs.test(base_reg, cache_line);
                                std::cout << "  [" << i << "] Load base reg: " << TI.MRI->getName(base_reg) 
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
                    updateSeenBaseRegs(inst_infos[i], global_seen_base_regs, *TI.MRI);
                    if (inst_infos[i].is_call()) {
                        for (unsigned ret_reg : return_regs) {
                            global_seen_base_regs.reset(ret_reg, *TI.MRI);
                        }
                    }
                }
                std::cout << "\n";

                // Evaluate windows for each load
                std::vector<float> mlp_vals(n, 1.0f);
                std::vector<float> mlp_r_vals(n, -1.0f);
                RegSet load_dep_regs(reg_count);
                
                std::cout << "--- MLP Evaluation (Forward Windows) ---\n";
                for (int load_idx : load_indices) {
                    std::cout << "  Evaluating window starting at load [" << load_idx << "]...\n";
                    
                    int current_loads = 0;
                    int actual_loads = 0;
                    
                    load_dep_regs.reset();
                    SeenBaseRegs window_seen_base_regs;
                    
                    for (int j = load_idx; j < n; ++j) {
                        const auto &J = inst_infos[j];
                        
                        bool has_dep = false;
                        if (opts::DepKind != DependencyKind::None) {
                            if (has_intersection(J.io_regs.inputs, load_dep_regs)) {
                                has_dep = true;
                            }
                        }
                        
                        if (has_dep) {
                            std::cout << "    [" << j << "] STALL due to register dependency\n";
                            break;
                        }
                        
                        if (J.is_load()) {
                            if (J.mem_info.is_stack_access()) {
                                std::cout << "    [" << j << "] Load is STACK access (ignored)\n";
                            } else {
                                bool is_hit = false;
                                if (opts::DepKind == DependencyKind::OOO) {
                                    unsigned base_reg = J.mem_info.valid() ? J.mem_info.base_reg : 0;
                                    if (base_reg != 0 && J.mem_info.is_constant_offset()) {
                                        int64_t cache_line = J.mem_info.offset / 64;
                                        is_hit = window_seen_base_regs.test(base_reg, cache_line);
                                    }
                                }
                                
                                if (is_hit) {
                                    std::cout << "    [" << j << "] Load hit (cache match) -> IGNORED from window count\n";
                                } else {
                                    actual_loads++;
                                    std::cout << "    [" << j << "] Load added to window (Count: " << actual_loads << ")\n";
                                }
                            }
                        }
                        
                        current_loads += J.num_uops;
                        if (current_loads > TI.WindowWidthVal) {
                            std::cout << "    [" << j << "] window width limit reached (" << current_loads << " > " << TI.WindowWidthVal << ")\n";
                            break;
                        }
                        
                        // update dependency registers
                        for (unsigned out : J.io_regs.outputs) {
                            set_reg(load_dep_regs, out, *TI.MRI);
                        }
                        for (unsigned in : J.io_regs.inputs) {
                            reset_reg(load_dep_regs, in, *TI.MRI);
                        }
                        
                        updateSeenBaseRegs(J, window_seen_base_regs, *TI.MRI);
                        if (J.is_call()) {
                            for (unsigned ret_reg : return_regs) {
                                window_seen_base_regs.reset(ret_reg, *TI.MRI);
                            }
                        }
                    }
                    
                    std::cout << "    Window actual MLP = " << actual_loads << "\n";
                    mlp_vals[load_idx] = actual_loads;
                    mlp_r_vals[load_idx] = 1.0f / actual_loads;
                }
                std::cout << "\n";
                
                std::cout << "--- Final Instruction Metrics ---\n";
                for (int i = 0; i < n; ++i) {
                    std::cout << "  [" << std::setw(2) << std::dec << i << "]  ";
                    if (inst_infos[i].is_load()) {
                        if (inst_infos[i].mem_info.is_stack_access()) {
                            std::cout << "MLP: STACK  ";
                        } else {
                            std::cout << "MLP: " << std::fixed << std::setw(5) << std::setprecision(2) << mlp_vals[i] << "  ";
                        }
                    } else {
                        std::cout << "            ";
                    }
                    std::cout << "0x" << std::hex << region_instrs[i].Addr << ": ";
                    printInstruction(*IP, region_instrs[i].Inst, region_instrs[i].Addr, *TI.STI);
                    std::cout << "\n";
                }
                break;
            }
        }
        if (Found) break;
    }

    if (!Found) {
        std::cout << "Target address 0x" << std::hex << TI.TargetAddress << " not found in any disassembled code regions.\n";
    }

    return 0;
}
