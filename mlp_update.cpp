#include "mca_common.h"
#include "frontend.h"
#include "custom_a55_sched.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCContext.h"
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

static uint64_t parseHex(const std::string &str) {
    return std::stoull(str, nullptr, 16);
}

int main(int argc, char **argv) {
    InitLLVM X(argc, argv);
    std::unique_ptr<ObjectFile> Obj;
    TargetInfo TI;
    if (!initializeFrontend(argc, argv, "mlp_update static MLP updater tool\n", Obj, TI)) {
        return 1;
    }

    if (opts::InputCsv.empty() || opts::OutputCsv.empty()) {
        WithColor::error() << "Input and output CSV files must be specified.\n";
        return 1;
    }

    // Disassemble all text sections and keep a single instruction list sorted by address
    std::vector<Instr> AllInstrs;
    for (const SectionRef &Section : Obj->sections()) {
        if (!Section.isText() || Section.getSize() == 0) continue;
        auto SectionInstrs = disassembleTextSection(Section, *TI.DisAsm, *TI.MCII, TI.MCIA.get());
        AllInstrs.insert(AllInstrs.end(), SectionInstrs.begin(), SectionInstrs.end());
    }

    std::sort(AllInstrs.begin(), AllInstrs.end(), [](const Instr &a, const Instr &b) {
        return a.Addr < b.Addr;
    });

    // Open CSV files
    std::ifstream infile(opts::InputCsv.c_str());
    if (!infile.is_open()) {
        WithColor::error() << "Failed to open input CSV: " << opts::InputCsv << "\n";
        return 1;
    }

    std::ofstream outfile(opts::OutputCsv.c_str());
    if (!outfile.is_open()) {
        WithColor::error() << "Failed to open output CSV: " << opts::OutputCsv << "\n";
        return 1;
    }

    std::string line;
    if (std::getline(infile, line)) {
        // Print header
        outfile << "start_address,end_address,length,loop,retired_instructions,load_instructions,cycles,mlp,mlp_r\n";
    }

    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }
        if (row.size() < 9) {
            outfile << line << "\n";
            continue;
        }

        uint64_t start_addr = parseHex(row[0]);
        uint64_t end_addr = parseHex(row[1]);
        bool is_loop = (row[3] == "1");

        // Binary search for the instruction index matching start_addr
        auto it = std::lower_bound(AllInstrs.begin(), AllInstrs.end(), start_addr, [](const Instr &inst, uint64_t addr) {
            return inst.Addr < addr;
        });

        if (it != AllInstrs.end() && it->Addr == start_addr) {
            size_t start_idx = std::distance(AllInstrs.begin(), it);
            size_t end_idx = start_idx;
            while (end_idx < AllInstrs.size() && AllInstrs[end_idx].Addr <= end_addr) {
                if (AllInstrs[end_idx].Addr == end_addr) {
                    break;
                }
                end_idx++;
            }

            if (end_idx < AllInstrs.size() && AllInstrs[end_idx].Addr == end_addr) {
                size_t region_size = end_idx - start_idx + 1;
                auto region_instrs = ArrayRef<Instr>(AllInstrs).slice(start_idx, region_size);

                bool mlpLoop = false;
                if (opts::MlpWindowLoop == MlpWindowLoopMode::Force) {
                    mlpLoop = true;
                } else if (opts::MlpWindowLoop == MlpWindowLoopMode::Default) {
                    mlpLoop = is_loop;
                } else if (opts::MlpWindowLoop == MlpWindowLoopMode::Disable) {
                    mlpLoop = false;
                }

                float mlp_r = 0.0f;
                float mlp = TI.Analyzer->compute_mlp(region_instrs, TI.WindowWidthVal, opts::DepKind, opts::AssignKind, *TI.STI, *TI.MCII, *TI.MRI, mlp_r, mlpLoop);
                size_t load_instrs = TI.Analyzer->countPotentialMissLoads(region_instrs, *TI.STI, *TI.MCII, *TI.MRI, opts::DepKind);

                // Update MLP and load instructions columns (row[5] = load_instructions, row[7] = mlp, row[8] = mlp_r)
                row[5] = std::to_string(load_instrs);
                char buf_mlp[32];
                char buf_mlp_r[32];
                std::snprintf(buf_mlp, sizeof(buf_mlp), "%.2f", mlp);
                std::snprintf(buf_mlp_r, sizeof(buf_mlp_r), "%.2f", mlp_r);
                row[7] = buf_mlp;
                row[8] = buf_mlp_r;
            }
        }

        // Write row to output CSV
        for (size_t i = 0; i < row.size(); ++i) {
            outfile << row[i];
            if (i + 1 < row.size()) outfile << ",";
        }
        outfile << "\n";
    }

    return 0;
}
