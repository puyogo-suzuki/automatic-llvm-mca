#include "mca_common.h"
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
static cl::opt<std::string> InputCsv("i", cl::desc("Input CSV file"), cl::Required);
static cl::opt<std::string> OutputCsv("o", cl::desc("Output CSV file"), cl::Required);
static cl::opt<std::string> MTriple("mtriple", cl::desc("Target triple"));
static cl::opt<std::string> MCPU("mcpu", cl::desc("Target CPU"));
static cl::opt<int> WindowWidth("window-width", cl::desc("MLP window width"), cl::init(4));
static cl::opt<DependencyKind> DepKind("dependency", cl::desc("Dependency mode"),
    cl::values(
        clEnumValN(DependencyKind::None, "none", "No dependency tracking"),
        clEnumValN(DependencyKind::IO, "io", "In-order dependency"),
        clEnumValN(DependencyKind::OOO, "ooo", "Out-of-order dependency"),
        clEnumValN(DependencyKind::Dependency, "dependency", "Load-use dependency distance")
    ), cl::init(DependencyKind::None));
static cl::opt<MLPWindowAssignmentKind> AssignKind("mlp-window-assignment", cl::desc("Per-load MLP assignment mode"),
    cl::values(
        clEnumValN(MLPWindowAssignmentKind::Forward, "forward", "Forward window"),
        clEnumValN(MLPWindowAssignmentKind::MaxContaining, "max-containing", "Max MLP of containing windows")
    ), cl::init(MLPWindowAssignmentKind::MaxContaining));
static cl::opt<MlpWindowLoopMode> MlpWindowLoop("mlp-window-loop",
    cl::desc("Loop back to the start of the basic block mode"),
    cl::values(
        clEnumValN(MlpWindowLoopMode::Default, "default", "Loop back to the start only for loops"),
        clEnumValN(MlpWindowLoopMode::Force, "force", "Always loop back to the start (even for non-loops)"),
        clEnumValN(MlpWindowLoopMode::Disable, "disable", "Never loop back to the start")
    ), cl::init(MlpWindowLoopMode::Default));

static uint64_t parseHex(const std::string &str) {
    return std::stoull(str, nullptr, 16);
}

int main(int argc, char **argv) {
    InitLLVM X(argc, argv);
    initializeTargets();

    cl::ParseCommandLineOptions(argc, argv, "mlp_update static MLP updater tool\n");

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
    MCContext Ctx(TT, *MAI, *MRI, *STI);
    std::unique_ptr<MCDisassembler> DisAsm(TheTarget->createMCDisassembler(*STI, Ctx));
    std::unique_ptr<MCInstrAnalysis> MCIA(TheTarget->createMCInstrAnalysis(MCII.get()));

    if (!MRI || !MAI || !MCII || !STI || !DisAsm || !MCIA) {
        WithColor::error() << "Failed to initialize LLVM components\n";
        return 1;
    }

    const MCSchedModel &SM = STI->getSchedModel();
    int windowWidth = WindowWidth;
    if (WindowWidth.getNumOccurrences() == 0 && !MCPU.empty()) {
        if (SM.MicroOpBufferSize > 0) {
            windowWidth = SM.MicroOpBufferSize;
        } else {
            windowWidth = SM.IssueWidth * SM.MispredictPenalty;
        }
    }

    std::unique_ptr<MLPAnalyzer> Analyzer = MLPAnalyzer::create(*STI);

    // Disassemble all text sections and keep a single instruction list sorted by address
    std::vector<Instr> AllInstrs;
    for (const SectionRef &Section : Obj.sections()) {
        if (!Section.isText() || Section.getSize() == 0) continue;
        auto SectionInstrs = disassembleTextSection(Section, *DisAsm, *MCII, MCIA.get());
        AllInstrs.insert(AllInstrs.end(), SectionInstrs.begin(), SectionInstrs.end());
    }

    std::sort(AllInstrs.begin(), AllInstrs.end(), [](const Instr &a, const Instr &b) {
        return a.Addr < b.Addr;
    });

    // Open CSV files
    std::ifstream infile(InputCsv.c_str());
    if (!infile.is_open()) {
        WithColor::error() << "Failed to open input CSV: " << InputCsv << "\n";
        return 1;
    }

    std::ofstream outfile(OutputCsv.c_str());
    if (!outfile.is_open()) {
        WithColor::error() << "Failed to open output CSV: " << OutputCsv << "\n";
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
                if (MlpWindowLoop == MlpWindowLoopMode::Force) {
                    mlpLoop = true;
                } else if (MlpWindowLoop == MlpWindowLoopMode::Default) {
                    mlpLoop = is_loop;
                } else if (MlpWindowLoop == MlpWindowLoopMode::Disable) {
                    mlpLoop = false;
                }

                float mlp_r = 0.0f;
                float mlp = Analyzer->compute_mlp(region_instrs, windowWidth, DepKind, AssignKind, *STI, *MCII, *MRI, mlp_r, mlpLoop);
                size_t load_instrs = Analyzer->countNonStackLoads(region_instrs, *STI, *MCII, *MRI, DepKind);

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
