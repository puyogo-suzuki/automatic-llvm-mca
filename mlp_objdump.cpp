#include "mca_common.h"
#include "frontend.h"
#include "custom_a55_sched.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

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

namespace {

using SpanKey = std::pair<size_t, size_t>;

struct RegionMetrics {
    double mlp = 0.0;
    double base_cpi = 0.0;
    bool valid = false;
};

void printSectionHeader(StringRef Name) {
    std::cout << "Disassembly of section " << Name.str() << ":\n";
}

std::string formatAddress(uint64_t Addr) {
    std::ostringstream OS;
    OS << "0x" << std::hex << std::setw(16) << std::setfill('0') << Addr;
    return OS.str();
}

std::string formatMetrics(const RegionMetrics &M) {
    if (!M.valid) return "  --.-  --.--";
    std::ostringstream OS;
    OS << std::fixed << std::setw(5) << std::setprecision(1) << M.mlp << "  " << std::setw(6)
       << std::setprecision(2) << M.base_cpi;
    return OS.str();
}

void mergeMetrics(RegionMetrics &Dst, const McaMetrics &Src) {
    if (!Src.Valid) return;
    if (!Dst.valid) {
        Dst.mlp = Src.MLP;
        Dst.base_cpi = Src.BaseCPI;
        Dst.valid = true;
        return;
    }
    Dst.mlp = std::max(Dst.mlp, static_cast<double>(Src.MLP));
    Dst.base_cpi = std::min(Dst.base_cpi, Src.BaseCPI);
}

} // namespace

int main(int argc, char **argv) {
    InitLLVM X(argc, argv);
    std::unique_ptr<ObjectFile> Obj;
    TargetInfo TI;
    if (!initializeFrontend(argc, argv, "mlp-objdump\n", Obj, TI)) {
        return 1;
    }

    std::unique_ptr<MCInstPrinter> IP(TI.TheTarget->createMCInstPrinter(Triple(TI.TripleName), 0, *TI.MAI, *TI.MCII, *TI.MRI));
    if (!IP) {
        WithColor::error() << "Failed to initialize MCInstPrinter\n";
        return 1;
    }

    IP->setPrintImmHex(true);
    IP->setPrintBranchImmAsAddress(true);

    FunctionBoundaries Boundaries = collectFunctionBoundaries(*Obj);
    bool FirstSection = true;

    for (const SectionRef &Section : Obj->sections()) {
        if (!Section.isText() || Section.getSize() == 0) continue;

        if (auto NameOrErr = Section.getName()) {
            StringRef Name = *NameOrErr;
            if (Name == ".plt" || Name == ".init" || Name == ".fini" || 
                Name.starts_with(".plt.") || Name == ".plt.got" || Name == ".plt.sec") {
                continue;
            }
        }

        auto SectionInstrs = disassembleTextSection(Section, *TI.DisAsm, *TI.MCII, TI.MCIA.get());
        if (SectionInstrs.empty()) continue;

        if (!FirstSection) std::cout << "\n";
        FirstSection = false;

        std::map<SpanKey, RegionMetrics> MetricsBySpan;
        std::vector<SpanKey> SpanForInstr(SectionInstrs.size(), SpanKey{0, 0});

        auto accumulateSpan = [&](const RegionSpan &Span, bool isLoop) {
            if (Span.Size == 0) return;
            
            auto region_instrs = ArrayRef<Instr>(SectionInstrs).slice(Span.Start, Span.Size);
            if (isAllNopRegion(region_instrs, *TI.MCII)) return;

            const SpanKey Key{Span.Start, Span.Size};
            bool ignore = false;
            if (opts::IgnoreLoopCarried == IgnoreLoopCarriedMode::Force) {
                ignore = true;
            } else if (opts::IgnoreLoopCarried == IgnoreLoopCarriedMode::Default) {
                ignore = !isLoop;
            } else if (opts::IgnoreLoopCarried == IgnoreLoopCarriedMode::Disable) {
                ignore = false;
            }
            bool mlpLoop = false;
            if (opts::MlpWindowLoop == MlpWindowLoopMode::Force) {
                mlpLoop = true;
            } else if (opts::MlpWindowLoop == MlpWindowLoopMode::Default) {
                mlpLoop = isLoop;
            } else if (opts::MlpWindowLoop == MlpWindowLoopMode::Disable) {
                mlpLoop = false;
            }
            auto Result = analyzeMcaRegion(ArrayRef<Instr>(SectionInstrs).slice(Span.Start, Span.Size), *TI.STI, *TI.MCII,
                                           *TI.MRI, TI.MCIA.get(), TI.PO, opts::Iterations, TI.WindowWidthVal, opts::DepKind, opts::AssignKind,
                                           *TI.Analyzer, ignore, opts::OverrideLoadLatency, mlpLoop);
            mergeMetrics(MetricsBySpan[Key], Result);
            for (size_t i = Span.Start; i < Span.Start + Span.Size; ++i) {
                if (SpanForInstr[i].second == 0) {
                    SpanForInstr[i] = Key;
                }
            }
        };

        auto rememberBasicBlock = [&](const RegionSpan &Span) {
            if (Span.Size == 0) return;
            const SpanKey Key{Span.Start, Span.Size};
            for (size_t i = Span.Start; i < Span.Start + Span.Size; ++i) SpanForInstr[i] = Key;
            accumulateSpan(Span, false);
        };

        walkRegions(SectionInstrs, Boundaries, opts::LoopMaxInstrs, opts::BBMaxInstrs, opts::NestLimitOuter, opts::NestLimitInner,
                    [&](const RegionSpan &Span) { accumulateSpan(Span, true); },
                    rememberBasicBlock);

        if (auto NameOrErr = Section.getName()) printSectionHeader(*NameOrErr);

        for (size_t i = 0; i < SectionInstrs.size(); ++i) {
            const auto &I = SectionInstrs[i];
            auto It = MetricsBySpan.find(SpanForInstr[i]);
            const RegionMetrics *M = (It != MetricsBySpan.end()) ? &It->second : nullptr;

            std::string InstText;
            raw_string_ostream OS(InstText);
            IP->printInst(&I.Inst, I.Addr, "", *TI.STI, OS);
            OS.flush();

            std::cout << formatMetrics(M ? *M : RegionMetrics{}) << " " << formatAddress(I.Addr) << ": "
                      << InstText << "\n";
        }
    }

    return 0;
}
