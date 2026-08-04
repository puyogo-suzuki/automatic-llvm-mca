#include "facile.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <set>
#include <vector>

namespace facile {

namespace {

struct DependencyEdge {
    size_t Target;
    double Latency;
    unsigned Distance; // 0 for intra-iteration, 1 for inter-iteration (loop-carried)
};

// 1. Calculate Dispatch / Issue Width Limit
double calculateIssueBound(const llvm::MCSchedModel &SM,
                            llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                            unsigned &TotalUops) {
    unsigned IssueWidth = SM.IssueWidth > 0 ? SM.IssueWidth : 1;
    TotalUops = 0;
    for (const auto &Inst : SimInstrs) {
        unsigned NumUops = Inst->getNumMicroOps();
        TotalUops += (NumUops > 0 ? NumUops : 1);
    }
    return static_cast<double>(TotalUops) / static_cast<double>(IssueWidth);
}

// 2. Calculate Execution Ports Contention Bound
double calculatePortUsageBound(const llvm::MCSubtargetInfo &STI,
                               const llvm::MCInstrInfo &MCII,
                               llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                               llvm::ArrayRef<const llvm::MCInst *> MCInsts,
                               std::string &BottleneckPortName) {
    const llvm::MCSchedModel &SM = STI.getSchedModel();
    unsigned NumProcResources = SM.NumProcResourceKinds;
    std::vector<double> ProcResUsage(NumProcResources, 0.0);

    for (size_t i = 0; i < SimInstrs.size(); ++i) {
        const auto &Inst = SimInstrs[i];
        const llvm::MCInst *MCI = (i < MCInsts.size()) ? MCInsts[i] : nullptr;
        const llvm::MCInstrDesc &MCID = MCII.get(Inst->getOpcode());
        
        unsigned ResolvedSchedClass = MCID.getSchedClass();
        const llvm::MCSchedClassDesc *SCDesc = SM.getSchedClassDesc(ResolvedSchedClass);
        unsigned PrevSchedClass = ResolvedSchedClass;
        
        while (SCDesc && SCDesc->isVariant()) {
            if (MCI) {
                ResolvedSchedClass = STI.resolveVariantSchedClass(ResolvedSchedClass, MCI, &MCII, SM.getProcessorID());
            }
            if (ResolvedSchedClass == PrevSchedClass) break;
            PrevSchedClass = ResolvedSchedClass;
            SCDesc = SM.getSchedClassDesc(ResolvedSchedClass);
        }
        if (!SCDesc) continue;

        for (const llvm::MCWriteProcResEntry *WPR = STI.getWriteProcResBegin(SCDesc);
             WPR != STI.getWriteProcResEnd(SCDesc); ++WPR) {
            unsigned ProcResIdx = WPR->ProcResourceIdx;
            unsigned Cycles = WPR->ReleaseAtCycle - WPR->AcquireAtCycle;
            if (Cycles == 0) Cycles = 1;
            if (ProcResIdx < NumProcResources) {
                ProcResUsage[ProcResIdx] += Cycles;
            }
        }
    }

    double MaxPortBound = 0.0;
    BottleneckPortName = "None";

    for (unsigned r = 1; r < NumProcResources; ++r) {
        const llvm::MCProcResourceDesc *PRD = SM.getProcResource(r);
        if (!PRD || PRD->NumUnits == 0) continue;
        double PortCycles = ProcResUsage[r] / static_cast<double>(PRD->NumUnits);
        if (PortCycles > MaxPortBound) {
            MaxPortBound = PortCycles;
            BottleneckPortName = PRD->Name ? PRD->Name : ("Port" + std::to_string(r));
        }
    }

    return MaxPortBound;
}

// 3. Build Read-After-Write (RAW) Register Dependency Graph
std::vector<std::vector<DependencyEdge>> buildDependencyGraph(
    llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs) {
    
    size_t N = SimInstrs.size();
    std::vector<std::vector<DependencyEdge>> Adj(N);
    std::map<unsigned, size_t> LastWriter;
    std::map<unsigned, size_t> FirstReader;
    std::set<unsigned> DefinedRegs;

    // Pass 1: Intra-iteration dependencies & Live-In reads
    for (size_t i = 0; i < N; ++i) {
        const auto &Inst = SimInstrs[i];

        for (const auto &Op : Inst->getUses()) {
            unsigned Reg = Op.getRegisterID();
            if (Reg == 0) continue;

            if (DefinedRegs.find(Reg) == DefinedRegs.end()) {
                if (FirstReader.find(Reg) == FirstReader.end()) {
                    FirstReader[Reg] = i;
                }
            }

            auto it = LastWriter.find(Reg);
            if (it != LastWriter.end()) {
                size_t WriterIdx = it->second;
                double Lat = static_cast<double>(SimInstrs[WriterIdx]->getLatency());
                if (Lat < 1.0) Lat = 1.0;
                if (WriterIdx < i) {
                    Adj[WriterIdx].push_back({i, Lat, 0});
                }
            }
        }

        for (const auto &Op : Inst->getDefs()) {
            unsigned Reg = Op.getRegisterID();
            if (Reg != 0) {
                LastWriter[Reg] = i;
                DefinedRegs.insert(Reg);
            }
        }
    }

    // Pass 2: Loop-carried dependencies (LastWriter -> FirstReader across iterations)
    for (const auto &entry : FirstReader) {
        unsigned Reg = entry.first;
        size_t ReaderIdx = entry.second;
        auto it = LastWriter.find(Reg);
        if (it != LastWriter.end()) {
            size_t WriterIdx = it->second;
            double Lat = static_cast<double>(SimInstrs[WriterIdx]->getLatency());
            if (Lat < 1.0) Lat = 1.0;
            Adj[WriterIdx].push_back({ReaderIdx, Lat, 1});
        }
    }

    return Adj;
}

// 4. Calculate Maximum Cycle Ratio (MCR) for Precedence Constraints
double calculatePrecedenceBound(
    size_t N,
    const std::vector<std::vector<DependencyEdge>> &Adj,
    llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs) {
    
    double MaxRatio = 0.0;

    // Fast check for self-loops (e.g., ADD X0, X0, #1)
    for (size_t u = 0; u < N; ++u) {
        for (const auto &E : Adj[u]) {
            if (E.Target == u && E.Distance > 0) {
                double Ratio = E.Latency / E.Distance;
                if (Ratio > MaxRatio) MaxRatio = Ratio;
            }
        }
    }

    // Floyd-Warshall-style MCR evaluation (guarded for N <= 1000)
    if (N <= 1000) {
        std::vector<double> Dist(N * N, -1e9);
        std::vector<unsigned> IterDist(N * N, 0);

        for (size_t u = 0; u < N; ++u) {
            for (const auto &E : Adj[u]) {
                size_t idx = u * N + E.Target;
                if (E.Latency > Dist[idx]) {
                    Dist[idx] = E.Latency;
                    IterDist[idx] = E.Distance;
                }
            }
        }

        for (size_t k = 0; k < N; ++k) {
            for (size_t i = 0; i < N; ++i) {
                double d_ik = Dist[i * N + k];
                if (d_ik < -1e8) continue;
                for (size_t j = 0; j < N; ++j) {
                    double d_kj = Dist[k * N + j];
                    if (d_kj < -1e8) continue;
                    double NewDist = d_ik + d_kj;
                    unsigned NewIter = IterDist[i * N + k] + IterDist[k * N + j];
                    size_t ij_idx = i * N + j;
                    if (NewDist > Dist[ij_idx]) {
                        Dist[ij_idx] = NewDist;
                        IterDist[ij_idx] = NewIter;
                    }
                }
            }
        }

        for (size_t i = 0; i < N; ++i) {
            size_t ii_idx = i * N + i;
            if (Dist[ii_idx] > 0 && IterDist[ii_idx] > 0) {
                double Ratio = Dist[ii_idx] / static_cast<double>(IterDist[ii_idx]);
                if (Ratio > MaxRatio) {
                    MaxRatio = Ratio;
                }
            }
        }
    } else {
        // Fast O(N) critical path estimation for large regions (N > 1000)
        std::vector<double> LongestPath(N, 0.0);
        for (size_t u = 0; u < N; ++u) {
            for (const auto &E : Adj[u]) {
                if (E.Distance == 0 && E.Target > u) {
                    LongestPath[E.Target] = std::max(LongestPath[E.Target], LongestPath[u] + E.Latency);
                }
            }
        }
        for (size_t u = 0; u < N; ++u) {
            if (LongestPath[u] > MaxRatio) MaxRatio = LongestPath[u];
        }
    }

    return MaxRatio;
}

// 5. Determine Dominant Bottleneck Category Name
std::string determineDominantBottleneck(double EstimatedCycles,
                                        double PrecedenceBound,
                                        double PortBound,
                                        const std::string &PortBottleneckName) {
    if (EstimatedCycles == PrecedenceBound && PrecedenceBound > 0.0) {
        return "Precedence Constraints (Dependency Chain)";
    }
    if (EstimatedCycles == PortBound && PortBound > 0.0) {
        return "Execution Ports (" + PortBottleneckName + ")";
    }
    return "Issue Width (Dispatch Limit)";
}

} // namespace

FacileResult computeFacilePrediction(const llvm::MCSubtargetInfo &STI,
                                     const llvm::MCInstrInfo &MCII,
                                     const llvm::MCRegisterInfo &MRI,
                                     llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                                     llvm::ArrayRef<const llvm::MCInst *> MCInsts) {
    FacileResult Res;
    if (SimInstrs.empty()) return Res;

    Res.TotalInstructions = SimInstrs.size();

    // 1. Issue Limit
    Res.IssueBound = calculateIssueBound(STI.getSchedModel(), SimInstrs, Res.TotalMicroOps);

    // 2. Execution Ports Limit
    Res.PortBound = calculatePortUsageBound(STI, MCII, SimInstrs, MCInsts, Res.PortBottleneckName);

    // 3. Precedence Constraints Limit
    auto Adj = buildDependencyGraph(SimInstrs);
    Res.PrecedenceBound = calculatePrecedenceBound(SimInstrs.size(), Adj, SimInstrs);

    // 4. Overall Max Bottleneck Prediction
    Res.EstimatedCycles = std::max({Res.IssueBound, Res.PortBound, Res.PrecedenceBound});
    Res.EstimatedCPI = Res.EstimatedCycles / static_cast<double>(Res.TotalInstructions);
    Res.DominantBottleneck = determineDominantBottleneck(Res.EstimatedCycles, Res.PrecedenceBound, Res.PortBound, Res.PortBottleneckName);

    return Res;
}

void printFacileResult(const FacileResult &Res, llvm::StringRef CPUName, llvm::raw_ostream &OS) {
    OS << "==================================================\n";
    OS << "Facile Static Analytical Throughput Prediction (AArch64)\n";
    OS << "==================================================\n";
    OS << "Target CPU:             " << CPUName << "\n";
    OS << "Total Instructions:     " << Res.TotalInstructions << "\n";
    OS << "Total MicroOps:         " << Res.TotalMicroOps << "\n";
    OS << "--------------------------------------------------\n";
    OS << "1. Issue Limit:         " << llvm::format("%.2f", Res.IssueBound) << " cycles/iter\n";
    OS << "2. Execution Ports:     " << llvm::format("%.2f", Res.PortBound) << " cycles/iter";
    if (!Res.PortBottleneckName.empty() && Res.PortBottleneckName != "None") {
        OS << "  [Bottleneck: " << Res.PortBottleneckName << "]";
    }
    OS << "\n";
    OS << "3. Precedence (RAW):    " << llvm::format("%.2f", Res.PrecedenceBound) << " cycles/iter\n";
    OS << "--------------------------------------------------\n";
    OS << "Dominant Bottleneck:    " << Res.DominantBottleneck << "\n";
    OS << "Estimated Throughput:   " << llvm::format("%.2f", Res.EstimatedCycles) << " cycles/iter\n";
    OS << "Estimated CPI:          " << llvm::format("%.3f", Res.EstimatedCPI) << "\n";
    OS << "==================================================\n";
}

} // namespace facile
