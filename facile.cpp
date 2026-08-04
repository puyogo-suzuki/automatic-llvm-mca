#include "facile.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <set>
#include <vector>

namespace facile {

FacileResult computeFacilePrediction(const llvm::MCSubtargetInfo &STI,
                                     const llvm::MCInstrInfo &MCII,
                                     const llvm::MCRegisterInfo &MRI,
                                     llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                                     llvm::ArrayRef<const llvm::MCInst *> MCInsts) {
    FacileResult Res;
    if (SimInstrs.empty()) return Res;

    Res.TotalInstructions = SimInstrs.size();

    // 1. Compute Total MicroOps and Issue Bound
    const llvm::MCSchedModel &SM = STI.getSchedModel();
    unsigned IssueWidth = SM.IssueWidth > 0 ? SM.IssueWidth : 1;

    unsigned TotalUops = 0;
    for (const auto &Inst : SimInstrs) {
        unsigned NumUops = Inst->getNumMicroOps();
        TotalUops += (NumUops > 0 ? NumUops : 1);
    }
    Res.TotalMicroOps = TotalUops;
    Res.IssueBound = static_cast<double>(TotalUops) / static_cast<double>(IssueWidth);

    // 2. Compute Execution Ports Bound (ProcResource contention)
    unsigned NumProcResources = SM.NumProcResourceKinds;
    std::vector<double> ProcResUsage(NumProcResources, 0.0);

    for (size_t i = 0; i < SimInstrs.size(); ++i) {
        const auto &Inst = SimInstrs[i];
        const llvm::MCInst *MCI = (i < MCInsts.size()) ? MCInsts[i] : nullptr;
        const llvm::MCInstrDesc &MCID = MCII.get(Inst->getOpcode());
        unsigned SchedClassID = MCID.getSchedClass();
        unsigned ResolvedSchedClass = SchedClassID;
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
    std::string BottleneckPortName = "None";

    for (unsigned r = 1; r < NumProcResources; ++r) {
        const llvm::MCProcResourceDesc *PRD = SM.getProcResource(r);
        if (!PRD || PRD->NumUnits == 0) continue;
        double PortCycles = ProcResUsage[r] / static_cast<double>(PRD->NumUnits);
        if (PortCycles > MaxPortBound) {
            MaxPortBound = PortCycles;
            BottleneckPortName = PRD->Name ? PRD->Name : ("Port" + std::to_string(r));
        }
    }
    Res.PortBound = MaxPortBound;
    Res.PortBottleneckName = BottleneckPortName;

    // 3. Compute Precedence Constraints (Loop-Carried RAW Dependency Chain)
    size_t N = SimInstrs.size();
    std::map<unsigned, size_t> LastWriter; // RegNum -> Instruction Index

    struct Edge {
        size_t Target;
        double Latency;
        unsigned Distance; // 0 for intra-iteration, 1 for inter-iteration
    };

    std::vector<std::vector<Edge>> Adj(N);

    // Pass 1: Identify Intra-iteration (Distance=0) and True Loop-Carried (Distance=1) dependencies
    std::map<unsigned, size_t> FirstReader; // Reg -> First instruction that reads Reg before writing it
    std::set<unsigned> DefinedRegs;

    for (size_t i = 0; i < N; ++i) {
        const auto &Inst = SimInstrs[i];

        // Process Uses (Reads)
        for (const auto &Op : Inst->getUses()) {
            unsigned Reg = Op.getRegisterID();
            if (Reg == 0) continue;

            // If Reg was not defined yet in this block, it is a Live-In read for loop-carried dep
            if (DefinedRegs.find(Reg) == DefinedRegs.end()) {
                if (FirstReader.find(Reg) == FirstReader.end()) {
                    FirstReader[Reg] = i;
                }
            }

            // Intra-iteration RAW dependency
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

        // Process Defs (Writes)
        for (const auto &Op : Inst->getDefs()) {
            unsigned Reg = Op.getRegisterID();
            if (Reg != 0) {
                LastWriter[Reg] = i;
                DefinedRegs.insert(Reg);
            }
        }
    }

    // Pass 2: Add True Loop-Carried Dependencies (LastWriter -> FirstReader, Distance=1)
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

    // Maximum Cycle Ratio (MCR) computation
    double MaxRatio = 0.0;
    
    // Check self-loops
    for (size_t u = 0; u < N; ++u) {
        for (const auto &E : Adj[u]) {
            if (E.Target == u && E.Distance > 0) {
                double Ratio = E.Latency / E.Distance;
                if (Ratio > MaxRatio) MaxRatio = Ratio;
            }
        }
    }

    // Floyd-Warshall-style MCR evaluation (guarded for N <= 1000 to prevent 20GB OOM)
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
            double u_lat = SimInstrs[u]->getLatency();
            if (u_lat < 1.0) u_lat = 1.0;
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

    Res.PrecedenceBound = MaxRatio;

    // 4. Overall Facile Prediction
    Res.EstimatedCycles = std::max({Res.IssueBound, Res.PortBound, Res.PrecedenceBound});
    Res.EstimatedCPI = Res.EstimatedCycles / static_cast<double>(Res.TotalInstructions);

    if (Res.EstimatedCycles == Res.PrecedenceBound && Res.PrecedenceBound > 0.0) {
        Res.DominantBottleneck = "Precedence Constraints (Dependency Chain)";
    } else if (Res.EstimatedCycles == Res.PortBound && Res.PortBound > 0.0) {
        Res.DominantBottleneck = "Execution Ports (" + Res.PortBottleneckName + ")";
    } else {
        Res.DominantBottleneck = "Issue Width (Dispatch Limit)";
    }

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
