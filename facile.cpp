#include "facile.h"
#include "mca_common.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <vector>

namespace facile {

namespace {

struct DependencyEdge {
    size_t Target;
    double Latency;
    unsigned Distance; // 0 for intra-iteration, 1 for inter-iteration (loop-carried)
};

// Extract instruction latency with a minimum threshold of 1.0 cycle
double getInstLatency(const llvm::mca::Instruction &Inst) {
    double Lat = static_cast<double>(Inst.getLatency());
    return std::max(Lat, 1.0);
}

// Resolve dynamic variant scheduling classes via subtarget info and MCInst if available
const llvm::MCSchedClassDesc *resolveSchedClass(const llvm::MCSubtargetInfo &STI,
                                                const llvm::MCInstrInfo &MCII,
                                                unsigned SchedClass,
                                                const llvm::MCInst *MCI) {
    const llvm::MCSchedModel &SM = STI.getSchedModel();
    const llvm::MCSchedClassDesc *SCDesc = SM.getSchedClassDesc(SchedClass);
    unsigned PrevSchedClass = SchedClass;
    unsigned CurrSchedClass = SchedClass;

    while (SCDesc && SCDesc->isVariant()) {
        if (MCI) {
            CurrSchedClass = STI.resolveVariantSchedClass(CurrSchedClass, MCI, &MCII, SM.getProcessorID());
        }
        if (CurrSchedClass == PrevSchedClass) break;
        PrevSchedClass = CurrSchedClass;
        SCDesc = SM.getSchedClassDesc(CurrSchedClass);
    }
    return SCDesc;
}

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
        
        const llvm::MCSchedClassDesc *SCDesc = resolveSchedClass(STI, MCII, MCID.getSchedClass(), MCI);
        if (!SCDesc) continue;

        for (const llvm::MCWriteProcResEntry *WPR = STI.getWriteProcResBegin(SCDesc);
             WPR != STI.getWriteProcResEnd(SCDesc); ++WPR) {
            unsigned ProcResIdx = WPR->ProcResourceIdx;
            unsigned Cycles = WPR->ReleaseAtCycle - WPR->AcquireAtCycle;
            if (Cycles == 0) continue; // Skip entries that consume 0 resource cycles
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
                double Lat = getInstLatency(*SimInstrs[WriterIdx]);
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
            double Lat = getInstLatency(*SimInstrs[WriterIdx]);
            Adj[WriterIdx].push_back({ReaderIdx, Lat, 1});
        }
    }

    return Adj;
}

// Check if a cycle ratio mu is achievable using SPFA negative cycle detection on transformed weights: c = mu * distance - latency
bool isAchievableRatio(double mu,
                       size_t N,
                       const std::vector<std::vector<DependencyEdge>> &Adj) {
    std::vector<double> dist(N, 0.0);
    std::vector<unsigned> count(N, 0);
    std::vector<bool> inQueue(N, true);
    std::queue<size_t> q;
    for (size_t i = 0; i < N; ++i) {
        q.push(i);
    }

    size_t maxOps = N * std::max<size_t>(10, N);
    size_t ops = 0;

    while (!q.empty()) {
        size_t u = q.front();
        q.pop();
        inQueue[u] = false;

        if (++ops > maxOps) {
            return true;
        }

        for (const auto &E : Adj[u]) {
            double weight = mu * E.Distance - E.Latency;
            if (dist[u] + weight < dist[E.Target] - 1e-9) {
                dist[E.Target] = dist[u] + weight;
                count[E.Target] = count[u] + 1;
                if (count[E.Target] >= N) {
                    return true;
                }
                if (!inQueue[E.Target]) {
                    q.push(E.Target);
                    inQueue[E.Target] = true;
                }
            }
        }
    }

    return false;
}

// 4. Calculate Maximum Cycle Ratio (MCR) for Precedence Constraints
double calculatePrecedenceBound(
    size_t N,
    const std::vector<std::vector<DependencyEdge>> &Adj) {
    
    if (N == 0) return 0.0;

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

    // Determine upper bound for binary search.
    // Any cycle includes intra-iteration edges and at least one inter-iteration edge.
    // Therefore MCR <= sum of all edge latencies in the graph.
    double maxLatencySum = 0.0;
    for (size_t u = 0; u < N; ++u) {
        for (const auto &E : Adj[u]) {
            if (E.Latency > 0.0) {
                maxLatencySum += E.Latency;
            }
        }
    }

    if (maxLatencySum <= 0.0) {
        return MaxRatio;
    }

    double low = MaxRatio;
    double high = std::max({MaxRatio + 1.0, maxLatencySum, 1.0});

    // Binary search for exact MCR
    for (int iter = 0; iter < 30; ++iter) {
        double mid = low + (high - low) / 2.0;
        if (isAchievableRatio(mid, N, Adj)) {
            low = mid;
        } else {
            high = mid;
        }
    }

    return low;
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

    // For Coalesced ROB CPUs (Apple Firestorm / Icestorm), normalize NumMicroOps to 1 (MOP-based)
    if (isCoalescedROBCPU(STI.getCPU())) {
        for (const auto &Inst : SimInstrs) {
            llvm::mca::InstrDesc &MutableDesc = const_cast<llvm::mca::InstrDesc &>(Inst->getDesc());
            MutableDesc.NumMicroOps = 1;
        }
    }

    // 1. Issue Limit
    Res.IssueBound = calculateIssueBound(STI.getSchedModel(), SimInstrs, Res.TotalMicroOps);

    // 2. Execution Ports Limit
    Res.PortBound = calculatePortUsageBound(STI, MCII, SimInstrs, MCInsts, Res.PortBottleneckName);

    // 3. Precedence Constraints Limit
    auto Adj = buildDependencyGraph(SimInstrs);
    Res.PrecedenceBound = calculatePrecedenceBound(SimInstrs.size(), Adj);

    // 4. Overall Max Bottleneck Prediction
    Res.EstimatedCycles = std::max({Res.IssueBound, Res.PortBound, Res.PrecedenceBound});
    Res.EstimatedCPI = Res.EstimatedCycles / static_cast<double>(Res.TotalInstructions);
    Res.DominantBottleneck = determineDominantBottleneck(Res.EstimatedCycles, Res.PrecedenceBound, Res.PortBound, Res.PortBottleneckName);

    if (Res.EstimatedCycles == Res.PrecedenceBound && Res.PrecedenceBound > 0.0) {
        Res.FacileReason = "prec";
    } else if (Res.EstimatedCycles == Res.PortBound && Res.PortBound > Res.IssueBound) {
        Res.FacileReason = "exec";
    } else {
        Res.FacileReason = "inst";
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
