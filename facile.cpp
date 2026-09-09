#include "facile.h"
#include "mca_common.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
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

// Cortex-A78 Software Optimization Guide sec. 4.14 "Instruction fusion":
// specific adjacent AArch64 instruction pairs execute as a single fused
// operation on Cortex-A78 ("These instruction pairs must be adjacent to each
// other in program code"). The second instruction of a fused pair consumes
// no additional issue-width or execution-port resource. Confirmed this is
// A78-specific: the Cortex-A76 SOG documents no equivalent CMP/CSEL or
// CMP/B.cond fusion (only unrelated FP fused-multiply-accumulate), so this
// must not be applied there.
//   1. CMP/CMN (immediate) + B.cond      6. CMP (register) + CSET
//   2. CMP/CMN (register) + B.cond       7. TST (immediate) + B.cond
//   3. CMP (immediate) + CSEL            8. TST (register) + B.cond
//   4. CMP (register) + CSEL             9. BICS (register) + B.cond
//   5. CMP (immediate) + CSET            10. NOP + Any instruction
// CMP/CMN/TST are flag-setting SUBS/ADDS/ANDS with a discarded (WZR/XZR)
// destination; only those forms qualify, not general flag-setting ops with
// a real destination register.
bool isA78FusionCandidate(const llvm::MCSubtargetInfo &STI) {
    return STI.getCPU() == "cortex-a78" || STI.getCPU() == "cortex-a78ae" || STI.getCPU() == "cortex-a78c";
}

// mca::Instruction::getDefs() elides writes to the discarded zero register
// (nothing can ever depend on WZR/XZR's value, so the register-renaming
// dependency tracker doesn't bother recording it) -- so whether a
// SUBS/ADDS/ANDS is a true CMP/CMN/TST (destination discarded) vs. a general
// flag-setting op with a real destination can only be checked on the raw
// MCInst operand list, not on the mca::Instruction.
bool writesZeroReg(const llvm::MCInst *MCI) {
    if (!MCI || MCI->getNumOperands() == 0 || !MCI->getOperand(0).isReg())
        return false;
    llvm::MCRegister R = MCI->getOperand(0).getReg();
    return R == llvm::AArch64::WZR || R == llvm::AArch64::XZR;
}

bool isA78FusibleProducer(const llvm::MCInstrInfo &MCII, const llvm::mca::Instruction &Inst,
                          const llvm::MCInst *MCI) {
    llvm::StringRef Name = MCII.getName(Inst.getOpcode());
    bool IsCmpCmnTst = Name.starts_with_insensitive("SUBS") || Name.starts_with_insensitive("ADDS") ||
                        Name.starts_with_insensitive("ANDS");
    bool IsBics = Name.starts_with_insensitive("BICS");
    if (IsBics) return true; // BICS(register)+B.cond: always the register form
    if (!IsCmpCmnTst) return false;
    return writesZeroReg(MCI);
}

bool isA78FusibleConsumer(const llvm::MCInstrInfo &MCII, const llvm::mca::Instruction &Inst) {
    llvm::StringRef Name = MCII.getName(Inst.getOpcode());
    // CSET is an alias for CSINC Wd, WZR, WZR, invert(cond) -- detecting the
    // alias precisely would need operand inspection; only the plain CSEL
    // form (which this codebase can verify directly) is matched here.
    return Name.equals_insensitive("Bcc") || Name.starts_with_insensitive("CSEL");
}

// Returns a mask, one entry per instruction, true if that instruction is the
// *second* half of an adjacent fused pair (and should be excluded from
// issue-width/port-resource accounting).
std::vector<bool> computeA78FusedMask(const llvm::MCSubtargetInfo &STI,
                                      const llvm::MCInstrInfo &MCII,
                                      llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                                      llvm::ArrayRef<const llvm::MCInst *> MCInsts) {
    std::vector<bool> Fused(SimInstrs.size(), false);
    if (!isA78FusionCandidate(STI) || SimInstrs.size() < 2)
        return Fused;
    for (size_t i = 0; i + 1 < SimInstrs.size(); ++i) {
        const llvm::MCInst *MCI = (i < MCInsts.size()) ? MCInsts[i] : nullptr;
        if (isA78FusibleProducer(MCII, *SimInstrs[i], MCI) && isA78FusibleConsumer(MCII, *SimInstrs[i + 1]))
            Fused[i + 1] = true;
    }
    return Fused;
}

// 1. Calculate Dispatch / Issue Width Limit
double calculateIssueBound(const llvm::MCSchedModel &SM,
                            llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                            unsigned &TotalUops,
                            unsigned DispatchWidthOverride,
                            const std::vector<bool> &FusedMask) {
    unsigned IssueWidth = DispatchWidthOverride > 0 ? DispatchWidthOverride
                         : (SM.IssueWidth > 0 ? SM.IssueWidth : 1);
    TotalUops = 0;
    for (size_t i = 0; i < SimInstrs.size(); ++i) {
        if (i < FusedMask.size() && FusedMask[i])
            continue;
        unsigned NumUops = SimInstrs[i]->getNumMicroOps();
        TotalUops += (NumUops > 0 ? NumUops : 1);
    }
    return static_cast<double>(TotalUops) / static_cast<double>(IssueWidth);
}

// 2. Calculate Execution Ports Contention Bound
double calculatePortUsageBound(const llvm::MCSubtargetInfo &STI,
                               const llvm::MCInstrInfo &MCII,
                               llvm::ArrayRef<std::unique_ptr<llvm::mca::Instruction>> SimInstrs,
                               llvm::ArrayRef<const llvm::MCInst *> MCInsts,
                               std::string &BottleneckPortName,
                               const std::vector<bool> &FusedMask) {
    const llvm::MCSchedModel &SM = STI.getSchedModel();
    unsigned NumProcResources = SM.NumProcResourceKinds;
    std::vector<double> ProcResUsage(NumProcResources, 0.0);

    for (size_t i = 0; i < SimInstrs.size(); ++i) {
        if (i < FusedMask.size() && FusedMask[i])
            continue;
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
                                     llvm::ArrayRef<const llvm::MCInst *> MCInsts,
                                     unsigned DispatchWidth) {
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

    std::vector<bool> FusedMask = computeA78FusedMask(STI, MCII, SimInstrs, MCInsts);

    // 1. Issue Limit
    Res.IssueBound = calculateIssueBound(STI.getSchedModel(), SimInstrs, Res.TotalMicroOps, DispatchWidth, FusedMask);

    // 2. Execution Ports Limit
    Res.PortBound = calculatePortUsageBound(STI, MCII, SimInstrs, MCInsts, Res.PortBottleneckName, FusedMask);

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
