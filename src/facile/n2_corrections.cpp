#include "facile/n2_corrections.h"
#include "cpu/cpu_traits.h"

namespace facile {

// A78 SOG correction: integer register-offset loads/stores do not use an integer
// (I) pipe on the A78 (only halfword forms with a shift do), unlike
// NeoverseN2Model's WriteLDIdx/WriteSTIdx.  See docs.md section 4.D.

// A78 SOG correction: NeoverseN2Model occupies M0 for several cycles for
// SCVTF/UCVTF (gen->vec), FMOV (gen->vec) and DUP (from GPR), which the SOG lists
// as fully pipelined; it models LDPSW with no load pipe; and it occupies the
// divider for the worst case where NeoverseN1Model (A76) uses the best case.
// classifyA78ResOverride() corrects these.  See docs.md section 4.D.


A78ResOverride classifyA78ResOverride(const llvm::MCSubtargetInfo &STI, const llvm::MCInstrInfo &MCII,
                                             const llvm::mca::Instruction &Inst) {
    if (!getCpuTraits(STI.getCPU()).N2ModelCorrections) return A78ResOverride::None;
    llvm::StringRef N = MCII.getName(Inst.getOpcode());
    if ((N == "SDIVWr" || N == "UDIVWr" || N == "SDIVXr" || N == "UDIVXr"))
        return A78ResOverride::DivCap5;
    if (N == "LDPSWi" || N == "LDPSWpost" || N == "LDPSWpre") return A78ResOverride::LDPSW;
    if (N == "FMOVWHr" || N == "FMOVXHr" || N == "FMOVWSr" || N == "FMOVXDr") return A78ResOverride::OccupancyCap1;
    if ((N.starts_with("SCVTF") || N.starts_with("UCVTF")) && N.ends_with("ri") && N.size() == 10)
        return A78ResOverride::OccupancyCap1;  // [SU]CVTF[SU][WX][HSD]ri
    if (N.starts_with("DUPv") && N.ends_with("gpr")) return A78ResOverride::OccupancyCap1;
    return A78ResOverride::None;
}

bool isA78RegOffsetNoAlu(const llvm::MCSubtargetInfo &STI, const llvm::MCInstrInfo &MCII,
                                const llvm::mca::Instruction &Inst, const llvm::MCInst *MCI) {
    if (!getCpuTraits(STI.getCPU()).N2ModelCorrections) return false;
    llvm::StringRef N = MCII.getName(Inst.getOpcode());
    if (!(N.ends_with("roW") || N.ends_with("roX"))) return false;
    llvm::StringRef B = N.drop_back(3);
    static const char *always[] = {"LDRBB", "LDRW", "LDRX", "LDRSBW", "LDRSBX", "LDRSW", "PRFM",
                                   "STRBB", "STRW", "STRX"};
    for (const char *a : always) if (B == a) return true;
    static const char *half[] = {"LDRHH", "LDRSHW", "LDRSHX", "STRHH"};
    for (const char *h : half) {
        if (B == h) {
            // operand 4 = "do shift" flag of the addressing mode (Rt, Rn, Rm, sign-ext, shift)
            if (!MCI || MCI->getNumOperands() < 5 || !MCI->getOperand(4).isImm()) return false;
            return MCI->getOperand(4).getImm() == 0;  // unscaled halfword: L only
        }
    }
    return false;  // FP/SIMD reg-offset forms: left to the model (see SOG Table 3-18)
}

} // namespace facile
