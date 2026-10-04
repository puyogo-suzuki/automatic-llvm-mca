#include "mca/steady_state_tracker.h"
#include "mca/private_access.h"

using namespace llvm;
using mca_common_detail::ReadState_IsReady_Tag;

namespace {

void applyLoopCarriedIgnore(const mca::HWInstructionEvent &Event, unsigned LoopSize, unsigned &CurrentIteration, mca::RegisterFile *PRF) {
    if (LoopSize == 0) return;
    mca::Instruction &Inst = *const_cast<mca::Instruction *>(Event.IR.getInstruction());
    unsigned readerIID = Event.IR.getSourceIndex();
    unsigned readerIteration = readerIID / LoopSize;
    unsigned limitIID = readerIteration * LoopSize;
    
    if (readerIteration > CurrentIteration) {
        CurrentIteration = readerIteration;
        if (PRF) {
            auto member_ptr = get_mappings(RegisterFile_RegisterMappings_Tag{});
            auto &mappings = (*PRF).*member_ptr;
            for (auto &mapping : mappings) {
                if (mapping.first.isValid()) {
                    unsigned writerIID = mapping.first.getSourceIndex();
                    if (writerIID < limitIID) {
                        mapping.first = mca::WriteRef();
                    }
                }
            }
        }
    }

    for (mca::ReadState &RS : Inst.getUses()) {
        if (!RS.isReady()) {
            const mca::CriticalDependency &CRD = RS.getCriticalRegDep();
            if (CRD.IID < limitIID) {
                RS.*get(ReadState_IsReady_Tag{}) = true;
                RS.setIndependentFromDef();
            }
        }
    }
}

} // namespace

void LoopCarriedIgnoreSteadyStateTracker::onInstructionDispatched(const mca::HWInstructionEvent &Event) {
    applyLoopCarriedIgnore(Event, LoopSize, CurrentIteration, PRF);
}
