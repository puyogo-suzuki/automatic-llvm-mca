#ifndef CUSTOM_A55_SCHED_H
#define CUSTOM_A55_SCHED_H

namespace llvm {
class MCSubtargetInfo;
void overrideCortexA55SchedModel(llvm::MCSubtargetInfo &STI);
}

#endif
