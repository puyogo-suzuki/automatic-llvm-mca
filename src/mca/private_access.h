#ifndef MCA_PRIVATE_ACCESS_H
#define MCA_PRIVATE_ACCESS_H

#include "llvm/MCA/HardwareUnits/RegisterFile.h"
#include "llvm/MCA/Instruction.h"

// --- Shared private-member accessor hacks (Colvin/Gibbons "friend injection"
// technique). Explicit template instantiation gives these friend functions
// vague/COMDAT linkage, so it is safe for this header to be included by
// multiple translation units. ---
struct RegisterFile_RegisterMappings_Tag {};
auto get_mappings(RegisterFile_RegisterMappings_Tag);

template <typename Tag, auto M>
struct RobStoreMappings {
  friend auto get_mappings(Tag) {
    return M;
  }
};
template struct RobStoreMappings<RegisterFile_RegisterMappings_Tag, &llvm::mca::RegisterFile::RegisterMappings>;

namespace mca_common_detail {

template <typename Tag, typename Tag::type M>
struct RobStore {
  friend typename Tag::type get(Tag) { return M; }
};

struct ReadState_IsReady_Tag {
  typedef bool llvm::mca::ReadState::*type;
  friend type get(ReadState_IsReady_Tag);
};
template struct RobStore<ReadState_IsReady_Tag, &llvm::mca::ReadState::IsReady>;

} // namespace mca_common_detail

#endif // MCA_PRIVATE_ACCESS_H
