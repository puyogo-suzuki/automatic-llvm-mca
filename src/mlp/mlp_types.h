#ifndef MLP_MLP_TYPES_H
#define MLP_MLP_TYPES_H

// Per-instruction memory-access / register-dependence descriptors used by the
// MLP analysis (and by Facile's store->load edges), and the "seen" trackers
// behind the always-hit load heuristics.

#include <bitset>
#include <vector>
#include "core/types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCRegisterInfo.h"

struct MemAccessInfo {
    std::bitset<7> flags;
    unsigned base_reg = 0;
    int64_t offset = 0;

    bool valid() const { return flags.test(0); }
    void set_valid(bool val) { flags.set(0, val); }

    bool is_pc_relative() const { return flags.test(1); }
    void set_is_pc_relative(bool val) { flags.set(1, val); }

    bool is_stack_access() const { return flags.test(2); }
    void set_is_stack_access(bool val) { flags.set(2, val); }

    bool is_load() const { return flags.test(3); }
    void set_is_load(bool val) { flags.set(3, val); }

    bool is_store() const { return flags.test(4); }
    void set_is_store(bool val) { flags.set(4, val); }

    bool is_writeback() const { return flags.test(5); }
    void set_is_writeback(bool val) { flags.set(5, val); }

    bool is_constant_offset() const { return flags.test(6); }
    void set_is_constant_offset(bool val) { flags.set(6, val); }
};

struct RegDeps {
    llvm::SmallVector<unsigned, 4> inputs;
    llvm::SmallVector<unsigned, 4> outputs;
};

struct MLPInstInfo {
    std::bitset<3> flags; // bit 0: is_load, bit 1: is_store, bit 2: is_call
    short num_uops = 1;
    RegDeps io_regs;
    MemAccessInfo mem_info;

    bool is_load() const { return flags.test(0); }
    bool is_store() const { return flags.test(1); }
    bool is_call() const { return flags.test(2); }
    void set_is_load(bool val) { flags.set(0, val); }
    void set_is_store(bool val) { flags.set(1, val); }
    void set_is_call(bool val) { flags.set(2, val); }
};

#include "llvm/ADT/SmallVector.h"

struct SeenBaseRegs {
    struct Element {
        unsigned base_reg = 0;
        int64_t cache_line = 0;
        bool is_completed = false;
        llvm::SmallVector<unsigned, 4> pending_dest_regs;
    };
    llvm::SmallVector<Element, 16> data;
    bool test(unsigned reg, int64_t cache_line) const;
    void set(unsigned reg, int64_t cache_line, const llvm::MCRegisterInfo &MRI);
    void reset(unsigned reg, const llvm::MCRegisterInfo &MRI);
    void add_load(unsigned reg, int64_t cache_line, const llvm::SmallVectorImpl<unsigned> &dests, const llvm::MCRegisterInfo &MRI);
    void check_uses(const llvm::SmallVectorImpl<unsigned> &uses, const llvm::MCRegisterInfo &MRI);
};

// Tracks the most recent store to a given (base_reg, exact byte offset) pair,
// so that a later load from the *exact same address* (not merely the same
// 64-byte cache line) can be recognized as a store-to-load forwarding hit.
// Unlike SeenBaseRegs (which matches at cache-line granularity and is used to
// approximate "recently loaded" cache residency), this struct requires an
// exact offset match, and it is armed by stores rather than by loads.
// Entries for a base register are invalidated whenever that register is
// redefined (its value -- and therefore the address it denotes -- may have
// changed), mirroring SeenBaseRegs::reset.
struct SeenStoreAddrs {
    struct Element {
        unsigned base_reg = 0;
        int64_t offset = 0;
    };
    llvm::SmallVector<Element, 16> data;
    bool test(unsigned reg, int64_t offset) const;
    void set(unsigned reg, int64_t offset, const llvm::MCRegisterInfo &MRI);
    void reset(unsigned reg, const llvm::MCRegisterInfo &MRI);
};

// Fills the load/store flags of `info` from the instruction descriptor.
inline void finalizeMemAccessInfo(MemAccessInfo &info, const llvm::MCInstrDesc &MCID) {
    info.set_is_load(MCID.mayLoad());
    info.set_is_store(MCID.mayStore());
}

#endif // MLP_MLP_TYPES_H
