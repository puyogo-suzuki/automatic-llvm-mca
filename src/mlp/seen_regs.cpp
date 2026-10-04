#include "mlp/mlp_analyzer.h"
#include <algorithm>
#include <bitset>
#include <cmath>
#include <iostream>
#include <string>
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

bool SeenBaseRegs::test(unsigned reg, int64_t cache_line) const {
    for (const auto &el : data) {
        if (el.base_reg == reg && el.cache_line == cache_line && el.is_completed)
            return true;
    }
    return false;
}

void SeenBaseRegs::set(unsigned reg, int64_t cache_line, const llvm::MCRegisterInfo &MRI) {
    if (reg == 0) return;
    for (llvm::MCRegAliasIterator AI(reg, &MRI, true); AI.isValid(); ++AI) {
        unsigned alias = *AI;
        bool found = false;
        for (auto &el : data) {
            if (el.base_reg == alias && el.cache_line == cache_line) {
                el.is_completed = true;
                found = true;
                break;
            }
        }
        if (!found) {
            Element el;
            el.base_reg = alias;
            el.cache_line = cache_line;
            el.is_completed = true;
            data.push_back(el);
        }
    }
}

void SeenBaseRegs::reset(unsigned reg, const llvm::MCRegisterInfo &MRI) {
    if (reg == 0) return;
    for (llvm::MCRegAliasIterator AI(reg, &MRI, true); AI.isValid(); ++AI) {
        unsigned alias = *AI;
        data.erase(std::remove_if(data.begin(), data.end(),
                                  [alias](const Element &el) {
                                      return el.base_reg == alias;
                                  }),
                   data.end());
    }
}

void SeenBaseRegs::add_load(unsigned reg, int64_t cache_line, const llvm::SmallVectorImpl<unsigned> &dests, const llvm::MCRegisterInfo &MRI) {
    if (reg == 0) return;
    for (llvm::MCRegAliasIterator AI(reg, &MRI, true); AI.isValid(); ++AI) {
        unsigned alias = *AI;
        bool found = false;
        for (auto &el : data) {
            if (el.base_reg == alias && el.cache_line == cache_line) {
                for (unsigned dest : dests) {
                    for (llvm::MCRegAliasIterator DAI(dest, &MRI, true); DAI.isValid(); ++DAI) {
                        unsigned d_alias = *DAI;
                        if (std::find(el.pending_dest_regs.begin(), el.pending_dest_regs.end(), d_alias) == el.pending_dest_regs.end()) {
                            el.pending_dest_regs.push_back(d_alias);
                        }
                    }
                }
                found = true;
                break;
            }
        }
        if (!found) {
            Element el;
            el.base_reg = alias;
            el.cache_line = cache_line;
            el.is_completed = false;
            for (unsigned dest : dests) {
                for (llvm::MCRegAliasIterator DAI(dest, &MRI, true); DAI.isValid(); ++DAI) {
                    el.pending_dest_regs.push_back(*DAI);
                }
            }
            data.push_back(el);
        }
    }
}

void SeenBaseRegs::check_uses(const llvm::SmallVectorImpl<unsigned> &uses, const llvm::MCRegisterInfo &MRI) {
    if (uses.empty()) return;
    for (unsigned use : uses) {
        for (llvm::MCRegAliasIterator AI(use, &MRI, true); AI.isValid(); ++AI) {
            unsigned alias = *AI;
            for (auto &el : data) {
                if (el.is_completed) continue;
                if (std::find(el.pending_dest_regs.begin(), el.pending_dest_regs.end(), alias) != el.pending_dest_regs.end()) {
                    el.is_completed = true;
                    el.pending_dest_regs.clear();
                }
            }
        }
    }
}

void updateSeenBaseRegs(const MLPInstInfo &inst_info, SeenBaseRegs &seen_base_regs, const llvm::MCRegisterInfo &MRI) {
    seen_base_regs.check_uses(inst_info.io_regs.inputs, MRI);

    bool is_valid_load = inst_info.is_load() && inst_info.mem_info.valid();
    unsigned base_reg = is_valid_load ? inst_info.mem_info.base_reg : 0;
    if (is_valid_load && base_reg != 0 && inst_info.mem_info.is_constant_offset()) {
        int64_t cache_line = inst_info.mem_info.offset / 64;
        for (unsigned reg : inst_info.io_regs.outputs) {
            for (llvm::MCRegAliasIterator AI(base_reg, &MRI, true); AI.isValid(); ++AI) {
                if (*AI == reg) goto end_base_regs_set;
            }
        }
        seen_base_regs.add_load(base_reg, cache_line, inst_info.io_regs.outputs, MRI);
    }
  end_base_regs_set:
    for (unsigned reg : inst_info.io_regs.outputs) {
        seen_base_regs.reset(reg, MRI);
    }
}

bool SeenStoreAddrs::test(unsigned reg, int64_t offset) const {
    for (const auto &el : data) {
        if (el.base_reg == reg && el.offset == offset)
            return true;
    }
    return false;
}

void SeenStoreAddrs::set(unsigned reg, int64_t offset, const llvm::MCRegisterInfo &MRI) {
    if (reg == 0) return;
    for (llvm::MCRegAliasIterator AI(reg, &MRI, true); AI.isValid(); ++AI) {
        unsigned alias = *AI;
        bool found = false;
        for (auto &el : data) {
            if (el.base_reg == alias && el.offset == offset) {
                found = true;
                break;
            }
        }
        if (!found) {
            Element el;
            el.base_reg = alias;
            el.offset = offset;
            data.push_back(el);
        }
    }
}

void SeenStoreAddrs::reset(unsigned reg, const llvm::MCRegisterInfo &MRI) {
    if (reg == 0) return;
    for (llvm::MCRegAliasIterator AI(reg, &MRI, true); AI.isValid(); ++AI) {
        unsigned alias = *AI;
        data.erase(std::remove_if(data.begin(), data.end(),
                                  [alias](const Element &el) {
                                      return el.base_reg == alias;
                                  }),
                   data.end());
    }
}

void updateSeenStoreAddrs(const MLPInstInfo &inst_info, SeenStoreAddrs &seen_store_addrs, const llvm::MCRegisterInfo &MRI) {
    bool is_valid_store = inst_info.is_store() && inst_info.mem_info.valid();
    if (is_valid_store && inst_info.mem_info.base_reg != 0 && inst_info.mem_info.is_constant_offset()) {
        // Record the address this store writes to (base_reg + exact byte offset),
        // using the base register's value as of *before* this instruction executes.
        seen_store_addrs.set(inst_info.mem_info.base_reg, inst_info.mem_info.offset, MRI);
    }
    // Any instruction that redefines a register invalidates stored addresses keyed
    // on the old value of that register (e.g. a writeback store/load that
    // post/pre-increments its own base register, or an unrelated add/mov).
    for (unsigned reg : inst_info.io_regs.outputs) {
        seen_store_addrs.reset(reg, MRI);
    }
}
