#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>
#include <fcntl.h>

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/MCA/Context.h"
#include "llvm/MCA/InstrBuilder.h"
#include "llvm/MCA/Pipeline.h"
#include "llvm/MCA/SourceMgr.h"
#include "llvm/MCA/CustomBehaviour.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/MC/MCSchedule.h"

using namespace llvm;

// Dummy diagnostic handler to suppress warnings/notes
void mcaDiagHandler(const SMDiagnostic &Diag, void *Context) {
  // Do nothing
}

// Custom streamer to catch MCInsts from the parser
class MCInstCatcher : public MCStreamer {
public:
  std::vector<MCInst> Insts;
  MCInstCatcher(MCContext &Ctx) : MCStreamer(Ctx) {}
  void emitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI) override {
    Insts.push_back(Inst);
  }
  
  // Minimal stubs for pure virtuals
  bool emitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override { return true; }
  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size, Align ByteAlignment) override {}
  void emitZerofill(MCSection *Section, MCSymbol *Symbol, uint64_t Size, Align ByteAlignment, SMLoc Loc) override {}
  
  void emitLabel(MCSymbol *Symbol, SMLoc Loc = SMLoc()) override {}
};

struct MCARunner {
  Triple TT;
  MCTargetOptions MCOPT;
  const Target *TheTarget;
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<MCInstrInfo> MCII;
  std::unique_ptr<MCSubtargetInfo> STI;
  std::unique_ptr<MCInstrAnalysis> MCIA;
  mca::PipelineOptions PO;

  MCARunner(const std::string &TripleStr, const std::string &CPU, const std::string &Features)
      : TT(TripleStr), PO(0, 0, 0, 0, 0, 0, true) {
    std::string Error;
    TheTarget = TargetRegistry::lookupTarget(TT, Error);
    if (!TheTarget) return;

    MRI.reset(TheTarget->createMCRegInfo(TT));
    MAI.reset(TheTarget->createMCAsmInfo(*MRI, TT, MCOPT));
    MCII.reset(TheTarget->createMCInstrInfo());
    STI.reset(TheTarget->createMCSubtargetInfo(TT, CPU, Features));
    MCIA.reset(TheTarget->createMCInstrAnalysis(MCII.get()));
    
    if (!MRI || !MAI || !MCII || !STI || !MCIA) {
        TheTarget = nullptr;
        return;
    }
    
    const MCSchedModel &SM = STI->getSchedModel();
    PO.MicroOpQueueSize = SM.MicroOpBufferSize;
    PO.DispatchWidth = SM.IssueWidth;
    PO.RegisterFileSize = 0; 
    PO.LoadQueueSize = 0;
    PO.StoreQueueSize = 0;
    PO.AssumeNoAlias = true;
  }
};

extern "C" {

void* mca_create(const char* triple, const char* cpu, const char* features) {
  static bool Initialized = false;
  if (!Initialized) {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86AsmParser();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeAArch64TargetInfo();
    LLVMInitializeAArch64Target();
    LLVMInitializeAArch64TargetMC();
    LLVMInitializeAArch64AsmParser();
    LLVMInitializeAArch64AsmPrinter();
    LLVMInitializeARMTargetInfo();
    LLVMInitializeARMTarget();
    LLVMInitializeARMTargetMC();
    LLVMInitializeARMAsmParser();
    LLVMInitializeARMAsmPrinter();
    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVTarget();
    LLVMInitializeRISCVTargetMC();
    LLVMInitializeRISCVAsmParser();
    LLVMInitializeRISCVAsmPrinter();
    Initialized = true;
  }
  auto Runner = new MCARunner(triple, cpu, features);
  if (!Runner->TheTarget) {
    delete Runner;
    return nullptr;
  }
  return Runner;
}

int mca_run(void* handle, const char* asm_code, int iterations, int* retired, int* cycles) {
  auto Runner = static_cast<MCARunner*>(handle);
  if (!Runner) return -1;

  llvm::SourceMgr SrcMgr;
  SrcMgr.setDiagHandler(mcaDiagHandler);
  SrcMgr.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(asm_code), SMLoc());

  MCContext Ctx(Runner->TT, *Runner->MAI, *Runner->MRI, *Runner->STI, &SrcMgr);
  std::unique_ptr<MCObjectFileInfo> MOFI(Runner->TheTarget->createMCObjectFileInfo(Ctx, false));
  Ctx.setObjectFileInfo(MOFI.get());

  MCInstCatcher Catcher(Ctx);
  std::unique_ptr<MCAsmParser> Parser(createMCAsmParser(SrcMgr, Ctx, Catcher, *Runner->MAI));
  
  std::unique_ptr<MCTargetAsmParser> TAP(Runner->TheTarget->createMCAsmParser(*Runner->STI, *Parser, *Runner->MCII));
  if (!TAP) return -6;
  Parser->setTargetParser(*TAP);

  if (Parser->Run(false)) return -2; 
  if (Catcher.Insts.empty()) return -3;

  mca::InstrumentManager IM(*Runner->STI, *Runner->MCII);
  mca::InstrBuilder IB(*Runner->STI, *Runner->MCII, *Runner->MRI, Runner->MCIA.get(), IM, /*CallLatency=*/0);
  
  std::vector<mca::SourceMgr::UniqueInst> Sequence;
  SmallVector<mca::Instrument *> IVec; 
  for (const auto &Inst : Catcher.Insts) {
    auto ExpectedInst = IB.createInstruction(Inst, IVec);
    if (!ExpectedInst) return -4;
    Sequence.push_back(std::move(*ExpectedInst));
  }

  mca::CircularSourceMgr CSM(Sequence, iterations);
  mca::CustomBehaviour CB(*Runner->STI, CSM, *Runner->MCII);
  
  mca::Context MCAContext(*Runner->MRI, *Runner->STI);

  std::unique_ptr<mca::Pipeline> P;
  try {
    if (Runner->STI->getSchedModel().isOutOfOrder()) {
      P = MCAContext.createDefaultPipeline(Runner->PO, CSM, CB);
    } else {
      P = MCAContext.createInOrderPipeline(Runner->PO, CSM, CB);
    }

    if (!P) return -9;

    auto ExpectedCycles = P->run();
    if (!ExpectedCycles) {
        consumeError(ExpectedCycles.takeError());
        return -5;
    }

    *cycles = (int)(*ExpectedCycles);
    *retired = (int)(Catcher.Insts.size() * iterations);
  } catch (...) {
    return -10;
  }

  return 0;
}

void mca_free(void* handle) {
  delete static_cast<MCARunner*>(handle);
}

}
