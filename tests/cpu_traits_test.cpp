#include "sog_test_support.h"

// The per-CPU table (cpu_traits.cpp) that replaced the CPU-name ladders.
TEST(CpuTraitsTest, TableContents) {
    // Variants share their core's row.
    for (const char *n : {"cortex-a78", "cortex-a78ae", "cortex-a78c"}) {
        const CpuTraits &T = getCpuTraits(n);
        EXPECT_EQ(T.Model, CpuModel::A78) << n;
        EXPECT_EQ(T.UopDispatchWidth, 12u) << n;
        EXPECT_EQ(T.MopDispatchWidth, 6u) << n;
        EXPECT_EQ(T.Fusion, FusionTable::A78) << n;
        EXPECT_TRUE(T.ZeroLatencyMov) << n;
    }
    for (const char *n : {"cortex-x1", "cortex-x1c"}) {
        const CpuTraits &T = getCpuTraits(n);
        EXPECT_EQ(T.Model, CpuModel::X1) << n;
        EXPECT_EQ(T.UopDispatchWidth, 16u) << n;
        EXPECT_EQ(T.MopDispatchWidth, 8u) << n;
        EXPECT_EQ(T.Fusion, FusionTable::X1) << n;
        EXPECT_TRUE(T.ZeroLatencyMov) << n;
    }
    // Stock-model cores that only share a uop width keep no Mop cap and no local model.
    EXPECT_EQ(getCpuTraits("neoverse-v1").UopDispatchWidth, 16u);
    EXPECT_EQ(getCpuTraits("neoverse-v1").MopDispatchWidth, 0u);
    EXPECT_EQ(getCpuTraits("neoverse-v1").Model, CpuModel::None);
    EXPECT_EQ(getCpuTraits("cortex-a710").UopDispatchWidth, 10u);
    EXPECT_EQ(getCpuTraits("cortex-a710").MopDispatchWidth, 0u);
    EXPECT_EQ(getCpuTraits("cortex-a76").MopDispatchWidth, 4u);
    EXPECT_FALSE(getCpuTraits("cortex-a76").ZeroLatencyMov);  // N1 uses Latency==0 as a marker
    EXPECT_EQ(getCpuTraits("cortex-a720").MopDispatchWidth, 5u);
    EXPECT_EQ(getCpuTraits("cortex-a720").Fusion, FusionTable::A720);
    // Unknown names, including "generic", get the neutral default.
    for (const char *n : {"generic", "", "cortex-a999"}) {
        const CpuTraits &T = getCpuTraits(n);
        EXPECT_EQ(T.Model, CpuModel::None) << n;
        EXPECT_EQ(T.UopDispatchWidth, 0u) << n;
        EXPECT_EQ(T.Fusion, FusionTable::None) << n;
    }
}


// Apple's cores are recognised from the installed sched model: LLVM reports
// getCPU() == "apple-a14" for both, so the CPU name cannot be used.
TEST(CpuTraitsTest, CoalescedROBSubtargetFromSchedModel) {
    initLLVMAArch64();
    {
        AArch64TestContext fire("firestorm");
        EXPECT_TRUE(isCoalescedROBSubtarget(*fire.STI));
    }
    {
        AArch64TestContext ice("icestorm");
        EXPECT_TRUE(isCoalescedROBSubtarget(*ice.STI));
    }
    {
        AArch64TestContext a78("cortex-a78");
        EXPECT_FALSE(isCoalescedROBSubtarget(*a78.STI));
    }
}


TEST(CpuTraitsTest, AppleWindowResources) {
    const WindowResources &F = getCpuTraits("firestorm").MlpWindow;
    EXPECT_TRUE(F.enabled());
    EXPECT_EQ(F.RenameEntries, 623u);
    EXPECT_EQ(F.LoadSlots, 130u);
    EXPECT_EQ(F.StoreSlots, 60u);
    EXPECT_EQ(F.BranchSlots, 144u);
    const WindowResources &I = getCpuTraits("icestorm").MlpWindow;
    EXPECT_TRUE(I.enabled());
    EXPECT_EQ(I.RenameEntries, 111u);
    EXPECT_EQ(I.LoadSlots, 30u);
    EXPECT_EQ(I.StoreSlots, 18u);
    EXPECT_EQ(I.BranchSlots, 0u);   // the source gives no branch limit: unlimited
    EXPECT_FALSE(getCpuTraits("cortex-a78").MlpWindow.enabled());
}


// Every --mcpu spelling resolves to one canonical name (the CpuTraits key); only the short Apple
// aliases differ from the spelling itself.  LLVM has no per-core Apple CPU: both cores are created
// as "apple-m1".
TEST(CpuTraitsTest, CanonicalCpuName) {
    EXPECT_EQ(canonicalCpuName("ice"), "icestorm");
    EXPECT_EQ(canonicalCpuName("fire"), "firestorm");
    for (const char *n : {"icestorm", "firestorm", "cortex-a55", "cortex-a520", "cortex-a76", "cortex-a78",
                          "cortex-a78c", "cortex-a720", "cortex-x1", "cortex-x1c", "neoverse-n1",
                          "generic", "apple-m1"})
        EXPECT_EQ(canonicalCpuName(n), n) << n;
    EXPECT_EQ(llvmCpuName("icestorm"), "apple-m1");
    EXPECT_EQ(llvmCpuName("firestorm"), "apple-m1");
    for (const char *n : {"cortex-a55", "cortex-a78c", "cortex-x1c", "neoverse-v1", "generic"})
        EXPECT_EQ(llvmCpuName(n), n) << n;
    // Aliases are not keys of the table: only their canonical name is.
    EXPECT_EQ(getCpuTraits("ice").Model, CpuModel::None);
    EXPECT_EQ(getCpuTraits("fire").Model, CpuModel::None);
    EXPECT_EQ(&getCpuTraits(canonicalCpuName("ice")), &getCpuTraits("icestorm"));
    EXPECT_EQ(&getCpuTraits(canonicalCpuName("fire")), &getCpuTraits("firestorm"));
}

// The subtarget the tools create for a CPU gives back its canonical name and traits, so code that
// only has the MCSubtargetInfo (Facile, the MCA driver) keys by the same name as the frontend.
// For the Apple cores this needs the installed model: getCPU() is LLVM's "apple-a14" for both.
TEST(CpuTraitsTest, SubtargetCanonicalNameRoundTrip) {
    initLLVMAArch64();
    for (const char *mcpu : {"cortex-a55", "cortex-a520", "cortex-a520ae", "cortex-a76", "cortex-a76ae",
                             "neoverse-n1", "cortex-a78", "cortex-a78ae", "cortex-a78c", "cortex-a720",
                             "cortex-a720ae", "cortex-a710", "cortex-a715", "neoverse-n2", "cortex-x1",
                             "cortex-x1c", "neoverse-v1", "icestorm", "firestorm", "ice", "fire",
                             "generic"}) {
        AArch64TestContext TC("cortex-a76");
        const llvm::StringRef canon = canonicalCpuName(mcpu);
        auto STI = llvm::createLocalSubtargetInfo(*TC.TheTarget, TC.TT, canon, *TC.MCII);
        ASSERT_TRUE(STI) << mcpu;
        EXPECT_EQ(canonicalCpuName(*STI), canon) << mcpu;
        EXPECT_EQ(&getCpuTraits(*STI), &getCpuTraits(canon)) << mcpu;
    }
    AArch64TestContext TC("cortex-a76");
    auto Fire = llvm::createLocalSubtargetInfo(*TC.TheTarget, TC.TT, "firestorm", *TC.MCII);
    EXPECT_EQ(Fire->getCPU(), "apple-a14");
}

// The aliases get the local Apple tables, not the stock apple-m1 model: FMADD (D) has latency 4
// in the local Firestorm and Icestorm tables (Dougall Johnson's tables) and 10 in libLLVM's stock
// apple-m1 model.
TEST(CpuTraitsTest, AppleAliasesInstallLocalTables) {
    const std::string Fmadd = "fmadd d0, d1, d2, d3";
    const SogInstInfo Stock = sogDesc("apple-m1", Fmadd);  // apple-m1 itself has no local model
    ASSERT_TRUE(Stock.Valid);
    EXPECT_EQ(Stock.MaxLatency, 10u);
    for (const char *mcpu : {"ice", "icestorm", "fire", "firestorm"}) {
        const SogInstInfo Local = sogDesc(mcpu, Fmadd);
        ASSERT_TRUE(Local.Valid) << mcpu;
        EXPECT_EQ(Local.MaxLatency, 4u) << mcpu;
        AArch64TestContext TC(mcpu);
        EXPECT_TRUE(isCoalescedROBSubtarget(*TC.STI)) << mcpu;
    }
    // Icestorm and Firestorm differ on UDIV (W): 10 vs 11 in the local tables.
    EXPECT_EQ(sogDesc("ice", "udiv w0, w1, w2").MaxLatency, sogDesc("icestorm", "udiv w0, w1, w2").MaxLatency);
    EXPECT_EQ(sogDesc("fire", "udiv w0, w1, w2").MaxLatency, sogDesc("firestorm", "udiv w0, w1, w2").MaxLatency);
    EXPECT_NE(sogDesc("ice", "udiv w0, w1, w2").MaxLatency, sogDesc("fire", "udiv w0, w1, w2").MaxLatency);
}

// The MLP window limits the frontend hands to the analyzer (getCpuTraits(TI.CPU).MlpWindow, with
// TI.CPU the canonical name) are the Apple core's for its alias too.
TEST(CpuTraitsTest, AppleAliasesSelectWindowResources) {
    const WindowResources &F = getCpuTraits(canonicalCpuName("fire")).MlpWindow;
    EXPECT_TRUE(F.enabled());
    EXPECT_EQ(F.RenameEntries, 623u);
    EXPECT_EQ(F.BranchSlots, 144u);
    EXPECT_EQ(F.MaxInsts, 330u * 7);
    const WindowResources &I = getCpuTraits(canonicalCpuName("ice")).MlpWindow;
    EXPECT_TRUE(I.enabled());
    EXPECT_EQ(I.RenameEntries, 111u);
    EXPECT_EQ(I.LoadSlots, 30u);
    EXPECT_EQ(I.MaxInsts, 60u * 7);
}
