#include "test_support.h"

// The per-CPU table (cpu_traits.cpp) that replaced the CPU-name ladders.
TEST(CpuTraitsTest, TableContents) {
    // Variants share their core's row.
    for (const char *n : {"cortex-a78", "cortex-a78ae", "cortex-a78c"}) {
        const CpuTraits &T = getCpuTraits(n);
        EXPECT_EQ(T.Model, CpuModel::A78) << n;
        EXPECT_EQ(T.UopDispatchWidth, 12u) << n;
        EXPECT_EQ(T.MopDispatchWidth, 6u) << n;
        EXPECT_EQ(T.Fusion, FusionTable::A78) << n;
        EXPECT_TRUE(T.ZeroLatencyMov && T.N2ModelCorrections) << n;
    }
    for (const char *n : {"cortex-x1", "cortex-x1c"}) {
        const CpuTraits &T = getCpuTraits(n);
        EXPECT_EQ(T.Model, CpuModel::X1) << n;
        EXPECT_EQ(T.UopDispatchWidth, 16u) << n;
        EXPECT_EQ(T.MopDispatchWidth, 8u) << n;
        EXPECT_EQ(T.Fusion, FusionTable::X1) << n;
        EXPECT_TRUE(T.ZeroLatencyMov) << n;
        EXPECT_FALSE(T.N2ModelCorrections) << n;  // V1 model has no N2 defects
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
