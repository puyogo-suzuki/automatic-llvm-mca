#include "test_support.h"

// === walkRegions Region Partitioning Tests ===

TEST(MLPTest, SplitterBasicBlockPartitioning) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC, "movq %rax, %rbx\naddq $1, %rcx\nsubq $1, %rdx");
    ASSERT_EQ(instrs.size(), 3u);
    
    FunctionBoundaries empty_bounds;
    std::vector<RegionSpan> bbs;
    std::vector<RegionSpan> loops;

    walkRegions(instrs, empty_bounds,
                [&](const RegionSpan &Span) { loops.push_back(Span); },
                [&](const RegionSpan &Span) { bbs.push_back(Span); });

    // There are no branches, so it should resolve as a single basic block of size 3.
    EXPECT_TRUE(loops.empty());
    ASSERT_EQ(bbs.size(), 1u);
    EXPECT_EQ(bbs[0].Start, 0u);
    EXPECT_EQ(bbs[0].Size, 3u);
}


TEST(MLPTest, SplitterBasicBlockSplittingLimit) {
    initLLVMX86();
    TestContext TC;
    auto instrs = parseAsm(TC, "movq %rax, %rbx\naddq $1, %rcx\nsubq $1, %rdx");
    ASSERT_EQ(instrs.size(), 3u);
    
    FunctionBoundaries empty_bounds;
    std::vector<RegionSpan> bbs;
    std::vector<RegionSpan> loops;

    // bbMaxInstrs = 2, so a block of size 3 should split into a block of size 2 and size 1.
    walkRegions(instrs, empty_bounds,
                [&](const RegionSpan &Span) { loops.push_back(Span); },
                [&](const RegionSpan &Span) { bbs.push_back(Span); });

    EXPECT_TRUE(loops.empty());
#if OPT_MERGE_BB
    ASSERT_EQ(bbs.size(), 1u);
    EXPECT_EQ(bbs[0].Start, 0u);
    EXPECT_EQ(bbs[0].Size, 3u);
#else
    ASSERT_EQ(bbs.size(), 2u);
    EXPECT_EQ(bbs[0].Start, 0u);
    EXPECT_EQ(bbs[0].Size, 2u);
    EXPECT_EQ(bbs[1].Start, 2u);
    EXPECT_EQ(bbs[1].Size, 1u);
#endif
}


TEST(MLPTest, SplitterNopInstructionCheck) {
    initLLVMAArch64();
    AArch64TestContext TC;
    auto instrs = parseAsm(TC, "nop\nnop\nnop");
    ASSERT_EQ(instrs.size(), 3u);
    EXPECT_TRUE(isAllNopRegion(instrs, *TC.MCII));

    auto mixed_instrs = parseAsm(TC, "nop\nadd x0, x1, x2\nnop");
    ASSERT_EQ(mixed_instrs.size(), 3u);
    EXPECT_FALSE(isAllNopRegion(mixed_instrs, *TC.MCII));
}


TEST(MLPTest, SplitterNestedLoopNestingLimits) {
    FunctionBoundaries empty_bounds;
    std::vector<Instr> instrs(6);
    for (size_t i = 0; i < 6; ++i) instrs[i].Addr = i * 4;

    instrs[3].IsBranch = true;
    instrs[3].BranchTarget = 8; // Loop 1: [2, 3]

    instrs[5].IsBranch = true;
    instrs[5].BranchTarget = 4; // Loop 2: [1, 5]

    std::vector<RegionSpan> outer_loops;
    std::vector<RegionSpan> outer_bbs;
    walkRegions(instrs, empty_bounds,
                [&](const RegionSpan &Span) {
                    if (Span.Start == Span.AnalysisStart) outer_loops.push_back(Span);
                },
                [&](const RegionSpan &Span) { outer_bbs.push_back(Span); });

    ASSERT_EQ(outer_loops.size(), 2u);
    EXPECT_EQ(outer_loops[0].Start, 1u);
    EXPECT_EQ(outer_loops[0].Size, 5u);
    EXPECT_EQ(outer_loops[1].Start, 2u);
    EXPECT_EQ(outer_loops[1].Size, 2u);
}


TEST(MLPTest, SplitterPostDominatorLoopMerging) {
    initLLVMX86();
    TestContext TC;
    // 0: nop (addr = 0)
    // 1: addq $1, %rax (addr = 4)   <-- Loop pre-header (post-dominated by loop header 2)
    // 2: subq $1, %rbx (addr = 8)   <-- Loop Header
    // 3: jne -8 (addr = 12)         <-- Loop Latch (back-edge to 2)
    // 4: movq %rax, %rcx (addr = 16) <-- Post-loop BB (NOT post-dominated by loop)
    // 5: retq (addr = 20)           <-- Function EXIT
    std::vector<Instr> instrs(6);
    for (size_t i = 0; i < 6; ++i) instrs[i].Addr = i * 4;
    instrs[3].IsBranch = true;
    instrs[3].BranchTarget = 8; // target = 2 (subq)

    FunctionBoundaries empty_bounds;
    std::vector<RegionSpan> loops;
    std::vector<RegionSpan> bbs;

    walkRegions(instrs, empty_bounds,
                [&](const RegionSpan &Span) {
                    if (Span.Start == Span.AnalysisStart) loops.push_back(Span);
                },
                [&](const RegionSpan &Span) { bbs.push_back(Span); });

    // Loop should be detected (IID 2 to 3, size = 2)
    ASSERT_EQ(loops.size(), 1u);
    EXPECT_EQ(loops[0].Start, 2u);
    EXPECT_EQ(loops[0].Size, 2u);

    // IID 0 (nop) and IID 1 (pre-header) are post-dominated by loop header (IID 2)
    // because all paths from them must pass through the loop.  They are merged
    // into the loop region and NOT emitted as basic blocks.
    //
    // IID 4 (movq) and IID 5 (retq) are post-loop BBs.  Their post-dominator
    // chain goes directly to virtual_exit without passing through the loop header,
    // so they are NOT merged and ARE emitted as basic blocks.
    //
    // Expected: 1 BB spanning IID 4–5 (Start=4, Size=2).
    EXPECT_EQ(bbs.size(), 0u);
}


TEST(MLPTest, SplitterAbabMerging) {
    opts::ChainThreshold = 1;
    FunctionBoundaries empty_bounds;

    std::vector<Instr> instrs(5);
    for (size_t i = 0; i < 5; ++i) instrs[i].Addr = i * 4;

    instrs[3].IsBranch = true;
    instrs[3].BranchTarget = 4; // Loop A: [1, 3]

    instrs[4].IsBranch = true;
    instrs[4].BranchTarget = 8; // Loop B: [2, 4]

    std::vector<RegionSpan> loops;
    std::vector<RegionSpan> bbs;
    walkRegions(instrs, empty_bounds,
                [&](const RegionSpan &Span) {
                    if (Span.Start == Span.AnalysisStart) loops.push_back(Span);
                },
                [&](const RegionSpan &Span) { bbs.push_back(Span); });

    // Pre-merged loops A and B should be discarded and merged into 1 bounding span [1, 4] (size 4)
    ASSERT_EQ(loops.size(), 1u);
    EXPECT_EQ(loops[0].Start, 1u);
    EXPECT_EQ(loops[0].Size, 4u);
}


TEST(MLPTest, SplitterAbccabNestedPreservation) {
    opts::ChainThreshold = 1;
    FunctionBoundaries empty_bounds;

    std::vector<Instr> instrs(8);
    for (size_t i = 0; i < 8; ++i) instrs[i].Addr = i * 4;

    instrs[4].IsBranch = true;
    instrs[4].BranchTarget = 12; // Loop C: [3, 4] (fully nested in A and B)

    instrs[6].IsBranch = true;
    instrs[6].BranchTarget = 4;  // Loop A: [1, 6]

    instrs[7].IsBranch = true;
    instrs[7].BranchTarget = 8;  // Loop B: [2, 7]

    std::vector<RegionSpan> loops;
    std::vector<RegionSpan> bbs;
    walkRegions(instrs, empty_bounds,
                [&](const RegionSpan &Span) {
                    if (Span.Start == Span.AnalysisStart) loops.push_back(Span);
                },
                [&](const RegionSpan &Span) { bbs.push_back(Span); });

    // Loop C is nested (not abab partial overlap), so C must be PRESERVED.
    // Loops A and B are abab overlapping, so they are merged into bounding span [1, 7].
    // Total actual loop regions: 2 (Loop C [3, 4] and Merged AB [1, 7])
    ASSERT_EQ(loops.size(), 2u);

    // First loop: Loop C [3, 4]
    EXPECT_EQ(loops[0].Start, 3u);
    EXPECT_EQ(loops[0].Size, 2u);

    // Second loop: Merged AB [1, 7]
    EXPECT_EQ(loops[1].Start, 1u);
    EXPECT_EQ(loops[1].Size, 7u);
}


TEST(MLPTest, SplitterThresholdBoundary) {
    FunctionBoundaries empty_bounds;

    std::vector<Instr> instrs(5);
    for (size_t i = 0; i < 5; ++i) instrs[i].Addr = i * 4;

    instrs[3].IsBranch = true;
    instrs[3].BranchTarget = 4; // Loop A: [1, 3]

    instrs[4].IsBranch = true;
    instrs[4].BranchTarget = 8; // Loop B: [2, 4]

    // 1) High threshold (K=100): abab merging disabled, both A and B are kept as separate loops
    {
        opts::ChainThreshold = 100;
        std::vector<RegionSpan> loops;
        std::vector<RegionSpan> bbs;
        walkRegions(instrs, empty_bounds,
                    [&](const RegionSpan &Span) {
                        if (Span.Start == Span.AnalysisStart) loops.push_back(Span);
                    },
                    [&](const RegionSpan &Span) { bbs.push_back(Span); });

        ASSERT_EQ(loops.size(), 2u);
        EXPECT_EQ(loops[0].Start, 1u);
        EXPECT_EQ(loops[0].Size, 3u);
        EXPECT_EQ(loops[1].Start, 2u);
        EXPECT_EQ(loops[1].Size, 3u);
    }

    // 2) Low threshold (K=1): abab merging enabled, A and B merged into bounding span [1, 4]
    {
        opts::ChainThreshold = 1;
        std::vector<RegionSpan> loops;
        std::vector<RegionSpan> bbs;
        walkRegions(instrs, empty_bounds,
                    [&](const RegionSpan &Span) {
                        if (Span.Start == Span.AnalysisStart) loops.push_back(Span);
                    },
                    [&](const RegionSpan &Span) { bbs.push_back(Span); });

        ASSERT_EQ(loops.size(), 1u);
        EXPECT_EQ(loops[0].Start, 1u);
        EXPECT_EQ(loops[0].Size, 4u);
    }

    // Reset default ChainThreshold to 5
    opts::ChainThreshold = 5;
}


TEST(MLPTest, SplitterDirectRetExclusion) {
    FunctionBoundaries bounds;
    bounds[0x1000] = 0x2000;

    // Case 1: A: ldr; ldr; ldr; ret; add; add; goto A; (Should NOT be a loop)
    {
        std::vector<Instr> instrs(7, Instr{});
        for (size_t i = 0; i < 7; ++i) instrs[i].Addr = 0x1000 + i * 4;
        instrs[3].IsReturn = true;
        instrs[6].IsBranch = true;
        instrs[6].BranchTarget = 0x1000; // backward jump to 0x1000 (A)

        std::vector<RegionSpan> loops;
        walkRegions(instrs, bounds,
                    [&](const RegionSpan &Span) {
                        if (Span.Start == Span.AnalysisStart) loops.push_back(Span);
                    },
                    [](const RegionSpan &) {});

        EXPECT_EQ(loops.size(), 0u);
    }

    // Case 2: A: ldr; ldr; b.gt goto B; ret; B: add; add; goto A; (Should BE a loop)
    {
        std::vector<Instr> instrs(7, Instr{});
        for (size_t i = 0; i < 7; ++i) instrs[i].Addr = 0x1000 + i * 4;
        instrs[2].IsBranch = true;
        instrs[2].BranchTarget = 0x1010; // forward branch to 0x1010 (B)
        instrs[3].IsReturn = true;
        instrs[6].IsBranch = true;
        instrs[6].BranchTarget = 0x1000; // backward jump to 0x1000 (A)

        std::vector<RegionSpan> loops;
        walkRegions(instrs, bounds,
                    [&](const RegionSpan &Span) {
                        if (Span.Start == Span.AnalysisStart) loops.push_back(Span);
                    },
                    [](const RegionSpan &) {});

        EXPECT_EQ(loops.size(), 1u);
        if (!loops.empty()) {
            EXPECT_EQ(loops[0].Start, 0u);
            EXPECT_EQ(loops[0].Size, 7u);
        }
    }

    // Case 3: B: add; add; A: ldr; ldr; b.gt goto B; ret; add; add; goto A; (Should NOT be a loop)
    {
        std::vector<Instr> instrs(9, Instr{});
        for (size_t i = 0; i < 9; ++i) instrs[i].Addr = 0x1000 + i * 4;
        instrs[4].IsBranch = true;
        instrs[4].BranchTarget = 0x1000; // backward branch to 0x1000 (B)
        instrs[5].IsReturn = true;
        instrs[8].IsBranch = true;
        instrs[8].BranchTarget = 0x1008; // backward jump to 0x1008 (A)

        std::vector<RegionSpan> loops;
        walkRegions(instrs, bounds,
                    [&](const RegionSpan &Span) {
                        if (Span.Start == Span.AnalysisStart) loops.push_back(Span);
                    },
                    [](const RegionSpan &) {});

        // Loop B [0, 4] is kept, while candidate A [2, 8] is excluded due to direct ret
        EXPECT_EQ(loops.size(), 1u);
        if (!loops.empty()) {
            EXPECT_EQ(loops[0].Start, 0u);
            EXPECT_EQ(loops[0].Size, 5u);
        }
    }
}


// --merge-same-header: two back-edges to the SAME header are one natural loop.
// Layout mirrors mcf primal_bea_mpp: hdr=1, short early-continue back-edge at 3 -> 1,
// full-body back-edge at 6 -> 1, plus an unrelated DIFFERENT-header inner loop [4,5].
static std::vector<RegionSpan> collectLoopSpans(std::vector<Instr> &instrs) {
    FunctionBoundaries empty_bounds;
    std::vector<RegionSpan> loops, bbs;
    walkRegions(instrs, empty_bounds,
                [&](const RegionSpan &S) { if (S.Start == S.AnalysisStart) loops.push_back(S); },
                [&](const RegionSpan &S) { bbs.push_back(S); });
    return loops;
}


TEST(MLPTest, SplitterMergeSameHeader) {
    opts::ChainThreshold = 5;
    std::vector<Instr> instrs(8);
    for (size_t i = 0; i < 8; ++i) instrs[i].Addr = i * 4;
    instrs[3].IsBranch = true; instrs[3].BranchTarget = 4;   // [1,3] short same-header path
    instrs[5].IsBranch = true; instrs[5].BranchTarget = 16;  // [4,5] different header (real inner loop)
    instrs[6].IsBranch = true; instrs[6].BranchTarget = 4;   // [1,6] full body

    opts::MergeSameHeader = false;
    auto off = collectLoopSpans(instrs);
    ASSERT_EQ(off.size(), 3u);  // pre-existing behavior unchanged when flag off

    opts::MergeSameHeader = true;
    auto on = collectLoopSpans(instrs);
    opts::MergeSameHeader = false;
    ASSERT_EQ(on.size(), 2u);
    bool has16 = false, has45 = false, has13 = false;
    for (auto &s : on) {
        if (s.Start == 1 && s.Size == 6) has16 = true;
        if (s.Start == 4 && s.Size == 2) has45 = true;
        if (s.Start == 1 && s.Size == 3) has13 = true;
    }
    EXPECT_TRUE(has16);
    EXPECT_TRUE(has45);   // different-header nested loop is preserved
    EXPECT_FALSE(has13);  // same-header partial path is dropped
}


TEST(MLPTest, SplitterMergeSameHeaderKeepsAbabSpansIdentical) {
    // abab pair [1,3],[2,4] with threshold 1 merges to [1,4]; the flag must not alter it.
    std::vector<Instr> instrs(5);
    for (size_t i = 0; i < 5; ++i) instrs[i].Addr = i * 4;
    instrs[3].IsBranch = true; instrs[3].BranchTarget = 4;
    instrs[4].IsBranch = true; instrs[4].BranchTarget = 8;
    for (int th : {1, 100}) {
        opts::ChainThreshold = th;
        opts::MergeSameHeader = false;
        auto a = collectLoopSpans(instrs);
        opts::MergeSameHeader = true;
        auto b = collectLoopSpans(instrs);
        opts::MergeSameHeader = false;
        ASSERT_EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            EXPECT_EQ(a[i].Start, b[i].Start);
            EXPECT_EQ(a[i].Size, b[i].Size);
        }
    }
    opts::ChainThreshold = 5;
}
