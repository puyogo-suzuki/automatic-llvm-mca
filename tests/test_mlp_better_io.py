
import os
import sys
import pytest

# Make the package root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze import _compute_mlp, X86Arch

def test_io_mlp_better_estimation():
    arch = X86Arch()
    
    # Case 1: Used after 2 instructions
    # 0: movq (%rdi), %rax  (load)
    # 1: addq $1, %rbx
    # 2: addq %rax, %rcx    (use %rax)
    instrs1 = [
        (0x0, "movq", "(%rdi), %rax"),
        (0x4, "addq", "$1, %rbx"),
        (0x8, "addq", "%rax, %rcx"),
    ]
    # n=3. load at 0. use at 2. dist = 2 - 0 = 2.
    # MLP = min(4, 2) / 1 = 2.0
    mlp = _compute_mlp(instrs1, decode_width=4, arch=arch, dependency="io")
    assert mlp == 2.0

    # Case 2: No use
    # 0: movq (%rdi), %rax  (load)
    # 1: addq $1, %rbx
    instrs2 = [
        (0x0, "movq", "(%rdi), %rax"),
        (0x4, "addq", "$1, %rbx"),
    ]
    # n=2. load at 0. no use. dist = 2 - 1 - 0 = 1.
    # MLP = min(4, 1) / 1 = 1.0
    mlp = _compute_mlp(instrs2, decode_width=4, arch=arch, dependency="io")
    assert mlp == 1.0

    # Case 3: Multiple loads
    # 0: movq (%rdi), %rax  (load)
    # 1: movq (%rsi), %rbx  (load)
    # 2: addq %rax, %rcx    (use %rax)
    # 3: addq %rbx, %rdx    (use %rbx)
    instrs3 = [
        (0x0, "movq", "(%rdi), %rax"),
        (0x4, "movq", "(%rsi), %rbx"),
        (0x8, "addq", "%rax, %rcx"),
        (0xc, "addq", "%rbx, %rdx"),
    ]
    # Load 0: dist = 2 - 0 = 2.
    # Load 1: dist = 3 - 1 = 2.
    # MLP = (min(4, 2) + min(4, 2)) / 2 = 2.0
    mlp = _compute_mlp(instrs3, decode_width=4, arch=arch, dependency="io")
    assert mlp == 2.0

    # Case 4: Single load
    instrs4 = [
        (0x0, "movq", "(%rdi), %rax"),
    ]
    # n=1. dist = 1 - 1 - 0 = 0.
    mlp = _compute_mlp(instrs4, decode_width=4, arch=arch, dependency="io")
    assert mlp == 0.0

    # Case 5: min(W, dist)
    instrs5 = [
        (0x0, "movq", "(%rdi), %rax"),
        (0x4, "nop", ""),
        (0x8, "nop", ""),
        (0xc, "nop", ""),
        (0x10, "addq", "%rax, %rcx"),
    ]
    # dist = 4 - 0 = 4.
    assert _compute_mlp(instrs5, decode_width=2, arch=arch, dependency="io") == 2.0
    assert _compute_mlp(instrs5, decode_width=4, arch=arch, dependency="io") == 4.0
    assert _compute_mlp(instrs5, decode_width=8, arch=arch, dependency="io") == 4.0

def test_ooo_mlp_remains_same():
    # Verify that ooo still works as before (OLD mission)
    from analyze import _compute_mlp, X86Arch
    arch = X86Arch()
    
    # 0: movq (%rdi), %rax (load)
    # 1: movq (%rsi), %rbx (load, independent)
    # 2: movq (%rdx), %rcx (load, independent)
    instrs = [
        (0x0, "movq", "(%rdi), %rax"),
        (0x4, "movq", "(%rsi), %rbx"),
        (0x8, "movq", "(%rdx), %rcx"),
    ]
    # For load 0, W=4:
    # mlp_0 = 1 (itself) + 1 (load 1) + 1 (load 2) = 3
    # For load 1, W=4:
    # mlp_1 = 1 (itself) + 1 (load 2) = 2
    # For load 2, W=4:
    # mlp_2 = 1 (itself) = 1
    # Average = (3 + 2 + 1) / 3 = 2.0
    mlp = _compute_mlp(instrs, decode_width=4, arch=arch, dependency="ooo")
    assert mlp == 2.0
