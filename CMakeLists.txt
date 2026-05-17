cmake_minimum_required(VERSION 3.10)
project(mca_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(LLVM 23 REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_library(mca_runner SHARED mca_runner.cpp)

# Link against the necessary LLVM components
llvm_map_components_to_libnames(llvm_libs 
    mca 
    mc 
    mcparser 
    support 
    core
    x86desc
    x86info
    x86asmparser
    aarch64desc
    aarch64info
    aarch64asmparser
    armdesc
    arminfo
    armasmparser
    riscvdesc
    riscvinfo
    riscvasmparser
    x86codegen
    aarch64codegen
    armcodegen
    riscvcodegen
)

target_link_libraries(mca_runner PRIVATE ${llvm_libs})
