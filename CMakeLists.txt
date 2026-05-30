cmake_minimum_required(VERSION 3.10)
project(mca_tool)

set(CMAKE_CXX_STANDARD 17)

find_package(LLVM 23 REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Common library for shared logic between tool and tests
add_library(mca_common_lib STATIC mlp_logic.cpp mca.cpp splitter.cpp)
target_link_libraries(mca_common_lib PUBLIC LLVM)

add_executable(mca_tool mca_main.cpp)
add_executable(mlp_objdump mlp_objdump.cpp)
add_executable(analyze_distribution analyze_distribution.cpp)
set_target_properties(mlp_objdump PROPERTIES OUTPUT_NAME "mlp-objdump")

# Explicitly link all necessary components for each supported architecture
llvm_map_components_to_libnames(llvm_libs 
    mca 
    mc 
    mcparser 
    mcdisassembler
    support 
    core
    object
    x86info x86desc x86codegen x86asmparser x86disassembler
    aarch64info aarch64desc aarch64codegen aarch64asmparser aarch64disassembler
    arminfo armdesc armcodegen armasmparser armdisassembler
    riscvinfo riscvdesc riscvcodegen riscvasmparser riscvdisassembler
)

target_link_libraries(mca_tool PRIVATE mca_common_lib ${llvm_libs})
target_link_libraries(mlp_objdump PRIVATE mca_common_lib ${llvm_libs})
target_link_libraries(analyze_distribution PRIVATE mca_common_lib ${llvm_libs})

# GoogleTest setup
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(mca_unit_tests tests/mlp_test.cpp)
target_link_libraries(mca_unit_tests PRIVATE gtest_main mca_common_lib ${llvm_libs})

include(GoogleTest)
gtest_discover_tests(mca_unit_tests)
add_executable(convergence_test convergence_test.cpp)
target_link_libraries(convergence_test PRIVATE mca_common_lib ${llvm_libs})

add_executable(mca_insts_info mca_insts_info.cpp)
target_link_libraries(mca_insts_info PRIVATE mca_common_lib ${llvm_libs})
set_target_properties(mca_insts_info PROPERTIES OUTPUT_NAME "mca-insts-info")

