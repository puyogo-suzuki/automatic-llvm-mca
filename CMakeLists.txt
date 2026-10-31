cmake_minimum_required(VERSION 3.10)
project(mca_tool)

set(CMAKE_CXX_STANDARD 17)

find_package(LLVM 22.1.8 REQUIRED CONFIG)

# Custom command to run llvm-tblgen and generate AArch64GenSubtargetInfo.inc
set(A55_TD_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ModifiedTarget/AArch64/AArch64SchedA55.td)
set(A76_TD_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ModifiedTarget/AArch64/AArch64SchedNeoverseN1.td)
set(A78_TD_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ModifiedTarget/AArch64/AArch64SchedNeoverseN2.td)
set(X1_TD_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ModifiedTarget/AArch64/AArch64SchedNeoverseV1.td)
set(ICESTORM_TD_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ModifiedTarget/AArch64/AArch64SchedIcestorm.td)
set(FIRESTORM_TD_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ModifiedTarget/AArch64/AArch64SchedFirestorm.td)
set(A520_TD_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ModifiedTarget/AArch64/AArch64SchedA520.td)
set(A720_TD_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ModifiedTarget/AArch64/AArch64SchedA720.td)
set(A76_PROC_TD_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ModifiedTarget/AArch64/AArch64Processors.td)
# Every TableGen output is derived from the whole AArch64.td include tree, and the
# ModifiedTarget/AArch64/*.td files shadow their llvm-source counterparts (the -I
# order below).  AArch64GenInstrInfo.inc in particular carries the per-opcode
# SchedClass index, so it must be regenerated whenever an InstRW is added or
# removed in any scheduling model -- otherwise the opcode -> SchedClass mapping in
# the binary disagrees with AArch64GenSubtargetInfo.inc and unrelated
# latencies/resources are reported.  All three outputs therefore depend on all
# of these .td files.
file(GLOB AARCH64_MODIFIED_TD_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ModifiedTarget/AArch64/*.td)
file(GLOB AARCH64_UPSTREAM_TD_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/llvm-source/llvm/lib/Target/AArch64/*.td)
set(AARCH64_TD_DEPENDS ${AARCH64_MODIFIED_TD_FILES} ${AARCH64_UPSTREAM_TD_FILES}
    ${A55_TD_FILE} ${A76_TD_FILE} ${A78_TD_FILE} ${X1_TD_FILE} ${ICESTORM_TD_FILE} ${FIRESTORM_TD_FILE}
    ${A520_TD_FILE} ${A720_TD_FILE} ${A76_PROC_TD_FILE})
set(GEN_INC_FILE ${CMAKE_CURRENT_BINARY_DIR}/AArch64GenSubtargetInfo.inc)
set(GEN_REG_INC_FILE ${CMAKE_CURRENT_BINARY_DIR}/AArch64GenRegisterInfo.inc)
set(GEN_INSTR_INC_FILE ${CMAKE_CURRENT_BINARY_DIR}/AArch64GenInstrInfo.inc)

add_custom_command(
    OUTPUT ${GEN_INC_FILE}
    COMMAND /usr/bin/llvm-tblgen-22 -gen-subtarget
            -I ${CMAKE_CURRENT_SOURCE_DIR}/ModifiedTarget/AArch64
            -I ${CMAKE_CURRENT_SOURCE_DIR}/llvm-source/llvm/lib/Target/AArch64
            -I ${CMAKE_CURRENT_SOURCE_DIR}/llvm-source/llvm/include
            -I /usr/include/llvm-22
            ${CMAKE_CURRENT_SOURCE_DIR}/llvm-source/llvm/lib/Target/AArch64/AArch64.td
            -o ${GEN_INC_FILE}
    DEPENDS ${AARCH64_TD_DEPENDS}
    COMMENT "Generating AArch64GenSubtargetInfo.inc using llvm-tblgen"
)

add_custom_command(
    OUTPUT ${GEN_REG_INC_FILE}
    COMMAND /usr/bin/llvm-tblgen-22 -gen-register-info
            -I ${CMAKE_CURRENT_SOURCE_DIR}/ModifiedTarget/AArch64
            -I ${CMAKE_CURRENT_SOURCE_DIR}/llvm-source/llvm/lib/Target/AArch64
            -I ${CMAKE_CURRENT_SOURCE_DIR}/llvm-source/llvm/include
            -I /usr/include/llvm-22
            ${CMAKE_CURRENT_SOURCE_DIR}/llvm-source/llvm/lib/Target/AArch64/AArch64.td
            -o ${GEN_REG_INC_FILE}
    DEPENDS ${AARCH64_TD_DEPENDS}
    COMMENT "Generating AArch64GenRegisterInfo.inc using llvm-tblgen"
)

add_custom_command(
    OUTPUT ${GEN_INSTR_INC_FILE}
    COMMAND /usr/bin/llvm-tblgen-22 -gen-instr-info
            -I ${CMAKE_CURRENT_SOURCE_DIR}/ModifiedTarget/AArch64
            -I ${CMAKE_CURRENT_SOURCE_DIR}/llvm-source/llvm/lib/Target/AArch64
            -I ${CMAKE_CURRENT_SOURCE_DIR}/llvm-source/llvm/include
            -I /usr/include/llvm-22
            ${CMAKE_CURRENT_SOURCE_DIR}/llvm-source/llvm/lib/Target/AArch64/AArch64.td
            -o ${GEN_INSTR_INC_FILE}
    DEPENDS ${AARCH64_TD_DEPENDS}
    COMMENT "Generating AArch64GenInstrInfo.inc using llvm-tblgen"
)

include_directories(${LLVM_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Common library for shared logic between the tools and the tests.  Sources are
# grouped by module under src/ (headers are included as "<module>/<file>.h").
set(MCA_SOURCES
    src/core/object_loader.cpp
    src/core/region_splitter.cpp
    src/cpu/cpu_traits.cpp
    src/cpu/sched_model.cpp
    src/a55/a55_adaptations.cpp
    src/a55/a55_issue_stage.cpp
    src/mca/mca_simulation.cpp
    src/mca/steady_state_tracker.cpp
    src/mlp/mlp_aarch64.cpp
    src/mlp/mlp_analyzer.cpp
    src/mlp/mlp_riscv.cpp
    src/mlp/mlp_window.cpp
    src/mlp/mlp_x86.cpp
    src/mlp/seen_regs.cpp
    src/facile/facile.cpp
    src/facile/fusion.cpp
    src/facile/n2_corrections.cpp
    src/frontend/frontend.cpp
    src/frontend/options.cpp)
add_library(mca_common_lib STATIC ${MCA_SOURCES} ${GEN_INC_FILE} ${GEN_REG_INC_FILE} ${GEN_INSTR_INC_FILE})
target_include_directories(mca_common_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(mca_common_lib PUBLIC LLVM)

add_executable(mca_tool apps/mca_main.cpp)
add_executable(mlp_objdump apps/mlp_objdump.cpp)
add_executable(analyze_distribution apps/analyze_distribution.cpp)
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

add_executable(mca_unit_tests tests/mlp_test.cpp tests/splitter_test.cpp tests/facile_test.cpp tests/cpu_traits_test.cpp
               tests/a55_sog_test.cpp)
target_link_libraries(mca_unit_tests PRIVATE gtest_main mca_common_lib ${llvm_libs})

include(GoogleTest)
gtest_discover_tests(mca_unit_tests)

add_executable(mca_insts_info apps/mca_insts_info.cpp)
target_link_libraries(mca_insts_info PRIVATE mca_common_lib ${llvm_libs})
set_target_properties(mca_insts_info PROPERTIES OUTPUT_NAME "mca-insts-info")


add_executable(mlp_check apps/mlp_check.cpp)
target_link_libraries(mlp_check PRIVATE mca_common_lib ${llvm_libs})
set_target_properties(mlp_check PROPERTIES OUTPUT_NAME "mlp-check")






