# The outer CMake needs to rely on MLIR${dialect}IncGen to generate mair_ops.h.inc
function(add_mair_dialect dialect)
  message(STATUS "mair_dialect = ${CMAKE_CURRENT_SOURCE_DIR}")
  set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}.h.inc -gen-op-decls)
  mlir_tablegen(${dialect}.cc.inc -gen-op-defs)
  add_public_tablegen_target(MLIR${dialect}IncGen)
  add_dependencies(mlir-headers MLIR${dialect}IncGen)
endfunction()

add_mair_dialect(mair_ops)

include_directories(BEFORE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MAIN_SRC_DIR}/ir)

add_mlir_dialect_library(AirDialect
        mair_ops.cc

        DEPENDS
        MLIRmair_opsIncGen
)
target_link_libraries(AirDialect PUBLIC
        MLIRInferTypeOpInterface        
        MLIRSideEffectInterfaces
        MLIRBytecodeOpInterface
        MLIRIR
)
