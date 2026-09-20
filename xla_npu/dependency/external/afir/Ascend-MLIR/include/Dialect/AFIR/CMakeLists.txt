# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

include_directories(${PYASC_DIR}/include/ascir/Dialect/Asc/IR)
include_directories(${PYASC_DIR}/include/ascir/API)
include_directories(${CMAKE_BINARY_DIR}/externals/pyasc/include)

# Dialect definitions
set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(AFIRDialectIncGen)

# Enum definitions (independent from AttrDefs to avoid redefinition)
set(LLVM_TARGET_DEFINITIONS Core/Enums.td)
mlir_tablegen(Enums.h.inc -gen-enum-decls)
mlir_tablegen(Enums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(AFIREnumsIncGen)
add_dependencies(AFIREnumsIncGen AFIRDialectIncGen)

# Attribute definitions (independent from enums in TableGen, but uses enum types in C++)
set(LLVM_TARGET_DEFINITIONS Core/Attrs.td)
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(AFIRAttrsIncGen)
add_dependencies(AFIRAttrsIncGen AFIRDialectIncGen)

# Operation definitions
set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(AFIROpsIncGen)
add_dependencies(AFIROpsIncGen AFIRDialectIncGen AFIRAttrsIncGen AFIREnumsIncGen MLIRAscIncGen AscIRAPITypesIncGen)

# Fix TableGen bug in generated Ops.cpp.inc
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Ops.cpp.inc.fixed
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/cmake/fix_tablegen_bug.cmake ${CMAKE_CURRENT_BINARY_DIR}/Ops.cpp.inc
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Ops.cpp.inc
        COMMENT "Fixing TableGen bug in Ops.cpp.inc"
)
add_custom_target(FixAfiROpsInc DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Ops.cpp.inc.fixed)
add_dependencies(AFIROpsIncGen FixAfiROpsInc)

# Make sure TableGen can find the generated Types.td.inc file
set(MLIR_TABLEGEN_FLAGS "${MLIR_TABLEGEN_FLAGS} -I${PYASC_DIR}/include/ascir/API")

add_mlir_doc(AFIRDialect AFIRDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(AFIRAttrs AFIRAttrs Dialects/ -gen-attrdef-doc)
add_mlir_doc(AFIROps AFIROps Dialects/ -gen-op-doc)

add_subdirectory(Transforms)
add_subdirectory(TransformOps)
