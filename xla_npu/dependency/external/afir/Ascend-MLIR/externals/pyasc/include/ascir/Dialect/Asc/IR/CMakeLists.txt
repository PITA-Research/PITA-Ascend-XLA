# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(AscendCDialect.h.inc -gen-dialect-decls -dialect=ascendc)
mlir_tablegen(AscendCDialect.cpp.inc -gen-dialect-defs -dialect=ascendc)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(AscendCOps.h.inc -gen-op-decls)
mlir_tablegen(AscendCOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS Interfaces.td)
mlir_tablegen(AscendCOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(AscendCOpInterfaces.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS Core/Types.td)
mlir_tablegen(AscendCTypes.h.inc -gen-typedef-decls -dialect=ascendc -typedefs-dialect=ascendc)
mlir_tablegen(AscendCTypes.cpp.inc -gen-typedef-defs -dialect=ascendc -typedefs-dialect=ascendc)

set(LLVM_TARGET_DEFINITIONS Core/Interfaces.td)
mlir_tablegen(AscendCTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(AscendCTypeInterfaces.cpp.inc -gen-type-interface-defs)

set(LLVM_TARGET_DEFINITIONS Core/Attributes.td)
mlir_tablegen(AscendCAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=ascendc)
mlir_tablegen(AscendCAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ascendc)
mlir_tablegen(AscendCEnums.h.inc -gen-enum-decls)
mlir_tablegen(AscendCEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(MLIRAscIncGen)
add_dependencies(MLIRAscIncGen AscIRAPITypesIncGen)

add_mlir_doc(Doc AscendC Dialects/ -gen-dialect-doc -dialect ascendc)
add_dependencies(AscendCDocGen AscIRAPITypesIncGen)

set(LLVM_TARGET_DEFINITIONS Ops.td)
tablegen(AscIR AscOpBindings.h.inc -gen-pybind-defs)
add_public_tablegen_target(AscPybindGen)

set(LLVM_TARGET_DEFINITIONS Ops.td)
tablegen(AscIR AscendCOpEmit.h.inc -gen-opemit-decls)
tablegen(AscIR AscendCOpEmit.cpp.inc -gen-opemit-defs)
add_public_tablegen_target(AscendCOpEmitsGen)
