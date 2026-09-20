# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(EmitAscDialect.h.inc -gen-dialect-decls -dialect=emitasc)
mlir_tablegen(EmitAscDialect.cpp.inc -gen-dialect-defs -dialect=emitasc)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(EmitAscOps.h.inc -gen-op-decls)
mlir_tablegen(EmitAscOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS Types.td)
mlir_tablegen(EmitAscTypes.h.inc -gen-typedef-decls -dialect=emitasc -typedefs-dialect=emitasc)
mlir_tablegen(EmitAscTypes.cpp.inc -gen-typedef-defs -dialect=emitasc -typedefs-dialect=emitasc)

set(LLVM_TARGET_DEFINITIONS Attributes.td)
mlir_tablegen(EmitAscAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=emitasc)
mlir_tablegen(EmitAscAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=emitasc)
mlir_tablegen(EmitAscEnums.h.inc -gen-enum-decls)
mlir_tablegen(EmitAscEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(MLIREmitAscIncGen)

add_mlir_doc(Doc EmitAsc Dialects/ -gen-dialect-doc -dialect emitasc)
