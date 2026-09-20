/* Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ASCIR_DIALECT_EMITASC_IR_EMITASC_H
#define ASCIR_DIALECT_EMITASC_IR_EMITASC_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "ascir/Dialect/EmitAsc/IR/EmitAscDialect.h.inc"

#include "ascir/Dialect/EmitAsc/IR/EmitAscEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ascir/Dialect/EmitAsc/IR/EmitAscAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ascir/Dialect/EmitAsc/IR/EmitAscTypes.h.inc"

#define GET_OP_CLASSES
#include "ascir/Dialect/EmitAsc/IR/EmitAscOps.h.inc"

namespace mlir {
namespace emitasc {

void registerExternalModels(DialectRegistry &registry);

}  // namespace emitasc
}  // namespace mlir

#endif  // ASCIR_DIALECT_EMITASC_IR_EMITASC_H
