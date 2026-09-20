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

//===- AFIR.h - AFIR dialect declaration -----------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFIR_AFIR_H
#define MLIR_DIALECT_AFIR_AFIR_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

// Include dialect definition
#include "Dialect/AFIR/Dialect.h.inc"

// Include enums (must be before attributes)
#include "Dialect/AFIR/Enums.h.inc"

// Include attribute definitions
#define GET_ATTRDEF_CLASSES
#include "Dialect/AFIR/Attrs.h.inc"

// Include operation definitions
#define GET_OP_CLASSES
#include "Dialect/AFIR/Ops.h.inc"

#endif  // MLIR_DIALECT_AFIR_AFIR_H