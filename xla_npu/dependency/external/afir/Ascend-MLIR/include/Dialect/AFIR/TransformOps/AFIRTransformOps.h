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

//===- AFIRTransformOps.h - AFIR transformation ops ----------------------===//
//
// Part of Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFIR_TRANSFORMOPS_AFIRTRANSFORMOPS_H
#define MLIR_DIALECT_AFIR_TRANSFORMOPS_AFIRTRANSFORMOPS_H

#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Dialect/AFIR/TransformOps/AFIRTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace afir {

void registerTransformDialectExtension(DialectRegistry &registry);

}  // namespace afir
}  // namespace mlir

#endif  // MLIR_DIALECT_AFIR_TRANSFORMOPS_AFIRTRANSFORMOPS_H
