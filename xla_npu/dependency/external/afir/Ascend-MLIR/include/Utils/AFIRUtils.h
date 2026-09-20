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

//===- AFIRUtils.h - AFIR utility functions ---------------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_UTILS_AFIRUTILS_H
#define MLIR_UTILS_AFIRUTILS_H

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace afir {
namespace utils {

/// Check if a type is a tensor type.
bool isTensorType(Type type);

/// Check if a type has a static shape.
bool hasStaticShape(Type type);

/// Get the shape of a tensor type. Returns empty vector for non-tensor types.
SmallVector<int64_t> getShape(Type type);

/// Get the element type of a shaped type. Returns the type itself for scalars.
Type getElementType(Type type);

/// Get the number of elements in a tensor. Returns -1 for dynamic shapes.
int64_t getNumElements(Type type);

}  // namespace utils
}  // namespace afir
}  // namespace mlir

#endif  // MLIR_UTILS_AFIRUTILS_H
