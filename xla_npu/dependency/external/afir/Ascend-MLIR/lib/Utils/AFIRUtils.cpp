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

//===- AFIRUtils.cpp - AFIR utility functions -------------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Utils/AFIRUtils.h"
#include "Dialect/AFIR/AFIR.h"

namespace mlir {
namespace afir {
namespace utils {

bool isTensorType(Type type) {
  return mlir::isa<RankedTensorType, UnrankedTensorType>(type);
}

bool hasStaticShape(Type type) {
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type))
    return tensorType.hasStaticShape();
  return false;
}

SmallVector<int64_t> getShape(Type type) {
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type))
    return SmallVector<int64_t>(tensorType.getShape());
  return {};
}

Type getElementType(Type type) {
  if (auto shapedType = mlir::dyn_cast<ShapedType>(type))
    return shapedType.getElementType();
  return type;
}

int64_t getNumElements(Type type) {
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type)) {
    if (!tensorType.hasStaticShape()) return -1;
    int64_t numElements = 1;
    for (int64_t dim : tensorType.getShape()) numElements *= dim;
    return numElements;
  }
  return -1;
}

}  // namespace utils
}  // namespace afir
}  // namespace mlir
