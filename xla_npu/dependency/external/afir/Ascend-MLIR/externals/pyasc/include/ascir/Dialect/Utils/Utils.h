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

#ifndef ASCIR_DIALECT_UTILS_UTILS_H
#define ASCIR_DIALECT_UTILS_UTILS_H

#include "mlir/IR/Value.h"

#include <cstddef>
#include <unordered_map>
#include <unordered_set>

namespace mlir {

template <typename T>
struct PointerLikeTypeHash {
  std::hash<const void *> h;
  size_t operator()(const T &op) const { return h(op.getAsOpaquePointer()); }
};

template <typename T>
using ValueMap = std::unordered_map<Value, T, PointerLikeTypeHash<Value>>;

using ValueSet = std::unordered_set<Value, PointerLikeTypeHash<Value>>;

using ValueVector = SmallVector<Value>;

}  // namespace mlir

#endif  // ASCIR_DIALECT_UTILS_UTILS_H
