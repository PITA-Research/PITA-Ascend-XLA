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

#ifndef ASCIR_TARGET_ASC_UTILS_H
#define ASCIR_TARGET_ASC_UTILS_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

/// Convenience functions to produce interleaved output with functions returning a LogicalResult.
/// This is different than those in STLExtras as functions used on each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline LogicalResult interleaveWithError(ForwardIterator beginIter,
                                         ForwardIterator endIter,
                                         UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  if (beginIter == endIter) return success();
  if (failed(eachFn(*beginIter))) return failure();
  ++beginIter;
  for (; beginIter != endIter; ++beginIter) {
    betweenFn();
    if (failed(eachFn(*beginIter))) return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &container,
                                         UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  return interleaveWithError(std::cbegin(container), std::cend(container),
                             eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c,
                                              raw_ostream &os,
                                              UnaryFunctor eachFn) {
  return interleaveWithError(std::cbegin(c), std::cend(c), eachFn,
                             [&]() { os << ", "; });
}

}  // namespace mlir

#endif  // ASCIR_TARGET_ASC_UTILS_H
