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

//===- Elementwise.h - AFIR Elementwise ops to ASC-IR conversion -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file declares the conversion patterns for AFIR elementwise operations
// to ASC-IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_AFIRTOASCIR_MATH_ELEMENTWISE_H
#define MLIR_CONVERSION_AFIRTOASCIR_MATH_ELEMENTWISE_H

namespace mlir {

class MLIRContext;
class RewritePatternSet;
class TypeConverter;

namespace afir {

void populateLoweringAFIRElementwiseOpToASCIRPattern(
    RewritePatternSet &patterns, MLIRContext *ctx,
    TypeConverter &typeConverter);

}  // namespace afir
}  // namespace mlir

#endif  // MLIR_CONVERSION_AFIRTOASCIR_MATH_ELEMENTWISE_H
