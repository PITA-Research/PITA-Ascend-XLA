/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at
*    http://www.apache.org/licenses/LICENSE-2.0
*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
 */

//===- transpose.cpp - AFIR transpose operation implementations -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::afir;

//===----------------------------------------------------------------------===//
// Transpose Operations
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());

  if (!inputType || !resultType) {
    return emitOpError("requires ranked tensor input and result");
  }

  int64_t rank = inputType.getRank();
  if (resultType.getRank() != rank) {
    return emitOpError("result rank must equal input rank");
  }

  auto permutation = getPermutation();
  if (static_cast<int64_t>(permutation.size()) != rank) {
    return emitOpError("permutation length ")
           << permutation.size() << " must equal input rank " << rank;
  }

  // Check permutation is a valid permutation of [0, rank)
  llvm::SmallVector<bool> seen(rank, false);
  for (auto attr : permutation) {
    int64_t dim = llvm::cast<IntegerAttr>(attr).getInt();
    if (dim < 0 || dim >= rank) {
      return emitOpError("permutation dimension ")
             << dim << " out of range [0, " << rank << ")";
    }
    if (seen[dim]) {
      return emitOpError("duplicate dimension ") << dim << " in permutation";
    }
    seen[dim] = true;
  }

  // Check result shape matches permuted input shape
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  for (int64_t i = 0; i < rank; ++i) {
    int64_t srcDim = llvm::cast<IntegerAttr>(permutation[i]).getInt();
    if (inputShape[srcDim] != ShapedType::kDynamic &&
        resultShape[i] != ShapedType::kDynamic &&
        inputShape[srcDim] != resultShape[i]) {
      return emitOpError("result shape mismatch at dimension ") << i;
    }
  }

  return success();
}

LogicalResult TransposeOp::inferReturnTypeComponents(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::afir::TransposeOpAdaptor adaptor,
    llvm::SmallVectorImpl<mlir::ShapedTypeComponents> &inferredReturnType) {
  auto inputType =
      llvm::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (!inputType) {
    return failure();
  }

  int64_t rank = inputType.getRank();
  auto permutation = adaptor.getPermutation();
  if (static_cast<int64_t>(permutation.size()) != rank) {
    return failure();
  }

  ArrayRef<int64_t> inputShape = inputType.getShape();
  SmallVector<int64_t> resultShape(rank);
  for (int64_t i = 0; i < rank; ++i) {
    int64_t srcDim = llvm::cast<IntegerAttr>(permutation[i]).getInt();
    resultShape[i] = inputShape[srcDim];
  }

  inferredReturnType.push_back(
      ShapedTypeComponents(resultShape, inputType.getElementType()));
  return success();
}
