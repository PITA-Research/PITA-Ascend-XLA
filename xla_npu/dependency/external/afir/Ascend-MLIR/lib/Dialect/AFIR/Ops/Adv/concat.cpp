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

//===- concat.cpp - AFIR concat operation implementations -*- C++ -*-===//
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
// Concat Operations
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::verify() {
  if (getInputs().empty()) {
    return emitOpError("requires at least one input tensor");
  }

  int64_t axis = getConcatAxisAttr().getInt();
  auto firstTensorType =
      llvm::dyn_cast<RankedTensorType>(getInputs()[0].getType());
  if (!firstTensorType) {
    return emitOpError("requires ranked tensor inputs");
  }

  int64_t rank = firstTensorType.getRank();

  if (axis < 0 || axis >= rank) {
    return emitOpError("axis ")
           << axis << " is out of bounds [0, " << rank << ")";
  }

  ArrayRef<int64_t> referenceShape = firstTensorType.getShape();
  Type referenceElementType = firstTensorType.getElementType();

  int64_t concatDimSize = referenceShape[axis];

  for (auto input : llvm::drop_begin(getInputs(), 1)) {
    auto tensorType = llvm::dyn_cast<RankedTensorType>(input.getType());
    if (!tensorType) {
      return emitOpError("all inputs must be ranked tensors");
    }

    if (tensorType.getRank() != rank) {
      return emitOpError("all inputs must have the same rank");
    }

    if (tensorType.getElementType() != referenceElementType) {
      return emitOpError("all inputs must have the same element type");
    }

    ArrayRef<int64_t> shape = tensorType.getShape();
    for (int64_t i = 0; i < rank; ++i) {
      if (i == axis) {
        if (shape[i] != ShapedType::kDynamic &&
            concatDimSize != ShapedType::kDynamic) {
          concatDimSize += shape[i];
        } else {
          concatDimSize = ShapedType::kDynamic;
        }
      } else {
        if (shape[i] != referenceShape[i] && shape[i] != ShapedType::kDynamic &&
            referenceShape[i] != ShapedType::kDynamic) {
          return emitOpError(
                     "all inputs must have the same shape except on axis ")
                 << axis;
        }
      }
    }
  }

  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  SmallVector<int64_t> expectedShape(referenceShape.begin(),
                                     referenceShape.end());
  expectedShape[axis] = concatDimSize;

  if (verifyCompatibleShape(expectedShape, resultType.getShape()).failed()) {
    return emitOpError("result shape mismatch");
  }
  // if (resultType.getShape() != ArrayRef<int64_t>(expectedShape)) {
  //   return emitOpError("result shape mismatch");
  // }

  return success();
}

LogicalResult ConcatOp::inferReturnTypeComponents(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::afir::ConcatOpAdaptor adaptor,
    llvm::SmallVectorImpl<mlir::ShapedTypeComponents> &inferredReturnType) {
  int64_t rank = -1;
  for (auto oper : adaptor.getInputs()) {
    if (auto opType = llvm::dyn_cast<RankedTensorType>(oper.getType())) {
      if (rank == -1) {
        rank = opType.getRank();
      } else if (rank != opType.getRank()) {
        return failure();
      }
    } else {
      auto eleType =
          llvm::cast<UnrankedTensorType>(oper.getType()).getElementType();
      inferredReturnType.push_back(ShapedTypeComponents(eleType));
      return success();
    }
  }
  auto axis = adaptor.getConcatAxis();
  if (axis < 0 || axis >= rank) {
    return failure();
  }
  SmallVector<int64_t> newShape(rank, 0);
  for (size_t i = 0; i < adaptor.getInputs().size(); i++) {
    auto shape =
        llvm::dyn_cast<RankedTensorType>(adaptor.getInputs()[i].getType())
            .getShape();
    for (int j = 0; j < rank; j++) {
      if (shape[j] == ShapedType::kDynamic) {
        newShape[j] = ShapedType::kDynamic;
      } else {
        if (j != axis) {
          if (newShape[j] == 0 || shape[j] == ShapedType::kDynamic) {
            newShape[j] = shape[j];
          } else if (newShape[j] != shape[j] &&
                     newShape[j] != ShapedType::kDynamic) {
            return failure();
          }
        } else {
          if (newShape[j] != ShapedType::kDynamic) {
            newShape[j] += shape[j];
          }
        }
      }
    }
  }
  auto eleType = llvm::cast<RankedTensorType>(adaptor.getInputs()[0].getType())
                     .getElementType();
  inferredReturnType.push_back(ShapedTypeComponents(newShape, eleType));
  return success();
}