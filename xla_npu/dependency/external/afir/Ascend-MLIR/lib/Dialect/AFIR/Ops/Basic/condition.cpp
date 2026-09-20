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

//===- condition.cpp - AFIR condition operation implementations -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::afir;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

namespace {

static LogicalResult verifyTernaryOp(Operation *op) {
  if (op->getNumOperands() != 3) {
    return op->emitOpError("expected 3 operands");
  }
  if (op->getNumResults() != 1) {
    return op->emitOpError("expected 1 result");
  }

  auto conditionType = mlir::dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto input1Type = mlir::dyn_cast<ShapedType>(op->getOperand(1).getType());
  auto input2Type = mlir::dyn_cast<ShapedType>(op->getOperand(2).getType());
  auto resultType = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());

  if (!conditionType || !input1Type || !input2Type || !resultType) {
    return op->emitOpError("expected tensor operands and result");
  }

  if (input1Type.hasRank() && input2Type.hasRank() && resultType.hasRank()) {
    if (input1Type.getRank() != input2Type.getRank() ||
        input1Type.getRank() != resultType.getRank()) {
      return op->emitOpError(
          "value operands and result must have the same rank");
    }

    ArrayRef<int64_t> input1Shape = input1Type.getShape();
    ArrayRef<int64_t> input2Shape = input2Type.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();

    for (size_t i = 0; i < input1Shape.size(); ++i) {
      if ((input1Shape[i] != ShapedType::kDynamic &&
           input2Shape[i] != ShapedType::kDynamic &&
           input1Shape[i] != input2Shape[i]) ||
          (input1Shape[i] != ShapedType::kDynamic &&
           resultShape[i] != ShapedType::kDynamic &&
           input1Shape[i] != resultShape[i])) {
        return op->emitOpError(
            "value operands and result must have compatible shapes");
      }
    }
  }

  if (conditionType.hasRank() && resultType.hasRank()) {
    ArrayRef<int64_t> conditionShape = conditionType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();

    size_t conditionRank = conditionShape.size();
    size_t resultRank = resultShape.size();

    if (resultRank < conditionRank) {
      return op->emitOpError("condition rank must be <= result rank");
    }

    size_t offset = resultRank - conditionRank;
    for (size_t i = 0; i < conditionRank; ++i) {
      int64_t conditionDim = conditionShape[i];
      int64_t resultDim = resultShape[i + offset];

      if (conditionDim != ShapedType::kDynamic &&
          resultDim != ShapedType::kDynamic && conditionDim != 1 &&
          conditionDim != resultDim) {
        return op->emitOpError(
            "condition shape must be broadcastable to result shape");
      }
    }
  }

  return success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// Condition Operations
//===----------------------------------------------------------------------===//

LogicalResult SelectOp::verify() { return verifyTernaryOp(getOperation()); }
LogicalResult WhereOp::verify() { return verifyTernaryOp(getOperation()); }
