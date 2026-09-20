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

//===- broadcast.cpp - AFIR broadcast operation implementations -*- C++ -*-===//
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

// Verify broadcast compatibility using indexing_maps attribute.
// This supports HLO-style broadcasts with explicit dimension mapping.
static LogicalResult verifyBroadcastViaIndexingMaps(
    Operation *op, ArrayRef<int64_t> inputShape, ArrayRef<int64_t> resultShape,
    size_t inputRank, size_t resultRank) {
  auto broadcastOp = mlir::dyn_cast<mlir::afir::BroadcastOp>(op);
  if (!broadcastOp) return failure();

  auto indexingMapsOpt = broadcastOp.getIndexingMaps();
  if (!indexingMapsOpt.has_value()) return failure();

  auto indexingMaps = *indexingMapsOpt;
  if (indexingMaps.size() < 2) return failure();

  // The first map is for the operand, the second is for the result
  auto operandMapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(indexingMaps[0]);
  auto resultMapAttr = mlir::dyn_cast<mlir::AffineMapAttr>(indexingMaps[1]);
  if (!operandMapAttr || !resultMapAttr) return failure();

  auto operandMap = operandMapAttr.getValue();
  auto resultMap = resultMapAttr.getValue();

  // Validate map structure
  if (operandMap.getNumInputs() != static_cast<unsigned>(resultRank) ||
      operandMap.getNumResults() != static_cast<unsigned>(inputRank) ||
      resultMap.getNumInputs() != static_cast<unsigned>(resultRank) ||
      resultMap.getNumResults() != static_cast<unsigned>(resultRank)) {
    return failure();
  }

  // Check that the operand dimensions are compatible with result dimensions
  // according to the operand_map
  for (size_t i = 0; i < inputRank; ++i) {
    int64_t inputDim = inputShape[i];
    if (inputDim == ShapedType::kDynamic) continue;

    auto expr = operandMap.getResult(i);
    if (auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
      // This input dimension maps to a result dimension
      size_t resultDimIdx = dimExpr.getPosition();
      if (resultDimIdx < resultShape.size()) {
        int64_t resultDim = resultShape[resultDimIdx];
        if (resultDim != ShapedType::kDynamic && inputDim != 1 &&
            inputDim != resultDim) {
          return op->emitOpError("input dimension ")
                 << i << " (size " << inputDim << ") is not broadcastable to "
                 << "result dimension " << resultDimIdx << " (size "
                 << resultDim << ")";
        }
      }
    } else if (mlir::dyn_cast<mlir::AffineConstantExpr>(expr)) {
      // This is a constant expression, typically 0 for broadcast dimensions
      // No need to check
    } else {
      // More complex expression, skip detailed check
    }
  }

  return success();
}

static LogicalResult verifyUnaryOp(Operation *op) {
  if (op->getNumOperands() != 1) {
    return op->emitOpError("expected 1 operand");
  }
  if (op->getNumResults() != 1) {
    return op->emitOpError("expected 1 result");
  }

  auto inputType = mlir::dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto resultType = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());

  if (!inputType || !resultType) {
    return op->emitOpError("expected tensor operand and result");
  }

  if (inputType.hasRank() && resultType.hasRank()) {
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();

    size_t inputRank = inputShape.size();
    size_t resultRank = resultShape.size();

    if (resultRank < inputRank) {
      return op->emitOpError("result rank must be >= input rank for broadcast");
    }

    // Try to verify using indexing_maps attribute (HLO-style broadcast)
    if (succeeded(verifyBroadcastViaIndexingMaps(op, inputShape, resultShape,
                                                 inputRank, resultRank))) {
      return success();
    }

    // Fall back to numpy-style right-aligned broadcast validation
    size_t offset = resultRank - inputRank;
    for (size_t i = 0; i < inputRank; ++i) {
      int64_t inputDim = inputShape[i];
      int64_t resultDim = resultShape[i + offset];

      if (inputDim != ShapedType::kDynamic &&
          resultDim != ShapedType::kDynamic && inputDim != 1 &&
          inputDim != resultDim) {
        return op->emitOpError(
            "input shape must be broadcastable to result shape");
      }
    }
  }

  return success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// Broadcast Operations
//===----------------------------------------------------------------------===//

LogicalResult BroadcastOp::verify() { return verifyUnaryOp(getOperation()); }
