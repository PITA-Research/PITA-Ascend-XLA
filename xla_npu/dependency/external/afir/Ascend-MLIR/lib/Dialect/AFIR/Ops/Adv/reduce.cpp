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

//===- reduce.cpp - AFIR reduce operation implementations -*- C++ -*-===//
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
    if (resultType.getRank() > inputType.getRank()) {
      return op->emitOpError("result rank must be <= input rank for reduce");
    }
  }

  return success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// Reduce Operations
//===----------------------------------------------------------------------===//

LogicalResult MaxOp::verify() { return verifyUnaryOp(getOperation()); }
LogicalResult MinOp::verify() { return verifyUnaryOp(getOperation()); }
LogicalResult SumOp::verify() { return verifyUnaryOp(getOperation()); }
LogicalResult MeanOp::verify() { return verifyUnaryOp(getOperation()); }
LogicalResult ProdOp::verify() { return verifyUnaryOp(getOperation()); }
LogicalResult AnyOp::verify() { return verifyUnaryOp(getOperation()); }
LogicalResult AllOp::verify() { return verifyUnaryOp(getOperation()); }

template <typename OpAdaptor>
LogicalResult inferReduceReturnTypeComponents(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    OpAdaptor adaptor,
    llvm::SmallVectorImpl<mlir::ShapedTypeComponents> &inferredReturnType) {
  auto opType = llvm::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (!opType) {
    return failure();
  }
  auto rank = opType.getRank();
  auto axis = adaptor.getAxis();
  if (axis < 0 || axis >= rank) {
    return failure();
  }
  SmallVector<int64_t> newShape(opType.getShape().begin(),
                                opType.getShape().end());
  newShape.erase(newShape.begin() + axis);
  auto eleType = opType.getElementType();
  inferredReturnType.push_back(ShapedTypeComponents(newShape, eleType));
  return success();
}

#define REGISTERREDUCEINFER(OP)                                      \
  LogicalResult OP::inferReturnTypeComponents(                       \
      MLIRContext *context, ::std::optional<Location> location,      \
      OP##Adaptor adaptor,                                           \
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) { \
    return inferReduceReturnTypeComponents<OP##Adaptor>(             \
        context, location, adaptor, inferredReturnShapes);           \
  }

REGISTERREDUCEINFER(afir::MaxOp)
REGISTERREDUCEINFER(afir::MinOp)
REGISTERREDUCEINFER(afir::SumOp)
REGISTERREDUCEINFER(afir::MeanOp)
REGISTERREDUCEINFER(afir::ProdOp)
REGISTERREDUCEINFER(afir::AnyOp)
REGISTERREDUCEINFER(afir::AllOp)