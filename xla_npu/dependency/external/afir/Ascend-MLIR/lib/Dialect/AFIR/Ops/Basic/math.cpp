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

//===- math.cpp - AFIR math operation implementations -------*- C++ -*-===//
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

static LogicalResult verifyUnaryElementwiseOp(Operation *op) {
  auto inputType = mlir::dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto resultType = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());

  if (!inputType || !resultType)
    return op->emitOpError("expected tensor operand and result");

  if (inputType.hasRank() && resultType.hasRank()) {
    if (inputType.getRank() != resultType.getRank())
      return op->emitOpError("operand and result must have the same rank");

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();

    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (inputShape[i] != ShapedType::kDynamic &&
          resultShape[i] != ShapedType::kDynamic &&
          inputShape[i] != resultShape[i]) {
        return op->emitOpError(
            "operand and result must have compatible shapes");
      }
    }
  }

  return success();
}

static LogicalResult verifyBinaryElementwiseOp(Operation *op) {
  auto lhsType = mlir::dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto rhsType = mlir::dyn_cast<ShapedType>(op->getOperand(1).getType());
  auto resultType = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());

  if (!lhsType || !rhsType || !resultType)
    return op->emitOpError("expected tensor operands and results");

  if (lhsType.hasRank() && rhsType.hasRank()) {
    /*
  if (lhsType.hasRank() && rhsType.hasRank()) {
    if (lhsType.getRank() != rhsType.getRank()) return op->emitOpError("operands must have the same rank");

    ArrayRef<int64_t> lhsShape = lhsType.getShape();
    ArrayRef<int64_t> rhsShape = rhsType.getShape();

    for (size_t i = 0; i < lhsShape.size(); ++i) {
      if (lhsShape[i] != ShapedType::kDynamic && rhsShape[i] != ShapedType::kDynamic && lhsShape[i] != 1 &&
          rhsShape[i] != 1 && lhsShape[i] != rhsShape[i]) {
        return op->emitOpError("operands must have compatible shapes");
      }
    }
  }
*/
    SmallVector<int64_t> largeShape;
    SmallVector<int64_t> smallShape;
    if (lhsType.getRank() >= rhsType.getRank()) {
      largeShape.assign(lhsType.getShape().begin(), lhsType.getShape().end());
      smallShape.assign(rhsType.getShape().begin(), rhsType.getShape().end());
    } else {
      smallShape.assign(lhsType.getShape().begin(), lhsType.getShape().end());
      largeShape.assign(rhsType.getShape().begin(), rhsType.getShape().end());
    }
    int rankOffset = largeShape.size() - smallShape.size();
    for (int i = 0; i < rankOffset; i++) {
      if (resultType.getShape()[i] != largeShape[i]) {
        return op->emitOpError("expected tensor operands and results");
      }
    }
    for (int i = rankOffset; i < largeShape.size(); i++) {
      if (largeShape[i] != ShapedType::kDynamic &&
          smallShape[i - rankOffset] != ShapedType::kDynamic &&
          largeShape[i] != 1 && smallShape[i - rankOffset] != 1 &&
          largeShape[i] != smallShape[i - rankOffset]) {
        return op->emitOpError("operands must have compatible shapes");
      }
    }
    return success();
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

LogicalResult AbsOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult ExpOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult LnOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult SqrtOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult RsqrtOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult ReciprocalOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult ErfOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult TanhOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult ReluOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult NegOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult LogicalNotOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult SigmoidOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult IsnanOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult IsFiniteOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

LogicalResult AddOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult SubOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult MulOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult DivOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult MinimumOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult MaximumOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult TrueDivOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult PowOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult LeakyReluOp::verify() {
  return verifyUnaryElementwiseOp(getOperation());
}
LogicalResult BitwiseAndOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult FloorDivOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult GeluOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult SignOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult LogicalOrOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult LogicalAndOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}

//===----------------------------------------------------------------------===//
// Compare Operations
//===----------------------------------------------------------------------===//

LogicalResult GeOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult EqOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult NeOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult GtOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult LeOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}
LogicalResult LtOp::verify() {
  return verifyBinaryElementwiseOp(getOperation());
}

//===----------------------------------------------------------------------===//
// Ternary Operations
//===----------------------------------------------------------------------===//

LogicalResult ClipByValue::verify() {
  auto inputType = mlir::dyn_cast<ShapedType>(getOperand(0).getType());
  auto minType = mlir::dyn_cast<ShapedType>(getOperand(1).getType());
  auto maxType = mlir::dyn_cast<ShapedType>(getOperand(2).getType());
  auto resultType = mlir::dyn_cast<ShapedType>(getResult().getType());

  if (!inputType || !minType || !maxType || !resultType) {
    return emitOpError("expected tensor operands and result");
  }

  if (inputType.hasRank() && minType.hasRank()) {
    if (inputType.getRank() != minType.getRank()) {
      return emitOpError("input and min must have the same rank");
    }
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> minShape = minType.getShape();
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (inputShape[i] != ShapedType::kDynamic &&
          minShape[i] != ShapedType::kDynamic && inputShape[i] != minShape[i] &&
          inputShape[i] != 1 && minShape[i] != 1) {
        return emitOpError("input and min must have compatible shapes");
      }
    }
  }

  if (inputType.hasRank() && maxType.hasRank()) {
    if (inputType.getRank() != maxType.getRank()) {
      return emitOpError("input and max must have the same rank");
    }
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> maxShape = maxType.getShape();
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (inputShape[i] != ShapedType::kDynamic &&
          maxShape[i] != ShapedType::kDynamic && inputShape[i] != maxShape[i] &&
          inputShape[i] != 1 && maxShape[i] != 1) {
        return emitOpError("input and max must have compatible shapes");
      }
    }
  }

  if (inputType.hasRank() && resultType.hasRank()) {
    if (inputType.getRank() != resultType.getRank()) {
      return emitOpError("input and result must have the same rank");
    }
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (inputShape[i] != ShapedType::kDynamic &&
          resultShape[i] != ShapedType::kDynamic &&
          inputShape[i] != resultShape[i] && inputShape[i] != 1 &&
          resultShape[i] != 1) {
        return emitOpError("input and result must have compatible shapes");
      }
    }
  }

  return success();
}

template <typename OPAdaptor>

LogicalResult inferBroadCastReturnTypes(
    MLIRContext *context, std::optional<mlir::Location> location,
    OPAdaptor adaptor, Type elementType,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
  int64_t newShapeRank = 0;
  for (auto oper : adaptor.getOperands()) {
    if (auto opType = llvm::dyn_cast<RankedTensorType>(oper.getType())) {
      newShapeRank = std::max(newShapeRank, opType.getRank());
    } else {
      inferredReturnTypes.push_back(ShapedTypeComponents(elementType));
      return success();
    }
  }
  SmallVector<int64_t> newShape(newShapeRank, 1);
  SmallVector<int64_t> operOffset(adaptor.getOperands().size());
  for (size_t i = 0; i < adaptor.getOperands().size(); i++) {
    operOffset[i] = newShapeRank - llvm::dyn_cast<RankedTensorType>(
                                       adaptor.getOperands()[i].getType())
                                       .getRank();
  }
  for (size_t i = newShapeRank; i != 0; i--) {
    int index = i - 1;
    for (int j = 0; j < adaptor.getOperands().size(); j++) {
      if (index < operOffset[j]) {
        continue;
      }
      auto shape =
          llvm::dyn_cast<RankedTensorType>(adaptor.getOperands()[j].getType())
              .getShape();
      if (shape[index - operOffset[j]] == 1) {
        continue;
      }
      if (newShape[index] == ShapedType::kDynamic || newShape[index] == 1) {
        newShape[index] = shape[index - operOffset[j]];
      }
      if (shape[index - operOffset[j]] == ShapedType::kDynamic) {
        continue;
      }
      if (newShape[index] != shape[index - operOffset[j]]) {
        return failure();
      }
    }
  }
  inferredReturnTypes.push_back(ShapedTypeComponents(newShape, elementType));
  return success();
}

// LogicalResult inferBroadCastReturnTypes(MLIRContext *context, std::optional<mlir::Location> location, OPAdaptor
// adaptor,
//                                         Type elementType, SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes)
//                                         {
//   int64_t newShapeRank = -1;
//   for (auto oper : adaptor.getOperands()) {
//     if (auto opType = llvm::dyn_cast<RankedTensorType>(oper.getType())) {
//       if (newShapeRank == -1) {
//         newShapeRank = opType.getRank();
//       } else if (newShapeRank != opType.getRank()) {
//         return failure();
//       }
//     } else {
//       inferredReturnTypes.push_back(ShapedTypeComponents(elementType));
//       return success();
//     }
//   }
//   SmallVector<int64_t> newShape(newShapeRank, 1);
//
//   for (size_t j = 0; j < adaptor.getOperands().size(); j++) {
//     auto shape = llvm::dyn_cast<RankedTensorType>(adaptor.getOperands()[j].getType()).getShape();
//     for (int64_t i = 0; i < newShapeRank; i++) {
//       if (shape[i] == 1) {
//         continue;
//       }
//       if (newShape[i] == ShapedType::kDynamic || newShape[i] == 1) {
//         newShape[i] = shape[i];
//       }
//       if (shape[i] == ShapedType::kDynamic) {
//         continue;
//       }
//       if (newShape[i] != shape[i]) {
//         return failure();
//       }
//     }
//   }
//   inferredReturnTypes.push_back(ShapedTypeComponents(newShape, elementType));
//   return success();
// }

#define REGISTERBROADCASTINFER(OP)                                             \
  LogicalResult OP::inferReturnTypeComponents(                                 \
      MLIRContext *context, ::std::optional<Location> location,                \
      OP##Adaptor adaptor,                                                     \
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {           \
    TensorType tensorType =                                                    \
        llvm::dyn_cast<TensorType>(adaptor.getOperands()[0].getType());        \
    if (!tensorType) {                                                         \
      return failure();                                                        \
    }                                                                          \
    return inferBroadCastReturnTypes<OP##Adaptor>(context, location, adaptor,  \
                                                  tensorType.getElementType(), \
                                                  inferredReturnShapes);       \
  }

REGISTERBROADCASTINFER(afir::AddOp)
REGISTERBROADCASTINFER(afir::SubOp)
REGISTERBROADCASTINFER(afir::MulOp)
REGISTERBROADCASTINFER(afir::DivOp)
REGISTERBROADCASTINFER(afir::MinimumOp)
REGISTERBROADCASTINFER(afir::MaximumOp)
REGISTERBROADCASTINFER(afir::TrueDivOp)
REGISTERBROADCASTINFER(afir::PowOp)
REGISTERBROADCASTINFER(afir::BitwiseAndOp)
REGISTERBROADCASTINFER(afir::FloorDivOp)
REGISTERBROADCASTINFER(afir::GeluOp)
REGISTERBROADCASTINFER(afir::SignOp)
REGISTERBROADCASTINFER(afir::ClipByValue)

// REGISTERBROADCASTINFER(afir::StoreOp)

#define REGISTERBROADCASTLOGICALINFER(OP)                            \
  LogicalResult OP::inferReturnTypeComponents(                       \
      MLIRContext *context, ::std::optional<Location> location,      \
      OP##Adaptor adaptor,                                           \
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) { \
    return inferBroadCastReturnTypes<OP##Adaptor>(                   \
        context, location, adaptor,                                  \
        IntegerType::get(context, 8, IntegerType::Unsigned),         \
        inferredReturnShapes);                                       \
  }

REGISTERBROADCASTLOGICALINFER(afir::LogicalAndOp)
REGISTERBROADCASTLOGICALINFER(afir::LogicalOrOp)
REGISTERBROADCASTLOGICALINFER(afir::GeOp)
REGISTERBROADCASTLOGICALINFER(afir::EqOp)
REGISTERBROADCASTLOGICALINFER(afir::NeOp)
REGISTERBROADCASTLOGICALINFER(afir::GtOp)
REGISTERBROADCASTLOGICALINFER(afir::LeOp)
REGISTERBROADCASTLOGICALINFER(afir::LtOp)
