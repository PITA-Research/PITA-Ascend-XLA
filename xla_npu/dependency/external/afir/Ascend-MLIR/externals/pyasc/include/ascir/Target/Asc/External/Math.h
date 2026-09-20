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

#ifndef ASCIR_TARGET_ASC_MLIR_MATH_H
#define ASCIR_TARGET_ASC_MLIR_MATH_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {

template <typename UnaryOpType>
auto printOperation(CodeEmitter &emitter, UnaryOpType op)
    -> LogicalResultForT<UnaryOpType, math::AbsFOp, math::SqrtOp, math::ExpOp,
                         math::LogOp, math::CosOp, math::SinOp, math::Log2Op,
                         math::ErfOp, math::CeilOp, math::FloorOp,
                         math::RsqrtOp, math::Exp2Op, math::RoundOp> {
  if (failed(isScalarOperation(op)) ||
      failed(emitter.emitAssignPrefix(*op.getOperation()))) {
    return failure();
  }
  auto &os = emitter.ostream();
  if constexpr (std::is_same_v<UnaryOpType, math::AbsFOp>) {
    auto lhs = emitter.getOrCreateName(op.getOperand());
    os << "(" << lhs << " > static_cast<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getType()));
    os << ">(0)) ? " << lhs << " : -" << lhs;
  } else if constexpr (std::is_same_v<UnaryOpType, math::SqrtOp>) {
    os << "sqrt(" << emitter.getOrCreateName(op.getOperand()) << ")";
  } else if constexpr (std::is_same_v<UnaryOpType, math::ExpOp>) {
    os << ascNamespace << "::Exp(" << emitter.getOrCreateName(op.getOperand())
       << ")";
  } else if constexpr (std::is_same_v<UnaryOpType, math::CosOp>) {
    os << ascNamespace << "::Cos(" << emitter.getOrCreateName(op.getOperand())
       << ")";
  } else if constexpr (std::is_same_v<UnaryOpType, math::SinOp>) {
    os << ascNamespace << "::Sin(" << emitter.getOrCreateName(op.getOperand())
       << ")";
  } else if constexpr (std::is_same_v<UnaryOpType, math::LogOp>) {
    os << ascNamespace << "::Log(" << emitter.getOrCreateName(op.getOperand())
       << ")";
  } else if constexpr (std::is_same_v<UnaryOpType, math::Log2Op>) {
    os << ascNamespace << "::Log2(" << emitter.getOrCreateName(op.getOperand())
       << ")";
  } else if constexpr (std::is_same_v<UnaryOpType, math::ErfOp>) {
    os << ascNamespace << "::Erf(" << emitter.getOrCreateName(op.getOperand())
       << ")";
  } else if constexpr (std::is_same_v<UnaryOpType, math::CeilOp>) {
    os << ascNamespace << "::Ceil(" << emitter.getOrCreateName(op.getOperand())
       << ")";
  } else if constexpr (std::is_same_v<UnaryOpType, math::FloorOp>) {
    os << ascNamespace << "::Floor(" << emitter.getOrCreateName(op.getOperand())
       << ")";
  } else if constexpr (std::is_same_v<UnaryOpType, math::RsqrtOp>) {
    os << ascNamespace << "::Rsqrt(" << emitter.getOrCreateName(op.getOperand())
       << ")";
  } else if constexpr (std::is_same_v<UnaryOpType, math::Exp2Op>) {
    os << ascNamespace << "::Exp(" << emitter.getOrCreateName(op.getOperand())
       << " * Log(2))";
  } else if constexpr (std::is_same_v<UnaryOpType, math::RoundOp>) {
    os << ascNamespace << "::Round(" << emitter.getOrCreateName(op.getOperand())
       << ")";
  } else {
    llvm_unreachable("not implemented");
  }
  return success();
}

template <typename BinaryOpType>
LogicalResultForT<BinaryOpType, math::Atan2Op> printOperation(
    CodeEmitter &emitter, BinaryOpType op) {
  FAIL_OR(isScalarOperation(op));
  FAIL_OR(emitter.emitAssignPrefix(*op.getOperation()));
  auto &os = emitter.ostream();
  if constexpr (std::is_same_v<BinaryOpType, math::Atan2Op>) {
    auto lhs = emitter.getOrCreateName(op.getLhs());
    auto rhs = emitter.getOrCreateName(op.getRhs());
    os << ascNamespace << "::Atan2(" << lhs << ", " << rhs << ")";
    return success();
  } else {
    llvm_unreachable("not implemented");
  }
  return failure();
}

LogicalResult printOperation(CodeEmitter &emitter, math::FmaOp op);

LogicalResult printOperation(CodeEmitter &emitter, math::CopySignOp op);

}  // namespace mlir

#endif  // ASCIR_TARGET_ASC_MLIR_MATH_H
