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

#ifndef ASCIR_TARGET_ASC_ADV_MATH_LIBS_H
#define ASCIR_TARGET_ASC_ADV_MATH_LIBS_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Unary math library operations
//===----------------------------------------------------------------------===//

template <typename UnaryMathOp>
auto printOperation(CodeEmitter &emitter, UnaryMathOp op) -> LogicalResultForT<
    UnaryMathOp, ascendc::AcoshOp, ascendc::AcosOp, ascendc::AsinhOp,
    ascendc::AsinOp, ascendc::AtanhOp, ascendc::AtanOp, ascendc::CeilOp,
    ascendc::CoshOp, ascendc::CosOp, ascendc::DigammaOp, ascendc::ErfcOp,
    ascendc::ErfOp, ascendc::FloorOp, ascendc::FracOp, ascendc::LgammaOp,
    ascendc::LogOp, ascendc::RoundOp, ascendc::SignOp, ascendc::SinhOp,
    ascendc::SinOp, ascendc::TanhOp, ascendc::TanOp, ascendc::TruncOp> {
  auto &os = emitter.ostream();
  os << ascNamespace << "::" << op.getAPIName();
  os << "<";
  auto dstType = op.getDst().getType().getElementType();
  FAIL_OR(emitter.emitType(op.getLoc(), dstType));
  os << ", " << emitter.getOrCreateName(op.getIsReuseSource()) << ">";
  os << "(" << emitter.getOrCreateName(op.getDst()) << ", "
     << emitter.getOrCreateName(op.getSrc());
  if (auto sharedTmpBuffer = op.getSharedTmpBuffer()) {
    os << ", " << emitter.getOrCreateName(sharedTmpBuffer);
  }
  if (auto calCount = op.getCalCount()) {
    os << ", " << emitter.getOrCreateName(calCount);
  }
  os << ")";
  return success();
}

//===----------------------------------------------------------------------===//
// Binary math library operations
//===----------------------------------------------------------------------===//

template <typename BinaryMathOp>
LogicalResultForT<BinaryMathOp, ascendc::PowerOp, ascendc::XorOp>
printOperation(CodeEmitter &emitter, BinaryMathOp op) {
  auto &os = emitter.ostream();
  os << ascNamespace << "::" << op.getAPIName();
  os << "<";
  auto dstType = op.getDst().getType().getElementType();
  FAIL_OR(emitter.emitType(op.getLoc(), dstType));
  os << ", " << emitter.getOrCreateName(op.getIsReuseSource()) << ">";
  os << "(" << emitter.getOrCreateName(op.getDst()) << ", "
     << emitter.getOrCreateName(op.getSrc0()) << ", "
     << emitter.getOrCreateName(op.getSrc1());
  if (auto sharedTmpBuffer = op.getSharedTmpBuffer()) {
    os << ", " << emitter.getOrCreateName(sharedTmpBuffer);
  }
  if (auto calCount = op.getCalCount()) {
    os << ", " << emitter.getOrCreateName(calCount);
  }
  os << ")";
  return success();
}

//===----------------------------------------------------------------------===//
// Other math library operations
//===----------------------------------------------------------------------===//

template <typename Clamp>
LogicalResultForT<Clamp, ascendc::ClampMaxOp, ascendc::ClampMinOp>
printOperation(CodeEmitter &emitter, Clamp op) {
  auto &os = emitter.ostream();
  os << ascNamespace << "::" << op.getAPIName();
  os << "<";
  auto dstType = op.getDst().getType().getElementType();
  FAIL_OR(emitter.emitType(op.getLoc(), dstType));
  os << ", " << emitter.getOrCreateName(op.getIsReuseSource()) << ">";
  os << "(" << emitter.getOrCreateName(op.getDst()) << ", "
     << emitter.getOrCreateName(op.getSrc());
  if (auto sharedTmpBuffer = op.getSharedTmpBuffer()) {
    os << ", " << emitter.getOrCreateName(sharedTmpBuffer);
  }
  os << ", " << emitter.getOrCreateName(op.getScalar()) << ", "
     << emitter.getOrCreateName(op.getCalCount()) << ")";

  return success();
}

LogicalResult printOperation(CodeEmitter &emitter, ascendc::ExpOp op);

LogicalResult printOperation(CodeEmitter &emitter, ascendc::AxpyOp op);

LogicalResult printOperation(CodeEmitter &emitter, ascendc::CumSumOp op);

}  // namespace ascendc
}  // namespace mlir

#endif  // ASCIR_TARGET_ASC_ADV_MATH_LIBS_H
