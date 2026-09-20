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

#ifndef ASCIR_TARGET_ASC_BASIC_UNARY_H
#define ASCIR_TARGET_ASC_BASIC_UNARY_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Vector unary operations
//===----------------------------------------------------------------------===//

template <typename UnaryOp>
auto printUnaryL0Params(CodeEmitter &emitter, UnaryOp op) {
  auto &os = emitter.ostream();
  os << "(" << emitter.getOrCreateName(op.getDst()) << ", "
     << emitter.getOrCreateName(op.getSrc()) << ", "
     << emitter.getOrCreateName(op.getMask()) << ", "
     << emitter.getOrCreateName(op.getRepeatTimes()) << ", "
     << emitter.getOrCreateName(op.getRepeatParams()) << ")";
}

template <typename UnaryOp>
auto printUnaryL1Params(CodeEmitter &emitter, UnaryOp op, StringRef maskName) {
  auto &os = emitter.ostream();
  os << "(" << emitter.getOrCreateName(op.getDst()) << ", "
     << emitter.getOrCreateName(op.getSrc()) << ", " << maskName << ", "
     << emitter.getOrCreateName(op.getRepeatTimes()) << ", "
     << emitter.getOrCreateName(op.getRepeatParams()) << ")";
}

template <typename UnaryOp>
auto printUnaryL2Params(CodeEmitter &emitter, UnaryOp op) {
  auto &os = emitter.ostream();
  os << "(" << emitter.getOrCreateName(op.getDst()) << ", "
     << emitter.getOrCreateName(op.getSrc()) << ", "
     << emitter.getOrCreateName(op.getCalCount()) << ")";
}

template <typename UnaryL0Op>
auto printOperation(CodeEmitter &emitter, UnaryL0Op op)
    -> LogicalResultForT<UnaryL0Op, ascendc::AbsL0Op, ascendc::ExpL0Op,
                         ascendc::LnL0Op, ascendc::NotL0Op,
                         ascendc::ReciprocalL0Op, ascendc::ReluL0Op,
                         ascendc::RsqrtL0Op, ascendc::SqrtL0Op> {
  auto &os = emitter.ostream();
  FAIL_OR(printIsSetMaskTemplate(emitter, op));
  printUnaryL0Params(emitter, op);
  return success();
}

template <typename UnaryL1Op>
auto printOperation(CodeEmitter &emitter, UnaryL1Op op)
    -> LogicalResultForT<UnaryL1Op, ascendc::AbsL1Op, ascendc::ExpL1Op,
                         ascendc::LnL1Op, ascendc::NotL1Op,
                         ascendc::ReciprocalL1Op, ascendc::ReluL1Op,
                         ascendc::RsqrtL1Op, ascendc::SqrtL1Op> {
  auto &os = emitter.ostream();
  auto maskName = printMask(emitter, op);
  FAIL_OR(printIsSetMaskTemplate(emitter, op));
  printUnaryL1Params(emitter, op, maskName);
  return success();
}

template <typename UnaryL2Op>
auto printOperation(CodeEmitter &emitter, UnaryL2Op op) -> LogicalResultForT<
    UnaryL2Op, ascendc::AbsL2Op, ascendc::ExpL2Op, ascendc::LnL2Op,
    ascendc::NotL2Op, ascendc::ReciprocalL2Op, ascendc::ReluL2Op,
    ascendc::RsqrtL2Op, ascendc::SqrtL2Op, ascendc::NegL2Op> {
  auto &os = emitter.ostream();
  os << ascNamespace << "::" << op.getAPIName();
  printUnaryL2Params(emitter, op);
  return success();
}

}  // namespace ascendc
}  // namespace mlir

#endif  // ASCIR_TARGET_ASC_BASIC_UNARY_H
