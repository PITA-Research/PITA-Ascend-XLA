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

#ifndef ASCIR_TARGET_ASC_BASIC_VECTOR_BINARY_H
#define ASCIR_TARGET_ASC_BASIC_VECTOR_BINARY_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Vector binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp>
auto printBinaryL0Params(CodeEmitter &emitter, BinaryOp op) {
  auto &os = emitter.ostream();
  os << "(" << emitter.getOrCreateName(op.getDst()) << ", "
     << emitter.getOrCreateName(op.getSrc0()) << ", "
     << emitter.getOrCreateName(op.getSrc1()) << ", "
     << emitter.getOrCreateName(op.getMask()) << ", "
     << emitter.getOrCreateName(op.getRepeatTimes()) << ", "
     << emitter.getOrCreateName(op.getRepeatParams()) << ")";
}

template <typename BinaryOp>
auto printBinaryL1Params(CodeEmitter &emitter, BinaryOp op,
                         StringRef maskName) {
  auto &os = emitter.ostream();
  os << "(" << emitter.getOrCreateName(op.getDst()) << ", "
     << emitter.getOrCreateName(op.getSrc0()) << ", "
     << emitter.getOrCreateName(op.getSrc1()) << ", " << maskName << ", "
     << emitter.getOrCreateName(op.getRepeatTimes()) << ", "
     << emitter.getOrCreateName(op.getRepeatParams()) << ")";
}

template <typename BinaryOp>
auto printBinaryL2Params(CodeEmitter &emitter, BinaryOp op) {
  auto &os = emitter.ostream();
  os << "(" << emitter.getOrCreateName(op.getDst()) << ", "
     << emitter.getOrCreateName(op.getSrc0()) << ", "
     << emitter.getOrCreateName(op.getSrc1()) << ", "
     << emitter.getOrCreateName(op.getCalCount()) << ")";
}

template <typename BinaryL0Op>
auto printOperation(CodeEmitter &emitter, BinaryL0Op op)
    -> LogicalResultForT<BinaryL0Op, ascendc::MulCastL0Op> {
  auto &os = emitter.ostream();
  os << ascNamespace << "::" << op.getAPIName();
  printBinaryL0Params(emitter, op);
  return success();
}

template <typename BinaryL1Op>
auto printOperation(CodeEmitter &emitter, BinaryL1Op op)
    -> LogicalResultForT<BinaryL1Op, ascendc::MulCastL1Op> {
  auto &os = emitter.ostream();
  auto maskName = printMask(emitter, op);
  os << ascNamespace << "::" << op.getAPIName();
  printBinaryL1Params(emitter, op, maskName);
  return success();
}

template <typename BinaryL2Op>
auto printOperation(CodeEmitter &emitter, BinaryL2Op op) -> LogicalResultForT<
    BinaryL2Op, ascendc::AddL2Op, ascendc::AddDeqReluL2Op, ascendc::AddReluL2Op,
    ascendc::AddReluCastL2Op, ascendc::AndL2Op, ascendc::DivL2Op,
    ascendc::FusedAbsSubL2Op, ascendc::FusedExpSubL2Op,
    ascendc::FusedMulAddL2Op, ascendc::FusedMulAddReluL2Op, ascendc::MaxL2Op,
    ascendc::MinL2Op, ascendc::MulL2Op, ascendc::MulAddDstL2Op,
    ascendc::MulCastL2Op, ascendc::OrL2Op, ascendc::PreluL2Op, ascendc::SubL2Op,
    ascendc::SubReluL2Op, ascendc::SubReluCastL2Op> {
  auto &os = emitter.ostream();
  os << ascNamespace << "::" << op.getAPIName();
  printBinaryL2Params(emitter, op);
  return success();
}

template <typename BinaryTemplateL0Op>
auto printOperation(CodeEmitter &emitter, BinaryTemplateL0Op op)
    -> LogicalResultForT<BinaryTemplateL0Op, ascendc::AddL0Op,
                         ascendc::AddReluL0Op, ascendc::AndL0Op,
                         ascendc::DivL0Op, ascendc::FusedMulAddL0Op,
                         ascendc::FusedMulAddReluL0Op, ascendc::MaxL0Op,
                         ascendc::MinL0Op, ascendc::MulL0Op, ascendc::OrL0Op,
                         ascendc::SubL0Op, ascendc::SubReluL0Op> {
  auto &os = emitter.ostream();
  FAIL_OR(printIsSetMaskTemplate(emitter, op));
  printBinaryL0Params(emitter, op);
  return success();
}

template <typename BinaryTemplateL1Op>
auto printOperation(CodeEmitter &emitter, BinaryTemplateL1Op op)
    -> LogicalResultForT<BinaryTemplateL1Op, ascendc::AddL1Op,
                         ascendc::AddReluL1Op, ascendc::AndL1Op,
                         ascendc::DivL1Op, ascendc::FusedMulAddL1Op,
                         ascendc::FusedMulAddReluL1Op, ascendc::MaxL1Op,
                         ascendc::MinL1Op, ascendc::MulL1Op, ascendc::OrL1Op,
                         ascendc::SubL1Op, ascendc::SubReluL1Op> {
  auto &os = emitter.ostream();
  auto maskName = printMask(emitter, op);
  FAIL_OR(printIsSetMaskTemplate(emitter, op));
  printBinaryL1Params(emitter, op, maskName);
  return success();
}

template <typename BinaryCastL0Op>
auto printOperation(CodeEmitter &emitter, BinaryCastL0Op op)
    -> LogicalResultForT<BinaryCastL0Op, ascendc::AddDeqReluL0Op,
                         ascendc::AddReluCastL0Op, ascendc::SubReluCastL0Op,
                         ascendc::MulAddDstL0Op> {
  auto &os = emitter.ostream();
  FAIL_OR(printIsSetMaskCastTemplate(emitter, op));
  printBinaryL0Params(emitter, op);
  return success();
}

template <typename BinaryCastL1Op>
auto printOperation(CodeEmitter &emitter, BinaryCastL1Op op)
    -> LogicalResultForT<BinaryCastL1Op, ascendc::AddDeqReluL1Op,
                         ascendc::AddReluCastL1Op, ascendc::SubReluCastL1Op,
                         ascendc::MulAddDstL1Op> {
  auto &os = emitter.ostream();
  auto maskName = printMask(emitter, op);
  FAIL_OR(printIsSetMaskCastTemplate(emitter, op));
  printBinaryL1Params(emitter, op, maskName);
  return success();
}

template <typename BinaryL3Op>
auto printOperation(CodeEmitter &emitter, BinaryL3Op op)
    -> LogicalResultForT<BinaryL3Op, ascendc::AddL3Op, ascendc::DivL3Op,
                         ascendc::MulL3Op, ascendc::SubL3Op> {
  auto &os = emitter.ostream();
  os << emitter.getOrCreateName(op.getDst()) << " = "
     << emitter.getOrCreateName(op.getSrc0()) << "." << op.getAPIName() << "("
     << emitter.getOrCreateName(op.getSrc1()) << ")";
  return success();
}

LogicalResult printOperation(CodeEmitter &emitter,
                             ascendc::BilinearInterpolationL0Op op);

LogicalResult printOperation(CodeEmitter &emitter,
                             ascendc::BilinearInterpolationL1Op op);

}  // namespace ascendc
}  // namespace mlir

#endif  // ASCIR_TARGET_ASC_BASIC_VECTOR_BINARY_H
