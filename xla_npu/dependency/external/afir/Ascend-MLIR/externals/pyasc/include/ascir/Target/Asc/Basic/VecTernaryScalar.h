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

#ifndef ASCIR_TARGET_ASC_BASIC_TERNARY_SCALAR_INSTR_H
#define ASCIR_TARGET_ASC_BASIC_TERNARY_SCALAR_INSTR_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Vector-ternary-scalar operations
//===----------------------------------------------------------------------===//

template <typename VecScalarL0Op>
auto printOperation(CodeEmitter &emitter, VecScalarL0Op op)
    -> LogicalResultForT<VecScalarL0Op, ascendc::AxpyL0Op> {
  auto &os = emitter.ostream();
  auto dstTensorType =
      cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
  auto srcTensorType =
      cast<ascendc::LocalTensorType>(op.getSrc().getType()).getElementType();
  os << ascNamespace << "::" << op.getAPIName() << "<";
  FAIL_OR(emitter.emitType(op.getLoc(), dstTensorType));
  os << ", ";
  FAIL_OR(emitter.emitType(op.getLoc(), srcTensorType));
  os << ", " << op.getIsSetMask() << ">";
  os << "(" << emitter.getOrCreateName(op.getDst()) << ", "
     << emitter.getOrCreateName(op.getSrc()) << ", "
     << emitter.getOrCreateName(op.getScalar()) << ", "
     << emitter.getOrCreateName(op.getMask()) << ", "
     << emitter.getOrCreateName(op.getRepeatTimes()) << ", "
     << emitter.getOrCreateName(op.getRepeatParams()) << ")";
  return success();
}

template <typename VecScalarL1Op>
auto printOperation(CodeEmitter &emitter, VecScalarL1Op op)
    -> LogicalResultForT<VecScalarL1Op, ascendc::AxpyL1Op> {
  auto &os = emitter.ostream();
  auto maskName = printMask(emitter, op);
  auto dstTensorType =
      cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
  auto srcTensorType =
      cast<ascendc::LocalTensorType>(op.getSrc().getType()).getElementType();
  os << ascNamespace << "::" << op.getAPIName() << "<";
  FAIL_OR(emitter.emitType(op.getLoc(), dstTensorType));
  os << ", ";
  FAIL_OR(emitter.emitType(op.getLoc(), srcTensorType));
  os << ", " << op.getIsSetMask() << ">";
  os << "(" << emitter.getOrCreateName(op.getDst()) << ", "
     << emitter.getOrCreateName(op.getSrc()) << ", "
     << emitter.getOrCreateName(op.getScalar()) << ", " << maskName << ", "
     << emitter.getOrCreateName(op.getRepeatTimes()) << ", "
     << emitter.getOrCreateName(op.getRepeatParams()) << ")";
  return success();
}

template <typename VecScalarL2Op>
auto printOperation(CodeEmitter &emitter, VecScalarL2Op op)
    -> LogicalResultForT<VecScalarL2Op, ascendc::AxpyL2Op> {
  auto &os = emitter.ostream();
  os << ascNamespace << "::" << op.getAPIName();
  os << "(" << emitter.getOrCreateName(op.getDst()) << ", "
     << emitter.getOrCreateName(op.getSrc()) << ", "
     << emitter.getOrCreateName(op.getScalar()) << ", "
     << emitter.getOrCreateName(op.getCalCount()) << ")";
  return success();
}

}  // namespace ascendc
}  // namespace mlir

#endif  // ASCIR_TARGET_ASC_BASIC_TERNARY_SCALAR_INSTR_H
