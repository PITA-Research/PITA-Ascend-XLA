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

#ifndef ASCIR_TARGET_ASC_BASIC_VCONV_H
#define ASCIR_TARGET_ASC_BASIC_VCONV_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

// ===----------------------------------------------------------------------===//
// Type conversion operations
// ===----------------------------------------------------------------------===//

template <typename OpType>
LogicalResult printCastDeqL01Template(CodeEmitter &emitter, OpType op) {
  auto &os = emitter.ostream();
  auto dstType =
      cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
  auto srcType =
      cast<ascendc::LocalTensorType>(op.getSrc().getType()).getElementType();
  os << ascNamespace << "::" << op.getAPIName() << "<";
  FAIL_OR(emitter.emitType(op.getLoc(), dstType));
  os << ", ";
  FAIL_OR(emitter.emitType(op.getLoc(), srcType));
  os << ", " << op.getIsSetMask() << ", " << op.getIsVecDeq() << ", "
     << op.getHalfBlock() << ">";
  return success();
}

template <typename OpType>
LogicalResult printCastDeqL2Template(CodeEmitter &emitter, OpType op) {
  auto &os = emitter.ostream();
  auto dstType =
      cast<ascendc::LocalTensorType>(op.getDst().getType()).getElementType();
  auto srcType =
      cast<ascendc::LocalTensorType>(op.getSrc().getType()).getElementType();
  os << ascNamespace << "::" << op.getAPIName() << "<";
  FAIL_OR(emitter.emitType(op.getLoc(), dstType));
  os << ", ";
  FAIL_OR(emitter.emitType(op.getLoc(), srcType));
  os << ", " << op.getIsVecDeq() << ", " << op.getHalfBlock() << ">";
  return success();
}

LogicalResult printOperation(CodeEmitter &emitter, ascendc::CastL0Op op);

LogicalResult printOperation(CodeEmitter &emitter, ascendc::CastL1Op op);

LogicalResult printOperation(CodeEmitter &emitter, ascendc::CastL2Op op);

LogicalResult printOperation(CodeEmitter &emitter, ascendc::CastDeqL0Op op);

LogicalResult printOperation(CodeEmitter &emitter, ascendc::CastDeqL1Op op);

LogicalResult printOperation(CodeEmitter &emitter, ascendc::CastDeqL2Op op);

LogicalResult printOperation(CodeEmitter &emitter, ascendc::SetDeqScaleOp op);

}  // namespace ascendc
}  // namespace mlir

#endif  // ASCIR_TARGET_ASC_BASIC_VCONV_H
