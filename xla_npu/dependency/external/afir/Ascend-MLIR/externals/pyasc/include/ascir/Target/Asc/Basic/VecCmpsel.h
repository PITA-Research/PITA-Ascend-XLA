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

#ifndef ASCIR_TARGET_ASC_BASIC_VECTOR_SELECT_INSTR_H
#define ASCIR_TARGET_ASC_BASIC_VECTOR_SELECT_INSTR_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Select operations
//===----------------------------------------------------------------------===//

template <typename SelectL2Op>
LogicalResultForT<SelectL2Op, ascendc::SelectL2Op, ascendc::SelectScalarL2Op>
printOperation(CodeEmitter &emitter, SelectL2Op op) {
  auto &os = emitter.ostream();
  os << ascNamespace << "::" << op.getAPIName() << "("
     << emitter.getOrCreateName(op.getDst()) << ", "
     << emitter.getOrCreateName(op.getSelMask()) << ", "
     << emitter.getOrCreateName(op.getSrc0()) << ", "
     << emitter.getOrCreateName(op.getSrc1()) << ", AscendC::SELMODE::"
     << ascendc::stringifySelMode(op.getSelMode()).upper() << ", "
     << emitter.getOrCreateName(op.getCalCount()) << ")";
  return success();
}

template <typename SelectL0Op>
LogicalResultForT<SelectL0Op, ascendc::SelectL0Op, ascendc::SelectScalarL0Op>
printOperation(CodeEmitter &emitter, SelectL0Op op) {
  auto &os = emitter.ostream();
  os << ascNamespace << "::" << op.getAPIName() << "("
     << emitter.getOrCreateName(op.getDst()) << ", "
     << emitter.getOrCreateName(op.getSelMask()) << ", "
     << emitter.getOrCreateName(op.getSrc0()) << ", "
     << emitter.getOrCreateName(op.getSrc1()) << ", "
     << "AscendC::SELMODE::" << ascendc::stringifySelMode(op.getMode()).upper()
     << ", " << emitter.getOrCreateName(op.getMask()) << ", "
     << emitter.getOrCreateName(op.getRepeatTimes())
     << ", AscendC::BinaryRepeatParams("
     << emitter.getOrCreateName(op.getDstBlkStride()) << ", "
     << emitter.getOrCreateName(op.getSrc0BlkStride()) << ", "
     << emitter.getOrCreateName(op.getSrc1BlkStride()) << ", "
     << emitter.getOrCreateName(op.getDstRepStride()) << ", "
     << emitter.getOrCreateName(op.getSrc0RepStride()) << ", "
     << emitter.getOrCreateName(op.getSrc1RepStride()) << "))";
  return success();
}

LogicalResult printOperation(CodeEmitter &emitter, ascendc::CompareL1Op op);

LogicalResult printOperation(CodeEmitter &emitter, ascendc::CompareRL1Op op);

LogicalResult printOperation(CodeEmitter &emitter,
                             ascendc::CompareScalarL1Op op);

}  // namespace ascendc
}  // namespace mlir

#endif  // ASCIR_TARGET_ASC_BASIC_VECTOR_SELECT_INSTR_H
