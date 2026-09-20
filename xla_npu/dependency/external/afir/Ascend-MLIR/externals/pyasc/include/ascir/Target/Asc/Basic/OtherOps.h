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

#ifndef ASCIR_TARGET_ASC_BASIC_OTHER_OPS_H
#define ASCIR_TARGET_ASC_BASIC_OTHER_OPS_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Other operations
//===----------------------------------------------------------------------===//

template <typename CVOp>
LogicalResultForT<CVOp, ascendc::AscendIsAICOp, ascendc::AscendIsAIVOp>
printOperation(CodeEmitter &emitter, CVOp op) {
  FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
  auto &os = emitter.ostream();
  os << " = g_coreType == AscendC::";
  if (isa<ascendc::AscendIsAICOp>(op)) {
    os << "AIC";
  } else {
    os << "AIV";
  }
  return success();
}

LogicalResult printOperation(CodeEmitter &emitter, ascendc::ConstructOp op);

LogicalResult printOperation(CodeEmitter &emitter,
                             ascendc::FftsCrossCoreSyncOp op);

LogicalResult printOperation(CodeEmitter &emitter,
                             ascendc::PopStackBufferOp op);

LogicalResult printOperation(CodeEmitter &emitter,
                             ascendc::SetFftsBaseAddrOp op);

LogicalResult printOperation(CodeEmitter &emitter, ascendc::ResetMaskOp op);

}  // namespace ascendc

LogicalResult printOperation(CodeEmitter &emitter, LLVM::UndefOp op);

}  // namespace mlir

#endif  // ASCIR_TARGET_ASC_BASIC_OTHER_OPS_H