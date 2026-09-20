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

#ifndef ASCIR_TARGET_ASC_EMITASC_H
#define ASCIR_TARGET_ASC_EMITASC_H

#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace emitasc {

//===----------------------------------------------------------------------===//
// EmitAsc operations
//===----------------------------------------------------------------------===//

LogicalResult printOperation(CodeEmitter &emitter, emitasc::CallOpaqueOp op);

LogicalResult printOperation(CodeEmitter &emitter, emitasc::CopyStructOp op);

LogicalResult printOperation(CodeEmitter &emitter,
                             emitasc::DeclarePyStructOp op);

LogicalResult printOperation(CodeEmitter &emitter, emitasc::DereferenceOp op);

LogicalResult printOperation(CodeEmitter &emitter, emitasc::MemberOp op);

LogicalResult printOperation(CodeEmitter &emitter, emitasc::MemberPtrOp op);

LogicalResult printOperation(CodeEmitter &emitter, emitasc::MemberRefOp op);

LogicalResult printOperation(CodeEmitter &emitter, emitasc::PtrOffsetOp op);

LogicalResult printOperation(CodeEmitter &emitter,
                             emitasc::ReinterpretCastOp op);

LogicalResult printOperation(CodeEmitter &emitter, emitasc::SetMemberOp op);

LogicalResult printOperation(CodeEmitter &emitter, emitasc::VariableOp op);

LogicalResult printOperation(CodeEmitter &emitter, emitasc::VerbatimOp op);

}  // namespace emitasc
}  // namespace mlir

#endif  // ASCIR_TARGET_ASC_EMITASC_H
