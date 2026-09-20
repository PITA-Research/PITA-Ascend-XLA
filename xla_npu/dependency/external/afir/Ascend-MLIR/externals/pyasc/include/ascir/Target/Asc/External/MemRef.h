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

#ifndef ASCIR_TARGET_ASC_MLIR_MEMREF_H
#define ASCIR_TARGET_ASC_MLIR_MEMREF_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {

LogicalResult printOperation(CodeEmitter &emitter, memref::AllocaOp op);

LogicalResult printOperation(CodeEmitter &emitter, memref::LoadOp op);

LogicalResult printOperation(CodeEmitter &emitter, memref::StoreOp op);

LogicalResult printOperation(CodeEmitter &emitter, memref::CastOp op);

}  // namespace mlir

#endif  // ASCIR_TARGET_ASC_MLIR_MEMREF_H
