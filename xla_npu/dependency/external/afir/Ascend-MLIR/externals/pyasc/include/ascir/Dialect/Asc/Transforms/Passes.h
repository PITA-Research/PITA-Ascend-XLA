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

#ifndef ASCIR_DIALECT_ASC_TRANSFORMS_PASSES_H
#define ASCIR_DIALECT_ASC_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace ascendc {

#define GEN_PASS_DECL
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createDeclarePyStructPass();
std::unique_ptr<Pass> createDefineCubeOnlyPass();
std::unique_ptr<Pass> createDetectKernelTypePass();
std::unique_ptr<Pass> createEraseSyncPass();
std::unique_ptr<Pass> createGenerateBoilerplatePass();
std::unique_ptr<Pass> createHoistQueBindPass();
std::unique_ptr<Pass> createHoistUBAllocationPass();
std::unique_ptr<Pass> createInputOutputTensorPass();
std::unique_ptr<Pass> createInsertSyncPass();
std::unique_ptr<Pass> createLegalizeKernelArgsPass();
std::unique_ptr<Pass> createMaterializeTensorPass();
std::unique_ptr<Pass> createNoopPass();
std::unique_ptr<Pass> createPrivatizeFuncPass();
std::unique_ptr<Pass> createUnifyPipePass();
std::unique_ptr<Pass> createVerifySyncPass();
std::unique_ptr<Pass> createDetectEnableDebugPass();

}  // namespace ascendc

#define GEN_PASS_REGISTRATION
#include "ascir/Dialect/Asc/Transforms/Passes.h.inc"

}  // end namespace mlir

#endif  // ASCIR_DIALECT_ASC_TRANSFORMS_PASSES_H
