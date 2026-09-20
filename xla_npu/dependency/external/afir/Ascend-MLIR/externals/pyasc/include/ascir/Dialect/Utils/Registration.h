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

#ifndef ASCIR_DIALECT_UTILS_REGISTRATION_H
#define ASCIR_DIALECT_UTILS_REGISTRATION_H

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/Asc/Transforms/Passes.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"

namespace mlir {
namespace ascir {

inline void registerDialects(DialectRegistry &registry) {
  registerAllDialects(registry);
  registry.insert<ascendc::AscendCDialect, emitasc::EmitAscDialect>();
  ascendc::registerExternalModels(registry);
  emitasc::registerExternalModels(registry);
}

inline void registerExtensions(DialectRegistry &registry) {
  registerAllExtensions(registry);
}

inline void registerPasses() {
  registerAllPasses();
  registerascendcPasses();
}

}  // namespace ascir
}  // namespace mlir

#endif  // ASCIR_DIALECT_UTILS_REGISTRATION_H
