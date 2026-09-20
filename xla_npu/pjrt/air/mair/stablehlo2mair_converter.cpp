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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "transforms/mair_passes.h"
#include "stablehlo2mair_converter.h"

namespace xla {
namespace mair {
int Stablehlo2MairConverter(mlir::ModuleOp& module) {
  mlir::PassManager pm(module.getContext());

  // Add Stablehlo to Mair dialect conversion pass
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::air::createLegalizeStablehloToGePass());

  if (failed(pm.run(module))) {
    return 4;
  }

  return 0;
}
}  // namespace mair
}  // namespace xla
