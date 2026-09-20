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

#ifndef TEST_MAIR_GEN_H_
#define TEST_MAIR_GEN_H_

#include <iostream>
#include <vector>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/CAPI/Registration.h"

#include "mlir-c/IR.h"

#include "../ir/mair_ops.h"
#include "../translations/export_graphdef.h"
#include "../translations/status.h"
#include "../translations/utils.h"

namespace mlir::air {

class GraphGenTester {
 public:
  static Status Convert(mlir::ModuleOp& module, ge::GraphPtr& graphPtr);

  static void buildGeluModule(mlir::ModuleOp& module,
                              mlir::MLIRContext& context);
  static void buildAddModule(mlir::ModuleOp& module,
                             mlir::MLIRContext& context);
  static void buildAddMulModule(mlir::ModuleOp& module,
                                mlir::MLIRContext& context);
  static void buildAEmptyModule(mlir::ModuleOp& module,
                                mlir::MLIRContext& context);
  static void buildConstModule(mlir::ModuleOp& module,
                               mlir::MLIRContext& context);
};

}  // namespace mlir::air

#endif  //TEST_MAIR_GEN_H_
