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

#include "mair_ops.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#define GET_OP_CLASSES
#include "xla/xla_npu/pjrt/air/mair/ir/mair_ops.cc.inc"

namespace mlir {
namespace air {

void AirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xla/xla_npu/pjrt/air/mair/ir/mair_ops.cc.inc"
      >();
}

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

void GraphOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {}

}  // namespace air
}  // namespace mlir
