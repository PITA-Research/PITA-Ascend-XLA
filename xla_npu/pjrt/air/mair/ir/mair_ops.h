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

#ifndef MAIR_OPS_H_
#define MAIR_OPS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

namespace mlir {
namespace air {

class AirDialect : public Dialect {
 public:
  explicit AirDialect(MLIRContext* context)
      : Dialect(getDialectNamespace(), context, TypeID::get<AirDialect>()) {
    initialize();
  }
  static StringRef getDialectNamespace() { return "air"; }
  void initialize();
};

}  // end namespace air
}  // end namespace mlir

#define GET_OP_CLASSES
#include "xla/xla_npu/pjrt/air/mair/ir/mair_ops.h.inc"

#endif  //  MAIR_OPS_H_
