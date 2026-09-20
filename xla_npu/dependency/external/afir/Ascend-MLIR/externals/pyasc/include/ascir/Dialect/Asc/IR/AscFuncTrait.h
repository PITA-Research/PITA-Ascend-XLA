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

#ifndef ASCIR_DIALECT_ASC_IR_ASC_FUNC_TRAIT_H
#define ASCIR_DIALECT_ASC_IR_ASC_FUNC_TRAIT_H

namespace mlir {
namespace OpTrait {
template <typename ConcreteOp>
struct AscConstructorTrait : public TraitBase<ConcreteOp, AscConstructorTrait> {
  static mlir::LogicalResult verifyTrait(Operation *op) {
    if (op->getNumResults() != 1) {
      return op->emitError("AscConstructorTrait must have a result");
    }
    return success();
  }
};

template <typename ConcreteOp>
struct AscMemberFuncTrait : public TraitBase<ConcreteOp, AscMemberFuncTrait> {
  static mlir::LogicalResult verifyTrait(Operation *op) {
    if (op->getNumOperands() < 1 || op->getNumResults() > 1) {
      return op->emitError(
          "AscMemberFuncTrait must have more than one inputs and less than one "
          "return value");
    }
    return success();
  }
};

template <typename ConcreteOp>
struct AscFuncTrait : public TraitBase<ConcreteOp, AscFuncTrait> {
  static mlir::LogicalResult verifyTrait(Operation *op) {
    if (op->getNumResults() > 1) {
      return op->emitError(
          "AscFunc trait only support less than one return value");
    }
    return success();
  }
};

}  // namespace OpTrait
}  // namespace mlir

#endif