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

#ifndef ASCIR_DIALECT_ASC_UTILS_UTILS_H
#define ASCIR_DIALECT_ASC_UTILS_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace ascendc {

template <typename OpT>
struct HoistOpPattern : public OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;

  virtual bool hoistable(OpT) const { return true; }

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    Operation *parent = op->getParentOp();
    if (isa<func::FuncOp>(parent)) return failure();
    if (!hoistable(op)) return failure();
    DominanceInfo di;
    bool dominatedByOperands = llvm::all_of(op->getOperands(), [&](Value opnd) {
      return di.dominates(opnd, parent);
    });
    if (!dominatedByOperands) return failure();
    rewriter.setInsertionPoint(parent);
    rewriter.replaceOp(op, rewriter.clone(*op.getOperation())->getResults());
    return success();
  }
};

bool opPrecedes(Operation *lhs, Operation *rhs);

bool opPrecedes(Operation *lhs, Operation *rhs, DominanceInfo &di);

void registerInlinerInterfaces(DialectRegistry &registry);

}  // namespace ascendc
}  // namespace mlir

#endif  // ASCIR_DIALECT_ASC_UTILS_UTILS_H
