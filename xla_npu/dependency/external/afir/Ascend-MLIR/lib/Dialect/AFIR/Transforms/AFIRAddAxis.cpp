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

//===- AFIRAddAxis.cpp - AFIR shape inference pass -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIR.h"
#include "Dialect/AFIR/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <functional>
namespace mlir {
namespace afir {

#define GEN_PASS_DEF_AFIRADDAXISPASS
#include "Dialect/AFIR/Transforms/Passes.h.inc"

namespace {

// Structure to hold axis addition information
struct AxisAddInfo {
  int64_t target_rank;
  int64_t
      insert_position;  // Where to insert the new dimension (-1 means append at front)
};

void getNeedToAddAxis(Operation *op,
                      llvm::DenseMap<Value, AxisAddInfo> &needToAddAxis) {
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());

  // For concat operations, we need to add dimension at the concat_axis position
  int64_t concat_axis = -1;
  if (auto concatOp = dyn_cast<afir::ConcatOp>(op)) {
    concat_axis = concatOp.getConcatAxisAttr().getInt();
  }

  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    if (auto opType = dyn_cast<RankedTensorType>(operand.getType())) {
      Value addOperand = operand;
      if (opType.getRank() != resultType.getRank()) {
        while (addOperand.getDefiningOp() &&
               isa<afir::LoadOp>(addOperand.getDefiningOp())) {
          addOperand =
              dyn_cast<afir::LoadOp>(addOperand.getDefiningOp()).getInput();
        }
        AxisAddInfo info;
        info.target_rank = resultType.getRank();
        info.insert_position =
            concat_axis;  // For concat, insert at concat_axis; for others, -1 (front)
        needToAddAxis[addOperand] = info;
      }
    }
  }
}

void addAxis(llvm::DenseMap<Value, AxisAddInfo> &needToAddAxis,
             IRRewriter &rewriter) {
  for (auto &[val, info] : needToAddAxis) {
    auto originType = dyn_cast<RankedTensorType>(val.getType());
    SmallVector<int64_t> newDims;

    if (info.insert_position == -1) {
      // Add dimensions at the front
      for (int64_t i = 0; i < info.target_rank - originType.getRank(); ++i) {
        newDims.push_back(1);
      }
      newDims.append(originType.getShape().begin(),
                     originType.getShape().end());
    } else {
      // Add dimension at specific position (for concat)
      int64_t num_dims_to_add = info.target_rank - originType.getRank();
      int64_t current_idx = 0;
      for (int64_t i = 0; i < info.target_rank; ++i) {
        if (i >= info.insert_position &&
            i < info.insert_position + num_dims_to_add) {
          // Insert dimension of size 1 at the concat_axis position
          newDims.push_back(1);
        } else {
          // Copy from original shape
          newDims.push_back(originType.getShape()[current_idx++]);
        }
      }
    }

    auto newType = RankedTensorType::get(newDims, originType.getElementType(),
                                         originType.getEncoding());
    val.setType(newType);
    if (auto blockArg = dyn_cast<BlockArgument>(val)) {
      if (auto func =
              dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
        auto originFuncType = func.getFunctionType();
        SmallVector<Type> inputs(originFuncType.getInputs());
        inputs[blockArg.getArgNumber()] = newType;
        func.setFunctionType(FunctionType::get(func.getContext(), inputs,
                                               func.getResultTypes()));
      }
    }
    for (auto user : val.getUsers()) {
      if (auto loadOp = dyn_cast<afir::LoadOp>(user)) {
        loadOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(
            {rewriter.getMultiDimIdentityMap(info.target_rank)}));
        loadOp.getResult().setType(newType);
      }
    }
  }
}

void addBroadcast(Operation *op, IRRewriter &rewriter) {
  rewriter.setInsertionPoint(op);
  SmallVector<Value> newOperands;
  bool needToAdd = false;
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  SmallVector<AffineMap> newIndexingMaps(
      dyn_cast<ArrayAttr>(op->getAttr("indexing_maps"))
          .getAsValueRange<AffineMapAttr>());

  // For concat operations, get the concat axis to skip shape checking on that axis
  int64_t concatAxis = -1;
  if (auto concatOp = dyn_cast<afir::ConcatOp>(op)) {
    concatAxis = concatOp.getConcatAxisAttr().getInt();
  }

  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    if (!isa<RankedTensorType>(operand.getType())) {
      newOperands.push_back(operand);
    } else {
      auto opType = cast<RankedTensorType>(operand.getType());
      Value addOperand = operand;
      if (newIndexingMaps[idx].getNumResults() < resultType.getRank()) {
        newIndexingMaps[idx] =
            rewriter.getMultiDimIdentityMap(resultType.getRank());
      }
      if (addOperand.getType() != resultType) {
        needToAdd = true;
        auto newMap = rewriter.getMultiDimIdentityMap(resultType.getRank());
        auto originMap = rewriter.getMultiDimIdentityMap(resultType.getRank());
        for (int pos = 0; pos < resultType.getRank(); pos++) {
          // For concat operations, skip shape checking on the concat axis
          // because concat inputs are allowed to have different sizes on that axis
          if (pos != concatAxis &&
              opType.getShape()[pos] != resultType.getShape()[pos]) {
            originMap = originMap.replace(rewriter.getAffineDimExpr(pos),
                                          rewriter.getAffineConstantExpr(0),
                                          resultType.getRank(), 0);
          }
        }
        SmallVector<AffineMap> broadcastMap{originMap, newMap};
        ArrayAttr broadAttr = rewriter.getAffineMapArrayAttr(broadcastMap);
        addOperand = rewriter.create<afir::BroadcastOp>(
            op->getLoc(), resultType, addOperand, broadAttr, 0,
            rewriter.getDictionaryAttr({}), rewriter.getArrayAttr({}),
            rewriter.getArrayAttr({}));
      }
      newOperands.push_back(addOperand);
    }
  }
  if (needToAdd) {
    IRMapping mapper;
    mapper.map(op->getOperands(), newOperands);
    auto newOp = rewriter.clone(*op, mapper);
    rewriter.replaceOp(op, newOp);
    newIndexingMaps.back() =
        rewriter.getMultiDimIdentityMap(resultType.getRank());
    newOp->setAttr("indexing_maps",
                   rewriter.getAffineMapArrayAttr(newIndexingMaps));
  }
}

struct AFIRAddAxisPass : public impl::AFIRAddAxisPassBase<AFIRAddAxisPass> {
  void runOnOperation() override {
    SmallVector<Operation *> opList;
    llvm::DenseMap<Value, AxisAddInfo> needToAddAxis;
    getOperation().walk([&](Operation *op) {
      if (op->getDialect()->getNamespace() ==
              AFIRDialect::getDialectNamespace() &&
          op->getOperands().size() > 1 && !isa<afir::MaxOp>(op)) {
        opList.push_back(op);
      }
    });
    IRRewriter rewriter(&getContext());
    do {
      needToAddAxis.clear();
      for (auto op : opList) {
        getNeedToAddAxis(op, needToAddAxis);
      }
      addAxis(needToAddAxis, rewriter);
    } while (needToAddAxis.size());
    for (auto op : opList) {
      addBroadcast(op, rewriter);
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createAFIRAddAxisPass() {
  return std::make_unique<AFIRAddAxisPass>();
}

}  // namespace afir
}  // namespace mlir
