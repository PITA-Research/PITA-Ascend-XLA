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

//===- AFIRShapeInference.cpp - AFIR shape inference pass -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIR.h"
#include "Dialect/AFIR/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <functional>
#include "Utils/AFIRShapeInferUtils.h"
namespace mlir {
namespace afir {

#define GEN_PASS_DEF_AFIRSHAPEINFERENCEPASS
#include "Dialect/AFIR/Transforms/Passes.h.inc"

namespace {

class TypeModificationState {
 public:
  TypeModificationState() = default;

  ~TypeModificationState() {
    // Ensure the recorded modifications are either committed or rolled back.
    assert(oldTypes.empty() && "unhandled type modifications");
  }

  // Update the state of the value and record the old type.
  void setType(Value value, Type type) {
    if (value.getType() != type) {
      oldTypes.emplace_back(value, value.getType());
      value.setType(type);
    }
  }

  // Roll back changes made to the types in the IR by setting all the affected
  // values to their old types.
  void rollBack() {
    for (auto [value, type] : oldTypes) value.setType(type);

    oldTypes.clear();
  }

  // Commit the changes to the types in the IR.
  // This requires inserting tensor.cast operations to mediate the newly
  // inferred result types with users that do not support type inference.
  void commit() {
    // For each use whose type changed, cast the value with the new type back to
    // the old type.
    /*
    for (auto [value, oldType] : oldTypes) {
      // The call to 'use->set()' in the body of the loop below invalidates the
      // iterator used to traverse op uses, so it is important to make a copy of
      // these first.
      llvm::SmallVector<OpOperand *> uses =
          llvm::map_to_vector(value.getUses(), [](OpOperand &use) -> OpOperand * { return &use; });

      // A 'tensor.cast' op is emitted only if needed. Once emitted, it is
      // cached and reused by all consumers.
      tensor::CastOp castValue;

      // Traverse all uses
      for (OpOperand *use : uses) {
        if (canBeRefined(use->getOwner())) continue;

        if (!castValue) {
          // Set the insertion point as far back as possible, since new
          // consumers of the 'tensor.cast' op generated in future iterations
          // are likely to be further up in the code due to the order in which
          // they appear in the use list.
          OpBuilder builder{value.getContext()};
          builder.setInsertionPointAfter(value.getDefiningOp());
          castValue = builder.create<tensor::CastOp>(value.getLoc(), oldType, value);
        }

        use->set(castValue);
      }
    }
      */

    oldTypes.clear();
  }

 private:
  // A record of each value whose type was updated along with that value's
  // previous type.
  llvm::SmallVector<std::pair<Value, Type>> oldTypes;
};

void propagateShapesInRegion(Region &region, TypeModificationState &state) {
  Dialect *afirDialect = region.getContext()->getLoadedDialect<AFIRDialect>();

  for (auto &block : region) {
    for (Operation &op : block) {
      if (op.getDialect() != afirDialect) continue;

      InferShapedTypeOpInterface shapeInterface =
          dyn_cast<InferShapedTypeOpInterface>(op);
      if (!shapeInterface) continue;

      SmallVector<ShapedTypeComponents> returnedShapes;

      if (shapeInterface
              .inferReturnTypeComponents(
                  op.getContext(), op.getLoc(), op.getOperands(),
                  op.getAttrDictionary(), op.getPropertiesStorage(),
                  op.getRegions(), returnedShapes)
              .succeeded()) {
        for (auto it : llvm::zip(op.getResults(), returnedShapes)) {
          Value result = std::get<0>(it);
          ShapedTypeComponents predictedShape = std::get<1>(it);

          // Determine the knowledge based on the output type.
          Type resultTy = result.getType();
          auto currentKnowledge =
              ValueKnowledge::getKnowledgeFromType(resultTy);

          // Compute the knowledge based on the inferred type.
          auto inferredKnowledge = ValueKnowledge::getPessimisticValueState();
          inferredKnowledge.dtype = cast<ShapedType>(resultTy).getElementType();
          inferredKnowledge.hasRank = predictedShape.hasRank();
          if (predictedShape.hasRank()) {
            for (auto dim : predictedShape.getDims()) {
              inferredKnowledge.sizes.push_back(dim);
            }
          }

          // Compute the new type based on the joined version.
          auto newKnowledge =
              ValueKnowledge::join(currentKnowledge, inferredKnowledge);
          if (!newKnowledge) continue;
          // Set new type
          state.setType(result, newKnowledge.getType());
        }
      }
    }
  }
}

void validateSameOperandsAndResultRankTrait(Region &region) {
  int errs = 0;
  for (auto &block : region) {
    for (auto &op : block) {
      if (!op.getDialect() ||
          op.getDialect()->getNamespace() != AFIRDialect::getDialectNamespace())
        continue;
      if (op.hasTrait<OpTrait::SameOperandsAndResultRank>()) {
        if (OpTrait::impl::verifySameOperandsAndResultRank(&op).failed()) {
          errs++;
          (void)errs;
        }
      }
    }
  }
}

struct AFIRShapeInferencePass
    : public impl::AFIRShapeInferencePassBase<AFIRShapeInferencePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](func::FuncOp func) {
      TypeModificationState state;
      propagateShapesInRegion(func.getBody(), state);
      state.commit();

      validateSameOperandsAndResultRankTrait(func.getBody());
      SmallVector<func::ReturnOp> returnOps;
      func.walk(
          [&](func::ReturnOp returnOp) { returnOps.push_back(returnOp); });

      if (returnOps.empty()) return;

      // Get the types of the values being returned
      auto firstReturn = returnOps[0];
      SmallVector<Type> newResultTypes;
      for (Value operand : firstReturn.getOperands()) {
        newResultTypes.push_back(operand.getType());
      }

      // Update function type
      auto funcType = func.getFunctionType();
      auto newFuncType = FunctionType::get(
          func.getContext(), funcType.getInputs(), newResultTypes);
      func.setFunctionType(newFuncType);
    });

    return;
    /*
    // Walk all operations and infer shapes
    module.walk([&](Operation *op) {
      if (auto shapeInfOp = dyn_cast<ShapeInferenceOpInterface>(op)) {
        // Infer shapes for operations implementing the interface
        std::function<void(mlir::Region &)> doShapeInference = [&](mlir::Region &region) {
          // Recursively process regions if needed
          region.walk([&](Operation *innerOp) {
            if (auto innerShapeInfOp = dyn_cast<ShapeInferenceOpInterface>(innerOp)) {
              (void)innerShapeInfOp.inferShapes(doShapeInference);
            }
          });
        };

        if (failed(shapeInfOp.inferShapes(doShapeInference))) {
          op->emitWarning("Shape inference failed");
        }
      }
    });

    // Update function signatures to match inferred return types
    module.walk([&](func::FuncOp funcOp) {
      // Collect all return operations
      SmallVector<func::ReturnOp> returnOps;
      funcOp.walk([&](func::ReturnOp returnOp) { returnOps.push_back(returnOp); });

      if (returnOps.empty()) return;

      // Get the types of the values being returned
      auto firstReturn = returnOps[0];
      SmallVector<Type> newResultTypes;
      for (Value operand : firstReturn.getOperands()) {
        newResultTypes.push_back(operand.getType());
      }

      // Update function type
      auto funcType = funcOp.getFunctionType();
      auto newFuncType = FunctionType::get(funcOp.getContext(), funcType.getInputs(), newResultTypes);
      funcOp.setFunctionType(newFuncType);
    });
    */
  }
};

}  // namespace

std::unique_ptr<Pass> createAFIRShapeInferencePass() {
  return std::make_unique<AFIRShapeInferencePass>();
}

}  // namespace afir
}  // namespace mlir
