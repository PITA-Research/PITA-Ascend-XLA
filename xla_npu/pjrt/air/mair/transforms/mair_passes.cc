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

#include "mair_passes.h"

#include <functional>
#include <memory>
#include <numeric>
#include <vector>
#include <iostream>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"          // from @llvm-project
#include "mlir/IR/MLIRContext.h"                // from @llvm-project
#include "mlir/IR/PatternMatch.h"               // from @llvm-project
#include "mlir/Pass/Pass.h"                     // from @llvm-project
#include "mlir/Support/LLVM.h"                  // from @llvm-project
#include "mlir/Support/LogicalResult.h"         // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project

#include "stablehlo/dialect/StablehloOps.h"

#include "../ir/mair_ops.h"

#include "xla/xla_npu/pjrt/air/mair/transforms/mair_passes.h.inc"

namespace mlir {
namespace air {
namespace {
std::uint32_t getReduceTypeString(mlir::stablehlo::AllReduceOp allReduceOp) {
  // Get computation area
  mlir::Region& computationRegion = allReduceOp.getComputation();

  // Check if the area is empty
  if (computationRegion.empty()) {
    return -1;
  }

  // Get the first block in the region (usually only one block)
  mlir::Block& block = computationRegion.front();

  // Traverse operations within a block
  for (mlir::Operation& op : block) {
    // Check if it is an add operation
    if (auto addOp = mlir::dyn_cast<mlir::stablehlo::AddOp>(&op)) {
      return 0;
    }
    // Check if it is a maximum operation (reduce max)
    else if (auto maxOp = mlir::dyn_cast<mlir::stablehlo::MaxOp>(&op)) {
      return 1;
    }
    // Check if it is a minimum operation (reduce min)
    else if (auto minOp = mlir::dyn_cast<mlir::stablehlo::MinOp>(&op)) {
      return 2;
    }
    // Check if it is a multiply operation (reduce product)
    else if (auto mulOp = mlir::dyn_cast<mlir::stablehlo::MulOp>(&op)) {
      return 3;
    }
  }
  return -1;
}
std::int64_t getAllGatherRankSize(stablehlo::AllGatherOp allGatherOp) {
  auto replicaGroupsAttr = allGatherOp.getReplicaGroupsAttr();
  int64_t rankSize = -1;
  if (replicaGroupsAttr && replicaGroupsAttr.getType().getRank() >= 2) {
    auto shape = replicaGroupsAttr.getType().getShape();
    rankSize = shape[1];
  }
  return rankSize;
}
class ConvertAddOp : public OpConversionPattern<stablehlo::AddOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), op.getLhs(),
                                       op.getRhs());
    return success();
  };
};
class ConvertMulOp : public OpConversionPattern<stablehlo::MulOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::MulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), op.getLhs(),
                                       op.getRhs());
    return success();
  };
};

// The situation that BroadcastInDim contains transpose has not been considered yet. Transpose needs to be processed by adding transpose.
class ConvertBroadcastInDimOp
    : public OpConversionPattern<stablehlo::BroadcastInDimOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::BroadcastInDimOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Value operand = op.getOperand();
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    auto broadcastDims = op.getBroadcastDimensions();
    auto operandShape = operandType.getShape();
    auto resultShape = resultType.getShape();
    // The dimension is reduced. If the dimension needs to be removed and is not 1, an error will be returned.
    if (operandType.getRank() > resultType.getRank()) {
      llvm::SmallDenseSet<int64_t> operandDimsUsedInResult(
          broadcastDims.begin(), broadcastDims.end());
      SmallVector<int64_t> axesToSqueeze;
      for (int i = 0; i < operandType.getRank(); ++i) {
        if (operandDimsUsedInResult.count(i) == 0) {
          if (operandShape[i] != 1) {
            return failure();
          }
          axesToSqueeze.push_back(i);
        }
      }

      auto axesTensorType = RankedTensorType::get(
          {static_cast<int64_t>(axesToSqueeze.size())}, rewriter.getI64Type());
      auto axesAttr = DenseIntElementsAttr::get(axesTensorType, axesToSqueeze);
      auto axesConstOp = rewriter.create<air::ConstantOp>(
          op.getLoc(), axesTensorType, axesAttr);
      auto newOp = rewriter.create<air::SqueezeV3Op>(
          op.getLoc(), resultType, operand, axesConstOp.getResult());
      rewriter.replaceOp(op, newOp.getResult());
      return success();
    }
    // There are two situations when the dimension is added. If the new dimension is 1, add it directly. If the new dimension is not 1, add the dimension first and then execute BroadcastOp.
    if (operandType.getRank() < resultType.getRank()) {
      llvm::SmallDenseSet<int64_t> resultDimsUsedInOperand(
          broadcastDims.begin(), broadcastDims.end());
      SmallVector<int64_t> axesToUnsqueeze;
      bool needBroadcastAfterUnsqueeze = false;
      for (int i = 0; i < resultType.getRank(); ++i) {
        if (resultDimsUsedInOperand.count(i) == 0) {
          if (resultShape[i] != 1) {
            needBroadcastAfterUnsqueeze = true;
          }
          axesToUnsqueeze.push_back(i);
        }
      }

      if (needBroadcastAfterUnsqueeze) {
        SmallVector<int64_t> intermediateShape;
        int operand_idx = 0;
        for (int i = 0; i < resultType.getRank(); ++i) {
          if (resultDimsUsedInOperand.count(i) > 0) {
            intermediateShape.push_back(operandShape[operand_idx]);
            operand_idx++;
          } else {
            intermediateShape.push_back(1);
          }
        }
        auto intermediateType = RankedTensorType::get(
            intermediateShape, resultType.getElementType());

        auto unsqueezeAxesTensorType = RankedTensorType::get(
            {static_cast<int64_t>(axesToUnsqueeze.size())},
            rewriter.getI64Type());
        auto unsqueezeAxesAttr =
            DenseIntElementsAttr::get(unsqueezeAxesTensorType, axesToUnsqueeze);
        auto unsqueezeAxesConstOp = rewriter.create<air::ConstantOp>(
            op.getLoc(), unsqueezeAxesTensorType, unsqueezeAxesAttr);

        auto unsqueezeOp = rewriter.create<air::UnsqueezeV3Op>(
            op.getLoc(), intermediateType, operand,
            unsqueezeAxesConstOp.getResult());
        Value intermediateValue = unsqueezeOp.getResult();

        auto targetShape = resultType.getShape();
        SmallVector<int64_t> shapeValues(targetShape.begin(),
                                         targetShape.end());
        auto shapeTensorType = RankedTensorType::get(
            {static_cast<int64_t>(shapeValues.size())}, rewriter.getI64Type());
        auto shapeAttr =
            DenseIntElementsAttr::get(shapeTensorType, shapeValues);
        auto shapeConstOp = rewriter.create<air::ConstantOp>(
            op.getLoc(), shapeTensorType, shapeAttr);
        auto newOp = rewriter.create<air::BroadcastOp>(
            op.getLoc(), resultType, intermediateValue,
            shapeConstOp.getResult());
        rewriter.replaceOp(op, newOp.getResult());
        return success();
      } else {
        // No Broadcast is required, the total number of elements remains unchanged, use Reshape
        auto targetShape = resultType.getShape();
        SmallVector<int64_t> shapeValues(targetShape.begin(), targetShape.end());
        
        auto shapeTensorType = RankedTensorType::get(
            {static_cast<int64_t>(shapeValues.size())}, rewriter.getI64Type());
        auto shapeAttr = DenseIntElementsAttr::get(shapeTensorType, shapeValues);
        auto shapeConstOp = rewriter.create<air::ConstantOp>(
            op.getLoc(), shapeTensorType, shapeAttr);
        
        auto newOp = rewriter.create<air::ReshapeOp>(
            op.getLoc(), resultType, operand, shapeConstOp.getResult(),
            rewriter.getI64IntegerAttr(0),
            rewriter.getI64IntegerAttr(-1));
        rewriter.replaceOp(op, newOp.getResult());
        return success();
      }
    }
    // Direct conversion to BroadcastOp with unchanged dimensions
    auto target_shape = resultType.getShape();
    SmallVector<int64_t> shapeValues(target_shape.begin(), target_shape.end());

    auto shapeTensorType = RankedTensorType::get(
        {static_cast<int64_t>(shapeValues.size())}, rewriter.getI64Type());
    auto shapeAttr = DenseIntElementsAttr::get(shapeTensorType, shapeValues);
    auto shapeConstOp = rewriter.create<air::ConstantOp>(
        op.getLoc(), shapeTensorType, shapeAttr);
    auto newOp = rewriter.create<air::BroadcastOp>(
        op.getLoc(), resultType, operand, shapeConstOp.getResult());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  };
};
class ConvertConstantOp : public OpConversionPattern<stablehlo::ConstantOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<ConstantOp>(op, op.getType(), op.getValue());
    return success();
  };
};
class ConvertTanhOp : public OpConversionPattern<stablehlo::TanhOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::TanhOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<TanhOp>(op, op.getType(), op.getOperand());
    return success();
  };
};
class ConvertAllReduceOp : public OpConversionPattern<stablehlo::AllReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::AllReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    std::uint32_t reduceType = getReduceTypeString(op);
    auto reduceTypeValue =
        rewriter
            .create<ConstantOp>(
                op.getLoc(), rewriter.getI32Type(),
                rewriter.getI32IntegerAttr(static_cast<int32_t>(reduceType)))
            .getResult();
    rewriter.replaceOpWithNewOp<HcomAllReduceOp>(
        op, op.getResult(0).getType(), op.getOperand(0), reduceTypeValue);
    return success();
  };
};
class ConvertAllGatherOp : public OpConversionPattern<stablehlo::AllGatherOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::AllGatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    std::int64_t rankSize = getAllGatherRankSize(op);
    auto rankSizeValue =
        rewriter
            .create<ConstantOp>(op.getLoc(), rewriter.getI64Type(),
                                rewriter.getI64IntegerAttr(rankSize))
            .getResult();
    auto allGatherResult =
        rewriter
            .create<HcomAllGatherOp>(op.getLoc(), op.getResult(0).getType(),
                                     op.getOperand(0), rankSizeValue)
            .getResult();
    // The splicing dimension needs to be determined based on the all_gather_dimension parameter
    std::uint64_t concatDim = op.getAllGatherDim();
    auto concatDimValue =
        rewriter
            .create<ConstantOp>(
                op.getLoc(), rewriter.getI64Type(),
                rewriter.getI64IntegerAttr(static_cast<int64_t>(concatDim)))
            .getResult();
    auto concatResult =
        rewriter
            .create<ConcatV2Op>(op.getLoc(),
                                op.getResult(0).getType(),  // final result type
                                allGatherResult, concatDimValue)
            .getResult();

    rewriter.replaceOp(op, concatResult);
    return success();
  };
};
class ConvertConcatV2Op : public OpConversionPattern<stablehlo::ConcatenateOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::ConcatenateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    mlir::ValueRange inputs = adaptor.getInputs();
    int64_t concatDim = op.getDimension();
    auto resultType = op.getType();
    auto indexType = rewriter.getI64Type();
    mlir::RankedTensorType scalarTensorType =
        mlir::RankedTensorType::get({}, indexType);
    mlir::DenseElementsAttr dimElementsAttr =
        mlir::DenseElementsAttr::get(scalarTensorType, concatDim);
    auto constantOp = rewriter.create<mlir::air::ConstantOp>(
        op.getLoc(), scalarTensorType, dimElementsAttr);
    mlir::Value concatDimValue = constantOp.getResult();
    auto newConcatOp = rewriter.create<mlir::air::ConcatV2Op>(
        op.getLoc(), resultType, inputs, concatDimValue,
        rewriter.getI64IntegerAttr(inputs.size()));
    rewriter.replaceOp(op, newConcatOp.getResult());
    return success();
  };
};
class ConvertDivOp : public OpConversionPattern<stablehlo::DivOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::DivOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<DivOp>(op, op.getType(), op.getLhs(),
                                       op.getRhs());
    return success();
  };
};
class ConvertExpOp : public OpConversionPattern<stablehlo::ExpOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::ExpOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<ExpOp>(op, op.getType(), op.getOperand());
    return success();
  };
};
class ConvertMaxOp : public OpConversionPattern<stablehlo::MaxOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::MaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<MaxOp>(op, op.getType(), op.getLhs(),
                                       op.getRhs());
    return success();
  };
};
class ConvertNegOp : public OpConversionPattern<stablehlo::NegOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::NegOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<NegOp>(op, op.getType(), op.getOperand());
    return success();
  };
};
class ConvertSubOp : public OpConversionPattern<stablehlo::SubtractOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::SubtractOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<SubOp>(op, op.getType(), op.getLhs(),
                                       op.getRhs());
    return success();
  };
};
class ConvertMatMulOp : public OpConversionPattern<stablehlo::DotGeneralOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::DotGeneralOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto resultType = op.getType();
    
    auto dimNumbers = op.getDotDimensionNumbers();
    auto lhsBatchingDims = dimNumbers.getLhsBatchingDimensions();
    auto rhsBatchingDims = dimNumbers.getRhsBatchingDimensions();
    auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
    auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();
    
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhsType || !rhsType) {
      return failure();
    }
    
    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();
    auto resultShape = dyn_cast<RankedTensorType>(resultType).getShape();
    
    llvm::SmallDenseSet<int64_t> lhsBatchSet(lhsBatchingDims.begin(), lhsBatchingDims.end());
    llvm::SmallDenseSet<int64_t> rhsBatchSet(rhsBatchingDims.begin(), rhsBatchingDims.end());
    llvm::SmallDenseSet<int64_t> lhsContractSet(lhsContractingDims.begin(), lhsContractingDims.end());
    llvm::SmallDenseSet<int64_t> rhsContractSet(rhsContractingDims.begin(), rhsContractingDims.end());
    
    SmallVector<int64_t> lhsNonBatchNonContractDims;
    SmallVector<int64_t> rhsNonBatchNonContractDims;
    
    for (int64_t i = 0; i < lhsType.getRank(); ++i) {
      if (lhsBatchSet.count(i) == 0 && lhsContractSet.count(i) == 0) {
        lhsNonBatchNonContractDims.push_back(i);
      }
    }
    
    for (int64_t i = 0; i < rhsType.getRank(); ++i) {
      if (rhsBatchSet.count(i) == 0 && rhsContractSet.count(i) == 0) {
        rhsNonBatchNonContractDims.push_back(i);
      }
    }
    
    SmallVector<int64_t> lhsPermutation;
    for (auto dim : lhsBatchingDims) lhsPermutation.push_back(dim);
    for (auto dim : lhsNonBatchNonContractDims) lhsPermutation.push_back(dim);
    for (auto dim : lhsContractingDims) lhsPermutation.push_back(dim);
    
    SmallVector<int64_t> rhsPermutation;
    for (auto dim : rhsBatchingDims) rhsPermutation.push_back(dim);
    for (auto dim : rhsContractingDims) rhsPermutation.push_back(dim);
    for (auto dim : rhsNonBatchNonContractDims) rhsPermutation.push_back(dim);
    
    Value lhsTransposed = transposeTensor(lhs, lhsPermutation, rewriter, op.getLoc());
    Value rhsTransposed = transposeTensor(rhs, rhsPermutation, rewriter, op.getLoc());
    
    SmallVector<int64_t> lhsReshapeShape;
    SmallVector<int64_t> rhsReshapeShape;
    
    int64_t lhsBatchSize = 1;
    for (auto dim : lhsBatchingDims) {
      lhsBatchSize *= lhsShape[dim];
    }
    
    int64_t lhsNonContractSize = 1;
    for (auto dim : lhsNonBatchNonContractDims) {
      lhsNonContractSize *= lhsShape[dim];
    }
    
    int64_t lhsContractSize = 1;
    for (auto dim : lhsContractingDims) {
      lhsContractSize *= lhsShape[dim];
    }
    
    int64_t rhsContractSize = 1;
    for (auto dim : rhsContractingDims) {
      rhsContractSize *= rhsShape[dim];
    }
    
    int64_t rhsNonContractSize = 1;
    for (auto dim : rhsNonBatchNonContractDims) {
      rhsNonContractSize *= rhsShape[dim];
    }
    
    int64_t rhsBatchSize = 1;
    for (auto dim : rhsBatchingDims) {
      rhsBatchSize *= rhsShape[dim];
    }
    
    // Verify that contract dimension sizes match
    if (lhsContractSize != rhsContractSize) {
      return rewriter.notifyMatchFailure(op, 
          "Contract dimension sizes must match: lhs=" + std::to_string(lhsContractSize) + 
          ", rhs=" + std::to_string(rhsContractSize));
    }
    
    // Verify batch dimension sizes match
    if (lhsBatchSize != rhsBatchSize) {
      return rewriter.notifyMatchFailure(op, 
          "Batch dimension sizes must match: lhs=" + std::to_string(lhsBatchSize) + 
          ", rhs=" + std::to_string(rhsBatchSize));
    }
    
    if (!lhsBatchingDims.empty()) {
      // With batch dimension, use BatchMatMul
      // lhs: [batch_size, non_contract_size, contract_size]
      // rhs: [batch_size, contract_size, non_contract_size]
      lhsReshapeShape = {lhsBatchSize, lhsNonContractSize, lhsContractSize};
      rhsReshapeShape = {rhsBatchSize, rhsContractSize, rhsNonContractSize};
    } else {
      // No batch dimension, use MatMul
      lhsReshapeShape = {lhsNonContractSize, lhsContractSize};
      rhsReshapeShape = {rhsContractSize, rhsNonContractSize};
    }
    
    auto lhsReshaped = reshapeTensor(lhsTransposed, lhsReshapeShape, rewriter, op.getLoc());
    auto rhsReshaped = reshapeTensor(rhsTransposed, rhsReshapeShape, rewriter, op.getLoc());
    
    SmallVector<int64_t> matmulResultShape;
    Value matmulResult;
    
    if (!lhsBatchingDims.empty()) {
      // Using BatchMatMul
      matmulResultShape = {lhsBatchSize, lhsNonContractSize, rhsNonContractSize};
      auto matmulResultType = RankedTensorType::get(matmulResultShape, lhsType.getElementType());
      matmulResult = rewriter.create<BatchMatMulOp>(
          op.getLoc(), matmulResultType, lhsReshaped, rhsReshaped, 
          false, false).getResult();
      
    } else {
      // Use MatMul
      matmulResultShape = {lhsNonContractSize, rhsNonContractSize};
      auto matmulResultType = RankedTensorType::get(matmulResultShape, lhsType.getElementType());
      matmulResult = rewriter.create<MatMulOp>(
          op.getLoc(), matmulResultType, lhsReshaped, rhsReshaped, 
          nullptr, false, false).getResult();
    }
    
    auto finalResult = reshapeTensor(matmulResult, 
                                     SmallVector<int64_t>(resultShape.begin(), resultShape.end()),
                                     rewriter, op.getLoc());    
    rewriter.replaceOp(op, finalResult);
    return success();
  }

 private:
  Value transposeTensor(Value input, ArrayRef<int64_t> permutation,
                       ConversionPatternRewriter& rewriter, Location loc) const {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) return input;
    
    bool needTranspose = false;
    for (size_t i = 0; i < permutation.size(); ++i) {
      needTranspose = true;
      break;
    }
    
    if (!needTranspose) return input;
    
    auto inputShape = inputType.getShape();
    SmallVector<int64_t> resultShape;
    for (auto idx : permutation) {
      resultShape.push_back(inputShape[idx]);
    }
    
    auto resultType = RankedTensorType::get(resultShape, inputType.getElementType());
    
    auto permTensorType = RankedTensorType::get(
        {static_cast<int64_t>(permutation.size())}, rewriter.getI64Type());
    auto permAttr = DenseIntElementsAttr::get(permTensorType, permutation);
    auto permConst = rewriter.create<ConstantOp>(loc, permTensorType, permAttr);
    
    return rewriter.create<TransposeOp>(loc, resultType, input, permConst.getResult()).getResult();
  }
  
  Value reshapeTensor(Value input, ArrayRef<int64_t> targetShape,
                     ConversionPatternRewriter& rewriter, Location loc) const {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) return input;
    
    auto inputShape = inputType.getShape();
    bool needReshape = false;
    if (inputShape.size() != targetShape.size()) {
      needReshape = true;
    } else {
      for (size_t i = 0; i < inputShape.size(); ++i) {
        if (inputShape[i] != targetShape[i]) {
          needReshape = true;
          break;
        }
      }
    }
    
    if (!needReshape) return input;
    
    auto resultType = RankedTensorType::get(targetShape, inputType.getElementType());
    
    auto shapeTensorType = RankedTensorType::get(
        {static_cast<int64_t>(targetShape.size())}, rewriter.getI64Type());
    auto shapeAttr = DenseIntElementsAttr::get(shapeTensorType, targetShape);
    auto shapeConst = rewriter.create<ConstantOp>(loc, shapeTensorType, shapeAttr);
    
    return rewriter.create<ReshapeOp>(loc, resultType, input, shapeConst.getResult(),
                                      rewriter.getI64IntegerAttr(0),
                                      rewriter.getI64IntegerAttr(-1)).getResult();
  }
};
class ConvertReshapeOp : public OpConversionPattern<stablehlo::ReshapeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::ReshapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    mlir::Value inputTensor = adaptor.getOperand();
    auto resultType = dyn_cast_or_null<mlir::RankedTensorType>(op.getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "Result must be a ranked tensor");
    }
    auto targetShape = resultType.getShape();
    auto shapeAttr = rewriter.getI64TensorAttr(targetShape);
    auto shapeTensorType =
        mlir::RankedTensorType::get({static_cast<int64_t>(targetShape.size())},
                                    rewriter.getIntegerType(64));
    auto shapeTensor = rewriter
                           .create<mlir::air::ConstantOp>(
                               op.getLoc(), shapeTensorType, shapeAttr)
                           .getResult();
    auto newReshapeOp = rewriter.create<mlir::air::ReshapeOp>(
        op.getLoc(), op.getType(), inputTensor, shapeTensor,
        rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(-1)
    );
    rewriter.replaceOp(op, newReshapeOp.getResult());
    return success();
  };
};
class ConvertReduceOp : public OpConversionPattern<stablehlo::ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::ReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto& body = op.getBody().front();
    auto it = body.begin();
    auto termIt = body.end();
    --termIt;

    if (it == termIt) {
      return rewriter.notifyMatchFailure(
          op, "Body must contain exactly one computation op.");
    }
    auto computeIt = it;
    auto afterComputeIt = std::next(computeIt);
    if (computeIt != body.begin() || afterComputeIt != termIt) {
      return rewriter.notifyMatchFailure(
          op, "Body must contain exactly one computation op.");
    }

    auto* singleComputeOp = &(*computeIt);

    bool isSum = isa<stablehlo::AddOp>(singleComputeOp);
    bool isMax = isa<stablehlo::MaxOp>(singleComputeOp);
    bool isMin = isa<stablehlo::MinOp>(singleComputeOp);
    bool isLogicalAnd = isa<stablehlo::AndOp>(singleComputeOp);
    bool isLogicalOr = isa<stablehlo::OrOp>(singleComputeOp);

    if (!isSum && !isMax && !isMin && !isLogicalAnd && !isLogicalOr) {
      return success();
    }

    auto inputsRange = adaptor.getInputs();
    if (inputsRange.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Only single-input reduce is supported.");
    }
    mlir::Value inputs = inputsRange[0];

    auto dimensionsArrayRef = op.getDimensions();
    if (dimensionsArrayRef.empty()) {
      return rewriter.notifyMatchFailure(op,
                                         "Dimensions list cannot be empty.");
    }
    auto elementType = rewriter.getIntegerType(64);
    llvm::SmallVector<int64_t, 4> shapeVector;
    shapeVector.push_back(dimensionsArrayRef.size());
    auto tensorType = mlir::RankedTensorType::get(shapeVector, elementType);

    auto dimensionsAttr = rewriter.getI64TensorAttr(dimensionsArrayRef);

    auto axesValue = rewriter
                         .create<mlir::air::ConstantOp>(op.getLoc(), tensorType,
                                                        dimensionsAttr)
                         .getResult();

    bool keepDims = false;
    auto keepDimsAttr = rewriter.getBoolAttr(keepDims);

    mlir::Operation* newOp = nullptr;
    if (isSum) {
      newOp = rewriter.create<mlir::air::ReduceSumOp>(
          op.getLoc(), op.getResultTypes(), inputs, axesValue, keepDimsAttr);
    } else if (isMax) {
      newOp = rewriter.create<mlir::air::ReduceMaxOp>(
          op.getLoc(), op.getResultTypes(), inputs, axesValue, keepDimsAttr);
    } else if (isMin) {
      newOp = rewriter.create<mlir::air::ReduceMinOp>(
          op.getLoc(), op.getResultTypes(), inputs, axesValue, keepDimsAttr);
    } else if (isLogicalAnd) {
      newOp = rewriter.create<mlir::air::ReduceAllOp>(
          op.getLoc(), op.getResultTypes(), inputs, axesValue, keepDimsAttr);
    } else if (isLogicalOr) {
      newOp = rewriter.create<mlir::air::ReduceAnyOp>(
          op.getLoc(), op.getResultTypes(), inputs, axesValue, keepDimsAttr);
    }
    if (newOp) {
      rewriter.replaceOp(op, newOp->getResults());
    }
    return success();
  };
};
class ConvertCastOp : public OpConversionPattern<stablehlo::ConvertOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::ConvertOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto inputValue = adaptor.getOperands()[0];
    auto dstType = op.getType();
    if (!dstType) {
      return rewriter.notifyMatchFailure(
          op, "failed to get destination type for convert op");
    }
    rewriter.replaceOpWithNewOp<CastOp>(op, dstType, inputValue,
                                         TypeAttr::get(dstType));
    return success();
  }
};
class ConvertAndOp : public OpConversionPattern<stablehlo::AndOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::AndOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<AndOp>(op, op.getType(), op.getLhs(),
                                       op.getRhs());
    return success();
  };
};
class ConvertCompareOp : public OpConversionPattern<stablehlo::CompareOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::CompareOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto direction = op.getComparisonDirection();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto resultType = op.getType();

    if (direction == stablehlo::ComparisonDirection::EQ) {
      rewriter.replaceOpWithNewOp<EqualOp>(op, resultType, lhs, rhs);
    } else if (direction == stablehlo::ComparisonDirection::NE) {
      rewriter.replaceOpWithNewOp<NotEqualOp>(op, resultType, lhs, rhs);
    } else if (direction == stablehlo::ComparisonDirection::GT) {
      rewriter.replaceOpWithNewOp<GreaterOp>(op, resultType, lhs, rhs);
    } else if (direction == stablehlo::ComparisonDirection::GE) {
      rewriter.replaceOpWithNewOp<GreaterEqualOp>(op, resultType, lhs, rhs);
    } else if (direction == stablehlo::ComparisonDirection::LT) {
      rewriter.replaceOpWithNewOp<LessOp>(op, resultType, lhs, rhs);
    } else if (direction == stablehlo::ComparisonDirection::LE) {
      rewriter.replaceOpWithNewOp<LessEqualOp>(op, resultType, lhs, rhs);
    } else {
      return failure();
    }
    return success();
  };
};
class ConvertCosineOp : public OpConversionPattern<stablehlo::CosineOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::CosineOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<CosOp>(op, op.getType(), op.getOperand());
    return success();
  };
};
class ConvertGatherOp : public OpConversionPattern<stablehlo::GatherOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::GatherOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto dimensionNumbers = op.getDimensionNumbers();
    auto sliceSizes = op.getSliceSizes();
    
    auto offsetDims = dimensionNumbers.getOffsetDims();
    auto collapsedSliceDims = dimensionNumbers.getCollapsedSliceDims();
    auto startIndexMap = dimensionNumbers.getStartIndexMap();
    auto indexVectorDim = dimensionNumbers.getIndexVectorDim();
    
    auto operand = op.getOperand();
    auto startIndices = op.getStartIndices();
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    auto indicesType = dyn_cast<RankedTensorType>(startIndices.getType());
    
    if (!operandType || !indicesType) {
      return rewriter.notifyMatchFailure(op, "Operand and indices must be ranked tensors");
    }
    
    // Simple case: single axis gather
    // Conditions:
    // 1. start_index_map has only one element
    // 2. collapsed_slice_dims has only one element
    // 3. collapsed_slice_dims[0] == start_index_map[0]
    // 4. slice_sizes is 1 in collapsed dimension
    
    if (startIndexMap.size() == 1 && 
        collapsedSliceDims.size() == 1 &&
        collapsedSliceDims[0] == startIndexMap[0]) {
      
      int64_t axis = startIndexMap[0];
      
      // Verify that slice_sizes is 1 in the collapsed dimension
      if (sliceSizes[axis] != 1) {
        return rewriter.notifyMatchFailure(op, 
            "Slice size on collapsed dimension must be 1, got " + 
            std::to_string(sliceSizes[axis]));
      }
      
      // Create axis constant
      auto axisType = RankedTensorType::get({}, rewriter.getI64Type());
      auto axisAttr = DenseIntElementsAttr::get(axisType, {axis});
      auto axisConst = rewriter.create<ConstantOp>(op.getLoc(), axisType, axisAttr);
      
      // Process indices (may require squeeze)
      Value indices = startIndices;
      auto indicesShape = indicesType.getShape();
      int64_t indicesRank = indicesType.getRank();
      
      // If index_vector_dim is the last dimension and size is 1, then squeeze
      if (indexVectorDim == indicesRank - 1 && indicesShape.back() == 1) {
        SmallVector<int64_t> newShape(indicesShape.begin(), indicesShape.end() - 1);
        
        auto newIndicesType = RankedTensorType::get(newShape, indicesType.getElementType());
        auto shapeType = RankedTensorType::get(
            {static_cast<int64_t>(newShape.size())}, rewriter.getI64Type());
        auto shapeAttr = DenseIntElementsAttr::get(shapeType, newShape);
        auto shapeConst = rewriter.create<ConstantOp>(op.getLoc(), shapeType, shapeAttr);
        
        indices = rewriter.create<ReshapeOp>(op.getLoc(), newIndicesType, indices, 
                                             shapeConst.getResult(),
                                             rewriter.getI64IntegerAttr(0),
                                             rewriter.getI64IntegerAttr(-1)).getResult();
      }
      
      // Create GatherV2
      auto gatherResult = rewriter.create<GatherV2Op>(
          op.getLoc(), op.getType(), operand, indices, 
          axisConst.getResult(), nullptr);
      rewriter.replaceOp(op, gatherResult.getResult());
      return success();
    }
    
    // Complications: More complex conversions are required
    return rewriter.notifyMatchFailure(op, 
        "Complex gather not yet supported. " +
        std::string("start_index_map size: ") + std::to_string(startIndexMap.size()) +
        ", collapsed_slice_dims size: " + std::to_string(collapsedSliceDims.size()));
  }
};
class ConvertPowOp : public OpConversionPattern<stablehlo::PowOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::PowOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<PowOp>(op, op.getType(), op.getLhs(),
                                        op.getRhs());
    return success();
  };
};
class ConvertRsqrtOp : public OpConversionPattern<stablehlo::RsqrtOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::RsqrtOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<RsqrtOp>(op, op.getType(), op.getOperand());
    return success();
  };
};
class ConvertSelectOp : public OpConversionPattern<stablehlo::SelectOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::SelectOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<SelectOp>(op, op.getType(), op.getPred(),
                                           op.getOnTrue(), op.getOnFalse());
    return success();
  };
};
class ConvertSineOp : public OpConversionPattern<stablehlo::SineOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::SineOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<SinOp>(op, op.getType(), op.getOperand());
    return success();
  };
};
class ConvertSliceOp : public OpConversionPattern<stablehlo::SliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::SliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto operand = op.getOperand();
    auto startIndices = op.getStartIndices();
    auto limitIndices = op.getLimitIndices();
    auto strides = op.getStrides();
    auto resultType = op.getType();

    for (auto stride : strides) {
      if (stride != 1) {
        return failure();
      }
    }

    SmallVector<int64_t> startValues(startIndices.begin(), startIndices.end());
    SmallVector<int64_t> limitValues(limitIndices.begin(), limitIndices.end());

    SmallVector<int64_t> sizeValues;
    for (size_t i = 0; i < startValues.size(); ++i) {
      sizeValues.push_back(limitValues[i] - startValues[i]);
    }

    auto startTensorType = RankedTensorType::get(
        {static_cast<int64_t>(startValues.size())}, rewriter.getI64Type());
    auto startTensorAttr = DenseIntElementsAttr::get(startTensorType, startValues);
    auto startConstOp = rewriter.create<ConstantOp>(
        op.getLoc(), startTensorType, startTensorAttr);

    auto sizeTensorType = RankedTensorType::get(
        {static_cast<int64_t>(sizeValues.size())}, rewriter.getI64Type());
    auto sizeTensorAttr = DenseIntElementsAttr::get(sizeTensorType, sizeValues);
    auto sizeConstOp = rewriter.create<ConstantOp>(
        op.getLoc(), sizeTensorType, sizeTensorAttr);

    rewriter.replaceOpWithNewOp<SliceOp>(op, resultType, operand,
                                         startConstOp.getResult(),
                                         sizeConstOp.getResult());
    return success();
  };
};
class ConvertTransposeOp : public OpConversionPattern<stablehlo::TransposeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::TransposeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto operand = op.getOperand();
    auto permutation = op.getPermutation();
    auto resultType = op.getType();

    SmallVector<int64_t> permValues;
    for (auto val : permutation) {
      permValues.push_back(val);
    }

    auto permTensorType = RankedTensorType::get(
        {static_cast<int64_t>(permValues.size())}, rewriter.getI64Type());
    auto permTensorAttr = DenseIntElementsAttr::get(permTensorType, permValues);
    auto permConstOp = rewriter.create<ConstantOp>(
        op.getLoc(), permTensorType, permTensorAttr);

    rewriter.replaceOpWithNewOp<TransposeOp>(op, resultType, operand,
                                              permConstOp.getResult());
    return success();
  };
};
}  // namespace

struct LegalizeToMairPass
    : PassWrapper<LegalizeToMairPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeToMairPass)
  StringRef getArgument() const override { return "stablehlo-to-mair"; }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<AirDialect>();
  }
  void runOnOperation() final;
};

void LegalizeToMairPass::runOnOperation() {
  MLIRContext& context = getContext();

  ConversionTarget target(context);

  target.addLegalDialect<AirDialect>();
  target.addIllegalDialect<stablehlo::StablehloDialect>();

  target.addLegalOp<stablehlo::ReturnOp>();

  RewritePatternSet patterns(&getContext());
  patterns.add<ConvertAddOp>(&getContext());
  patterns.add<ConvertMulOp>(&getContext());
  patterns.add<ConvertBroadcastInDimOp>(&getContext());
  patterns.add<ConvertConstantOp>(&getContext());
  patterns.add<ConvertTanhOp>(&getContext());
  patterns.add<ConvertAllReduceOp>(&getContext());
  patterns.add<ConvertAllGatherOp>(&getContext());
  patterns.add<ConvertDivOp>(&getContext());
  patterns.add<ConvertExpOp>(&getContext());
  patterns.add<ConvertMaxOp>(&getContext());
  patterns.add<ConvertNegOp>(&getContext());
  patterns.add<ConvertSubOp>(&getContext());
  patterns.add<ConvertMatMulOp>(&getContext());
  patterns.add<ConvertReshapeOp>(&getContext());
  patterns.add<ConvertReduceOp>(&getContext());
  patterns.add<ConvertConcatV2Op>(&getContext());
  patterns.add<ConvertCastOp>(&getContext());
  patterns.add<ConvertAndOp>(&getContext());
  patterns.add<ConvertCompareOp>(&getContext());
  patterns.add<ConvertCosineOp>(&getContext());
  patterns.add<ConvertGatherOp>(&getContext());
  patterns.add<ConvertPowOp>(&getContext());
  patterns.add<ConvertRsqrtOp>(&getContext());
  patterns.add<ConvertSelectOp>(&getContext());
  patterns.add<ConvertSineOp>(&getContext());
  patterns.add<ConvertSliceOp>(&getContext());
  patterns.add<ConvertTransposeOp>(&getContext());

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    getOperation().emitError("stablehlo to mair legalization failed.");
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createLegalizeStablehloToGePass() {
  return std::make_unique<LegalizeToMairPass>();
}

}  // namespace air
}  // end namespace mlir