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

#include "xla/xla_npu/dependency/interface/acl_nn_op_interface.h"

#include "xla/xla_npu/dependency/tool/function_loader.h"

namespace xla {
namespace dep {

REGISTER_LIBRARY(libopapi)
REGISTER_FUNCTION(libopapi, aclnnExpandGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnExpand)
REGISTER_FUNCTION(libopapi, aclnnAddGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnAdd)
REGISTER_FUNCTION(libopapi, aclnnTanGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnTan)
REGISTER_FUNCTION(libopapi, aclnnTanhGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnTanh)
REGISTER_FUNCTION(libopapi, aclnnMulGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnMul)
REGISTER_FUNCTION(libopapi, aclnnMatmulGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnMatmul)
REGISTER_FUNCTION(libopapi, aclnnDivGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnDiv)
REGISTER_FUNCTION(libopapi, aclnnPowTensorTensorGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnPowTensorTensor)
REGISTER_FUNCTION(libopapi, aclnnRemainderTensorTensorGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnRemainderTensorTensor)
REGISTER_FUNCTION(libopapi, aclnnSubGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnSub)
REGISTER_FUNCTION(libopapi, aclnnNegGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnNeg)
REGISTER_FUNCTION(libopapi, aclnnExpGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnExp)
REGISTER_FUNCTION(libopapi, aclnnMinimumGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnMinimum)
REGISTER_FUNCTION(libopapi, aclnnMaximumGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnMaximum)
REGISTER_FUNCTION(libopapi, aclnnClampGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnClamp)
REGISTER_FUNCTION(libopapi, aclnnClampTensorGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnClampTensor)
REGISTER_FUNCTION(libopapi, aclnnCatGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnCat)
REGISTER_FUNCTION(libopapi, aclnnReduceSumGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnReduceSum)
REGISTER_FUNCTION(libopapi, aclnnInplaceCopyGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnInplaceCopy)
REGISTER_FUNCTION(libopapi, aclnnAmaxGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnAmax)
REGISTER_FUNCTION(libopapi, aclnnAminGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnAmin)
REGISTER_FUNCTION(libopapi, aclnnMaxDimGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnMaxDim)

// All/Any operation: logical-AND/OR reduction over BOOL tensor
REGISTER_FUNCTION(libopapi, aclnnAllGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnAll)
REGISTER_FUNCTION(libopapi, aclnnAnyGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnAny)

// Rsqrt operation
REGISTER_FUNCTION(libopapi, aclnnRsqrtGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnRsqrt)

// Sin operation
REGISTER_FUNCTION(libopapi, aclnnSinGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnSin)

// Cos operation
REGISTER_FUNCTION(libopapi, aclnnCosGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnCos)

// Transpose/Permute operation
REGISTER_FUNCTION(libopapi, aclnnPermuteGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnPermute)

// Slice operation (V2 version)
REGISTER_FUNCTION(libopapi, aclnnSliceV2GetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnSliceV2)

// Gather operation
REGISTER_FUNCTION(libopapi, aclnnGatherGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnGather)

// Compare operations
REGISTER_FUNCTION(libopapi, aclnnEqTensorGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnEqTensor)
REGISTER_FUNCTION(libopapi, aclnnNeTensorGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnNeTensor)
REGISTER_FUNCTION(libopapi, aclnnLtTensorGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnLtTensor)
REGISTER_FUNCTION(libopapi, aclnnLeTensorGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnLeTensor)
REGISTER_FUNCTION(libopapi, aclnnGtTensorGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnGtTensor)
REGISTER_FUNCTION(libopapi, aclnnGeTensorGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnGeTensor)

// SWhere (Select/Where) operation
REGISTER_FUNCTION(libopapi, aclnnSWhereGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnSWhere)

// BitwiseNot operation
REGISTER_FUNCTION(libopapi, aclnnBitwiseNotGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnBitwiseNot)

// BitwiseAnd/LogicalAnd operation
REGISTER_FUNCTION(libopapi, aclnnBitwiseAndTensorGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnBitwiseAndTensor)

// BitwiseOr operation
REGISTER_FUNCTION(libopapi, aclnnBitwiseOrTensorGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnBitwiseOrTensor)

// BitwiseXor operation
REGISTER_FUNCTION(libopapi, aclnnBitwiseXorTensorGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnBitwiseXorTensor)

// IsFinite operation
REGISTER_FUNCTION(libopapi, aclnnIsFiniteGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnIsFinite)

// Abs operation
REGISTER_FUNCTION(libopapi, aclnnAbsGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnAbs)
REGISTER_FUNCTION(libopapi, aclnnSignGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnSign)
REGISTER_FUNCTION(libopapi, aclnnRoundGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnRound)

// Sqrt operation
REGISTER_FUNCTION(libopapi, aclnnSqrtGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnSqrt)

// Log operation
REGISTER_FUNCTION(libopapi, aclnnLogGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnLog)

// Log1p operation
REGISTER_FUNCTION(libopapi, aclnnLog1pGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnLog1p)

// RightShift operation
REGISTER_FUNCTION(libopapi, aclnnRightShiftGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnRightShift)

// Cast operation
REGISTER_FUNCTION(libopapi, aclnnCastGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnCast)

// IndexSelect operation
REGISTER_FUNCTION(libopapi, aclnnIndexSelectGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnIndexSelect)

// IndexCopy operation
REGISTER_FUNCTION(libopapi, aclnnIndexCopyGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnIndexCopy)

// ScatterNd operation
REGISTER_FUNCTION(libopapi, aclnnScatterNdGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnScatterNd)

// ScatterNdUpdate operation
REGISTER_FUNCTION(libopapi, aclnnScatterNdUpdateGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnScatterNdUpdate)

// Scatter operation
REGISTER_FUNCTION(libopapi, aclnnScatterGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnScatter)

// TfScatterAdd operation
REGISTER_FUNCTION(libopapi, aclnnTfScatterAddGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnTfScatterAdd)

// Sort operation
REGISTER_FUNCTION(libopapi, aclnnSortGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnSort)

// TopK operation
REGISTER_FUNCTION(libopapi, aclnnTopkGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnTopk)

// GatherV2 operation
REGISTER_FUNCTION(libopapi, aclnnGatherV2GetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnGatherV2)

// MaxPool operation for reduce-window max pooling
REGISTER_FUNCTION(libopapi, aclnnMaxPoolGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnMaxPool)

// Cumsum operation for reduce-window cumsum pattern
REGISTER_FUNCTION(libopapi, aclnnCumsumGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnCumsum)

// InplaceFillScalar operation
REGISTER_FUNCTION(libopapi, aclnnInplaceFillScalarGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnInplaceFillScalar)

// ConstantPadNd operation
REGISTER_FUNCTION(libopapi, aclnnConstantPadNdGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnConstantPadNd)

// Atan2 operation
REGISTER_FUNCTION(libopapi, aclnnAtan2GetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnAtan2)

// Cholesky decomposition operation
REGISTER_FUNCTION(libopapi, aclnnLinalgCholeskyGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnLinalgCholesky)

// Triangular solve operation
REGISTER_FUNCTION(libopapi, aclnnTriangularSolveGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnTriangularSolve)

// Flip (Reverse) operation
REGISTER_FUNCTION(libopapi, aclnnFlipGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnFlip)

// Tril operation
REGISTER_FUNCTION(libopapi, aclnnTrilGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnTril)
REGISTER_FUNCTION(libopapi, aclnnInplaceTrilGetWorkspaceSize)
REGISTER_FUNCTION(libopapi, aclnnInplaceTril)

}  // namespace dep
}  // namespace xla
