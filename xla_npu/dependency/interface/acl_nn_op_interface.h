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

#ifndef XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_NN_OP_INTERFACE_H_
#define XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_NN_OP_INTERFACE_H_

#include "xla/xla_npu/dependency/header/acl_nn/acl_meta.h"
#include "xla/xla_npu/dependency/tool/function_loader.h"

#define DECLARE_ACL_FUNC_IN_OPAPI(func_name, args_type, args_name) \
  DECLARE_ACL_FUNC(libopapi, aclnnStatus, func_name, args_type, args_name)

// The formal parameter declarations of all operator execution interfaces are exactly the same
#define DECLARE_ACL_EXEC_FUNC_IN_OPAPI(func_name)                        \
  DECLARE_ACL_FUNC_IN_OPAPI(                                             \
      func_name,                                                         \
      (void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, \
       const aclrtStream stream),                                        \
      (workspace, workspaceSize, executor, stream))

namespace xla {
namespace dep {

DECLARE_ACL_FUNC_IN_OPAPI(aclnnExpandGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* size,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, size, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnExpand)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnAddGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           const aclScalar* alpha, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, other, alpha, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnAdd)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnTanGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnTan)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnTanhGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnTanh)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnMulGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnMul)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnDivGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnDiv)

// Power operation: element-wise base^exponent
// Reference: acl_nn_op/aclnn_pow_tensor_tensor.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnPowTensorTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* exponent,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, exponent, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnPowTensorTensor)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnRemainderTensorTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnRemainderTensorTensor)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnSubGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           const aclScalar* alpha, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, other, alpha, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnSub)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnNegGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnNeg)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnExpGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnExp)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnMinimumGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnMinimum)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnMaximumGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnMaximum)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnClampGetWorkspaceSize,
                          (const aclTensor* self,
                           const aclScalar* clipValueMin,
                           const aclScalar* clipValueMax, aclTensor* out,
                           uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, clipValueMin, clipValueMax, out,
                           workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnClamp)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnClampTensorGetWorkspaceSize,
                          (const aclTensor* self,
                           const aclTensor* clipValueMin,
                           const aclTensor* clipValueMax, aclTensor* out,
                           uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, clipValueMin, clipValueMax, out,
                           workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnClampTensor)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnMatmulGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* mat2,
                           aclTensor* out, int8_t cubeMathType,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, mat2, out, cubeMathType, workspaceSize,
                           executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnMatmul)

DECLARE_ACL_FUNC_IN_OPAPI(aclnnCatGetWorkspaceSize,
                          (const aclTensorList* tensors, int64_t dim,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (tensors, dim, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnCat)

// Reduce sum operation
DECLARE_ACL_FUNC_IN_OPAPI(aclnnReduceSumGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* dims,
                           bool keepDims, aclDataType dtype, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dims, keepDims, dtype, out, workspaceSize,
                           executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnReduceSum)

// Copy operation for reshape
DECLARE_ACL_FUNC_IN_OPAPI(aclnnInplaceCopyGetWorkspaceSize,
                          (aclTensor * selfRef, const aclTensor* src,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (selfRef, src, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnInplaceCopy)

// MaxDim operation for reduce max
DECLARE_ACL_FUNC_IN_OPAPI(aclnnMaxDimGetWorkspaceSize,
                          (const aclTensor* self, int64_t dim, bool keepdim,
                           aclTensor* out, aclTensor* indices,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dim, keepdim, out, indices, workspaceSize,
                           executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnMaxDim)

// Amax operation for reduce max (returns max values only, no indices)
DECLARE_ACL_FUNC_IN_OPAPI(aclnnAmaxGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* dim,
                           bool keepDim, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dim, keepDim, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnAmax)

// Amin operation: min reduction over tensor along dims (mirror of aclnnAmax).
// Used for `minimum` reducer (jnp.min / reduce_min).
// Reference: acl_nn_op/aclnn_amin.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnAminGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* dim,
                           bool keepDim, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dim, keepDim, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnAmin)

// All operation: logical-AND reduction over BOOL tensor along dims.
// Used for PRED reduce with `and` reducer (jnp.all / allclose 的最终归约).
// Reference: acl_nn_op/aclnn_all.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnAllGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* dim,
                           bool keepdim, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dim, keepdim, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnAll)

// Any operation: logical-OR reduction over BOOL tensor along dims.
// Used for PRED reduce with `or` reducer (jnp.any).
// Reference: acl_nn_op/aclnn_any.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnAnyGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* dim,
                           bool keepdim, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dim, keepdim, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnAny)

// ==================== New operator interface declaration ====================

// Rsqrt operation: 1/sqrt(x)
// Reference: acl_nn_op/aclnn_rsqrt.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnRsqrtGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnRsqrt)

// Sin operation: sin(x)
// Reference: acl_nn_op/aclnn_sin.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnSinGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnSin)

// Cos operation: cos(x)
// Reference: acl_nn_op/aclnn_cos.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnCosGetWorkspaceSize,
                          (const aclTensor* input, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (input, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnCos)

// Transpose/Permute operation
// Reference: acl_nn_op/aclnn_permute.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnPermuteGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* dims,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, dims, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnPermute)

// Slice operation (V2 version with array parameters)
// Reference: acl_nn_op/aclnn_slice_v2.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnSliceV2GetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* starts,
                           const aclIntArray* ends, const aclIntArray* axes,
                           const aclIntArray* steps, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, starts, ends, axes, steps, out, workspaceSize,
                           executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnSliceV2)

// Gather operation
// Reference: acl_nn_op/aclnn_gather.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnGatherGetWorkspaceSize,
                          (const aclTensor* self, int64_t dim,
                           const aclTensor* index, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dim, index, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnGather)

// Compare operations (Tensor-Tensor comparison)
// Reference: acl_nn_op/aclnn_eq_tensor.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnEqTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnEqTensor)

// Reference: acl_nn_op/aclnn_ne_tensor.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnNeTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnNeTensor)

// Reference: acl_nn_op/aclnn_lt_tensor.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnLtTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnLtTensor)

// Reference: acl_nn_op/aclnn_le_tensor.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnLeTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnLeTensor)

// Reference: acl_nn_op/aclnn_gt_tensor.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnGtTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnGtTensor)

// Reference: acl_nn_op/aclnn_ge_tensor.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnGeTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnGeTensor)

// SWhere (Select/Where) operation: select elements from self or other based on condition
// Reference: acl_nn_op/aclnn_s_where.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnSWhereGetWorkspaceSize,
                          (const aclTensor* condition, const aclTensor* self,
                           const aclTensor* other, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (condition, self, other, out, workspaceSize,
                           executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnSWhere)

// BitwiseNot operation: element-wise logical/bitwise NOT
// Reference: acl_nn_op/aclnn_bitwise_not.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnBitwiseNotGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnBitwiseNot)

// BitwiseAnd/LogicalAnd operation: element-wise logical/bitwise AND
// Reference: acl_nn_op/aclnn_bitwise_and_tensor.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnBitwiseAndTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnBitwiseAndTensor)

// BitwiseOr operation: element-wise bitwise OR
// Reference: acl_nn_op/aclnn_bitwise_or_tensor.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnBitwiseOrTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnBitwiseOrTensor)

// BitwiseXor operation: element-wise bitwise XOR
// Reference: acl_nn_op/aclnn_bitwise_xor_tensor.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnBitwiseXorTensorGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnBitwiseXorTensor)

// IsFinite operation: test if each element is finite (not NaN, not Inf)
// Reference: acl_nn_op/aclnn_isfinite.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnIsFiniteGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnIsFinite)

// Abs operation: element-wise absolute value
// Reference: acl_nn_op/aclnn_abs.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnAbsGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnAbs)

// Sign operation: element-wise sign
// Reference: acl_nn_op/aclnn_sign.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnSignGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnSign)

// Round operation: element-wise round to nearest
// Reference: acl_nn_op/aclnn_round.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnRoundGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnRound)

// Sqrt operation: element-wise square root
// Reference: acl_nn_op/aclnn_sqrt.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnSqrtGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnSqrt)

// Log operation: element-wise natural logarithm
// Reference: acl_nn_op/aclnn_log.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnLogGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnLog)

// Log1p operation: element-wise log(1+x)
// Reference: acl_nn_op/aclnn_log1p.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnLog1pGetWorkspaceSize,
                          (const aclTensor* self, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnLog1p)

// RightShift operation: element-wise bitwise right shift
// Reference: acl_nn_op/aclnn_right_shift.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnRightShiftGetWorkspaceSize,
                          (const aclTensor* input, const aclTensor* shiftBits,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (input, shiftBits, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnRightShift)

// Cast operation: convert self into dtype and store into out.
// Reference: acl_nn_op/aclnn_cast.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnCastGetWorkspaceSize,
                          (const aclTensor* self, const aclDataType dtype,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, dtype, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnCast)

// IndexSelect operation: gather elements along specified dimension
// Reference: acl_nn_op/aclnn_index_select.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnIndexSelectGetWorkspaceSize,
                          (const aclTensor* self, int64_t dim,
                           const aclTensor* index, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dim, index, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnIndexSelect)

// IndexCopy operation: copy source into self along a 1D index on `dim`.
// Reference: acl_nn_op/aclnn_index_copy.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnIndexCopyGetWorkspaceSize,
                          (aclTensor* selfRef, int64_t dim,
                           const aclTensor* index, const aclTensor* source,
                           aclTensor* outRef, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (selfRef, dim, index, source, outRef, workspaceSize,
                           executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnIndexCopy)

// ScatterNd operation: scatter updates into data at specified indices (out-of-place)
// Reference: acl_nn_op/aclnn_scatter_nd.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnScatterNdGetWorkspaceSize,
                          (const aclTensor* data, const aclTensor* indices,
                           const aclTensor* updates, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (data, indices, updates, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnScatterNd)

// Scatter operation: scatter src along one dimension.
// Reference: acl_nn_op/aclnn_scatter.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnScatterGetWorkspaceSize,
                          (const aclTensor* self, int64_t dim,
                           const aclTensor* index, const aclTensor* src,
                           int64_t reduce, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dim, index, src, reduce, out, workspaceSize,
                           executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnScatter)

// ScatterNdUpdate operation: in-place scatter update
// Reference: acl_nn_op/aclnn_scatter_nd_update.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnScatterNdUpdateGetWorkspaceSize,
                          (aclTensor* varRef, const aclTensor* indices,
                           const aclTensor* updates, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (varRef, indices, updates, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnScatterNdUpdate)

// TfScatterAdd operation: scatter-add updates into varRef at specified indices (in-place)
// Reference: acl_nn_op/aclnn_tf_scatter_add.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnTfScatterAddGetWorkspaceSize,
                          (aclTensor* varRef, const aclTensor* indices,
                           const aclTensor* updates, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (varRef, indices, updates, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnTfScatterAdd)

// Sort operation: sort tensor along a dimension, returning sorted values and indices
// Reference: acl_nn_op/aclnn_sort.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnSortGetWorkspaceSize,
                          (const aclTensor* self, bool stable, int64_t dim,
                           bool descending, aclTensor* valuesOut,
                           aclTensor* indicesOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, stable, dim, descending, valuesOut, indicesOut,
                           workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnSort)

// TopK operation: select top-k values and indices along a dimension
// Reference: acl_nn_op/aclnn_topk.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnTopkGetWorkspaceSize,
                          (const aclTensor* self, int64_t k, int64_t dim,
                           bool largest, bool sorted, aclTensor* valuesOut,
                           aclTensor* indicesOut, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, k, dim, largest, sorted, valuesOut,
                           indicesOut, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnTopk)

// GatherV2 operation: gather elements along specified dimension
// Reference: acl_nn_op/aclnn_gather_v2.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnGatherV2GetWorkspaceSize,
                          (const aclTensor* self, int64_t dim,
                           const aclTensor* index, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, dim, index, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnGatherV2)

// Atan2 operation: element-wise arctangent of y/x with sign awareness
// Reference: acl_nn_op/aclnn_atan2.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnAtan2GetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* other,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, other, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnAtan2)

// MaxPool operation for reduce-window max pooling
// Reference: acl_nn_op/aclnn_max_pool.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnMaxPoolGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* kernelShape,
                           const aclIntArray* strides, const int64_t autoPad,
                           const aclIntArray* pads, const aclIntArray* dilations,
                           const int64_t ceilMode, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, kernelShape, strides, autoPad, pads, dilations,
                           ceilMode, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnMaxPool)

// Cumsum operation for reduce-window cumsum pattern
// Reference: acl_nn_op/aclnn_cumsum.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnCumsumGetWorkspaceSize,
                          (const aclTensor* self, int64_t dim, aclDataType dtype,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, dim, dtype, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnCumsum)

// InplaceFillScalar operation
// Reference: acl_nn_op/aclnn_fill_scalar.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnInplaceFillScalarGetWorkspaceSize,
                          (aclTensor* selfRef, const aclScalar* value,
                           uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (selfRef, value, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnInplaceFillScalar)

// ConstantPadNd operation for HLO pad
// Reference: acl_nn_op/aclnn_constant_pad_nd.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnConstantPadNdGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* pad,
                           const aclScalar* value, aclTensor* out,
                           uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, pad, value, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnConstantPadNd)

// Cholesky decomposition operation
// Reference: aclnnop/aclnn_linalg_cholesky.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnLinalgCholeskyGetWorkspaceSize,
                          (const aclTensor* self, bool upper, aclTensor* out,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, upper, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnLinalgCholesky)

// Flip (Reverse) operation: reverse elements along specified dimensions
// Reference: acl_nn_op/aclnn_flip.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnFlipGetWorkspaceSize,
                          (const aclTensor* self, const aclIntArray* dims,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, dims, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnFlip)

// Tril operation: lower triangular part
// Reference: aclnnop/aclnn_tril.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnTrilGetWorkspaceSize,
                          (const aclTensor* self, int64_t diagonal,
                           aclTensor* out, uint64_t* workspaceSize,
                           aclOpExecutor** executor),
                          (self, diagonal, out, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnTril)

// Inplace Tril operation
// Reference: aclnnop/aclnn_tril.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnInplaceTrilGetWorkspaceSize,
                          (const aclTensor* selfRef, int64_t diagonal,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (selfRef, diagonal, workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnInplaceTril)

// Triangular solve operation
// Reference: aclnnop/aclnn_triangular_solve.h
DECLARE_ACL_FUNC_IN_OPAPI(aclnnTriangularSolveGetWorkspaceSize,
                          (const aclTensor* self, const aclTensor* A,
                           bool upper, bool transpose, bool unitriangular,
                           aclTensor* xOut, aclTensor* mOut,
                           uint64_t* workspaceSize, aclOpExecutor** executor),
                          (self, A, upper, transpose, unitriangular, xOut, mOut,
                           workspaceSize, executor))

DECLARE_ACL_EXEC_FUNC_IN_OPAPI(aclnnTriangularSolve)

}  // namespace dep
}  // namespace xla

#endif  // XLA_XLA_NPU_DEPENDENCY_INTERFACE_ACL_NN_OP_INTERFACE_H_
