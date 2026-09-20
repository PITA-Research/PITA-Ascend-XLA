/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_REPEAT_INTERLEAVE_H_
#define OP_API_INC_LEVEL2_ACLNN_REPEAT_INTERLEAVE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

// No dim, tensor repeats
/**
 * The first interface of @brief aclnnRepeatInterleave calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the repeatinterleave operation on the input Tensor
 * @param [in] self: aclTensor on the NPU device side. The data type supports UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, and BFLOAT16 types.
 * Supports empty tensors and non-continuous tensors. The data format supports ND.
 * @param [in] repeats: aclTensor on the NPU device side. The data type supports INT64. repeats can only be 0D/1D Tensor. If it is a 1D Tensor, then
 * The size of repeats must be 1 or the number of elements of self. Supports empty tensors and non-continuous tensors. The data format supports ND.
 * @param [in] outputSize: The final size of the tensor after repetition. The data type is INT64. When there is only one element in repeats, outputSize =
 * The number of elements of self * the value of repeats. When there are multiple values ​​in repeats, outputSize = the sum of the values ​​of repeats.
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, and FLOAT32 types. data
 * The type needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnRepeatInterleaveGetWorkspaceSize(
    const aclTensor* self, const aclTensor* repeats, int64_t outputSize,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnRepeatInterleave is used to perform calculations
 *
 * Operator function: Complete the repeatinterleave operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnRepeatInterleaveGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRepeatInterleave(void* workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);

// There are dim, tensor repeats
/**
 * The first interface of @brief aclnnRepeatInterleaveWithDim calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the repeatinterleave operation on the input Tensor
 * @param [in] self: aclTensor on the NPU device side. The data type supports UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, and BFLOAT16 types.
 * Supports empty tensors and non-continuous tensors. The data format supports ND.
 * @param [in] repeats: aclTensor on the NPU device side. The data type supports INT64. repeats can only be 0D/1D tensor. If it is a 1D tensor, then
 * The size of repeats must be 1 or the size of the dim dimension of self. Supports empty tensors and non-continuous tensors. The data format supports ND.
 * @param [in] dim: The dimension to be repeated, the data type is INT64. The range is [-the number of dimensions of self, the number of dimensions of self-1].
 * @param [in] outputSize:
 * The final size of the dim dimension after iteration. The data type is INT64. If there are multiple values in repeats, the outputSize value must be repeats
 * The summation. If repeats has only one element, the outputSize value must be the dim dimension size of repeats * self.
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, and FLOAT32 types. data
 * The type needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnRepeatInterleaveWithDimGetWorkspaceSize(
    const aclTensor* self, const aclTensor* repeats, int64_t dim,
    int64_t outputSize, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnRepeatInterleaveWithDim is used to perform calculations
 *
 * Operator function: Complete the repeatinterleave operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnRepeatInterleaveWithDimGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRepeatInterleaveWithDim(void* workspace,
                                                   uint64_t workspaceSize,
                                                   aclOpExecutor* executor,
                                                   aclrtStream stream);

// no dim, int repeats
/**
 * The first interface of @brief aclnnRepeatInterleaveInt calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the repeatinterleave operation on the input Tensor
 * @param [in] self: aclTensor on the NPU device side. The data type supports UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, and BFLOAT16 types.
 * Supports empty tensors and non-continuous tensors. The data format supports ND.
 * @param [in] repeats: The number of repetitions. The data type is INT64. The value of repeats must be a natural number.
 * @param [in] outputSize: The final size of the tensor after repetition. The data type is INT64. outputSize must be equal to the number of elements of self * the value of repeats.
 * @param [in] out: aclTensor on the NPU device side. The data type supports UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, and FLOAT32 types.
 * The data type needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnRepeatInterleaveIntGetWorkspaceSize(
    const aclTensor* self, int64_t repeats, int64_t outputSize, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnRepeatInterleaveInt is used to perform calculations
 *
 * Operator function: Complete the repeatinterleave operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, determined by the first interface aclnnRepeatInterleaveIntGetWorkspaceSize
 * Get.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRepeatInterleaveInt(void* workspace,
                                               uint64_t workspaceSize,
                                               aclOpExecutor* executor,
                                               aclrtStream stream);

// There are dim, int repeats
/**
 * The first interface of @brief aclnnRepeatInterleaveIntWithDim calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the repeatinterleave operation on the input Tensor
 * @param [in] self: aclTensor on the NPU device side. The data type supports UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, and BFLOAT16 types.
 * Supports empty tensors and non-continuous tensors. The data format supports ND.
 * @param [in] repeats: The number of repetitions. The data type is INT64. The value of repeats must be a natural number.
 * @param [in] dim: The dimension to be repeated, the data type is INT64. The range is [-the number of dimensions of self, the number of dimensions of self-1].
 * @param [in] outputSize: The final size of the dim dimension after repetition. The data type is INT64. outputSize value must be repeats *
 * Dim dimension size of self.
 * @param [in] out: aclTensor on the NPU device side. The data type supports UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, and FLOAT32 types.
 * The data type needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnRepeatInterleaveIntWithDimGetWorkspaceSize(
    const aclTensor* self, int64_t repeats, int64_t dim, int64_t outputSize,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnRepeatInterleaveIntWithDim is used to perform calculations
 *
 * Operator function: Complete the repeatinterleave operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, determined by the first interface
 * aclnnRepeatInterleaveIntWithDimGetWorkspaceSize gets.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRepeatInterleaveIntWithDim(void* workspace,
                                                      uint64_t workspaceSize,
                                                      aclOpExecutor* executor,
                                                      aclrtStream stream);

// repeat_interleave.Tensor
/**
 * The first interface of @brief aclnnRepeatInterleaveTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the repeatinterleave operation on the input Tensor
 * @param [in] repeats: aclTensor on the NPU device side. The data type supports INT32 and INT64. repeats can only be 1D Tensor
 * (Including the scenario with shape=[0,]). Supports non-continuous tensors. The data format supports ND.
 * @param [in] outputSize: The final size of the tensor after repetition. The data type is INT64. outputSize must be equal to the sum of repeats values.
 * @param [in] out: aclTensor on the NPU device side, the data type supports INT64 type. The data type needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnRepeatInterleaveTensorGetWorkspaceSize(
    const aclTensor* repeats, int64_t outputSize, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnRepeatInterleaveTensor, used to perform calculations
 *
 * Operator function: Complete the repeatinterleave operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, determined by the first interface aclnnRepeatInterleaveTensorGetWorkspaceSize
 * Get.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRepeatInterleaveTensor(void* workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_REPEAT_INTERLEAVE_H_