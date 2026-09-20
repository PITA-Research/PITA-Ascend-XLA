/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_CUMPROD_H_
#define OP_API_INC_LEVEL2_ACLNN_CUMPROD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnCumprod calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * Operator function: Calculate the cumulative product of the input tensor (Tensor) input along the specified dimension. It returns a new tensor with the same shape as the input tensor
 * @param [in] input: aclTensor on the device side, indicating the input tensor to be used for cumulative product operation. The data type supports DT_FLOAT32, DT_FLOAT16, DT_BFLOAT16, DT_DOUBLE,
    DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_COMPLEX128, DT_COMPLEX64
 * @param [in] dim: aclScalar on the NPU device side, used to specify along which dimension the cumulative product operation is performed. The data type supports DT_INT32
 * @param [in] dtype aclDataType on the host side, the data type of the output tensor, which needs to be consistent with the out data type
 * @param [in] out: Returns a new tensor with the same shape as the input tensor, containing the cumulative product.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnCumprodGetWorkspaceSize(
    const aclTensor *input, const aclScalar *dim, const aclDataType dtype,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * The first interface of @brief aclnnCumprod calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * Operator function: Calculate the cumulative product of the input tensor (Tensor) input along the specified dimension. It returns a new tensor with the same shape as the input tensor
 * @param [in] input: aclTensor on the device side, indicating the input tensor to be used for cumulative product operation. The data type supports DT_FLOAT32, DT_FLOAT16, DT_BFLOAT16, DT_DOUBLE,
    DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_COMPLEX128, DT_COMPLEX64
 * @param [in] dim: aclScalar on the NPU device side, used to specify along which dimension the cumulative product operation is performed. The data type supports DT_INT32
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnInplaceCumprodGetWorkspaceSize(
    aclTensor *input, const aclScalar *dim, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief: The second interface of aclnnCumprod, used to perform calculations
 * @domain aclnn_ops_train
 * Operator function: Calculate the cumulative product of the input
 * This operator corresponds to the underlying tf operator
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnCumprodGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCumprod(void *workspace, uint64_t workspaceSize,
                                   aclOpExecutor *executor, aclrtStream stream);

/**
 * @brief: The second interface of aclnnCumprod is used to perform calculations and update input parameters in place.
 * @domain aclnn_ops_train
 * Operator function: Calculate the cumulative product of the input
 * This operator corresponds to the underlying tf operator
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceCumprodGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceCumprod(void *workspace,
                                          uint64_t workspaceSize,
                                          aclOpExecutor *executor,
                                          aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_CUMPROD_H_