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
#ifndef OP_API_INC_SINC_H_
#define OP_API_INC_SINC_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnSinc calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the sin operation on the input Tensor
 * @param [in] selfRef: aclTensor on the NPU device side, data type supports INT8, INT16, INT32, INT64, UINT8, BOOL, FLOAT, FLOAT16,
 * DOUBLE, COMPLEX64, COMPLEX128, shape is non-empty, non-continuous Tensor is supported, and the data format supports ND.
 * @param [in] out: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128, shape and self
 * Staying the same, the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnSincGetWorkspaceSize(const aclTensor* self,
                                                aclTensor* out,
                                                uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnSinc is used to perform calculations
 *
 * Operator function: Complete the sin operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSincGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSinc(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor, aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceSinc calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the sin operation on the input Tensor
 * @param [in] selfRef: aclTensor on the NPU device side, data type supports INT8, INT16, INT32, INT64, UINT8, BOOL, FLOAT,
 * FLOAT16, DOUBLE, COMPLEX64, COMPLEX128, shape is non-empty, non-continuous Tensor is supported, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnInplaceSincGetWorkspaceSize(
    aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnInplaceSinc is used to perform calculations
 *
 * Operator function: Complete the sin operation on the input Tensor in place
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceSincGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceSinc(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SINC_H_