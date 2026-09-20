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
#ifndef OP_API_INC_RSQRT_H_
#define OP_API_INC_RSQRT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnRsqrt calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the operation of finding the reciprocal arithmetic square root of the input Tensor
 * @param [in] self: aclTensor on NPU device side,
 * The data type supports floating point and integer types, the shape is non-empty, and non-continuous Tensor is supported. The data format supports ND and non-continuous Tensor.
 * @param [in] out: aclTensor on the NPU device side, the data type supports floating point type, the shape is the same as self, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnRsqrtGetWorkspaceSize(const aclTensor* self,
                                                 aclTensor* out,
                                                 uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);
/**
 * @brief: The second interface of aclnnRsqrt, used to perform calculations
 *
 * Operator function: Complete the operation of finding the reciprocal arithmetic square root of the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSigmoidGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRsqrt(void* workspace, uint64_t workspaceSize,
                                 aclOpExecutor* executor, aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceRsqrt calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the operation of finding the reciprocal arithmetic square root of the input Tensor
 * @param [in] self: aclTensor on the NPU device side, the data type supports floating point type, the shape is non-empty, supports non-continuous Tensor, and the data format supports ND
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnInplaceRsqrtGetWorkspaceSize(
    aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnInplaceRsqrt(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_RSQRT_H_