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
#ifndef OP_API_INC_SUB_H_
#define OP_API_INC_SUB_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnSub calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: complete subtraction calculation
 * Calculation formula:
 * $$ output_i = self_i - alpha * other_i $$
 *
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to form a mutual derivation relationship with other, and the shape needs to satisfy the broadcast relationship with other.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with other.
 * @param [in] other: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to form a mutual derivation relationship with self, and the shape needs to satisfy the broadcast relationship with self.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with self.
 * @param [in] alpha: aclScalar on the host side, the data type needs to be convertible into the data type derived from self and other.
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to be a convertible data type after self and other are deduced, and the shape needs to be self and other.
 * The data format of the shape after broadcast supports ND, and the data format needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSubGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, const aclScalar* alpha,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnSub is used to perform calculations.
 *
 * Operator function: complete subtraction calculation
 * Calculation formula:
 * $$ output_i = self_i - alpha * other_i $$
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSubGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSub(void* workspace, uint64_t workspaceSize,
                               aclOpExecutor* executor, aclrtStream stream);

/**
 * The first interface of @brief aclnnSubs calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnSubsGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, const aclScalar* alpha,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);
ACLNN_API aclnnStatus aclnnSubs(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor, aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceSub calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: complete subtraction calculation
 * Calculation formula:
 * $$ selfRef_{i} = selfRef_{i} - alpha \times other_{i} $$
 *
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to form a mutual derivation relationship with other, and the shape needs to satisfy the broadcast relationship with other.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with other.
 * @param [in] other: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to form a mutual derivation relationship with self, and the shape needs to satisfy the broadcast relationship with self.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with self.
 * @param [in] alpha: aclScalar on the host side, the data type needs to be convertible into the data type derived from self and other.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceSubGetWorkspaceSize(aclTensor* selfRef,
                                                      const aclTensor* other,
                                                      const aclScalar* alpha,
                                                      uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceSub is used to perform calculations.
 *
 * Operator function: complete subtraction calculation
 * Calculation formula:
 * $$ selfRef_{i} = selfRef_{i} - alpha \times other_{i} $$
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceSubGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceSub(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor,
                                      aclrtStream stream);

/**
 * @brief The first interface of aclnnInplaceSubs calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnInplaceSubsGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, const aclScalar* alpha,
    uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnInplaceSubs(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SUB_H_
