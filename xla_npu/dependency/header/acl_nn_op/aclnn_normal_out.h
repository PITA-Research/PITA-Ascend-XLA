/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OP_API_INC_NORMAL_OUT_H_
#define OP_API_INC_NORMAL_OUT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * The first interface of @brief aclnnNormal calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 * @param [in] mean: npu
 * The data type of aclTensor on the device side supports floating-point data types, and the data type needs to form a mutual derivation relationship with std, and the shape needs to satisfy the broadcast relationship with std.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with std.
 * @param [in] std: npu
 * The data type of aclTensor on the device side supports floating point types, and the data type needs to form a mutual derivation relationship with mean, and the shape needs to satisfy the broadcast relationship with mean.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with mean.
 * @param [in] seed: int64_t on the host side, the seed value of the pseudo-random number generator.
 * @param [in] offset: int64_t on the host side, offset of the pseudo-random number generator.
 * @param [out] out: int64_t on the host side, output tensor.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnNormalTensorTensorGetWorkspaceSize(
    const aclTensor* mean, const aclTensor* std, int64_t seed, int64_t offset,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnNormalTensorFloat calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 */
ACLNN_API aclnnStatus aclnnNormalTensorFloatGetWorkspaceSize(
    const aclTensor* mean, float std, int64_t seed, int64_t offset,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnNormalFloatTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 */
ACLNN_API aclnnStatus aclnnNormalFloatTensorGetWorkspaceSize(
    float mean, const aclTensor* std, int64_t seed, int64_t offset,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnNormalFloatFloat calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 */
ACLNN_API aclnnStatus aclnnNormalFloatFloatGetWorkspaceSize(
    float mean, float std, int64_t seed, int64_t offset, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnNormal is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceAddGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnNormalTensorTensor(void* workspace,
                                              uint64_t workspaceSize,
                                              aclOpExecutor* executor,
                                              aclrtStream stream);

ACLNN_API aclnnStatus aclnnNormalTensorFloat(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

ACLNN_API aclnnStatus aclnnNormalFloatTensor(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

ACLNN_API aclnnStatus aclnnNormalFloatFloat(void* workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_NORMAL_OUT_H_
