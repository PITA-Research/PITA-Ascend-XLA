/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef OP_API_INC_FUSED_LINEAR_ONLINE_MAX_SUM_H_
#define OP_API_INC_FUSED_LINEAR_ONLINE_MAX_SUM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnFusedLinearOnlineMaxSum calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * @param [in] input: aclTensor on the NPU device side, the data type supports FLOAT16, BFLOAT16, non-continuous Tensor, and the data format supports ND.
 * @param [in] weight: aclTensor on the NPU device side, the data type supports FLOAT16, BFLOAT16, non-continuous Tensor, and the data format supports ND.
 * @param [in] target: aclTensor on the NPU device side, the data type supports INT32, INT64, non-continuous Tensor, and the data format supports ND.
 * @param [in] vocabStartIndex: int64 type on the host side, indicating the start index allocated to this card.
 * @param [in] vocabEndIndex: int64 type on the host side, indicating the end index allocated to this card.
 * @param [in] logitsMaxLocalOut: aclTensor on the NPU device side, the data type supports FLOAT, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] sumExpLogitsLocalOut: aclTensor on the NPU device side, the data type supports FLOAT, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] predictedLogitsLocalOut: aclTensor on the NPU device side, the data type supports FLOAT, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] targetMaskOut: aclTensor on the NPU device side, the data type supports UINT8, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] maskedTargetOut: aclTensor on the NPU device side, the data type supports INT32, INT64, non-continuous Tensor, and the data format supports ND.
 * @param [in] vocabParallelLogitsOutOptional: aclTensor on the NPU device side, the data type supports FLOAT16, BFLOAT16, non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFusedLinearOnlineMaxSumGetWorkspaceSize(
    const aclTensor* input, const aclTensor* weight, const aclTensor* target,
    int64_t vocabStartIndex, int64_t vocabEndIndex,
    aclTensor* logitsMaxLocalOut, aclTensor* sumExpLogitsLocalOut,
    aclTensor* predictedLogitsLocalOut, aclTensor* targetMaskOut,
    aclTensor* maskedTargetOut, aclTensor* vocabParallelLogitsOutOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnFusedLinearOnlineMaxSum is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aaclnnFusedLinearOnlineMaxSumGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFusedLinearOnlineMaxSum(void* workspace,
                                                   uint64_t workspaceSize,
                                                   aclOpExecutor* executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_FUSED_LINEAR_ONLINE_MAX_SUM_H_
