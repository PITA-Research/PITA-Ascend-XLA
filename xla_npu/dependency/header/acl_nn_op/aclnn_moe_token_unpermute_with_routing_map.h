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

/* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef OP_API_INC_MOE_TOKEN_UNPERMUTE_WITH_ROTING_MAP_H_
#define OP_API_INC_MOE_TOKEN_UNPERMUTE_WITH_ROTING_MAP_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnMoeTokenUnpermuteWithRoutingMap calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * @param [in] permutedTokens: calculation input, Tensor, data type float16, bfloat16, float. Input token data.
 * @param [in] sortedIndices: Calculation input, Tensor, data type int32, only supports 1 dimension, and the data format supports ND.
 * @param [in] routingMapOptional: Calculation optional input, Tensor, data type int8, bool, data format supports ND. Represents the mapping relationship from token to expert.
 * @param [in] probsOptional: Calculation optional input, Tensor, data type float16, bfloat16, float, must be 2-dimensional, data format supports ND. Input prob data.
 * @param [in] paddedMode: Compute optional input, bool. Indicates whether to enable dropAndPad mode.
 * @param [out] unpermutedTokens: calculation output, Tensor, required output, data type supports float16, bfloat16, float, only supports 2 dimensions, data format supports ND.
 * @param [out] outIndex: calculation output, Tensor, required output, data type int32, only supports 1 dimension, and the data format supports ND.
 * @param [out] permuteTokenId: calculation output, Tensor, required output, data type int32, only supports 1 dimension, data format supports ND.
 * @param [out] permuteProbs: calculation output, Tensor, optional output, data type float16, bfloat16, float, only supports 1 dimension, data format supports ND, prob processed according to routingMap.
 * @param [out] workspaceSize: Output parameter, returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Out parameter, returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return value, return status code
 *
 */
ACLNN_API aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize(
    const aclTensor* permutedTokens, const aclTensor* sortedIndices,
    const aclTensor* routingMapOptional, const aclTensor* probsOptional,
    bool paddedMode, const aclIntArray* restoreShapeOptional,
    aclTensor* unpermutedTokens, aclTensor* outIndex, aclTensor* permuteTokenId,
    aclTensor* permuteProbs, uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * The second interface of @brief aclnnMoeTokenUnpermuteWithRoutingMap calculates the workspace size according to the specific calculation process.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */

ACLNN_API aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMap(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif