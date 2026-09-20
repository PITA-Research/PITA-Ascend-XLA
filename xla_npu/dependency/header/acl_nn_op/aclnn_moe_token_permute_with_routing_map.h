/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_H_
#define OP_API_INC_MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * The first interface of @brief aclnnMoeTokenPermuteWithRoutingMap calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * @param [in] tokens: calculation input, Tensor, data type float16, bfloat16, float. Input token data.
 * @param [in] routingMap: Calculation input, Tensor, data type int8, bool, data format supports ND. Represents the mapping relationship from token to expert.
 * @param [in] probsOptional: Calculation optional input, Tensor, data type float16, bfloat16, float, must be 2-dimensional, data format supports ND. Input prob data.
 * @param [in] numOutTokens: Compute optional input, int. Used to calculate the number of valid output tokens.
 * @param [in] dropAndPad: Calculate optional input, bool. Indicates whether to enable dropAndPad mode.
 * @param [out] permutedTokensOut: calculation output, Tensor, required output, data type supports float16,
 * bfloat16, float, only supports 2 dimensions, and the data format supports ND. According to the token characteristics processed by routingMap.
 * @param [out] sortedIndicesOut: calculation output, Tensor, required output, data type int32, only supports 1 dimension, and the data format supports ND. When quantMode is 0, the output is empty.
 * @param [out] permuteProbsOutOptional: calculation output, Tensor, optional output, data type float16,
 * bfloat16, float, only supports 1 dimension, the data format supports ND, and is based on the prob processed by routingMap.
 * @param [out] workspaceSize: Output parameter, returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Out parameter, returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return value, return status code
 *
 */
ACLNN_API aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize(
    const aclTensor* tokens, const aclTensor* routingMap,
    const aclTensor* probsOptional, int64_t numOutTokens, bool dropAndPad,
    aclTensor* permuteTokensOut, aclTensor* permuteProbsOutOptional,
    aclTensor* sortedIndicesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);
/**
 * The second interface of @brief aclnnMoeTokenPermuteWithRoutingMap is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus
aclnnMoeTokenPermuteWithRoutingMap(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
