/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_GRAD_H_
#define OP_API_INC_MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_GRAD_H_

#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnMoeTokenPermuteWithRoutingMapGrad calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * @param [in] permutedTokenOutputGrad:
 * Calculation input, Tensor, data type float16, bfloat16, float. Only supports 2 dimensions, and the data format supports ND. Input token gradient data.
 * @param [in] permutedProbsOutputGradOptional:
 * Calculation optional input, Tensor, data type float16, bfloat16, float, must be 2-dimensional, data format supports ND. Input prob gradient data.
 * @param [in] sortedIndices: Calculation input, Tensor, data type int32, only supports 1 dimension, and the data format supports ND.
 * @param [in] routingMapOptional:
 * Calculation optional input, Tensor, data type int8, bool, data format supports ND. Represents the mapping relationship from token to expert.
 * @param [in] expertsNum: calculation input, int. Indicates the number of experts participating in the calculation.
 * @param [in] tokensNum: calculation input, int. Indicates the number of tokens involved in the operation.
 * @param [in] dropAndPad: Calculation input, bool. Indicates whether to enable dropAndPad mode.
 * @param [out] tokensGradOut:
 * Calculation output, Tensor, required output, data type supports float16, bfloat16, float, only supports 2 dimensions, data format supports ND. According to the token characteristics processed by routingMap.
 * @param [out] probsGradOutOptional:
 * Calculation output, Tensor, optional output, data type float16, bfloat16, float, only supports 1 dimension, data format supports ND, based on the prob gradient processed by routingMap.
 * @param [out] workspaceSize: Output parameter, returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Out parameter, returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return value, return status code
 *
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize(
    const aclTensor* permutedTokenOutputGrad,
    const aclTensor* permutedProbsOutputGradOptional,
    const aclTensor* sortedIndices, const aclTensor* routingMapOptional,
    int64_t numExperts, int64_t tokensNum, bool dropAndPad,
    aclTensor* tokensGradOut, aclTensor* probsGradOutOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMoeTokenPermuteWithRoutingMapGrad's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnMoeTokenPermuteWithRoutingMapGrad(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif