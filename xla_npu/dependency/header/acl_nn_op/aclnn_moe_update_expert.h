/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MOE_UPDATE_EXPERT_
#define OP_API_INC_MOE_UPDATE_EXPERT_

#include <string>

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * Operator function: implements the MoeUpdateExpert function.
 * The first interface of @brief aclnnMoeUpdateExpert calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] expertIds: Calculation input, Tensor, data type int32, int64, must be 2 dimensions, data format supports ND.
 * @param [in] eplbTable: Calculation input, Tensor, data type int32, must be 2-dimensional, data format supports ND.
 * @param [in] expertScalesOptional: Calculation input, Tensor, data type float16, bfloat16, float32, must be 2 dimensions, data format supports ND.
 * @param [in] pruningThresholdOptional: Calculation input, Tensor, data type float32, must be 1-dimensional or 2-dimensional, and the data format supports ND.
 * @param [in] activeMaskOptional: Calculation input, Tensor, data type bool, must be 1-dimensional, data format supports ND.
 * @param [in] localRankId: calculation input, int. ep communication domain card ID.
 * @param [in] worldSize: calculation input, int. ep communication domain size. In the Shengteng 910_93 scenario, the value supports 8/16/32/64/128/144/256/288.
 * @param [in] balanceMode: Optional parameter, int. Balanced mode, currently defaults to 0, distributed according to rank.
 * @param [out] balancedExpertIds: Calculation output, Tensor, required output, data type supports int32, int64, only supports 2 dimensions, data format supports ND.
 * @param [out] balancedActiveMask: Calculation output, Tensor, required output, data type supports bool, only supports 2 dimensions, data format supports ND.
 * @param [out] workspaceSize: Output parameter, returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Out parameter, returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return value, return status code
 *
 */
ACLNN_API aclnnStatus aclnnMoeUpdateExpertGetWorkspaceSize(
    const aclTensor* expertIds, const aclTensor* eplbTable,
    const aclTensor* expertScalesOptional,
    const aclTensor* pruningThresholdOptional,
    const aclTensor* activeMaskOptional, int64_t localRankId, int64_t worldSize,
    int64_t balanceMode, aclTensor* balancedExpertIds,
    aclTensor* balancedActiveMask, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnMoeUpdateExpert is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnMoeUpdateExpertGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnMoeUpdateExpert(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MOE_UPDATE_EXPERT_