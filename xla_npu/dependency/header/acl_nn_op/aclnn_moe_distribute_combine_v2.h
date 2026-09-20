/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MOE_DISTRIBUTE_COMBINE_V2_
#define OP_API_INC_MOE_DISTRIBUTE_COMBINE_V2_

#include <string>

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: implements the MoeDistributeCombineV2 function.
 * The first interface of @brief aclnnMoeDistributeCombine calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] expandX: Calculation input, Tensor, data type float16, bfloat16, must be 2 dimensions, data format supports ND.
 * @param [in] expertIds: Calculation input, Tensor, data type int32, must be 2 dimensions, data format supports ND.
 * @param [in] assistInfoForCombine: Calculation input, Tensor, data type int32, must be 1-dimensional, data format supports ND.
 * @param [in] epSendCounts: Calculation input, Tensor, data type int32, must be 1-dimensional, data format supports ND.
 * @param [in] expertScales: Calculation input, Tensor, data type float32, must be 2-dimensional, data format supports ND.
 * @param [in] tpSendCountsOptional: Calculation input, Tensor, data type int32, must be 1-dimensional, data format supports ND. If there is TP domain communication, parameters need to be passed. If there is no TP domain communication, just pass a null pointer.
 * @param [in] xActiveMaskOptional: Calculation input, Tensor, data type bool, must be 1-dimensional, and the data format supports ND.
 * @param [in] activationScaleOptional: Calculation input, Tensor, data type float32, must be 1-dimensional, data format supports ND. Reserved parameters, not supported by the current version, just pass a null pointer.
 * @param [in] weightScaleOptional: Calculation input, Tensor, data type float32, must be 2 dimensions, data format supports ND. Reserved parameters, not used yet, just pass them empty.
 * @param [in] groupListOptional: Calculation input, Tensor, data type int64, must be 1 dimension, data format supports ND. Reserved parameters, not used yet, just pass them empty.
 * @param [in] expandScalesOptional: Calculation input, Tensor, data type float32, must be 1-dimensional, data format supports ND.
 * @param [in] sharedExpertXOptional: Calculation optional input, Tensor, data type float16, bfloat16, must be 2-dimensional, data format supports ND.
 * @param [in] groupEp: calculation input, str. ep communication domain name, expert parallel communication domain. It cannot be the same as groupTp.
 * @param [in] epWorldSize: calculation input, int. ep communication domain size.
 * @param [in] epRankId: calculation input, int. ep card ID. The epRankId of each card in the same EP communication domain does not overlap.
 * @param [in] moeExpertNum: calculation input, int. Number of MOE experts.
 * @param [in] groupTp: Compute optional input, str. tp communication domain name, data parallel communication domain.
 * @param [in] tpWorldSize: Compute optional input, int. tp communication domain size.
 * @param [in] tpRankId: Optional input for calculation, int. tpCard ID. The tpRankId of each card in the same TP communication domain cannot be repeated.
 * @param [in] expertShardType: Optional input for calculation, int. Expert sharing type. Currently only 0 is supported.
 * @param [in] sharedExpertNum: Compute optional input, int. Number of shared experts.
 * @param [in] sharedExpertRankNum: Compute optional input, int. The number of shared expert cards.
 * @param [in] globalBs: Compute optional input, int.
 * @param [in] outDtype: Compute optional input, int. Output data type. Reserved parameters, not used yet, just pass 0.
 * @param [in] commQuantMode: Compute optional input, int. Communication quantification type.
 * @param [in] groupListType: Compute optional input, int. groupList format. Reserved parameters, not used yet, just pass 0.
 * @param [in] commAlg: Compute optional input, str. Communication algorithm type. Reserved parameters, not used yet.
 * @param [out] xOut: Calculation output, Tensor, required output, data type supports float16, bfloat16, only supports 2 dimensions, data format supports ND.
 * @param [out] workspaceSize: Output parameter, returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Out parameter, returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return value, return status code
 *
 */
ACLNN_API aclnnStatus aclnnMoeDistributeCombineV2GetWorkspaceSize(
    const aclTensor* expandX, const aclTensor* expertIds,
    const aclTensor* assistInfoForCombine, const aclTensor* epSendCounts,
    const aclTensor* expertScales, const aclTensor* tpSendCountsOptional,
    const aclTensor* xActiveMaskOptional,
    const aclTensor* activationScaleOptional,
    const aclTensor* weightScaleOptional, const aclTensor* groupListOptional,
    const aclTensor* expandScalesOptional,
    const aclTensor* sharedExpertXOptional, const char* groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    const char* groupTp, int64_t tpWorldSize, int64_t tpRankId,
    int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype,
    int64_t commQuantMode, int64_t groupListType, const char* commAlg,
    aclTensor* xOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnMoeDistributeCombine is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMoeDistributeCombineGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnMoeDistributeCombineV2(void* workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MOE_DISTRIBUTE_COMBINE_V2_