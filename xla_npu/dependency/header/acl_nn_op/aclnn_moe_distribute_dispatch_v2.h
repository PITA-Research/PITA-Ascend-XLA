/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MOE_DISTRIBUTE_DISPATCH_V2_
#define OP_API_INC_MOE_DISTRIBUTE_DISPATCH_V2_

#include <string>

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: implements the MoeDistributeDispatch function, quantifies the Token data first, then performs alltoallv communication in the EP domain, and then performs allgatherv communication in the TP domain.
 * The first interface of @brief aclnnMoeDistributeDispatch calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] x: Calculation input, Tensor, data type float16, bfloat16, must be 2 dimensions, data format supports ND. Input token data.
 * @param [in] expertIds: Calculation input, Tensor, data type int32, must be 2 dimensions, data format supports ND. TopK expert indexes for each token.
 * @param [in] scalesOptional: Calculation optional input, Tensor, data type float32, must be 2-dimensional, data format supports ND. Smooth weight for each expert.
 * @param [in] xActiveMaskOptional: Calculation input, Tensor, data type Bool, must be 1-dimensional, and the data format supports ND.
 * @param [in] expertScalesOptional: Calculation input, Tensor, must be 2-dimensional, and the data format supports ND.
 * @param [in] groupEp: calculation input, str. ep communication domain name, expert parallel communication domain. It cannot be the same as groupTp.
 * @param [in] epWorldSize: calculation input, int. ep communication domain size.
 * @param [in] epRankId: calculation input, int. ep card ID. The epRankId of each card in the same EP communication domain cannot be repeated.
 * @param [in] moeExpertNum: calculation input, int. Number of MOE experts.
 * @param [in] groupTp: Compute optional input, str. tp communication domain name, data parallel communication domain. If there is no TP communication domain, it will be empty.
 * @param [in] tpWorldSize: Compute optional input, int. tp communication domain size.
 * @param [in] tpRankId: Optional input for calculation, int. tpCard ID.
 * @param [in] expertShardType: Optional input for calculation, int. Expert sharing type.
 * @param [in] sharedExpertNum: Compute optional input, int. Number of shared experts.
 * @param [in] sharedExpertRankNum: Compute optional input, int. The number of shared expert cards.
 * @param [in] quantMode: Compute optional input, int, quantization mode.
 * @param [in] globalBs: Compute optional input, int. The global batch size of the EP domain.
 * @param [in] expertTokenNumsType: Compute optional input, int. Output the semantic type of the value in expertTokenNums.
 * @param [in] commAlg: Compute optional input, str. Communication algorithm type. Reserved parameters, not used yet.
 * @param [out] expandXOut: Calculation output, Tensor, required output, data type supports float16, bfloat16, int8, only supports 2 dimensions, data format supports ND. According to
    ExpertIdx has extended token characteristics.
 * @param [out] dynamicScalesOut: Calculation output, Tensor, required output, data type float32, only supports 1 dimension, data format supports ND. When quantMode is 0, the output is empty.
 * @param [out] assistInfoForCombineOut: Calculation output, Tensor, required output, data type int32, only supports 1 dimension, data format supports ND, auxiliary information transmitted to the combine operator.
 * @param [out] expertTokenNumsOut: calculation output, Tensor, required output, data type int64, only supports 1 dimension, and the data format supports ND. The number of tokens received by each expert.
 * @param [out] epRecvCountsOut: Calculation output, Tensor, required output, data type int32, only supports 1 dimension, data format supports ND. Indicates the number of tokens received from each card.
 * @param [out] tpRecvCountsOut: Calculation output, Tensor, required output, data type int32, only supports 1 dimension, data format supports ND. The output is empty when there is no TP communication domain.
 * @param [out] expandScalesOut: Calculation output, Tensor, required output, data type float32, only supports 1 dimension, data format supports ND.
 * @param [out] workspaceSize: Output parameter, returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Out parameter, returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return value, return status code
 *
 */
ACLNN_API aclnnStatus aclnnMoeDistributeDispatchV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* expertIds,
    const aclTensor* scalesOptional, const aclTensor* xActiveMaskOptional,
    const aclTensor* expertScalesOptional, const char* groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    const char* groupTp, int64_t tpWorldSize, int64_t tpRankId,
    int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs,
    int64_t expertTokenNumsType, const char* commAlg, aclTensor* expandXOut,
    aclTensor* dynamicScalesOut, aclTensor* assistInfoForCombineOut,
    aclTensor* expertTokenNumsOut, aclTensor* epRecvCountsOut,
    aclTensor* tpRecvCountsOut, aclTensor* expandScalesOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnMoeDistributeDispatch is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMoeDistributeDispatchGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnMoeDistributeDispatchV2(void* workspace,
                                                   uint64_t workspaceSize,
                                                   aclOpExecutor* executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MOE_DISTRIBUTE_DISPATCH_