/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MOE_DISTRIBUTE_COMBINE_
#define OP_API_INC_MOE_DISTRIBUTE_COMBINE_

#include <string>

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: implements the MoeDistributeCombine function.
 * The first interface of @brief aclnnMoeDistributeCombine calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] expandX: Calculation input, Tensor, data type float16, bfloat16, must be 2 dimensions, data format supports ND.
 * @param [in] expertIds: Calculation input, Tensor, data type int32, must be 2 dimensions, data format supports ND.
 * @param [in] expandIdx: Calculation input, Tensor, data type int32, must be 1-dimensional, data format supports ND, how many MOE experts handle the token.
 * @param [in] epSendCounts: Calculation input, Tensor, data type int32, must be 1-dimensional, data format supports ND.
 * @param [in] expertScales: Calculation input, Tensor, data type float32, must be 2-dimensional, data format supports ND.
 * @param [in] tpSendCounts: Calculation input, Tensor, data type int32, must be 1-dimensional, data format supports ND. Pass empty when there is no tp domain communication.
 * @param [in] xActiveMask: Calculation input, Tensor, data type bool, must be 1-dimensional, data format supports ND. Reserved parameters, not used yet, just pass them empty.
 * @param [in] activationScale: Calculation input, Tensor, data type float32, must be 1-dimensional, data format supports ND,
 * Reserved parameters, not used yet, just pass them empty.
 * @param [in] weightScale:
 * Calculation input, Tensor, data type float32, must be 2-dimensional, data format supports ND, this parameter must be present. Reserved parameters, not used yet, just pass them empty.
 * @param [in] groupList: Calculation input, Tensor, data type int64, must be 1 dimension, data format supports ND, reserved parameters, not used yet, just pass empty.
 * @param [in] expandScales: Calculation input, Tensor, data type float32, must be 1-dimensional, data format supports ND.
 * @param [in] groupEp: calculation input, str. The name of the ep communication domain, the communication domain of expert parallelism, cannot be the same as groupTp.
 * @param [in] epWorldSize: calculation input, int. ep communication domain size.
 * @param [in] epRankId: calculation input, int. ep card ID. The epRankId of each card in the same EP communication domain does not overlap.
 * @param [in] moeExpertNum: calculation input, int. Number of MOE experts.
 * @param [in] groupTp: Compute optional input, str. tp communication domain name, data parallel communication domain.
 * @param [in] tpWorldSize: Compute optional input, int. tp communication domain size.
 * @param [in] tpRankId: Optional input for calculation, int. tpCard ID.
 * @param [in] expertShardType: Optional input for calculation, int. Expert sharing type.
 * @param [in] sharedExpertNum: Compute optional input, int. Number of shared experts.
 * @param [in] sharedExpertRankNum: Compute optional input, int. The number of shared expert cards.
 * @param [in] globalBs: Compute optional input, int.
 * @param [in] outDtype: Compute optional input, int. Output data type. Reserved parameters, not used yet, just pass 0.
 * @param [in] commQuantMode: Compute optional input, int. Communication quantification type.
 * @param [in] groupListType: Compute optional input, int. groupList format. Reserved parameters, not used yet, just pass 0.
 * @param [out] x: Calculation output, Tensor, required output, data type supports float16, bfloat16, only supports 2 dimensions, data format supports ND.
 * @param [out] workspaceSize: Output parameter, returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Out parameter, returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return value, return status code
 *
 */
ACLNN_API aclnnStatus aclnnMoeDistributeCombineGetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds,
    const aclTensor *expandIdx, const aclTensor *epSendCounts,
    const aclTensor *expertScales, const aclTensor *tpSendCounts,
    const aclTensor *xActiveMask, const aclTensor *activationScale,
    const aclTensor *weightScale, const aclTensor *groupList,
    const aclTensor *expandScales, const char *groupEp, int64_t epWorldSize,
    int64_t epRankId, int64_t moeExpertNum, const char *groupTp,
    int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType,
    int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t globalBs,
    int64_t outDtype, int64_t commQuantMode, int64_t groupListType,
    aclTensor *x, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * The second interface of @brief aclnnMoeDistributeCombine is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnMoeDistributeCombineGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnMoeDistributeCombine(void *workspace,
                                                uint64_t workspaceSize,
                                                aclOpExecutor *executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MOE_DISTRIBUTE_COMBINE_