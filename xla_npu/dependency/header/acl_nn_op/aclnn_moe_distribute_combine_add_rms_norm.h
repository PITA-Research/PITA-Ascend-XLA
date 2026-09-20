/**
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MOE_DISTRIBUTE_ADD_RMS_NORM_COMBINE
#define OP_API_INC_MOE_DISTRIBUTE_ADD_RMS_NORM_COMBINE

#include <string>

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: implements aclnnMoeDistributeCombineAddRmsNorm function.
 * The first interface of @brief aclnnMoeDistributeCombineAddRmsNorm calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] expandX: Calculation input, Tensor, data type bfloat16, must be 2-dimensional, and the data format supports ND.
 * @param [in] expertIds: Calculation input, Tensor, data type int32, must be 2 dimensions, data format supports ND.
 * @param [in] assistInfoForCombine: Calculation input, Tensor, data type int32, must be 1-dimensional, data format supports ND, auxiliary information transmitted to combine operator.
 * @param [in] epSendCounts: Calculation input, Tensor, data type int32, must be 1-dimensional, data format supports ND.
 * @param [in] expertScales: Calculation input, Tensor, data type float32, must be 2-dimensional, data format supports ND.
 * @param [in] residualX: calculation input, Tensor, data type bfloat16, must be 3-dimensional, and the data format supports ND.
 * @param [in] gamma: Calculation input, Tensor, data type bfloat16, must be 1-dimensional, and the data format supports ND.
 * @param [in] tpSendCountsOptional: Calculation input, Tensor, data type int32, must be 1-dimensional, data format supports ND. Pass empty when there is no tp domain communication.
 * @param [in] xActiveMaskOptional: Calculation input, Tensor, data type bool, must be 1-dimensional, and the data format supports ND. Reserved parameters, not used yet, just pass them empty.
 * @param [in] activationScaleOptional: Calculation input, Tensor, data type float32, must be 1-dimensional, data format supports ND, GMM externally thrown left matrix quantization coefficient,
    When the type of x is int32, this parameter is required. Reserved parameters, not used yet, just pass them empty.
 * @param [in] weightScaleOptional: Calculation input, Tensor, data type float32, must be 2 dimensions, data format supports ND, GMM externally thrown right matrix quantization coefficient,
    When the type of x is int32, this parameter is required. Reserved parameters, not used yet, just pass them empty.
 * @param [in] groupListOptional: Calculation input, Tensor, data type int64, must be 1 dimension, data format supports ND, GMM group size,
    When E>1 of weight scale, this parameter is required. Reserved parameters, not used yet, just pass them empty.
 * @param [in] expandScalesOptional: Calculation input, Tensor, data type float32, must be 1-dimensional, data format supports ND. It is not used in Shengteng 910_93 yet, just pass it as empty.
 * @param [in] sharedExpertXOptional: Calculation optional input, Tensor, data type float16, bfloat16, must be 2 dimensions, data format supports ND. The data type must be consistent with the expandX type,
    When sharedExpertRankNum is not 0, this parameter must be empty.
 * @param [in] groupEp: calculation input, str. ep communication domain name, expert parallel communication domain. The string length range is [1, 128) and cannot be the same as groupTp.
 * @param [in] epWorldSize: calculation input, int. ep communication domain size. In the Shengteng 910_93 scenario, the value supports 8/16/32/64/128/144/256/288.
 * @param [in] epRankId: calculation input, int. ep card ID. The value range is [0, epWorldSize). The epRankId of each card in the same EP communication domain cannot be repeated.
 * @param [in] moeExpertNum: calculation input, int. Number of MOE experts. The value range is [1, 512], and it needs to be
    Satisfies moeExpertNum%(epWorldSize-sharedExpertRankNum) equal to 0.
 * @param [in] groupTp: Compute optional input, str. tp communication domain name, data parallel communication domain. When there is no TP communication domain, empty is transmitted.
    When there is a tp communication domain, the string length range is [1, 128), which cannot be the same as groupEp.
 * @param [in] tpWorldSize: Compute optional input, int. tp communication domain size. The value range is [0, 2], 0 and 1 indicate no TP domain communication, and only 2 is supported when there is TP domain communication.
 * @param [in] tpRankId: Optional input for calculation, int. tpCard ID. The value range is [0, tpWorldSize). The tpRankId of each card in the same TP communication domain cannot be repeated.
 * @param [in] expertShardType: Optional input for calculation, int. Expert sharing type. Currently only 0 is supported.
 * @param [in] sharedExpertNum: Compute optional input, int. Number of shared experts. The value range is [0,1]. 0 means no shared experts.
 * @param [in] sharedExpertRankNum: Compute optional input, int. Number of shared experts. Support passing 0 means there is no shared expert card. If it is not 0, it must meet the requirements.
    sharedExpertRankNum<epWorldSize and epWorldSize%sharedExpertRankNum is equal to 0.
 * @param [in] globalBs: Compute optional input, int. In 910_93, when the number of Bs in each rank is the same, globalBs = Bs * epWorldSize or
    globalBs = 0; when the number of Bs in each rank is inconsistent, globalBs = maxBs * epWorldSize, where maxBs represents the maximum Bs of a single card.
 * @param [in] outDtype: Compute optional input, int. Output data type. Reserved parameters, not used yet, just pass 0.
 * @param [in] commQuantMode: Compute optional input, int. Communication quantification type. Reserved parameters, not used yet, just pass 0.
 * @param [in] groupListType: Compute optional input, int. groupList format. Reserved parameters, not used yet, just pass 0.
 * @param [in] commAlg: Compute optional input, str. Communication algorithm type. Reserved parameters, not used yet.
 * @param [in] normEps: Compute optional input, float. Used to prevent AddRmsNorm division by 0 errors. The data type only supports float, and the default value is 1e-6.
 * @param [out] yOut: Calculation output, Tensor, required output, data type supports bfloat16, only supports 3 dimensions, data format supports ND.
 * @param [out] rstdOut: Calculation output, Tensor, required output, data type supports float32, only supports 3 dimensions, data format supports ND.
 * @param [out] xOut: Calculation output, Tensor, required output, data type supports bfloat16, only supports 3 dimensions, data format supports ND.
 * @param [out] workspaceSize: Output parameter, returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Out parameter, returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return value, return status code
 *
 */
ACLNN_API aclnnStatus aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize(
    const aclTensor* expandX, const aclTensor* expertIds,
    const aclTensor* assistInfoForCombine, const aclTensor* epSendCounts,
    const aclTensor* expertScales, const aclTensor* residualX,
    const aclTensor* gamma, const aclTensor* tpSendCountsOptional,
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
    float normEps, aclTensor* yOut, aclTensor* rstdOut, aclTensor* xOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMoeDistributeCombineAddRmsNorm's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnMoeDistributeCombineAddRmsNorm(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MOE_DISTRIBUTE_ADD_RMS_NORM_COMBINE