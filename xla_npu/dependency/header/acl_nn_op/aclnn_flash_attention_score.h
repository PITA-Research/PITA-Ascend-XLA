/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_FLASH_ATTENTION_SCORE_H_
#define OP_API_INC_LEVEL2_ACLNN_FLASH_ATTENTION_SCORE_H_

#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnFlashAttentionScore calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *realShiftOptional, const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, double scaleValue, double keepProb,
    int64_t preTokens, int64_t nextTokens, int64_t headNum, char *inputLayout,
    int64_t innerPrecise, int64_t sparseMode, const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * The second interface of @brief aclnnFlashAttentionScore is used to perform calculations.
 */
aclnnStatus aclnnFlashAttentionScore(void *workspace, uint64_t workspaceSize,
                                     aclOpExecutor *executor,
                                     const aclrtStream stream);

/**
 * The first interface of @brief aclnnFlashAttentionVarLenScore calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionVarLenScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *realShiftOptional, const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
    char *inputLayout, int64_t innerPrecise, int64_t sparseMode,
    const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut, const aclTensor *attentionOutOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * The second interface of @brief aclnnFlashAttentionVarLenScore is used to perform calculations.
 */
aclnnStatus aclnnFlashAttentionVarLenScore(void *workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor *executor,
                                           const aclrtStream stream);

/**
 * The first interface of @brief aclnnFlashAttentionScoreV2 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionScoreV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *realShiftOptional, const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional, double scaleValue, double keepProb,
    int64_t preTokens, int64_t nextTokens, int64_t headNum, char *inputLayout,
    int64_t innerPrecise, int64_t sparseMode, int64_t pseType,
    const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut, const aclTensor *attentionOutOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnFlashAttentionScoreV2 is used to perform calculations.
*/
aclnnStatus aclnnFlashAttentionScoreV2(void *workspace, uint64_t workspaceSize,
                                       aclOpExecutor *executor,
                                       const aclrtStream stream);

/**
 * @brief The first interface of aclnnFlashAttentionScoreV3 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionScoreV3GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *realShiftOptional, const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclTensor *sinkOptional, const aclIntArray *prefixOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
    double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise,
    int64_t sparseMode, int64_t pseType, const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnFlashAttentionScoreV3 is used to perform calculations.
*/
aclnnStatus aclnnFlashAttentionScoreV3(void *workspace, uint64_t workspaceSize,
                                       aclOpExecutor *executor,
                                       const aclrtStream stream);

/**
 * The first interface of @brief aclnnFlashAttentionVarLenScoreV2 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionVarLenScoreV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *realShiftOptional, const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
    double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise,
    int64_t sparseMode, int64_t pseType, const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionVarLenScoreV2's second interface is used to perform calculations.
*/
aclnnStatus aclnnFlashAttentionVarLenScoreV2(void *workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor *executor,
                                             const aclrtStream stream);

/**
 * The first interface of @brief aclnnFlashAttentionVarLenScoreV3 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionVarLenScoreV3GetWorkspaceSize(
    const aclTensor *query, const aclTensor *queryRope, const aclTensor *key,
    const aclTensor *keyRope, const aclTensor *value,
    const aclTensor *realShiftOptional, const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
    double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise,
    int64_t sparseMode, int64_t pseType, const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionVarLenScoreV3's second interface is used to perform calculations.
*/
aclnnStatus aclnnFlashAttentionVarLenScoreV3(void *workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor *executor,
                                             const aclrtStream stream);

/**
 * The first interface of @brief aclnnFlashAttentionVarLenScoreV4 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionVarLenScoreV4GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *realShiftOptional, const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
    char *inputLayout, int64_t innerPrecise, int64_t sparseMode,
    char *softmaxOutLayout, const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionVarLenScoreV4's second interface is used to perform calculations.
 */
aclnnStatus aclnnFlashAttentionVarLenScoreV4(void *workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor *executor,
                                             const aclrtStream stream);

/**
 * The first interface of @brief aclnnFlashAttentionVarLenScoreV5 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize(
    const aclTensor *query, const aclTensor *queryRope, const aclTensor *key,
    const aclTensor *keyRope, const aclTensor *value,
    const aclTensor *realShiftOptional, const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclTensor *sinkOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
    double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise,
    int64_t sparseMode, int64_t pseType, char *softmaxOutLayout,
    const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut, const aclTensor *attentionOutOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionVarLenScoreV5's second interface is used to perform calculations.
 */
aclnnStatus aclnnFlashAttentionVarLenScoreV5(void *workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor *executor,
                                             const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_FLASH_ATTENTION_SCORE_H_
