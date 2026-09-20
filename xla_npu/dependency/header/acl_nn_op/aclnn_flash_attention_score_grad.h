/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_FLASH_ATTENTION_SCORE_GRAD_H_
#define OP_API_INC_FLASH_ATTENTION_SCORE_GRAD_H_

#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnFlashAttentionScoreGrad calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionScoreGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value,
    const aclTensor *dy, const aclTensor *pseShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional,
    const aclTensor *softmaxSumOptional, const aclTensor *softmaxInOptional,
    const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise,
    int64_t sparseMode, const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dpseOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionScoreGrad's second interface is used to perform calculations.
 */
aclnnStatus aclnnFlashAttentionScoreGrad(void *workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor *executor,
                                         const aclrtStream stream);

/**
 * The first interface of @brief aclnnFlashAttentionUnpaddingScoreGrad calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value,
    const aclTensor *dy, const aclTensor *pseShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional,
    const aclTensor *softmaxSumOptional, const aclTensor *softmaxInOptional,
    const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
    char *inputLayout, int64_t innerPrecise, int64_t sparseMode,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dpseOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionUnpaddingScoreGrad's second interface is used to perform calculations.
 */
aclnnStatus aclnnFlashAttentionUnpaddingScoreGrad(void *workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor *executor,
                                                  const aclrtStream stream);

/**
 * @brief aclnnFlashAttentionScoreGradV2's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionScoreGradV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value,
    const aclTensor *dy, const aclTensor *pseShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional,
    const aclTensor *softmaxSumOptional, const aclTensor *softmaxInOptional,
    const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
    double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise,
    int64_t sparseMode, int64_t pseType, const aclTensor *dqOut,
    const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionScoreGradV2's second interface is used to perform calculations.
*/
aclnnStatus aclnnFlashAttentionScoreGradV2(void *workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor *executor,
                                           const aclrtStream stream);

/**
 * @brief aclnnFlashAttentionScoreGradV3's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionScoreGradV3GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value,
    const aclTensor *dy, const aclTensor *pseShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional,
    const aclTensor *softmaxSumOptional, const aclTensor *softmaxInOptional,
    const aclTensor *attentionInOptional, const aclTensor *sinkInOptional,
    const aclIntArray *prefixOptional, const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional, double scaleValue, double keepProb,
    int64_t preTokens, int64_t nextTokens, int64_t headNum, char *inputLayout,
    int64_t innerPrecise, int64_t sparseMode, int64_t pseType,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dpseOut, const aclTensor *dsinkOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionScoreGradV3's second interface is used to perform calculations.
*/
aclnnStatus aclnnFlashAttentionScoreGradV3(void *workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor *executor,
                                           const aclrtStream stream);

/**
 * @brief aclnnFlashAttentionUnpaddingScoreGradV2's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value,
    const aclTensor *dy, const aclTensor *pseShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional,
    const aclTensor *softmaxSumOptional, const aclTensor *softmaxInOptional,
    const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
    double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise,
    int64_t sparseMode, int64_t pseType, const aclTensor *dqOut,
    const aclTensor *dkOut, const aclTensor *dvOut, const aclTensor *dpseOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionUnpaddingScoreGradV2's second interface is used to perform calculations.
*/
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV2(void *workspace,
                                                    uint64_t workspaceSize,
                                                    aclOpExecutor *executor,
                                                    const aclrtStream stream);

/**
  * @brief aclnnFlashAttentionUnpaddingScoreGradV3's first interface calculates the workspace size according to the specific calculation process.
  * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV3GetWorkspaceSize(
    const aclTensor *query, const aclTensor *queryRope, const aclTensor *keyIn,
    const aclTensor *keyInRope, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional,
    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
    double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise,
    int64_t sparseMode, int64_t pseType, const aclTensor *dqOut,
    const aclTensor *dqRopeOut, const aclTensor *dkOut,
    const aclTensor *dkRopeOut, const aclTensor *dvOut,
    const aclTensor *dpseOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
  * @brief aclnnFlashAttentionUnpaddingScoreGradV3's second interface is used to perform calculations.
 */
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV3(void *workspace,
                                                    uint64_t workspaceSize,
                                                    aclOpExecutor *executor,
                                                    const aclrtStream stream);

/**
 * @brief aclnnFlashAttentionUnpaddingScoreGradV4's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV4GetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value,
    const aclTensor *dy, const aclTensor *pseShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional,
    const aclTensor *softmaxSumOptional, const aclTensor *softmaxInOptional,
    const aclTensor *attentionInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, double scaleValue,
    double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum,
    char *inputLayout, int64_t innerPrecise, int64_t sparseMode,
    const aclTensor *dqOut, const aclTensor *dkOut, const aclTensor *dvOut,
    const aclTensor *dpseOut, char *softmaxInLayout, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionUnpaddingScoreGradV4's second interface is used to perform calculations.
 */
aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV4(void *workspace,
                                                    uint64_t workspaceSize,
                                                    aclOpExecutor *executor,
                                                    const aclrtStream stream);

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize(
    const aclTensor *query, const aclTensor *queryRope, const aclTensor *keyIn,
    const aclTensor *keyInRope, const aclTensor *value, const aclTensor *dy,
    const aclTensor *pseShiftOptional, const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclTensor *softmaxMaxOptional, const aclTensor *softmaxSumOptional,
    const aclTensor *softmaxInOptional, const aclTensor *attentionInOptional,
    const aclTensor *sinkInOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
    double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise,
    int64_t sparseMode, int64_t pseType, char *softmaxInLayout,
    const aclTensor *dqOut, const aclTensor *dqRopeOut, const aclTensor *dkOut,
    const aclTensor *dkRopeOut, const aclTensor *dvOut,
    const aclTensor *dpseOut, const aclTensor *dsinkOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV5(void *workspace,
                                                    uint64_t workspaceSize,
                                                    aclOpExecutor *executor,
                                                    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_FLASH_ATTENTION_SCORE_GRAD_H_
