/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_FLASH_ATTENTION_SCORE_GRAD_REGBASE_H_
#define OP_API_INC_FLASH_ATTENTION_SCORE_GRAD_REGBASE_H_

#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnFlashAttentionScoreGradVX's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
*/
aclnnStatus aclnnFlashAttentionScoreGradVXGetWorkspaceSize(
    const aclTensor *query, const aclTensor *keyIn, const aclTensor *value,
    const aclTensor *dy, const aclTensor *pseShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional,
    const aclTensor *attenMaskOptional, const aclTensor *softmaxMaxOptional,
    const aclTensor *softmaxSumOptional, const aclTensor *softmaxInOptional,
    const aclTensor *attentionInOptional, const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional, const aclTensor *dScaleQOptional,
    const aclTensor *dScaleKOptional, const aclTensor *dScaleVOptional,
    const aclTensor *dScaleDyOptional, const aclTensor *dScaleOOptional,
    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
    double scaleValueOptional, double keepProbOptional,
    int64_t preTokensOptional, int64_t nextTokensOptional, int64_t headNum,
    char *inputLayout, int64_t innerPreciseOptional, int64_t sparseModeOptional,
    int64_t pseTypeOptional, int64_t seed, int64_t offset,
    int64_t outDtypeOptional, const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dqRopeOut,
    const aclTensor *dkRopeOut, const aclTensor *dpseOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionScoreGradVX's second interface is used to perform calculations.
*/
aclnnStatus aclnnFlashAttentionScoreGradVX(void *workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor *executor,
                                           const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_FLASH_ATTENTION_SCORE_GRAD_REGBASE_H_
