/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_FLASH_ATTENTION_SCORE_REGBASE_H_
#define OP_API_INC_LEVEL2_ACLNN_FLASH_ATTENTION_SCORE_REGBASE_H_

#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief ACLNN The first interface of FlashAttentionScoreVX calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
*/
aclnnStatus aclnnFlashAttentionScoreVXGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *realShiftOptional, const aclTensor *dropMaskOptional,
    const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional,
    const aclTensor *dScaleQOptional, const aclTensor *dScaleKOptional,
    const aclTensor *dScaleVOptional, const aclIntArray *prefixOptional,
    const aclIntArray *actualSeqQLenOptional,  /*varlen only*/
    const aclIntArray *actualSeqKvLenOptional, /*varlen only*/
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
    double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise,
    int64_t sparseMode, int64_t outDtype, int64_t pseType,
    int64_t seed,   /*dropout ADD*/
    int64_t offset, /*dropout ADD*/
    const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut,
    const aclTensor *softmaxOutOut, const aclTensor *attentionOutOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashAttentionScoreVX's second interface is used to perform calculations.
*/
aclnnStatus aclnnFlashAttentionScoreVX(void *workspace, uint64_t workspaceSize,
                                       aclOpExecutor *executor,
                                       const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_FLASH_ATTENTION_SCORE_H_
