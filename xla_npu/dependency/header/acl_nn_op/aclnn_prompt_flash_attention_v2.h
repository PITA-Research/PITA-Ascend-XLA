/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_PROMPT_FLASH_ATTENTION_V2_H_
#define ACLNN_PROMPT_FLASH_ATTENTION_V2_H_
#include "acl_meta.h"
#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnPromptFlashAttentionV2 is used to calculate the workspace size based on the specific calculation process.
 * @domain aclnn_ops_infer
*/
__attribute__((visibility("default"))) aclnnStatus
aclnnPromptFlashAttentionV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *pseShift,
    const aclTensor *attenMask,  // attenMask of V2
    const aclIntArray *actualSeqLengths, const aclIntArray *actualSeqLengthsKv,
    const aclTensor *deqScale1, const aclTensor *quantScale1,
    const aclTensor *deqScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2,
    int64_t numHeads,  // q_n of V2
    double scaleValue, int64_t preTokens, int64_t nextTokens, char *inputLayout,
    int64_t numKeyValueHeads,
    int64_t sparseMode,  // sparse of V2
    const aclTensor *attentionOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnPromptFlashAttentionV2 is used to perform calculations.
*/
__attribute__((visibility("default"))) aclnnStatus aclnnPromptFlashAttentionV2(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif