/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_NSA_SELECTED_ATTENTION_H_
#define OP_API_INC_LEVEL2_ACLNN_NSA_SELECTED_ATTENTION_H_

#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnNsaSelectedAttention calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 */
aclnnStatus aclnnNsaSelectedAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *topkIndices, const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, double scaleValue,
    int64_t headNum, char *inputLayout, int64_t sparseMode,
    int64_t selectedBlockSize, int64_t selectedBlockCount,
    const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut,
    const aclTensor *attentionOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * The second interface of @brief aclnnNsaSelectedAttention is used to perform calculations.
 */
aclnnStatus aclnnNsaSelectedAttention(void *workspace, uint64_t workspaceSize,
                                      aclOpExecutor *executor,
                                      const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_NSA_SELECTED_ATTENTION_H_
