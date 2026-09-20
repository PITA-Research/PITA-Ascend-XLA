/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_NSA_COMPRESS_ATTENTION_H_
#define OP_API_INC_LEVEL2_ACLNN_NSA_COMPRESS_ATTENTION_H_

#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnNsaCompressAttention calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnNsaCompressAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *attenMaskOptional, const aclTensor *topkMaskOptional,
    const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualCmpSeqKvLenOptional,
    const aclIntArray *actualSelSeqKvLenOptional, double scaleValue,
    int64_t headNum, char *inputLayout, int64_t sparseMode,
    int64_t compressBlockSize, int64_t compressStride, int64_t selectBlockSize,
    int64_t selectBlockCount, const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut, const aclTensor *attentionOutOut,
    const aclTensor *topkIndicesOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * The second interface of @brief aclnnNsaCompressAttention is used to perform calculations.
 */
aclnnStatus aclnnNsaCompressAttention(void *workspace, uint64_t workspaceSize,
                                      aclOpExecutor *executor,
                                      const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_NSA_COMPRESS_ATTENTION_H_