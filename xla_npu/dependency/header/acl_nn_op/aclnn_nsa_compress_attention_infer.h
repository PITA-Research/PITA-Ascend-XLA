/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_NSA_COMPRESS_ATTENTION_INFER_H_
#define OP_API_INC_LEVEL2_ACLNN_NSA_COMPRESS_ATTENTION_INFER_H_

#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnNsaCompressAttentionInfer calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * funtion: aclnnNsaCompressAttentionInferGetWorkspaceSize
 * param [in] query : required
 * param [in] key : required
 * param [in] value : required
 * param [in] attentionMaskOptional : optional
 * param [in] blockTableOptional : optional
 * param [in] actualQSeqLenOptional : optional
 * param [in] actualCmpKvSeqLenOptional : optional
 * param [in] actualSelKvSeqLenOptional : optional
 * param [in] topkMaskOptional : optional
 * param [in] numHeads : optional
 * param [in] numKeyValueHeads : optional
 * param [in] selectBlockSize : optional
 * param [in] selectBlockCount : optional
 * param [in] compressBlockSize : optional
 * param [in] compressStride: optional [compressBlockStride in ACLNN]
 * param [in] scaleValue : optional
 * param [in] layoutOptional : optional
 * param [in] pageBlockSize : optional
 * param [in] sparseMode : optional
 * @param [out] output : required
 * @param [out] topkIndicesOut: required [topKOutput in ACLNN]
 * @param [out] workspaceSize : size of workspace(output).
 * @param [out] executor : executor context(output).
 * @return aclnnStatus: return status code
 */

aclnnStatus aclnnNsaCompressAttentionInferGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *attentionMaskOptional, const aclTensor *blockTableOptional,
    const aclIntArray *actualQSeqLenOptional,
    const aclIntArray *actualCmpKvSeqLenOptional,
    const aclIntArray *actualSelKvSeqLenOptional,
    const aclTensor *topKMaskOptional, int64_t numHeads,
    int64_t numKeyValueHeads, int64_t selectBlockSize, int64_t selectBlockCount,
    int64_t compressBlockSize, int64_t compressBlockStride, double scaleValue,
    char *layoutOptional, int64_t pageBlockSize, int64_t sparseMode,
    const aclTensor *output, const aclTensor *topKOutput,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * funtion: aclnnNsaCompressAttentionInfer
 * param [in] workspace : workspace memory addr(input).
 * param [in] workspaceSize : size of workspace(input).
 * param [in] executor : executor context(input).
 * param [in] stream : acl stream.
 * @return aclnnStatus: return status code
 */

aclnnStatus aclnnNsaCompressAttentionInfer(void *workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor *executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
