/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_INCRE_FLASH_ATTENTION_V3_H_
#define ACLNN_INCRE_FLASH_ATTENTION_V3_H_

#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnIncreFlashAttentionV3's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * funtion: aclnnIncreFlashAttentionV3GetWorkspaceSize
 * @param [in] query : required
 * @param [in] key : dynamic
 * @param [in] value : dynamic
 * @param [in] pseShift : optional
 * @param [in] attenMask : optional
 * @param [in] actualSeqLengths : optional
 * @param [in] dequantScale1 : optional
 * @param [in] quantScale1 : optional
 * @param [in] dequantScale2 : optional
 * @param [in] quantScale2 : optional
 * @param [in] quantOffset2 : optional
 * @param [in] antiquantScale : optional
 * @param [in] antiquantOffset : optional
 * @param [in] blocktable : optional
 * @param [in] numHeads : required
 * @param [in] scaleValue : optional
 * @param [in] inputLayout : optional
 * @param [in] numKeyValueHeads : optional
 * @param [in] blockSize : optional
 * @param [in] innerPrecise : optional
 * @param [out] attentionOut : required
 * @param [out] workspaceSize : size of workspace(output).
 * @param [out] executor : executor context(output).
 * @return aclnnStatus: return status code
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnIncreFlashAttentionV3GetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key,
    const aclTensorList *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclIntArray *actualSeqLengths,
    const aclTensor *dequantScale1, const aclTensor *quantScale1,
    const aclTensor *dequantScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, const aclTensor *antiquantScale,
    const aclTensor *antiquantOffset, const aclTensor *blocktable,
    int64_t numHeads, double scaleValue, char *inputLayout,
    int64_t numKeyValueHeads, int64_t blockSize, int64_t innerPrecise,
    const aclTensor *attentionOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * funtion: aclnnIncreFlashAttentionV3
 * @param [in] workspace : workspace memory addr(input).
 * @param [in] workspaceSize : size of workspace(input).
 * @param [in] executor : executor context(input).
 * @param [in] stream : acl stream.
 * @return aclnnStatus: return status code
 */
__attribute__((visibility("default"))) aclnnStatus aclnnIncreFlashAttentionV3(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif