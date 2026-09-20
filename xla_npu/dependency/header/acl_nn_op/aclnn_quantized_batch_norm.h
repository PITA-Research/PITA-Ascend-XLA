/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file aclnn_quantized_batch_norm.h
 * \brief
 */
#ifndef OP_API_INC_QUANTIZED_BATCH_NORM_H_
#define OP_API_INC_QUANTIZED_BATCH_NORM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnQuantizedBatchNorm calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnQuantizedBatchNormGetWorkspaceSize(
    const aclTensor* input, const aclTensor* mean, const aclTensor* var,
    const aclScalar* inputScale, const aclScalar* inputZeroPoint,
    const aclScalar* outputScale, const aclScalar* outputZeroPoint,
    aclTensor* weight, aclTensor* bias, float epsilon, aclTensor* output,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnQuantizedBatchNorm is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnQuantizedBatchNorm(void* workspace,
                                              uint64_t workspaceSize,
                                              aclOpExecutor* executor,
                                              aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANTIZED_BATCH_NORM_H_