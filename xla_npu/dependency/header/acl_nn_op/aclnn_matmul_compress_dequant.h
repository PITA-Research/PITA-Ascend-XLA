/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MM_UNZIP_H_
#define OP_API_INC_MM_UNZIP_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnMatmulCompressDequant calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnMatmulCompressDequantGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* compressIndex,
    const aclTensor* bias, const aclTensor* deqScale, const aclTensor* offsetW,
    int offsetX, const aclIntArray* compressInfo, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMatmulCompressDequant's second interface is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnMatmulCompressDequant(void* workspace,
                                                 uint64_t workspaceSize,
                                                 aclOpExecutor* executor,
                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MM_UNZIP_H_
