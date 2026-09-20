/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ADAPTIVE_MAX_POOL2D_H_
#define OP_API_INC_ADAPTIVE_MAX_POOL2D_H_
#include "aclnn_base.h"
#include "aclnn_util.h"
#include <array>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnAdaptiveMaxPool2d calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnAdaptiveMaxPool2dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* outputSize, aclTensor* outputOut,
    aclTensor* indicesOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnAdaptiveMaxPool2d is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnAdaptiveMaxPool2d(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif
