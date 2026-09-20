/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ACLNN_DYNAMIC_QUANT_V2_H_
#define ACLNN_DYNAMIC_QUANT_V2_H_

#include "acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnDynamicQuantV2GetWorkspaceSize calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnDynamicQuantV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* smoothScalesOptional,
    const aclTensor* groupIndexOptional, int64_t dstType, const aclTensor* yOut,
    const aclTensor* scaleOut, const aclTensor* offsetOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnDynamicQuantV2 is used to perform calculations.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDynamicQuantV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_DYNAMIC_QUANT_V2_H_