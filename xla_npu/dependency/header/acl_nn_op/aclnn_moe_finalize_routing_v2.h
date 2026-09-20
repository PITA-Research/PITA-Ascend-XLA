/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MOE_FINALIZE_ROUTING_V2_H_
#define OP_API_INC_MOE_FINALIZE_ROUTING_V2_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMoeFinalizeRoutingV2's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnMoeFinalizeRoutingV2GetWorkspaceSize(
    const aclTensor* expandedX, const aclTensor* expandedRowIdx,
    const aclTensor* x1Optional, const aclTensor* x2Optional,
    const aclTensor* biasOptional, const aclTensor* scalesOptional,
    const aclTensor* expertIdxOptional, int64_t dropPadMode,
    const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/* @brief aclnnMoeFinalizeRoutingV2's second interface is used to perform calculations. */
ACLNN_API aclnnStatus aclnnMoeFinalizeRoutingV2(void* workspace,
                                                uint64_t workspaceSize,
                                                aclOpExecutor* executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif