/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_IM2COL_BACKWARD_H_
#define OP_API_INC_IM2COL_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnIm2colBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 */
ACLNN_API aclnnStatus aclnnIm2colBackwardGetWorkspaceSize(
    const aclTensor *gradOutput, const aclIntArray *inputSize,
    const aclIntArray *kernelSize, const aclIntArray *dilation,
    const aclIntArray *padding, const aclIntArray *stride, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * The second interface of @brief aclnnIm2colBackward is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnIm2colBackward(void *workspace,
                                          uint64_t workspaceSize,
                                          aclOpExecutor *executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
