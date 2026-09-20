/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_AdaptiveAvgPool2dBackward_H_
#define OP_API_INC_AdaptiveAvgPool2dBackward_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnAdaptiveAvgPool2dBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: aclnnAdaptiveAvgPool2d reverse operation
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * @param [in] gradOutput: forward operation result, aclTensor on the NPU device side, the data type supports FLOAT16 and FLOAT32 data types,
 * And the data type is consistent with self. Supports non-continuous Tensor, the data format supports NCHW, and the data format needs to be consistent with self.
 * @param [in] self: forward input, aclTensor on the NPU device side, the data type supports FLOAT16 and FLOAT32 data types.
 * Supports non-continuous Tensor, and the data format supports NCHW
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */

ACLNN_API aclnnStatus aclnnAdaptiveAvgPool2dBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnAdaptiveAvgPool2dBackward is used to perform calculations.
 *
 * Operator function: aclnnAdaptiveAvgPool2d reverse operation
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnAdaptiveAvgPool2dBackwardGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAdaptiveAvgPool2dBackward(void* workspace,
                                                     uint64_t workspaceSize,
                                                     aclOpExecutor* executor,
                                                     const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_AdaptiveAvgPool2dBackward_H_
