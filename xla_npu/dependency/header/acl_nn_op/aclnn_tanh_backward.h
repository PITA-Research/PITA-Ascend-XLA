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

#ifndef OP_API_INC_TANH_BACKWARD_H_
#define OP_API_INC_TANH_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnTanhBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: the reverse of activation function tanh(x)
 *
 * Implementation instructions: API
 * The base path for calculation: as shown below
 * ```mermaid
 * graph LR
 *         A[(gradOutput)] -->B([l0op::Contiguous]) --> C([l0op::TanhGrad])
 *         D[(output)] -->E([l0op::Contiguous]) --> C
 *         C --> F([l0op::ViewCopy])
 *         F --> G[(gradInput)]
 * ```
 *
 * @param [in] gradOutput: aclTensor on the NPU device side. The data type supports FLOAT, BFLOAT16, FLOAT16, and DOUBLE.
 * The data type is the same as output, and the shape is the same as output. Supports [non-continuous Tensor] (#non-continuous Tensor description), and the data format supports ND.
 * @param [in] output: aclTensor on the NPU device side. The data type supports FLOAT, BFLOAT16, FLOAT16, and DOUBLE.
 * Supports [non-continuous Tensor] (#non-continuous Tensor description), and the data format supports ND.
 * @param [out] gradInput: aclTensor on the NPU device side. The data type supports FLOAT, BFLOAT16, FLOAT16, and DOUBLE.
 * Supports [non-continuous Tensor] (#non-continuous Tensor description), and the data format supports ND.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTanhBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* output, aclTensor* gradInput,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnTanhBackward is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnTanhBackwardGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTanhBackward(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_TANH_BACKWARD_H_
