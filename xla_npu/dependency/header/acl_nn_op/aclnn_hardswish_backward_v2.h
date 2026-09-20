/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_HARDSWISH_BACKWARD_V2_H_
#define OP_API_INC_HARDSWISH_BACKWARD_V2_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnHardswishBackwardV2's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: the reverse of activation function hardswish
 * Calculation formula:
 * $$ res_{i} = grad\_output_{i} \times grad\_self_{i} $$
 * $$
 * grad\_self_{i} = \begin{cases}
 * 0, & self_{i} \le -3, \\
 * self_{i} / 3 + 0.5, &   -3 \lt self_{i} \lt 3, \\
 * 1, & self_{i} \ge 3
 * \end{cases}
 * $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
graph LR
 *     A[(gradOutput)] --> B([l0op::Contiguous])
 *     B --> C([l0op::HardSwishGradV2])
 *     D[(self)] --> E([l0op::Contiguous])
 *     E --> C([l0op::HardSwishGradV2])
 *     C --> F([l0op::ViewCopy])
 *     F --> g[(out)]
```
 *
 * @param [in] gradOutput: npu
 * The data type of aclTensor on the device side supports floating point type, and the data type needs to be consistent with self, and the shape needs to be consistent with self.
 * Supports non-continuous Tensor, the data format supports ND, NCHW, NHWC, and the data format needs to be consistent with other.
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports floating point type, and the data type needs to be consistent with gradOutput, and the shape needs to be consistent with gradOutput.
 * Supports non-continuous Tensor, the data format supports ND, NCHW, and NHWC, and the data format needs to be consistent with gradOutput.
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports floating point type, and the data type needs to be consistent with gradOutput, and the shape needs to be consistent with gradOutput.
 * Supports non-continuous Tensor, the data format supports ND, NCHW, and NHWC, and the data format needs to be consistent with gradOutput.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnHardswishBackwardV2GetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnHardswishBackwardV2's second interface is used to perform calculations.
 *
 * Operator function: the reverse of activation function hardswish
 * Calculation formula:
 * $$ res_{i} = grad\_output_{i} \times grad\_self_{i} $$
 * $$
 * grad\_self_{i} = \begin{cases}
 * 0, & self_{i} \le -3, \\
 * self_{i} / 3 + 0.5, &   -3 \lt self_{i} \lt 3, \\
 * 1, & self_{i} \ge 3
 * \end{cases}
 * $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
graph LR
 *     A[(gradOutput)] --> B([l0op::Contiguous])
 *     B --> C([l0op::HardSwishGradV2])
 *     D[(self)] --> E([l0op::Contiguous])
 *     E --> C([l0op::HardSwishGradV2])
 *     C --> F([l0op::ViewCopy])
 *     F --> g[(out)]
```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSubGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnHardswishBackwardV2(void* workspace,
                                               uint64_t workspaceSize,
                                               aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_HARDSWISH_BACKWARD_V2_H_