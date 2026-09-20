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

#ifndef OP_API_INC_LEVEL2_ACLNN_DOT_H_
#define OP_API_INC_LEVEL2_ACLNN_DOT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: Calculate the dot product result of two one-dimensional tensors
 * Calculation formula: as follows
 * $$
 * self = [x_{1}, x_{2}, ..., x_{n}]
 * $$
 *
 * $$
 * tensor = [y_{1}, y_{2}, ..., y_{n}]
 * $$
 *
 * $$
 * out = x_{1}*y_{1} + x_{2}*y_{2} + ... + x_{n}*y_{n}
 * $$
 *
 * Implementation instructions:
 * Basic path 1 for API calculation (both self and tensor are not empty Tensor scenarios):
 * ```mermaid
 * graph LR
 *     A[(self)] --> B([l0op::Contiguous])
 *     B --> C([l0op::Dot])
 *     D[(tensor)] --> E([l0op::Contiguous])
 *     E --> C
 *     C --> F([l0op::ViewCopy])
 *     F --> G[(out)]
 * ```
 *
 * Basic path two for API calculation (both self and tensor are empty Tensor scenarios):
 * ```mermaid
 * graph LR
 *     A[(self)] --> B([l0op::Fill])
 *     C[(tensor)] --> B
 *     B --> D([l0op::ViewCopy])
 *     D --> E[(out)]
 * ```
 */

/**
 * The first interface of @brief aclnnDot calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: aclTensor on the NPU device side. The data type supports FLOAT, BFLOAT16, and FLOAT16, and the data type needs to be consistent with the tensor.
 * Supports non-continuous Tensor, the data format supports ND, the dimension is 1 dimension, and the shape needs to be consistent with the tensor.
 * @param [in] tensor: aclTensor on the NPU device side. The data type supports FLOAT, BFLOAT16, and FLOAT16, and the data type needs to be consistent with self.
 * Supports non-continuous Tensor, the data format supports ND, the dimension is 1 dimension, and the shape needs to be consistent with self.
 * @param [in] out: aclTensor on the NPU device side, the data type supports FLOAT, BFLOAT16, FLOAT16, and the data type needs to be the same as self, tensor
 * Consistent, the data format supports ND, and the dimension is 0.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnDotGetWorkspaceSize(const aclTensor* self,
                                               const aclTensor* tensor,
                                               aclTensor* out,
                                               uint64_t* workspaceSize,
                                               aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnDot is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnDotGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnDot(void* workspace, uint64_t workspaceSize,
                               aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_DOT_H_
