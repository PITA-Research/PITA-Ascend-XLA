/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_HARDSIGMOID_H_
#define OP_API_INC_HARDSIGMOID_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnHardsigmoid calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: activation function calculation
 * $$
 * Hardsigmoid(x)=\begin{cases}
 * 1, & x\gt3 \\
 * 0, &  x\le -3 \\
 * x/6 + 1/2 , & otherwise
 * \end{cases}
 * $$
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([L0::Contiguous])
 *     B --> C([L0::Hardsigmoid])
 *     C --> D([L0::ViewCopy])
 *     D --> E[(out)]
 * ```
 *
 * @param [in] self: npu
 * The aclTensor on the device side supports data types such as FLOAT, FLOAT16, and INT32, and supports non-continuous Tensors.
 * Supports non-continuous Tensor, and the data format supports ND
 * @param [in] out: npu
 * The aclTensor on the device side supports FLOAT, FLOAT16, INT32 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnHardsigmoidGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);
/**
 * @brief aclnnHardsigmoid's second interface is used to perform calculations.
 *
 * Operator function: activation function calculation
 * $$
 * Hardsigmoid(x)=\begin{cases}
 * 1, & x\gt3 \\
 * 0, &  x\le -3 \\
 * x/6 + 1/2 , & otherwise
 * \end{cases}
 * $$
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([L0::Contiguous])
 *     B --> C([L0::Hardsigmoid])
 *     C --> D([L0::ViewCopy])
 *     D --> E[(out)]
 * ```
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnHardsigmoidGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnHardsigmoid(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       const aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceHardsigmoid calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: activation function calculation
 * $$
 * Hardsigmoid(x)=\begin{cases}
 * 1, & x\gt3 \\
 * 0, &  x\le -3 \\
 * x/6 + 1/2 , & otherwise
 * \end{cases}
 * $$
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([L0::Contiguous])
 *     B --> C([L0::Hardsigmoid])
 *     C --> D([L0::ViewCopy])
 *     D --> E[(out)]
```
 *
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, INT32, supports non-continuous Tensor, and the data format supports ND
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceHardsigmoidGetWorkspaceSize(
    const aclTensor* self, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceHardsigmoid is used to perform calculations.
 *
 * Operator function: activation function calculation
 * $$
 * Hardsigmoid(x)=\begin{cases}
 * 1, & x\gt3 \\
 * 0, &  x\le -3 \\
 * x/6 + 1/2 , & otherwise
 * \end{cases}
 * $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([L0::Contiguous])
 *     B --> C([L0::Hardsigmoid])
 *     C --> D([L0::ViewCopy])
 *     D --> E[(out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnInplaceHardsigmoidGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceHardsigmoid(void* workspace,
                                              uint64_t workspaceSize,
                                              aclOpExecutor* executor,
                                              const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_HARDSIGMOID_H
