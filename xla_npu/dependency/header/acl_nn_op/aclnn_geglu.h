/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_GEGLU_H_
#define OP_API_INC_LEVEL2_ACLNN_GEGLU_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGeGlu's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: Gaussian error linear unit activation function
 * Calculation formula:
 * $$ out_{i}=GeGlu(self_{i}) = A \cdot Gelu(B) $$
 * Among them, $A$ represents the first half of $self$, and $B$ represents the second half of $self$.
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *     A[(self)] --> B([l0op::Contiguous])
 *     B --> C([l0op::GeGlu])
 *     C --> D([l0op::ViewCopy])
 *     D --> E[(out)]
 *     D --> F[(outGelu)]
 *
 *     G((dim)) --> C
 *     H((approximate)) --> C
```
 */

/**
 * @param [in] self: input parameter to be calculated by GeGlu, NPU
 * aclTensor on the device side, the data type supports FLOAT32, FLOAT16, BFLOAT16 (supported by 910B), supported
 * [Discontinuous Tensor](#), the data format supports ND([Reference](#)).
 * @param [in] dim: Optional parameter, set slice axis, data type supports INT64.
 * @param [in] approximate: Optional parameter, the activation function index used in GeGlu calculation, 0 means using `tanh`, 1 means using `none`, the data type supports INT64.
 * @param [in] out: Output parameter calculated by GeGlu, NPU
 * The data type of aclTensor on the device side needs to be consistent with self, supports [non-continuous Tensor](#), and the data format supports ND([reference](#)).
 * @param [in] outGelu: Output parameter calculated by Gelu(B), NPU
 * The data type of aclTensor on the device side needs to be consistent with self, supports [non-continuous Tensor](#), and the data format supports ND([reference](#)).
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGeGluGetWorkspaceSize(
    const aclTensor* self, int64_t dim, int64_t approximate, aclTensor* out,
    aclTensor* outGelu, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @param [in] self: input parameter to be calculated by GeGlu, NPU
 * @domain aclnn_ops_infer
 * aclTensor on the device side, the data type supports FLOAT32, FLOAT16, BFLOAT16 (supported by 910B), supported
 * [Discontinuous Tensor](#), the data format supports ND([Reference](#)).
 * @param [in] dim: Optional parameter, set slice axis, data type supports INT64.
 * @param [in] approximate: Optional parameter, the activation function index used in GeGlu calculation, 0 means using `tanh`, 1 means using `none`, the data type supports INT64.
 * @param [in] out: Output parameter calculated by GeGlu, NPU
 * The data type of aclTensor on the device side needs to be consistent with self, supports [non-continuous Tensor](#), and the data format supports ND([reference](#)).
 * @param [in] activateLeft:
 * Computed attribute, the Boolean value on the host side indicates the direction in which the activation function operates the data block. The default value is false, which means activate on the right side.
 * @param [in] outGelu: Output parameter calculated by Gelu(B), NPU
 * The data type of aclTensor on the device side needs to be consistent with self, supports [non-continuous Tensor](#), and the data format supports ND([reference](#)).
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGeGluV3GetWorkspaceSize(
    const aclTensor* self, int64_t dim, int64_t approximate, bool activateLeft,
    aclTensor* out, aclTensor* outGelu, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnGeGlu is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGeGluGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGeGlu(void* workspace, uint64_t workspaceSize,
                                 aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief The second interface of aclnnGeGlu is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGeGluGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGeGluV3(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_GEGLU_H_
