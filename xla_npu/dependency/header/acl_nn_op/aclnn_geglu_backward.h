/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_GEGLU_BACKWARD_H_
#define OP_API_INC_LEVEL2_ACLNN_GEGLU_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGeGluBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: Complete the reverse of GeGlu.
 *
 * Implementation instructions
 * API calculation base path:
 * ```mermaid
 * graph LR
 *     A[(gradOutput)] --> B([l0op::Contiguous])
 *     C[(self)] --> D([l0op::Contiguous])
 *     E[(gelu)] --> F([l0op::Contiguous])
 *     B --> G([l0op::GeGluV2Grad])
 *     D --> G
 *     F --> G
 *     H((dim)) --> G
 *     I((approximate)) --> G
 *     G --> J([l0op::ViewCopy])
 *     J --> K[(gradInput)]
 * ```
 */

/**
 * @param [in] gradOutput: calculation input, NPU
 * The data type of aclTensor on the device side supports FLOAT16. Except for the dim dimension, the sizes of other dimensions in the shape are the same as self.
 * The size of dim dimension is half of self, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] self: calculation input, NPU
 * The data type of aclTensor on the device side supports FLOAT16. Except for the dim dimension, the sizes of other dimensions in the shape are the same as gradOutput.
 * The size of dim dimension is twice that of gradOutput, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] gelu: calculation input, NPU
 * For aclTensor on the device side, the data type supports FLOAT16, the shape needs to be the same as gradOutput, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] dim: Calculated attribute, an integer on the host side, the data type supports INT64, and the current value only supports -1.
 * @param [in] approximate: calculated attribute, an integer on the host side, the data type supports INT64, the value range is 0 ('none'), 1 ('tanh'), the current value only supports
 * 1('tanh') 。
 * @param [in] activateLeft:
 * Computed attribute, the Boolean value on the host side indicates the direction in which the activation function operates the data block. The default value is false, which means activate on the right side.
 * @param [out] gradInput: calculation output, NPU
 * The data type of aclTensor on the device side supports FLOAT16, the shape needs to be the same as self, supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGeGluBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gelu,
    int64_t dim, int64_t approximate, aclTensor* gradInput,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnGeGluV3Backward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * @param [in] gradOutput: calculation input, NPU
 * The data type of aclTensor on the device side supports FLOAT16. Except for the dim dimension, the sizes of other dimensions in the shape are the same as self.
 * The size of dim dimension is half of self, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] self: calculation input, NPU
 * The data type of aclTensor on the device side supports FLOAT16. Except for the dim dimension, the sizes of other dimensions in the shape are the same as gradOutput.
 * The size of dim dimension is twice that of gradOutput, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] gelu: calculation input, NPU
 * For aclTensor on the device side, the data type supports FLOAT16, the shape needs to be the same as gradOutput, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] dim: Calculated attribute, an integer on the host side, the data type supports INT64, and the current value only supports -1.
 * @param [in] approximate: calculated attribute, an integer on the host side, the data type supports INT64, the value range is 0 ('none'), 1 ('tanh'), the current value only supports
 * 1('tanh') 。
 * @param [in] activateLeft:
 * Computed attribute, the Boolean value on the host side indicates the direction in which the activation function operates the data block. The default value is false, which means activate on the right side.
 * @param [out] gradInput: calculation output, NPU
 * The data type of aclTensor on the device side supports FLOAT16, the shape needs to be the same as self, supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGeGluV3BackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gelu,
    int64_t dim, int64_t approximate, bool activateLeft, aclTensor* gradInput,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnGeGluBackward is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGeGluBackwardGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGeGluBackward(void* workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         aclrtStream stream);

/**
 * The second interface of @brief aclnnGeGluBackward is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGeGluBackwardGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGeGluV3Backward(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_GEGLU_BACKWARD_H_
