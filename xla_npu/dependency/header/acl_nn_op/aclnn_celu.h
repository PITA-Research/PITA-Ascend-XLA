/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_CELU_H_
#define OP_API_INC_LEVEL2_ACLNN_CELU_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: Call the continuously differentiable exponential linear unit activation function CELU for each element x in the input tensor self, and store the result in the output tensor out.
 * Calculation formula: as follows
 * $$
 * CELU(x) = max(0, x) + min(0, \alpha \ast (exp(x / \alpha) - 1))
 * $$
 *
 * Implementation instructions: as follows
 *
 * Calculation diagram: as follows
 *
 * ```mermaid
 * graph LR
 *     A[(Self)] --> B([l0op::Contiguous])
 *     B --> C([l0op::CeluV2])
 *     C --> D([l0op::Cast])
 *     D --> E([l0op::ViewCopy])
 *     E --> F[(out)]
 *     G((alpha)) --> C
 * ```
 */

/**
 * The first interface of @brief aclnnCelu calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] self: Represents the input of the CELU activation function, aclTensor on the NPU device side, the data type supports FLOAT, FLOAT16, BFLOAT16, and supports non-continuous
 * Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in] alpha: Represents the activation coefficient of the CELU activation function, aclScalar on the host side, the data type needs to be a data type convertible to FLOAT.
 * @param [in] out: Represents the output of the CELU activation function, NPU
 * For aclTensor on the device side, the data type needs to be a data type that can be converted by self, and the shape needs to be consistent with self.
 * Supports non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCeluGetWorkspaceSize(const aclTensor* self,
                                                const aclScalar* alpha,
                                                aclTensor* out,
                                                uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnCelu is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnCeluGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCelu(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor, aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceCelu calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] selfRef: represents the input of the CELU activation function, NPU
 * The data type of aclTensor on the device side supports FLOAT and FLOAT16, and supports non-continuous Tensor. The data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in] alpha: Represents the activation coefficient of the CELU activation function, aclScalar on the host side, the data type needs to be a data type convertible to FLOAT.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceCeluGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* alpha, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceCelu is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceCeluGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceCelu(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_CELU_H_
