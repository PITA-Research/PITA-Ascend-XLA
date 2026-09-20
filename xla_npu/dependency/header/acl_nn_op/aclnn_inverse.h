/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_INVERSE_H_
#define OP_API_INC_LEVEL2_ACLNN_INVERSE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: Get the inverse matrix of the square matrix.
 * Calculation formula: as follows
 * $$
 * {A}^{-1}A = A{A}^{-1} = {I}_{n}
 * $$
 * Where A is the input tensor, ${A}^{-1}$ is the inverse of A, and ${I}_{n}$ is the n-dimensional identity matrix.
 *
 * Calculation diagram 1: as follows
 * Scenario: The data type of input self is not FLOAT16.
 *
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0op::Contiguous])
 *   B --> C([l0op::MatrixInverse])
 *   C --> D([l0op::Cast])
 *   D --> E([l0op::ViewCopy])
 *   E --> F[(out)]
 * ```
 *
 * Calculation diagram 2: as follows
 * Scenario: When the data type of the input self is FLOAT16, FLOAT16 needs to be converted into FLOAT32 and passed to the operator for calculation.
 *
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0op::Contiguous])
 *   B --> C([l0op::Cast])
 *   C --> D([l0op::MatrixInverse])
 *   D --> E([l0op::Cast])
 *   E --> F([l0op::ViewCopy])
 *   F --> G[(out)]
 * ```
 */

/**
 * The first interface of @brief aclnnInverse calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports FLOAT, DOUBLE, COMPLEX64, COMPLEX128, FLOAT16, and the shape is at least 2 dimensions.
 * And the sizes of the last two dimensions must be the same. Non-continuous Tensors are supported. The data format supports ND. The data dimension does not support more than 8 dimensions.
 * @param [in] out: aclTensor on the NPU device side, the data type supports FLOAT, FLOAT16, DOUBLE, BFLOAT16, COMPLEX64, COMPLEX128, and the data
 * The type needs to be a data type that can be converted by self, the shape needs to be consistent with the shape of self, support non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInverseGetWorkspaceSize(const aclTensor* self,
                                                   aclTensor* out,
                                                   uint64_t* workspaceSize,
                                                   aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnInverse is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInverseGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInverse(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_INVERSE_H_