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

#ifndef OP_API_INC_LEVEL2_ISFINITE_H_
#define OP_API_INC_LEVEL2_ISFINITE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: Determine whether the elements of the input tensor are bounded.
 * Calculation diagram: as follows
 * Scenario 1: Non-floating point number, must be bounded, return True directly
 *
 * ```mermaid
 * graph LR
 *     A[(self)] --> C([l0op::Fill]) --> H([l0op::ViewCopy]) --> K[(out)]
 * ```
 *
 * Scenario 2: In the case of floating point numbers, directly adjust IsFinite
 *
 * ```mermaid
 * graph LR
 *     A[(self)] --> B([l0op::Contiguous]) --> C([l0op::IsFinite]) --> H([l0op::ViewCopy]) --> K[(out)]
 * ```
 */

/**
 * The first interface of @brief aclnnIsFinite calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: Original tensor. aclTensor on the NPU device side,
 * Data types support FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL, and support non-continuous Tensors. The data format supports ND.
 * @param [out] out: aclTensor on the NPU device side, the data type can only be BOOL, the shape is consistent with self, and non-continuous Tensors are supported. The data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnIsFiniteGetWorkspaceSize(const aclTensor* self,
                                                    aclTensor* out,
                                                    uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnIsFinite is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnIsFiniteGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnIsFinite(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ISFINITE_H_
