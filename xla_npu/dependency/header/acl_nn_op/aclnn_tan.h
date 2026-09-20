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
#ifndef OP_API_INC_TAN_H_
#define OP_API_INC_TAN_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnTan calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Calculate the tangent value of each element in the input tensor self and store the result in the tensor out
 * Calculation formula: $$ out[i] = tan(self[i]) $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([l0op::Contiguous])
 *     B --> C([l0op::Tan])
 *     C --> D([l0op::ViewCopy])
 *     D --> E[(out)]
 * ```
 *
 * @param [in] self: aclTensor on NPU device side,
 * Data types support FLOAT, BFLOAT16, FLOAT16, INT32, DOUBLE, COMPLEX64, COMPLEX128, support non-continuous Tensor, and data format supports ND.
 * @param [in] out: aclTensor on NPU device side,
 * Data types support FLOAT, BFLOAT16, FLOAT16, INT32, DOUBLE, COMPLEX64, COMPLEX128, support non-continuous Tensor, and data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTanGetWorkspaceSize(const aclTensor* self,
                                               aclTensor* out,
                                               uint64_t* workspaceSize,
                                               aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnInplaceTan calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Calculate the tangent value of each element in the input tensor selfRef and store the result in the tensor selfRef
 * Calculation formula: $$ out[i] = tan(self[i]) $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(selfRef)] -->B([l0op::Contiguous])
 *     B --> C([l0op::Tan])
 *     C --> D([l0op::ViewCopy])
 *     D --> E[(selfRef)]
 * ```
 *
 * @param [in] selfRef: aclTensor on NPU device side,
 * Data types support FLOAT, FLOAT16, INT32, DOUBLE, COMPLEX64, COMPLEX128, support non-continuous Tensor, and data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceTanGetWorkspaceSize(const aclTensor* selfRef,
                                                      uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnTan is used to perform calculations.
 *
 * Operator function: Calculate the tangent value of each element in the input tensor self and store the result in the tensor out
 * Calculation formula: $$ out[i] = tan(self[i]) $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([l0op::Contiguous])
 *     B --> C([l0op::Tan])
 *     C --> D([l0op::ViewCopy])
 *     D --> E[(out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnTanhGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTan(void* workspace, uint64_t workspaceSize,
                               aclOpExecutor* executor,
                               const aclrtStream stream);

/**
 * The second interface of @brief aclnnInplaceTan is used to perform calculations.
 *
 * Operator function: Calculate the tangent value of each element in the input tensor selfRef and store the result in the tensor selfRef
 * Calculation formula: $$ out[i] = tan(self[i]) $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(selfRef)] -->B([l0op::Contiguous])
 *     B --> C([l0op::Tan])
 *     C --> D([l0op::ViewCopy])
 *     D --> E[(selfRef)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnTanhGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceTan(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor,
                                      const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_TANH_H