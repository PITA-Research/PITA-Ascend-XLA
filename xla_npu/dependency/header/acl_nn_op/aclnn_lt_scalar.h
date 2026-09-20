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
#ifndef OP_API_INC_LTSCALAR_H_
#define OP_API_INC_LTSCALAR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnLtScalarGetWorkspaceSize calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Calculate whether the elements in self Tensor are less than (<) other
 * Elements in Scalar return a Bool type Tensor. If self < other, it is True, otherwise it is False. Calculation formula: $$ out = (self < other)? [True]
 * : [False] $$
 *
 * Implementation description: Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(Self)] -->B([Contiguous])
 * B-->C1([Cast])-->D([Less])
 * E[(other)] -->F([Contiguous])
 * F --> C2([Cast])-->D
 * D -->F1([ViewCopy])--> J[(out)]
 * ```
 *
 * @param [in] self: aclTensor on NPU device side,
 * Data types support FLOAT16, FLOAT, INT64, INT32, UINT8, BOOL, UINT64, UINT32, DOUBLE, UINT16 data types,
 * Shape needs to satisfy the broadcast relationship with other, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] other: aclScalar on the host side, the shape needs to satisfy the broadcast relationship with other.
 * @param [in] out: aclTensor on the NPU device side, outputs a Tensor whose data type is BOOL, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */

ACLNN_API aclnnStatus aclnnLtScalarGetWorkspaceSize(const aclTensor* self,
                                                    const aclScalar* other,
                                                    aclTensor* out,
                                                    uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnLtScalar is used to perform calculations.
 *
 * Operator function: Calculate whether the elements in self Tensor are less than (<) other
 * Elements in Scalar return a Bool type Tensor. If self < other, it is True, otherwise it is False. Calculation formula: $$ out = (self < other)? [True]
 * : [False] $$
 *
 * Implementation description: Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(Self)] -->B([Contiguous])
 * B-->C1([Cast])-->D([Less])
 * E[(other)] -->F([Contiguous])
 * F --> C2([Cast])-->D
 * D -->F1([ViewCopy])--> J[(out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnLtScalarGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLtScalar(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceLtScalarGetWorkspaceSize calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Calculate whether the elements in selfRef Tensor are less than (<)other
 * Elements in Scalar return the modified selfRef. If selfRef < other, it is True, otherwise it is False. Calculation formula: $$ out = (selfRef < other)?
 * [True] : [False] $$
 *
 * Implementation description: Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(SelfRef)] --> B([l0op::Contiguous])
 * B-->C1([l0op::Cast])--> D([l0op::Less])
 * E((other)) --> C2([l0op::Cast])
 * C2 -->D
 * D --> C3([l0op::Cast])
 * C3 --> F1([l0op::ViewCopy])
 * F1 --> J[(selfRef)]
 * ```
 *
 * @param [in] selfRef: aclTensor on NPU device side,
 * Data types support FLOAT16, FLOAT, INT64, INT32, UINT8, BOOL, UINT64, UINT32, DOUBLE, UINT16 data types,
 * Shape needs to satisfy the broadcast relationship with other, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] other: aclScalar on the host side, the shape needs to satisfy the broadcast relationship with other.
 * @param [in] out: aclTensor on the NPU device side, outputs a Tensor whose data type is BOOL, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */

ACLNN_API aclnnStatus aclnnInplaceLtScalarGetWorkspaceSize(
    const aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnLtScalar is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnLtScalarGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceLtScalar(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LTSCALAR_H_
