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

#ifndef OP_API_INC_LEVEL2_ACLNN_BITWISE_OR_SCALAR_H_
#define OP_API_INC_LEVEL2_ACLNN_BITWISE_OR_SCALAR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *
 Operator function: Calculate the bitwise OR of each element in the input tensor self and the input scalar other. The inputs self and other must be of type integer or boolean. For boolean types, a logical OR is calculated.
 * Calculation formula: as follows
 * $$
 * \text{out}_i =
 * \text{self}_i \, | \, \text{other}
 * $$
 *
 * Implementation instructions: as follows
 * Calculation diagram 1: as follows
 * Scenario: After type derivation, when the data types of self and other are both BOOL types, the l0::LogicalOr interface needs to be called for calculation.

 * ```mermaid
 * graph LR
 * A[(self)] --> B([l0op::Contiguous])
 * B --> C([l0op::Cast])
 * C --> D([l0op::LogicalOr])
 * E((other)) --> D
 * D --> F([l0op::Cast])
 * F --> G([l0op::ViewCopy])
 * G --> H[(out)]
 * ```
 *
 * Calculation diagram 2: as follows
 * Scenario: When the conditions for calculating Figure 1 are not met, the l0::BitwiseOr interface will be called for calculation.
 *
 * ```mermaid
 * graph LR
 * A[(self)] --> B([l0op::Contiguous])
 * B --> C([l0op::Cast])
 * C --> D([l0op::BitwiseOr])
 * E((other)) --> D
 * D --> F([l0op::Cast])
 * F --> G([l0op::ViewCopy])
 * G --> H[(out)]
 * ```
 */

/**
 * The first interface of @brief aclnnBitwiseOrScalar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports BOOL, INT8, INT16, UINT16, INT32, INT64, UINT8, and the data type is the same as the data type of other
 * It needs to meet the data type derivation rules, support non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in]
 * other: aclScalar on the host side, the data type supports BOOL, INT8, INT16, UINT16, INT32, INT64, UINT8, and the data type must meet the data type of self.
 * Data type derivation rules.
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports BOOL, INT8, INT16, UINT16, INT32, INT64, UINT8, FLOAT, FLOAT16, DOUBLE,
 * BFLOAT16, COMPLEX64, COMPLEX128, and the data type needs to be a convertible data type after derivation of self and other, the shape needs to be consistent with self, and non-continuous is supported.
 * Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBitwiseOrScalarGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnBitwiseOrScalar is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnBitwiseOrScalarGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBitwiseOrScalar(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceBitwiseOrScalar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: aclTensor on the NPU device side, the data type supports BOOL, INT8, INT16, UINT16, INT32, INT64, UINT8, and the data type is the same as other
 * The data type must meet the data type derivation rules, support non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in]
 * other: aclScalar on the host side, the data type supports BOOL, INT8, INT16, UINT16, INT32, INT64, UINT8, and the data type must meet the data type of self.
 * Data type derivation rules.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceBitwiseOrScalarGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnInplaceBitwiseOrScalar is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnInplaceBitwiseOrScalarGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceBitwiseOrScalar(void* workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_BITWISE_OR_SCALAR_H_
