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

#ifndef OP_API_INC_LEVEL2_ACLNN_BITWISE_OR_TENSOR_H_
#define OP_API_INC_LEVEL2_ACLNN_BITWISE_OR_TENSOR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnBitwiseOrTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 *
 Operator function: Calculate the bitwise OR of each element in the tensor self and the element at the corresponding position in the other tensor. The tensor must be of type integer or boolean, for boolean types a logical OR is calculated.
 * Calculation formula: as follows
 * $$
 * \text{out}_i =
 * \text{self}_i \, | \, \text{other}_i
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
 * E[(other)] --> D
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
 * E[(other)] --> D
 * D --> F([l0op::Cast])
 * F --> G([l0op::ViewCopy])
 * G --> H[(out)]
 * ```
 *
 * @param [in] self: npu
 aclTensor on the device side, the data type supports BOOL, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64,
 * The data type needs to form a mutual derivation relationship with other, supports non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in] other：npu
 aclTensor on the device side, the data type supports BOOL, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64,
 * And the data type needs to form a mutual derivation relationship with self.
 * @param [in] out: npu
 aclTensor on the device side, the data type supports BOOL, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64,
 * FLOAT, FLOAT16, DOUBLE, BFLOAT16, COMPLEX64, COMPLEX128, and the data type needs to be a convertible data type after derivation of self and other.
 * The shape needs to be consistent with self, supports non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBitwiseOrTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnBitwiseOrTensor is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnBitwiseOrTensorGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBitwiseOrTensor(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceBitwiseOrTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: aclTensor on the NPU device side, data type supports BOOL, INT8, INT16, INT32, INT64, UINT8,
 * The data type needs to form a mutual derivation relationship with other, supports non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in] other: aclTensor on the NPU device side, data type supports BOOL, INT8, INT16, INT32, INT64, UINT8,
 * And the data type needs to form a mutual derivation relationship with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceBitwiseOrTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnInplaceBitwiseOrTensor is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, determined by the first interface
 * aclnnInplaceBitwiseOrTensorGetWorkspaceSize gets.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceBitwiseOrTensor(void* workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_BITWISE_OR_TENSOR_H_
