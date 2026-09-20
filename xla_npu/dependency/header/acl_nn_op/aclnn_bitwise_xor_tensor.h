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

#ifndef OP_API_INC_LEVEL2_ACLNN_BITWISE_XOR_TENSOR_H_
#define OP_API_INC_LEVEL2_ACLNN_BITWISE_XOR_TENSOR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: Calculate the bitwise XOR of each element in the input tensor self and the corresponding position element in the input tensor other. The input self and other must be integers or Boolean types. For Boolean types,
 * Compute logical XOR.
 * Calculation formula: as follows
 * $$
 * \text{out}_i =
 * \text{self}_i \, \bigoplus\, \text{other}_i
 * $$
 *
 * Implementation instructions: as follows
 * Calculation diagram 1: as follows
 * Scenario: When the data type after type derivation is BOOL, the l0::NotEqual interface needs to be called for calculation.
 *
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0op::Contiguous])
 *   B --> C([l0op::NotEqual])
 *   D[(other)] --> E([l0op::Contiguous])
 *   E --> C
 *   C --> F([l0op::Cast])
 *   F --> G([l0op::ViewCopy])
 *   G --> H[(out)]
 * ```
 *
 * Calculation diagram 2: as follows
 * Scenario: When the conditions for calculating Figure 1 are not met, the l0::BitwiseXor interface will be called for calculation.
 *
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0op::Contiguous])
 *   B --> C([l0op::Cast])
 *   C --> D([l0op::BitwiseXor])
 *   E[(other)] --> F([l0op::Contiguous])
 *   F --> G([l0op::Cast])
 *   G --> D
 *   D --> H([l0op::Cast])
 *   H --> I([l0op::ViewCopy])
 *   I --> J[(out)]
 * ```
 */

/**
 * The first interface of @brief aclnnBitwiseXorTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports BOOL, INT8, INT16, INT32, INT64, UINT8, and the data type is the same as the data type of other
 * It needs to meet the data type derivation rules, shape needs to satisfy the broadcast relationship with other, supports non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in] other：npu
 * aclTensor on the device side, the data type supports BOOL, INT8, INT16, INT32, INT64, UINT8, and the data type is the same as the data type of self
 * It needs to meet the data type derivation rules, shape needs to satisfy the broadcast relationship with self, supports non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports BOOL, INT8, INT16, INT32, INT64, UINT8, FLOAT, FLOAT16, DOUBLE,
 * BFLOAT16, COMPLEX64, COMPLEX128, and the data type needs to be a convertible data type after self and other are deduced, and the shape needs to be self and other
 * The shape after broadcast supports non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBitwiseXorTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnBitwiseXorTensor's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnBitwiseXorTensorGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBitwiseXorTensor(void* workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceBitwiseXorTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] selfRef: aclTensor on the NPU device side, the data type supports BOOL, INT8, INT16, INT32, INT64, UINT8, and the data type is the same as other
 * The data type needs to meet the data type derivation rules, and the deduced data type needs to be converted into selfRef's own data type. The shape needs to satisfy the broadcast relationship with other, and
 * The shape after broadcast needs to be the same as the shape of selfRef itself, supports non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in] other: aclTensor on the NPU device side, the data type supports BOOL, INT8, INT16, INT32, INT64, UINT8, and the data type is the same as selfRef
 * The data type needs to meet the data type derivation rules, the shape needs to meet the broadcast relationship with selfRef, supports non-continuous Tensor, the data format supports ND, and the data dimension does not support 8 dimensions.
 * above.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceBitwiseXorTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnInplaceBitwiseXorTensor's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnInplaceBitwiseXorTensorGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceBitwiseXorTensor(void* workspace,
                                                   uint64_t workspaceSize,
                                                   aclOpExecutor* executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_BITWISE_XOR_TENSOR_H_