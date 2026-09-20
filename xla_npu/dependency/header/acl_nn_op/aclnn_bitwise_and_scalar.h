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

#ifndef OP_API_INC_BITWISE_AND_SCALAR_H_
#define OP_API_INC_BITWISE_AND_SCALAR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnBitwiseAndScalar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: complete bitwise AND or logical AND calculation
 * Calculation formula:
 * $$ output_i = self_i\&other_i $$
 *
 * Implementation instructions
 * Calculation diagram 1
 * Scenario 1: After type derivation, when the data types of self and other are both BOOL types, the l0::LogicalAnd interface needs to be called for calculation:
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([LogicalAnd])
 *     D[(other)] -->G
 *     G --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 * Calculation diagram 2
 * Scenario 2: After type derivation, when the data types of self and other are both INT types, the l0::BitwiseAnd interface needs to be called for calculation:
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([BitwiseAnd])
 *     D[(other)] -->G
 *     G --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 *
 * @param [in] self: aclTensor on the NPU device side, data type supports INT16, UINT16, INT32, INT64, INT8, UINT8, BOOL,
 * And the data type needs to form a mutual derivation relationship with other, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] other:
 * The data type of aclScalar on the host side supports INT16, UINT16, INT32, INT64, INT8, UINT8, BOOL, and the data type needs to form a mutual derivation relationship with self.
 * @param [in] out: aclTensor on the NPU device side, data type supports INT16, UINT16, INT32, INT64, INT8, UINT8, BOOL,
 * And the data type needs to be a convertible data type after self and other are deduced. The shape is consistent with self. The data format supports ND and non-continuous Tensor.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBitwiseAndScalarGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnBitwiseAndScalar is used to perform calculations.
 *
 * Operator function: complete bitwise AND or logical AND calculation
 * Calculation formula:
 * $$ output_i = self_i\&other_i $$
 *
 * Implementation instructions
 * Calculation diagram 1
 * Scenario 1: After type derivation, when the data types of self and other are both BOOL types, the l0::LogicalAnd interface needs to be called for calculation:
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([LogicalAnd])
 *     D[(other)] -->G
 *     G --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 * Calculation diagram 2
 * Scenario 2: After type derivation, when the data types of self and other are both INT types, the l0::BitwiseAnd interface needs to be called for calculation:
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([BitwiseAnd])
 *     D[(other)] -->G
 *     G --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnBitwiseAndScalarGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBitwiseAndScalar(void* workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceBitwiseAndScalar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: complete bitwise AND or logical AND calculation
 * Calculation formula:
 * $$ output_i = self_i\&other_i $$
 *
 * Implementation instructions
 * Calculation diagram 1
 * Scenario 1: After type derivation, when the data types of self and other are both BOOL types, the l0::LogicalAnd interface needs to be called for calculation:
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([LogicalAnd])
 *     D[(other)] -->G
 *     G --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 * Calculation diagram 2
 * Scenario 2: After type derivation, when the data types of self and other are both INT types, the l0::BitwiseAnd interface needs to be called for calculation:
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([BitwiseAnd])
 *     D[(other)] -->G
 *     G --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 *
 * @param [in] self: aclTensor on the NPU device side, data type supports INT16, UINT16, INT32, INT64, INT8, UINT8, BOOL,
 * And the data type needs to form a mutual derivation relationship with other, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] other:
 * The data type of aclScalar on the host side supports INT16, UINT16, INT32, INT64, INT8, UINT8, BOOL, and the data type needs to form a mutual derivation relationship with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceBitwiseAndScalarGetWorkspaceSize(
    const aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnInplaceBitwiseAndScalar's second interface is used to perform calculations.
 *
 * Operator function: complete bitwise AND or logical AND calculation
 * Calculation formula:
 * $$ output_i = self_i\&other_i $$
 *
 * Implementation instructions
 * Calculation diagram 1
 * Scenario 1: After type derivation, when the data types of self and other are both BOOL types, the l0::LogicalAnd interface needs to be called for calculation:
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([LogicalAnd])
 *     D[(other)] -->G
 *     G --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 * Calculation diagram 2
 * Scenario 2: After type derivation, when the data types of self and other are both INT types, the l0::BitwiseAnd interface needs to be called for calculation:
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([BitwiseAnd])
 *     D[(other)] -->G
 *     G --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnInplaceBitwiseAndScalarGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceBitwiseAndScalar(void* workspace,
                                                   uint64_t workspaceSize,
                                                   aclOpExecutor* executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_BITWISE_AND_SCALAR_H_
