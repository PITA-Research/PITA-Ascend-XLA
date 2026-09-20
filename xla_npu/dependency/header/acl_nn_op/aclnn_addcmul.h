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

#ifndef OP_API_INC_ADD_CMUL_H_
#define OP_API_INC_ADD_CMUL_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnAddcmul calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: complete multiplication and addition calculations
 * Calculation formula:
 * $$ output=self+ value \times tensor1 \times tensor2 $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(Self)]--> B([l0op::Contiguous]) --> C([l0op::Cast]) -->D([Addcmul])
 * E[(tensor1)]--> B1([l0op::Contiguous]) --> G([l0op::Cast]) --> D
 * E1[(tensor2)]--> B2([l0op::Contiguous]) --> G1([l0op::Cast])  --> D
 * D --> H([l0op::Cast]) --> I([l0op::ViewCopy]) --> J[(out)]
 * K((value)) --> L([l0op::Cast]) --> D
 * ```
 *
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to form a mutual derivation relationship with other inputs, and the shape needs to satisfy the broadcast relationship with other inputs.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with other inputs.
 * @param [in] tensor1: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to form a mutual derivation relationship with other inputs, and the shape needs to satisfy the broadcast relationship with other inputs.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with other inputs.
 * @param [in] tensor2: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to form a mutual derivation relationship with other inputs, and the shape needs to satisfy the broadcast relationship with other inputs.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with other inputs.
 * @param [in] value: aclScalar on the host side, the data type needs to be convertible into the data type derived from other inputs.
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to be a data type that can be converted after other input derivation, and the shape needs to be other inputs.
 * The data format of the shape after broadcast supports ND, and the data format needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAddcmulGetWorkspaceSize(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2,
    const aclScalar* value, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnAddcmul is used to perform calculations.
 *
 * Operator function: complete multiplication and addition calculations
 * Calculation formula:
 * $$ output=self+ value \times tensor1 \times tensor2 $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(Self)]--> B([l0op::Contiguous]) --> C([l0op::Cast]) -->D([Addcmul])
 * E[(tensor1)]--> B1([l0op::Contiguous]) --> G([l0op::Cast]) --> D
 * E1[(tensor2)]--> B2([l0op::Contiguous]) --> G1([l0op::Cast])  --> D
 * D --> H([l0op::Cast]) --> I([l0op::ViewCopy]) --> J[(out)]
 * K((value)) --> L([l0op::Cast]) --> D
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAddcmul(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor, aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceAddcmul calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: complete multiplication and addition calculations
 * Calculation formula:
 * $$ output=self+ value \times tensor1 \times tensor2 $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(Self)]--> B([l0op::Contiguous]) --> C([l0op::Cast]) -->D([Addcmul])
 * E[(tensor1)]--> B1([l0op::Contiguous]) --> G([l0op::Cast]) --> D
 * E1[(tensor2)]--> B2([l0op::Contiguous]) --> G1([l0op::Cast])  --> D
 * D --> H([l0op::Cast]) --> I([l0op::ViewCopy]) --> J[(out)]
 * K((value)) --> L([l0op::Cast]) --> D
 * ```
 *
 * @param [in] selfRef: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to form a mutual derivation relationship with other inputs, and the shape needs to satisfy the broadcast relationship with other inputs.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with other inputs.
 * @param [in] tensor1: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to form a mutual derivation relationship with other inputs, and the shape needs to satisfy the broadcast relationship with other inputs.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with other inputs.
 * @param [in] tensor2: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to form a mutual derivation relationship with other inputs, and the shape needs to satisfy the broadcast relationship with other inputs.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with other inputs.
 * @param [in] value: aclScalar on the host side, the data type needs to be convertible into the data type derived from other inputs.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceAddcmulGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* tensor1,
    const aclTensor* tensor2, const aclScalar* value, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceAddcmul is used to perform calculations.
 *
 * Operator function: complete multiplication and addition calculations
 * Calculation formula:
 * $$ output=self+ value \times tensor1 \times tensor2 $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(Self)]--> B([l0op::Contiguous]) --> C([l0op::Cast]) -->D([Addcmul])
 * E[(tensor1)]--> B1([l0op::Contiguous]) --> G([l0op::Cast]) --> D
 * E1[(tensor2)]--> B2([l0op::Contiguous]) --> G1([l0op::Cast])  --> D
 * D --> H([l0op::Cast]) --> I([l0op::ViewCopy]) --> J[(out)]
 * K((value)) --> L([l0op::Cast]) --> D
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceAddcmul(void* workspace,
                                          uint64_t workspaceSize,
                                          aclOpExecutor* executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ADD_CMUL_H_
