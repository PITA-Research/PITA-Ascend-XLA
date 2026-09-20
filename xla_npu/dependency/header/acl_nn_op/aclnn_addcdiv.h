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

#ifndef OP_API_INC_ADDCDIV_H_
#define OP_API_INC_ADDCDIV_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnAddcdiv calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: perform element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to input
 * Calculation formula:
 * $$ out_i = self_i + value \times {tensor1_i \over tensor2_i} $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 graph LR
 *  A[(self)] -->B([l0op::Contiguous])
 *  B --> K([l0op::Cast])
 *  K --> C([l0op::Addcdiv])
 *  D[(tensor1)] -->E([l0op::Contiguous])
 *  E --> L([l0op::Cast])
 *  L --> C
 *  F[(tensor2)] --> G([l0op::Contiguous])
 *  G --> M([l0op::Cast])
 *  M --> C
 *  H((value)) --> C
 *  C --> O([l0op::Cast])
 *  O --> I([l0op::ViewCopy])
 *  I --> J[(out)]
 * ```
 *
 * @param [in] self: npu
 * The tensor that needs to be accumulated is aclTensor on the NPU device side. The data type supports DT_BFLOAT16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, and DT_INT64.
 *
 The data type must support a deducible relationship with tensor1 and tensor2, the shape must satisfy the broadcast relationship with the tensor after tensor1 and tensor2 are divided, non-continuous Tensors must be supported, and the data format must support ND.
 * @param [in] tensor1: npu
 * Molecular tensor, aclTensor on the NPU device side, data type supports DT_BFLOAT16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64,
 * The data type must support deducible relationships with self and tensor2, the shape must satisfy the broadcast relationship with tensor2, support non-continuous Tensors, and the data format supports ND.
 * @param [in] tensor2: npu
 * Denominator tensor, aclTensor on the NPU device side, data type supports DT_BFLOAT16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64,
 * The data type must support deducible relationships with self and tensor1, the shape must satisfy the broadcast relationship with tensor1, support non-continuous Tensors, and the data format supports ND.
 * @param [in] value: aclScalar on the host side. The data type is consistent with the data types of self, tensor1, and tensor2. If they are inconsistent, they need to be forced to be consistent.
 * @param [in] out: npu
 * Output tensor, aclTensor on the NPU device side, data type supports DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64,
 * The data type must satisfy the derivation relationship with self, tensor1, and tensor2, and the shape is self, tensor1, and tensor2.
 The shape after broadcast supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAddcdivGetWorkspaceSize(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2,
    const aclScalar* value, const aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnAddcdiv is used to perform calculations.
 *
 * * Operator function: perform element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to input
 * Calculation formula:
 * $$ out_i = self_i + value \times {tensor1_i \over tensor2_i} $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 graph LR
 *  A[(self)] -->B([l0op::Contiguous])
 *  B --> K([l0op::Cast])
 *  K --> C([l0op::Addcdiv])
 *  D[(tensor1)] -->E([l0op::Contiguous])
 *  E --> L([l0op::Cast])
 *  L --> C
 *  F[(tensor2)] --> G([l0op::Contiguous])
 *  G --> M([l0op::Cast])
 *  M --> C
 *  H((value)) --> C
 *  C --> O([l0op::Cast])
 *  O --> I([l0op::ViewCopy])
 *  I --> J[(out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddcdivGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAddcdiv(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor,
                                   const aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceAddcdiv calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: perform element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to input
 * Calculation formula:
 * $$ out_i = self_i + value \times {tensor1_i \over tensor2_i} $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 graph LR
 *  A[(self)] -->B([l0op::Contiguous])
 *  B --> K([l0op::Cast])
 *  K --> C([l0op::Addcdiv])
 *  D[(tensor1)] -->E([l0op::Contiguous])
 *  E --> L([l0op::Cast])
 *  L --> C
 *  F[(tensor2)] --> G([l0op::Contiguous])
 *  G --> M([l0op::Cast])
 *  M --> C
 *  H((value)) --> C
 *  C --> O([l0op::Cast])
 *  O --> I([l0op::ViewCopy])
 *  I --> J[(out)]
 * ```
 *
 * @param [in] selfRef: npu
 * The tensor that needs to be accumulated is aclTensor on the NPU device side. The data type supports DT_BFLOAT16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, and DT_INT64.
 *
 The data type must support a deducible relationship with tensor1 and tensor2, the shape must satisfy the broadcast relationship with the tensor after tensor1 and tensor2 are divided, non-continuous Tensors must be supported, and the data format must support ND.
 * @param [in] tensor1: npu
 * Molecular tensor, aclTensor on the NPU device side, data type supports DT_BFLOAT16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64,
 * The data type must support deducible relationships with self and tensor2, the shape must satisfy the broadcast relationship with tensor2, support non-continuous Tensors, and the data format supports ND.
 * @param [in] tensor2: npu
 * Denominator tensor, aclTensor on the NPU device side, data type supports DT_BFLOAT16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64,
 * The data type must support deducible relationships with self and tensor1, the shape must satisfy the broadcast relationship with tensor1, support non-continuous Tensors, and the data format supports ND.
 * @param [in] value: aclScalar on the host side. The data type is consistent with the data types of self, tensor1, and tensor2. If they are inconsistent, they need to be forced to be consistent.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceAddcdivGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* tensor1,
    const aclTensor* tensor2, const aclScalar* value, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnInplaceAddcdiv is used to perform calculations.
 *
 * * Operator function: perform element-wise division of tensor1 by tensor2, multiply the result by the scalar value and add it to input
 * Calculation formula:
 * $$ out_i = self_i + value \times {tensor1_i \over tensor2_i} $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 graph LR
 *  A[(self)] -->B([l0op::Contiguous])
 *  B --> K([l0op::Cast])
 *  K --> C([l0op::Addcdiv])
 *  D[(tensor1)] -->E([l0op::Contiguous])
 *  E --> L([l0op::Cast])
 *  L --> C
 *  F[(tensor2)] --> G([l0op::Contiguous])
 *  G --> M([l0op::Cast])
 *  M --> C
 *  H((value)) --> C
 *  C --> O([l0op::Cast])
 *  O --> I([l0op::ViewCopy])
 *  I --> J[(out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddcdivGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceAddcdiv(void* workspace,
                                          uint64_t workspaceSize,
                                          aclOpExecutor* executor,
                                          const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ADDCDIV_H_
