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

#ifndef OP_API_INC_GTTENSOR_H_
#define OP_API_INC_GTTENSOR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGtTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Calculate whether the elements in self Tensor are greater than (>) the elements in other Tensor, and return a Tensor. If self > other, it is True (1.), otherwise it is False (0.):
 *
 * $$ out = (self_i > other_i)  ?  [True] : [False] $$
 *
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([l0op::Contiguous])
 *     B -->C1([l0op::Cast])-->D([l0op::Greater])--> C3([l0op::Cast])
 *     E[(other)] -->F([l0op::Contiguous])
 *     F --> C2([l0op::Cast])-->D
 *     C3 -->F1([l0op::ViewCopy])--> J[(out)]
 * ```
 *
 * @param [in] self: aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT, INT64, INT32, INT8, UINT8, BOOL, UINT64,
 * UINT32, INT16, DOUBLE, UINT16, BFLOAT16 data types, shape needs to satisfy the broadcast relationship with other,
 * The data type needs to meet the data type derivation rules with other, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] other: aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT, INT64, INT32, INT8, UINT8, BOOL, UINT64,
 * UINT32, INT16, DOUBLE, UINT16, BFLOAT16 data types, shape needs to satisfy the broadcast relationship with other,
 * The data type needs to meet the data type derivation rules with self, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] out: aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT, INT64, INT32, INT8, UINT8, BOOL, UINT64,
 * UINT32, INT16, DOUBLE, UINT16, BFLOAT16 data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGtTensorGetWorkspaceSize(const aclTensor* self,
                                                    const aclTensor* other,
                                                    aclTensor* out,
                                                    uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnGtTensor is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGtTensorGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGtTensor(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceGtTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Calculate whether the elements in selfRef Tensor are greater than (>) the elements in other Tensor, and return the result and copy it to selfRef:
 *
 * $$ selfRef = (selfRef_i > other_i)  ?  [True] : [False] $$
 *
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *    A[(SelfRef)] -->B([l0op::Contiguous])
 *    B -->C1([l0op::Cast])-->D([l0op::Greater])--> C3([l0op::Cast])
 *    E[(other)] -->F([l0op::Contiguous])
 *    F --> C2([l0op::Cast])-->D
 *    C3 -->F1([l0op::ViewCopy])--> J[(SelfRef)]
 * ```
 *
 * @param [in] selfRef(aclTensor*): The data type supports UINT8, INT8, DOUBLE, FLOAT, INT32, INT64, INT16, FLOAT16, BOOL, BFLOAT16 data types,
 * Shape needs to satisfy the broadcast relationship with other, and the data type needs to satisfy the data type derivation rules with other. Non-continuous Tensors are supported.
 * The data format supports ND; selfRef is also an output Tensor.
 * @param [in] other(aclTensor*): The data type supports UINT8, INT8, DOUBLE, FLOAT, INT32, INT64, INT16, FLOAT16, BOOL, BFLOAT16 data types,
 * The shape needs to satisfy the broadcast relationship with other, the data type needs to satisfy the data type derivation rules with selfRef, support non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceGtTensorGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnGtTensor is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGtTensorGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceGtTensor(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GTTENSOR_H_
