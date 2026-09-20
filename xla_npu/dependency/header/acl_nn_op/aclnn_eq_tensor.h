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

#ifndef OP_API_INC_EQTENSOR_H_
#define OP_API_INC_EQTENSOR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnEqTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Calculate whether the elements in two Tensors are equal and return a Tensor. If self=other, it is True(1.), otherwise it is False(0.):
 *
 * $$ out = (self == other)  ?  [True] : [False] $$
 *
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *    A[(Self)] -->B([l0op::Contiguous])
 *    B -->C1([l0op::Cast])-->D([l0op::Equal])
 *    E[(other)] -->F([l0op::Contiguous])
 *    F --> C2([l0op::Cast])-->D
 *    D --> C3([l0op::Cast])-->F1([l0op::ViewCopy])--> J[(out)]
 * ```
 *
 * @param [in] self: aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT, INT64, INT32, INT8, UINT8, BOOL, BFLOAT16,
 * DOUBLE, INT16, COMPLEX64, COMPLEX128 data types, shape needs to satisfy the broadcast relationship with other,
 * The data type needs to meet the data type derivation rules with other, support non-continuous Tensor, and the data format supports ND.
 * @param [in] other: aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT, INT64, INT32, INT8, UINT8, BOOL, BFLOAT16,
 * DOUBLE, INT16, COMPLEX64, COMPLEX128 data types, shape needs to satisfy the broadcast relationship with self.
 * The data type needs to meet the data type derivation rules with self, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] out: aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT, INT64, INT32, INT8, UINT8, BOOL, BFLOAT16,
 * DOUBLE, INT16, COMPLEX64, COMPLEX128, the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEqTensorGetWorkspaceSize(const aclTensor* self,
                                                    const aclTensor* other,
                                                    aclTensor* out,
                                                    uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnEqTensor is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnEqTensorGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEqTensor(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceEqTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Calculate whether the value of the element in selfRef is equal to the value of other, and return the comparison result of each element of selfRef to the value of other back to selfRef;
 *
 * Calculation formula: $$ selfRef = (selfRef == other)? [True]: [False] $$
 *
 * @param [in] selfRef: aclTensor on NPU device side,
 * Data types support FLOAT16, FLOAT, INT64, INT32, INT8, UINT8, BOOL, BFLOAT16 (only supported by 1971), DOUBLE, INT16, COMPLEX64, COMPLEX128,
 * The data format supports ND and non-continuous Tensor.
 * @param [in] other: aclTensor on NPU device side,
 * Data types support FLOAT16, FLOAT, INT64, INT32, INT8, UINT8, BOOL, BFLOAT16 (only supported by 1971), DOUBLE, INT16, COMPLEX64, COMPLEX128,
 * The data format supports ND and non-continuous Tensor.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceEqTensorGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnInplaceEqTensor's second interface is used to perform calculations.
 *
 * Operator function: Calculate whether the value of the element in selfRef is equal to the value of other, and return the comparison result of each element of selfRef to the value of other back to selfRef;
 *
 * Calculation formula: $$ selfRef = (selfRef == other)? [True]: [False] $$
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceAddGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceEqTensor(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_EQTENSOR_H_