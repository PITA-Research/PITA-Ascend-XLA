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

#ifndef OP_API_INC_LEVEL2_ACLNN_EQ_SCALAR_H_
#define OP_API_INC_LEVEL2_ACLNN_EQ_SCALAR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnEqScalar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Calculate whether the value of the element in self is equal to the value of other, and write the comparison result of each element of self with the value of other into out:
 *
 * $$ out_i = (self_i == \mathit{other} )  ?  [True] : [False] $$
 *
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 * A[(Self)] -->B([l0op::Contiguous])
 * B -->C1([l0op::Cast])-->D([l0op::Equal])
 *     F((other)) -->D
 *     D -->F1([l0op::ViewCopy])--> J[(out)]
 * ```
 * @param [in] self: aclTensor on NPU device side,
 * Data types support FLOAT16, FLOAT, INT64, UINT64, INT32, INT8, UINT8, BOOL, UINT32, BFLOAT16, DOUBLE, INT16, COMPLEX64,
 * COMPLEX128, the data format supports ND and non-continuous Tensor.
 * @param [in] other: aclScalar on the host side, data type supports FLOAT16, FLOAT, INT64, UINT64, INT32, INT8, UINT8, BOOL, UINT32,
 * BFLOAT16, DOUBLE, INT16, COMPLEX64, COMPLEX128
 * @param [in] out: aclTensor on the NPU device side, the data type is bool, the shape is consistent with the shape of self, the data format supports ND
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEqScalarGetWorkspaceSize(const aclTensor* self,
                                                    const aclScalar* other,
                                                    aclTensor* out,
                                                    uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnEqScalar is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnEqScalarGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEqScalar(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    const aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceEqScalar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Calculate whether the value of the element in self is equal to the value of other, and return the comparison result of each element of self to the value of other to self:
 *
 * $$ self_i = (self_i == \mathit{other} )  ?  [True] : [False] $$
 *
 * Calculation graph:
*```mermaid
* graph LR
* A[(SelfRef)] -->B([l0op::Contiguous])
* B -->C1([l0op::Cast])-->D([l0op::Equal])
* F((other)) -->D
* D -->F1([l0op::ViewCopy])--> J[(selfRef)]
```
 * @param [in] selfRef: aclTensor on NPU device side,
 * Data types support FLOAT16, FLOAT, INT64, UINT64, INT32, INT8, UINT8, BOOL, UINT32, BFLOAT16, DOUBLE, INT16, COMPLEX64,
 * COMPLEX128, the data format supports ND and non-continuous Tensor.
 * @param [in] other: aclScalar on the host side, data type supports FLOAT16, FLOAT, INT64, UINT64, INT32, INT8, UINT8, BOOL, UINT32,
 * BFLOAT16, DOUBLE, INT16, COMPLEX64, COMPLEX128
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceEqScalarGetWorkspaceSize(
    const aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceEqScalar is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnEqScalarGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceEqScalar(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_EQ_SCALAR_H_