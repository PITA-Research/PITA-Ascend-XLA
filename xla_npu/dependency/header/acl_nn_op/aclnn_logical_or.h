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
#ifndef OP_API_INC_LogicalOr_H_
#define OP_API_INC_LogicalOr_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnLogicalOr calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: complete logical or calculation
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A1[(self)] -->B1([Contiguous])-->C1([Cast])-->D([LogicalOr])
 * A2[(other)]-->B2([Contiguous])-->C2([Cast])-->D([LogicalOr])
 * D([LogicalOr])-->E([Cast])-->F([ViewCopy])-->G[(out)]
 * ```
 *
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL,
 * shape needs to satisfy the broadcast relationship with other. Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with other.
 * @param [in] other: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL,
 * Shape needs to satisfy the broadcast relationship with self. Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with self.
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL,
 * Shape is the shape after broadcasting both self and other. The data format supports ND, and the data format needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLogicalOrGetWorkspaceSize(const aclTensor* self,
                                                     const aclTensor* other,
                                                     aclTensor* out,
                                                     uint64_t* workspaceSize,
                                                     aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnLogicalOr is used to perform calculations.
 *
 * Operator function: complete logical or calculation
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A1[(self)] -->B1([Contiguous])-->C1([Cast])-->D([LogicalOr])
 * A2[(other)]-->B2([Contiguous])-->C2([Cast])-->D([LogicalOr])
 * D([LogicalOr])-->E([Cast])-->F([ViewCopy])-->G[(out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnLogicalOrGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLogicalOr(void* workspace, uint64_t workspaceSize,
                                     aclOpExecutor* executor,
                                     aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceLogicalOr calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] selfRef: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8,
 * BOOL, COMPLEX64, COMPLEX128, BFLOAT16 (supported by 910B), the shape needs to satisfy the broadcast relationship with other, and the shape after broadcast needs to be with selfRef
 * The shape of itself is the same. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] other: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL,
 * COMPLEX64, COMPLEX128, BFLOAT16 (supported by 910B), shape needs to satisfy the broadcast relationship with selfRef. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceLogicalOrGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnInplaceLogicalOr is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceLogicalOrGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceLogicalOr(void* workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LogicalOr_H_
