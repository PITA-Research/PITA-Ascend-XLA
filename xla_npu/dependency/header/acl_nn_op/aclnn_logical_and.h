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
#ifndef OP_API_INC_LogicalAnd_H_
#define OP_API_INC_LogicalAnd_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnLogicalAnd calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: complete logic and calculation
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A1[(self)] -->B1([Contiguous])-->C1([Cast])-->D([LogicalAnd])
 * A2[(other)]-->B2([Contiguous])-->C2([Cast])-->D([LogicalAnd])
 * D([LogicalAnd])-->E([Cast])-->F([ViewCopy])-->G[(out)]
 * ```
 *
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8,
 * BOOL、COMPLEX64、COMPLEX128，
 * shape needs to satisfy the broadcast relationship with other. Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with other.
 * @param [in] other: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8,
 * BOOL、COMPLEX64、COMPLEX128,
 * Shape needs to satisfy the broadcast relationship with self. Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with self.
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8,
 * BOOL、COMPLEX64、COMPLEX128，
 * Shape is the shape after broadcasting both self and other. The data format supports ND, and the data format needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLogicalAndGetWorkspaceSize(const aclTensor* self,
                                                      const aclTensor* other,
                                                      aclTensor* out,
                                                      uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnLogicalAnd is used to perform calculations.
 *
 * Operator function: complete logic and calculation
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A1[(self)] -->B1([Contiguous])-->C1([Cast])-->D([LogicalAnd])
 * A2[(other)]-->B2([Contiguous])-->C2([Cast])-->D([LogicalAnd])
 * D([LogicalAnd])-->E([Cast])-->F([ViewCopy])-->G[(out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnLogicalAndGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLogicalAnd(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor,
                                      aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceLogicalAnd calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: complete logic and calculation
 *
 * @param [in] selfRef: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8,
 * BOOL、COMPLEX64、COMPLEX128，
 * shape needs to satisfy the broadcast relationship with other. Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with other.
 * @param [in] other: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8,
 * BOOL、COMPLEX64、COMPLEX128,
 * Shape needs to satisfy the broadcast relationship with selfRef. Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with selfRef.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceLogicalAndGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnInplaceLogicalAnd is used to perform calculations.
 *
 * Operator function: complete logic and calculation
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnLogicalAndGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceLogicalAnd(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LogicalAnd_H_
