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
#ifndef OP_API_INC_RSUB_H_
#define OP_API_INC_RSUB_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnRsubs calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: complete subtraction calculation
 * Calculation formula:
 * $$ out_i = other - self_i * alpha $$
 *
 * @param [in] self: aclTensor on the NPU device side. The data type supports integers, floating point types, and complex numbers, and the data type needs to form a mutual derivation relationship with other.
 * shape needs to satisfy the broadcast relationship with other. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] other: aclScalar on the host side, the data type supports integers, floating point types, and complex numbers, and the data type needs to form a mutual derivation relationship with self.
 * @param [in] alpha: aclScalar on the host side, the data type needs to be convertible into the data type derived from self and other.
 * @param [in] out: aclTensor on the NPU device side. The data type supports integer, floating point, and complex numbers.
 * And the data type needs to be a convertible data type after derivation of self and other, and the shape must be consistent with self. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRsubsGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, const aclScalar* alpha,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnRsubs is used to perform calculations.
 *
 * Operator function: complete subtraction calculation
 * Calculation formula:
 * $$ out_i = other - self_i * alpha $$
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnRsubsGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRsubs(void* workspace, uint64_t workspaceSize,
                                 aclOpExecutor* executor, aclrtStream stream);

/**
 * The first interface of @brief aclnnRsub calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: complete subtraction calculation
 * Calculation formula:
 * $$ out_i = other_i - self_i * alpha $$
 *
 * @param [in] self: aclTensor on the NPU device side. The data type supports integers, floating point types, and complex numbers, and the data type needs to form a mutual derivation relationship with other.
 * shape needs to satisfy the broadcast relationship with other. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] other: aclTensor on the NPU device side, the data type supports integers, floating point types, and complex numbers, and the data type needs to form a mutual derivation relationship with self.
 * Shape needs to satisfy the broadcast relationship with self. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] alpha: aclScalar on the host side, the data type needs to be convertible into the data type derived from self and other.
 * @param [in] out: aclTensor on the NPU device side. The data type supports integer, floating point, and complex numbers.
 * And the data type needs to be a convertible data type after the derivation of self and other, and the shape needs to be the shape after the broadcast of self and other.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRsubGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, const aclScalar* alpha,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnRsub is used to perform calculations.
 *
 * Operator function: complete addition calculation
 * Calculation formula:
 * $$ out_i = other_i - self_i * alpha $$
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnRsubGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRsub(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_RSUB_H_
