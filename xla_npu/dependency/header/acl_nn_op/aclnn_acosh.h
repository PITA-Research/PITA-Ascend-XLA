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
#ifndef OP_API_INC_ACOSH_H_
#define OP_API_INC_ACOSH_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAcosh's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: perform inverse hyperbolic cosine operation on each element in the input Tensor and output it.
 * Calculation formula:
 * $$ out_{i}=cosh^{-1}(self_{i}) $$
 *
 * @param [in] self: Input Tensor, the data type supports integer, floating point, and complex number types, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] out: Output Tensor. The data type supports floating point and complex number types. It supports non-continuous Tensor. The data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAcoshGetWorkspaceSize(const aclTensor* self,
                                                 aclTensor* out,
                                                 uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnAcosh is used to perform calculations.
 *
 * Operator function: perform inverse hyperbolic cosine operation on each element in the input Tensor and output it.
 * Calculation formula:
 * $$ out_{i}=cosh^{-1}(self_{i}) $$
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAcoshGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAcosh(void* workspace, uint64_t workspaceSize,
                                 aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief The first interface of aclnnInplaceAcosh calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: perform inverse hyperbolic cosine operation on each element in the input Tensor and output it.
 * Calculation formula:
 * $$ selfRef_{i}=cosh^{-1}(selfRef_{i}) $$
 *
 * @param [in] selfRef: Input Tensor, the data type supports floating point, complex number type, non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceAcoshGetWorkspaceSize(
    aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceAcosh is used to perform calculations.
 *
 * Operator function: perform inverse hyperbolic cosine operation on each element in the input Tensor and output it.
 * Calculation formula:
 * $$ selfRef_{i}=cosh^{-1}(selfRef_{i}) $$
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceAcoshGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceAcosh(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ACOSH_H_
