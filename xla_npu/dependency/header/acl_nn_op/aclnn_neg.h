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

#ifndef OP_API_INC_NEG_H_
#define OP_API_INC_NEG_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnNeg calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: complete the inverse calculation for each element of the input
 * Calculation formula:
 * $$ output​=(-1) * self $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 *  graph LR
 *      A[(self)] -->B([Contiguous])
 *      B -->D([Neg])
 *      D-->E([Cast])
 *      E-->F([ViewCopy])
 *      F-->G[(out)]
 * ```
 *
 * @param [in] self: aclTensor on the NPU device side. The data type supports integer and floating point types. It supports non-continuous Tensor. The data format supports ND.
 * @param [in] out: aclTensor on the NPU device side, the data type supports integer and floating point types, the data format supports ND, and the data format needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnNegGetWorkspaceSize(const aclTensor* self,
                                               aclTensor* out,
                                               uint64_t* workspaceSize,
                                               aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnInplaceNeg calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: complete the inverse calculation for each element of the input
 * Calculation formula:
 * $$ selfRef ​=(-1) * selfRef $$
 *
 * @param [in] selfRef: aclTensor on the NPU device side. The data type supports integer and floating point types. It supports non-continuous Tensor. The data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceNegGetWorkspaceSize(aclTensor* selfRef,
                                                      uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnNeg is used to perform calculations.
 *
 * Operator function: complete addition calculation
 * Calculation formula:
 * $$ output_i = self_i+alpha*other_i $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 *  graph LR
 *      A[(self)] -->B([Contiguous])
 *      B -->D([Neg])
 *      D-->E([Cast])
 *      E-->F([ViewCopy])
 *      F-->G[(out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnNegGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */

ACLNN_API aclnnStatus aclnnNeg(void* workspace, uint64_t workspaceSize,
                               aclOpExecutor* executor,
                               const aclrtStream stream);

/**
 * @brief The second interface of aclnnInplaceNeg is used to perform calculations.
 *
 * Operator function: complete the inverse calculation for each element of the input
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceNegGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceNeg(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif /* OP_API_INC_NEG_H_ */