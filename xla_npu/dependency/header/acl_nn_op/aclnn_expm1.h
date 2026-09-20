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

#ifndef OP_API_INC_EXPM1_H_
#define OP_API_INC_EXPM1_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnExpm1 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Use the input self as the exponent, calculate the power of the natural constant e, and subtract 1 from the exponent calculation result. For scenarios where the value of self is small, it provides higher accuracy than directly calculating the results using formulas.
 * Calculation formula: $$ out_i = {e}^{self_i} - 1 $$
 * Implementation Description - Computational Graph
 *
 * ```mermaid
 * graph LR
 *  A[(self)] -->B([l0op::Contiguous])
 *  B --> D([l0op::Expm1])
 *  D --> H([l0op::Cast])
 *  H --> I([l0op::ViewCopy])
 *  I --> J[(out)]
 * ```
 *
 * @param [in] self: Enter `self` in the formula. The data type supports INT64, BOOL, FLOAT, BFLOAT16, and FLOAT16. The shape needs to be consistent with out.
 * Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#))
 * @param [out] out: Enter `out` in the formula, the data type supports FLOAT, BFLOAT16, and FLOAT16, and the shape needs to be consistent with self.
 * Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnExpm1GetWorkspaceSize(const aclTensor* self,
                                                 aclTensor* out,
                                                 uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnExpm1, used to perform calculations
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnExpm1GetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnExpm1(void* workspace, uint64_t workspace_size,
                                 aclOpExecutor* executor,
                                 const aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceExpm1 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Use the input self as the exponent, calculate the power of the natural constant e, and subtract 1 from the exponent calculation result. For scenarios where the value of self is small, it provides higher accuracy than directly calculating the results using formulas.
 * Calculation formula: $$ out_i = {e}^{self_i} - 1 $$
 * Implementation Description - Computational Graph
 *
 * ```mermaid
 * graph LR
 *  A[(selfRef)] -->B([l0op::Contiguous])
 *  B --> D([l0op::Expm1])
 *  D --> H([l0op::Cast])
 *  H --> I([l0op::ViewCopy])
 *  I --> J[(selfRef)]
 * ```
 *
 * @param [in] selfRef: Enter `selfRef` in the formula, and the data type supports FLOAT and FLOAT16.
 * Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#))
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceExpm1GetWorkspaceSize(
    aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnInplaceExpm1, used to perform calculations
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceExpm1GetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceExpm1(void* workspace,
                                        uint64_t workspace_size,
                                        aclOpExecutor* executor,
                                        const aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_EXPM1_H_
