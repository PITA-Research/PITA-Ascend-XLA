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

#ifndef OP_API_INC_LEVEL2_ACLNN_ERF_H_
#define OP_API_INC_LEVEL2_ACLNN_ERF_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnErf calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: Returns the value of the error function corresponding to each element in the input Tensor
 * Calculation formula:
 * $$ erf(x)=\frac{2}{\sqrt{\pi } } \int_{0}^{x} e^{-t^{2} } \mathrm{d}t $$
 *
 * Calculation diagram 1: as shown below
 * Scenario: When the input type is within the range supported by the Erf operator (FLOAT32, FLOAT16, BFLOAT16, FLOAT64), use the Erf operator to complete the calculation.
 * ```mermaid
 * graph LR
 *     A[(Self)]  --> B([l0op::Contiguous])
 *     B -->C([l0op::Erf])
 *     C --> D([l0op::ViewCopy])
 *     D --> E[(out)]
 * ```
 *
 * Calculation diagram 2: as shown below
 * Scenario: The data type of self is (BOOL, INT64), CAST the data type of self to FLOAT32, and then use the Erf operator to complete the calculation.
 * ```mermaid
 * graph LR
 *     A[(Self)]  --> B([l0op::Contiguous])
 *     B -->C([l0op::Cast])
 *     C -->D([l0op::Erf])
 *     D -->H([l0op::Cast])
 *     H --> E([l0op::ViewCopy])
 *     E --> F[(out)]
 * ```
 *
 * @param [in] self: input parameter to be calculated for erf. aclTensor on the NPU device side,
 * The data type supports FLOAT64, FLOAT32, FLOAT16, BFLOAT16, BOOL, and INT64, the data format supports ND, and non-continuous Tensors are supported.
 * @param [in] out: The output parameter of erf calculation. aclTensor on the NPU device side,
 * The data type supports FLOAT64, FLOAT32, FLOAT16, and BFLOAT16, the data format supports ND, and non-continuous Tensors are supported.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnErfGetWorkspaceSize(const aclTensor* self,
                                               aclTensor* out,
                                               uint64_t* workspaceSize,
                                               aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnErf is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnErfGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnErf(void* workspace, uint64_t workspaceSize,
                               aclOpExecutor* executor, aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceErf calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: Returns the value of the error function corresponding to each element in the input Tensor
 * Calculation formula:
 * $$ erf(x)=\frac{2}{\sqrt{\pi } } \int_{0}^{x} e^{-t^{2} } \mathrm{d}t $$
 *
 * Calculation graph: as shown below
 * ```mermaid
 * graph LR
 *     A[(Self)]  --> B{l0op::Contiguous}
 *     B -->C([l0op::Erf])
 *     C --> D{l0op::ViewCopy}
 *     D --> E[(out)]
 * ```
 *
 * @param [in] selfRef: input parameter to be calculated for erf. aclTensor on the NPU device side,
 * The data type supports FLOAT64, FLOAT32, FLOAT16, and BFLOAT16, the data format supports ND, and non-continuous Tensors are supported.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceErfGetWorkspaceSize(aclTensor* selfRef,
                                                      uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceErf is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnErfGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceErf(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ERF_H_