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
#ifndef OP_API_INC_LEVEL2_ACLNN_ERFC_H_
#define OP_API_INC_LEVEL2_ACLNN_ERFC_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnErfc calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: Returns the value of the error complementary function corresponding to each element in the input Tensor.
 * Calculation formula:
 * $$ erfc(x)=1-\frac{2}{\sqrt{\pi } } \int_{0}^{x} e^{-t^{2} } \mathrm{d}t $$
 *
 * Scenario: When the input type is within the range supported by the Erfc operator (FLOAT32, BFLOAT16, FLOAT16, FLOAT64), use the Erfc operator to complete the calculation.
 * Calculation graph:
 * ```mermaid
 * graph LR
 *     A[(Self)]  --> B([l0op::Contiguous])
 *     B --> C([l0op::Erfc])
 *     C --> D([l0op::Cast])
 *     D --> E([l0op::ViewCopy])
 *     E --> F[(out)]
 * ```
 *
 * Scenario: The data type of self is BOOL or INT64, CAST the data type of self to FLOAT32, and then use the Erfc operator to complete the calculation.
 * Calculation graph:
 * ```mermaid
 * graph LR
 *     A[(Self)]  --> B([l0op::Contiguous])
 *     B -->C([l0op::Cast])
 *     C -->D([l0op::Erfc])
 *     D --> E([l0op::Cast])
 *     E --> F([l0op::ViewCopy])
 *     F --> G[(out)]
 * ```
 *
 * @param [in] self: input parameter to be calculated by erfc. aclTensor on the NPU device side,
 * The data type supports FLOAT64, FLOAT32, BFLOAT16, FLOAT16, BOOL, and INT64, the data format supports ND, and non-continuous Tensors are supported.
 * @param [in] out: The output parameter calculated by erfc. aclTensor on the NPU device side,
 * The data type supports FLOAT64, FLOAT32, BFLOAT16, and FLOAT16. The default is consistent with self. If the self data type is BOOL or INT64, the data type of out defaults to FLOAT32.
 * The data format supports ND, is consistent with the shape of self, and supports non-continuous Tensors.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnErfcGetWorkspaceSize(const aclTensor* self,
                                                aclTensor* out,
                                                uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnErfc is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnErfcGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnErfc(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor,
                                const aclrtStream stream);

/**
 * @brief The first interface of aclnnInplaceErfc calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: Returns the value of the error complementary function corresponding to each element in the input Tensor
 * Calculation formula:
 * $$ erfc(x)=1-\frac{2}{\sqrt{\pi } } \int_{0}^{x} e^{-t^{2} } \mathrm{d}t $$
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *     A[(Self)]  --> B{l0op::Contiguous}
 *     B -->C([l0op::Erfc])
 *     C --> D{l0op::ViewCopy}
 *     D --> E[(out)]
 * ```
 *
 * @param [in] selfRef: input parameter to be calculated by erfc. aclTensor on the NPU device side,
 * The data type supports FLOAT64, FLOAT32, BFLOAT16, and FLOAT16, the data format supports ND, and non-continuous Tensors are supported.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceErfcGetWorkspaceSize(
    const aclTensor* selfRef, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceErfc is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceErfcGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceErfc(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ERFC_H_