/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_ERFINV_H_
#define OP_API_INC_LEVEL2_ACLNN_ERFINV_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnErfinv calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: erfinv is the inverse function of Gaussian error function erf. Returns the quantile of the standard normal distribution function corresponding to each element in the input Tensor.
 * Calculation formula: as follows
 * $$
 * y = erfinv(x) \\
 * x = erf(y)=\frac{2}{\sqrt{\pi } } \int_{0}^{y} e^{-t^{2} } \mathrm{d}t
 * $$
 *
 * Calculation diagram: as follows
 * Scenario: When the input type is within the range supported by the Erfinv operator (FLOAT32, FLOAT16, BFLOAT16), use the Erfinv operator to complete the calculation.
 * ```mermaid
 * graph LR
 * A[(Self)] --> B([l0op::Contiguous]) --> C([l0op::Erfinv])
 * C --> D([l0op::Cast]) --E D([l0op::ViewCopy]) --> F[(out)]
 * ```
 *
 * Integer types: BOOL, INT8, INT16, INT32, INT64, UINT8, first convert to FLOAT, and then calculate:
 * ```mermaid
 * graph LR
 * A[(Self)] --> B([l0op::Contiguous]) --> C([l0op::Cast]) --> D([l0op::Erfinv])
 * D --> E([l0op::Cast]) --> F([l0op::ViewCopy]) --> G[(out)]
 * ```
 *
 * @param [in] self: input parameter to be calculated by erfinv. aclTensor on the NPU device side,
 * The data types support FLOAT32, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, BOOL, the data format supports ND, and non-continuous Tensors are supported.
 * @param [in] out: The output parameter calculated by erfinv. aclTensor on the NPU device side,
 * The data type supports FLOAT32, FLOAT16, and BFLOAT16, the data format supports ND, the shape is consistent with self, and non-continuous Tensors are supported.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnErfinvGetWorkspaceSize(const aclTensor* self,
                                                  aclTensor* out,
                                                  uint64_t* workspaceSize,
                                                  aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnErfinv is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnErfinvGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnErfinv(void* workspace, uint64_t workspaceSize,
                                  aclOpExecutor* executor,
                                  const aclrtStream stream);

/**
 * @brief The first interface of aclnnInplaceErfinv calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: Returns the quantile of the standard normal distribution function corresponding to each element in the input Tensor
 *
 * @param [in] selfRef: input parameter to be calculated by erfinv. aclTensor on the NPU device side,
 * The data type supports FLOAT32, FLOAT16, and BFLOAT16, the data format supports ND, and non-continuous Tensors are supported.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceErfinvGetWorkspaceSize(
    const aclTensor* selfRef, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceinv is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnErfinvGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceErfinv(void* workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ERFINV_H_