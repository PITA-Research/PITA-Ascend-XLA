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

#ifndef OP_API_INC_LEVEL2_ACLNN_HARDTANH_H_
#define OP_API_INC_LEVEL2_ACLNN_HARDTANH_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnHardtanh calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: Limit all input elements to the range [clipValueMin, clipValueMax]. If the element is greater than max, limit it to max.
 * If the element is less than min, it is limited to min, otherwise it is equal to the element itself. The default value of min is -1.0 and the default value of max is 1.0.
 * Calculation formula: as shown below
 * $$
 * HardTanh(x) = \left\{\begin{matrix}
 * \begin{array}{l}
 * clipValueMax\ \ \ \ \ \ \ if \ \ x>clipValueMax \\
 * clipValueMin\ \ \ \ \ \ \ if\ \ x<clipValueMin \\
 * x\ \ \ \ \ \ \ \ \ \ \ otherwise \\
 * \end{array}
 * \end{matrix}\right.\begin{array}{l}
 * \end{array}
 * $$
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([l0::Contiguous])
 *     B --> E([l0::ClipByValue])
 *     C((clipValueMin)) --> E
 *     D((clipValueMax)) --> E
 *     E --> G([l0::ViewCopy])
 *     G --> H[(out)]
 * ```
 *
 * @param [in] self: input parameter to be calculated for erf. NPU
 * aclTensor on the device side, the data type supports FLOAT, BFLOAT16, FLOAT16, INT32, INT64, INT16, INT8,
 * UINT8, FLOAT64, the data format supports ND, and the data format needs to be consistent with out, and supports non-continuous Tensor.
 * @param [in] out: The output parameter of erf calculation. NPU
 * aclTensor on the device side, the data type supports FLOAT, BFLOAT16, FLOAT16, INT32, INT64, INT16, INT8, UINT8,
 * FLOAT64, the data format supports ND, and the data format needs to be consistent with self, and supports non-continuous Tensor.
 * @param [in] clipValueMin: aclScalar on the host side, the data type needs to be convertible to self.
 * @param [in] clipValueMax: aclScalar on the host side, the data type needs to be convertible to self.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnHardtanhGetWorkspaceSize(
    const aclTensor* self, const aclScalar* clipValueMin,
    const aclScalar* clipValueMax, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnHardtanh's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnHardtanhGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnHardtanh(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceHardtanh calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: Limit all input elements to the range [clipValueMin, clipValueMax]. If the element is greater than max, limit it to max.
 * If the element is less than min, it is limited to min, otherwise it is equal to the element itself. The default value of min is -1.0 and the default value of max is 1.0.
 * Calculation formula: as shown below
 * $$
 * HardTanh(x) = \left\{\begin{matrix}
 * \begin{array}{l}
 * clipValueMax\ \ \ \ \ \ \ if \ \ x>clipValueMax \\
 * clipValueMin\ \ \ \ \ \ \ if\ \ x<clipValueMin \\
 * x\ \ \ \ \ \ \ \ \ \ \ otherwise \\
 * \end{array}
 * \end{matrix}\right.\begin{array}{l}
 * \end{array}
 * $$
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([l0::Contiguous])
 *     B --> E([l0::ClipByValue])
 *     C((clipValueMin)) --> E
 *     D((clipValueMax)) --> E
 *     E --> G([l0::ViewCopy])
 *     G --> H[(out)]
 * ```
 *
 * @param [in] selfRef: input parameter to be calculated for erf. NPU
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, INT32, INT64, INT16, INT8,
 * UINT8, FLOAT64, the data format supports ND, and the data format needs to be consistent with out, and supports non-continuous Tensor.
 * @param [in] clipValueMin: aclScalar on the host side, the data type needs to be convertible to self.
 * @param [in] clipValueMax: aclScalar on the host side, the data type needs to be convertible to self.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceHardtanhGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* clipValueMin,
    const aclScalar* clipValueMax, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnInplaceHardtanh's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceHardtanhGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceHardtanh(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_HARDTANH_H_