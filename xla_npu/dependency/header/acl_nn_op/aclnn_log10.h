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
#ifndef OP_API_INC_LOG10_H_
#define OP_API_INC_LOG10_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnLog10 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: complete the calculation of the input base 10 logarithm
 * Calculation formula:
 * $$ output_i=log_{10}(self_i) $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([l0op::Contiguous])
 *     B -->C([l0op::Cast])
 *     C -->D([l0op::Log])
 *     D -->E([l0op::Cast])
 *     E -->I([l0op::ViewCopy])
 *     I -->J[(out)]
 * ```
 *
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports BOOL, INT8, INT16, INT32, INT64, UINT8, FLOAT, FLOAT16, and BFLOAT16.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] out: aclTensor on the NPU device side. The data type supports FLOAT, FLOAT16, and BFLOAT16. And the data type is a data type that can be converted by self.
 * The data shape is consistent with self. The data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLog10GetWorkspaceSize(const aclTensor* self,
                                                 aclTensor* out,
                                                 uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnLog10 is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnLog10GetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLog10(void* workspace, uint64_t workspaceSize,
                                 aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief The first interface of aclnnInplaceLog10 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the calculation of the base 10 logarithm of the input and update it on the original Tensor
 * Calculation formula:
 * $$ output_i=log_{10}(self_i) $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([l0op::Contiguous])
 *     B -->C([l0op::Cast])
 *     C -->D([l0op::Log])
 *     D -->E([l0op::Cast])
 *     E -->I([l0op::ViewCopy])
 *     I -->J[(out)]
 * ```
 *
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports BOOL, INT8, INT16, INT32, INT64, UINT8, FLOAT, FLOAT16, and BFLOAT16.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceLog10GetWorkspaceSize(
    aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceLog10 is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceLog10GetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceLog10(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LOG10_H_
