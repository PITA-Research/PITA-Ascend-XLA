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

#ifndef OP_API_INC_BITWISE_NOT_H_
#define OP_API_INC_BITWISE_NOT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnBitwiseNot calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: perform logical "NOT" or bitwise NOT operation on the input tensor

 * Calculation formula:
 * $$ output_i = \lnot self_i $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Invert])
 *     C --> D([ViewCopy])
 *     D --> E([out])
 * ```
 *
 * @param [in] self: npu
 * For aclTensor on the device side, the data type supports INT16, INT32, INT64, INT8, UINT8, BOOL, non-continuous Tensor, and the data format supports ND.
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports INT16, INT32, INT64, INT8, UINT8, BOOL, and the data type needs to be a self-convertible data type.
 * The shape is the same as self, the data format supports ND, and the data format needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBitwiseNotGetWorkspaceSize(const aclTensor* self,
                                                      aclTensor* out,
                                                      uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);
/**
 * The second interface of @brief aclnnBitwiseNot is used to perform calculations.
 *
 * Operator function: perform logical "NOT" or bitwise NOT operation on the input tensor
 * Calculation formula:
 * $$ output_i = \lnot self_i $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Invert])
 *     C --> D([ViewCopy])
 *     D --> E([out])
 * ```
 *
 * @param [in] workspace: the starting address of the workspace memory applied for on the NPU device side
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnBitwiseNotGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBitwiseNot(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_BITWISE_NOT_H_