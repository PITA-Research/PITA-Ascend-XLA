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
#ifndef OP_API_INC_STD_H_
#define OP_API_INC_STD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnStd calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: calculate sample standard deviation
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     D([dim]) --> C([l0op::ReduceMean])
 *     A[(self)] --> B([l0op::Contiguous])
 *     B --> C
 *     E([keepdim=true]) --> C
 *     C --> H([l0op::Expand])
 *     J([dim]) --> I
 *     K([correction]) --> I
 *     O[(self)] --> I
 *     P --> I([l0op::ReduceStdWithMean])
 *     L([keepdim]) --> I
 *     H --> P[(meanOut)]
 *     I --> M([l0op::ViewCopy])
 *     M --> N[(out)]
 * ```
 *
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports FLOAT, BFLOAT16, and FLOAT16. The data format supports ND. Supports non-contiguous Tensors.
 * @param [in] dim: aclIntArray on the host side. The supported data types are INT32 and INT64.
 * @param [in] correction: shaping and correction value on the host side.
 * @param [in] keepdim: Boolean type on the host side, whether to retain the dimensions of the input tensor in the output tensor.
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports FLOAT, BFLOAT16, and FLOAT16. The data format supports ND. Supports non-contiguous Tensors.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnStdGetWorkspaceSize(const aclTensor* self,
                                               const aclIntArray* dim,
                                               const int64_t correction,
                                               bool keepdim, aclTensor* out,
                                               uint64_t* workspaceSize,
                                               aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnStd is used to perform calculations.
 *
 * Operator function: calculate sample standard deviation
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     D([dim]) --> C([l0op::ReduceMean])
 *     A[(self)] --> B([l0op::Contiguous])
 *     B --> C
 *     E([keepdim=true]) --> C
 *     C --> H([l0op::Expand])
 *     J([dim]) --> I
 *     K([correction]) --> I
 *     O[(self)] --> I
 *     P --> I([l0op::ReduceStdWithMean])
 *     L([keepdim]) --> I
 *     H --> P[(meanOut)]
 *     I --> M([l0op::ViewCopy])
 *     M --> N[(out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSubGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnStd(void* workspace, uint64_t workspaceSize,
                               aclOpExecutor* executor,
                               const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_STD_H_
