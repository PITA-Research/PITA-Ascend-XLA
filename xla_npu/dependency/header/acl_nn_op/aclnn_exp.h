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

#ifndef OP_API_INC_LEVEL2_ACLNN_EXP_H_
#define OP_API_INC_LEVEL2_ACLNN_EXP_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: Returns a new tensor, each element of this tensor is the exponent of the corresponding element of the input tensor.
 * Calculation formula: as follows
 * $$
 *     out_{i} = e^{self_{i}}
 * $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([l0op::Contiguous])
 *     B --> C([l0op::Exp])
 *     C --> D([l0op::Cast])
 *     D --> E([l0op::ViewCopy])
 *     E --> F[(out)]
 * ```
 */

/**
 * The first interface of @brief aclnnExp calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16, DOUBLE, COMPLEX64, COMPLEX128, BOOL,
 * Supports non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in] out: npu
 * The data type of aclTensor on the device side needs to be a data type that can be converted by self. The shape needs to be consistent with self. It supports non-continuous Tensor and data.
 * The format supports ND, but the data dimension does not support more than 8 dimensions.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnExpGetWorkspaceSize(const aclTensor* self,
                                               aclTensor* out,
                                               uint64_t* workspaceSize,
                                               aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnExp is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnExpGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnExp(void* workspace, uint64_t workspaceSize,
                               aclOpExecutor* executor, aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceExp calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] selfRef: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16, DOUBLE, COMPLEX64, COMPLEX128, supported
 * For non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceExpGetWorkspaceSize(aclTensor* selfRef,
                                                      uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceExp is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceExpGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceExp(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_EXP_H_
