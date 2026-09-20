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

#ifndef OP_API_INC_LEVEL2_ACLNN_CEIL_H_
#define OP_API_INC_LEVEL2_ACLNN_CEIL_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnCeil calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Returns the result of rounding down each element in the input Tensor
 * Calculation formula:
 * $$ out_{i} =ceil(self_{i}) $$
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *     A[(Self)]  --> B{l0op::Contiguous}
 *     B -->C([l0op::ceil])
 *     C --> D{l0op::ViewCopy}
 *     D --> E[(out)]
 * ```
 *
 * @param [in] self: input parameter to be used for ceil calculation. aclTensor on the NPU device side,
 * The data type supports FLOAT64, FLOAT32, FLOAT16, and BFLOAT16. The data format supports ND, and the data format needs to be consistent with out. Non-continuous Tensors are supported.
 * @param [in] out: The output parameter of ceil calculation. aclTensor on the NPU device side,
 * The data type supports FLOAT64, FLOAT32, FLOAT16, and BFLOAT16. The data format supports ND, and the data format needs to be consistent with self. Non-continuous Tensors are supported.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCeilGetWorkspaceSize(const aclTensor* self,
                                                aclTensor* out,
                                                uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnCeil is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnCeilGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCeil(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor,
                                const aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceCeil calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] selfRef: input parameter to be used for ceil calculation. aclTensor on the NPU device side,
 * The data type supports FLOAT64, FLOAT32, FLOAT16, and BFLOAT16. The data format supports ND, and the data format needs to be consistent with out. Non-continuous Tensors are supported.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceCeilGetWorkspaceSize(
    aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceCeil is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnCeilGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceCeil(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_CEIL_H_