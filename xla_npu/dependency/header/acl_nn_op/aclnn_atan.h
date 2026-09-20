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
#ifndef OP_API_INC_ATAN_H_
#define OP_API_INC_ATAN_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnAtan calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Function description: Output after performing arctangent operation on each element of the input matrix.
 * Calculation formula:
 * out_{i}=tan^{-1}(input_{i})
 * Parameter description:
 * @param [in]   input
 * Input Tensor, the data type supports FLOAT, BFLOAT16, FLOAT16, DOUBLE. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in]   out
 * Output Tensor, data type supports FLOAT, BFLOAT16, FLOAT16, DOUBLE. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor returns the op executor, including the operator calculation process.
 * @return aclnnStatus return status code
 */
ACLNN_API aclnnStatus aclnnAtanGetWorkspaceSize(const aclTensor* input,
                                                aclTensor* out,
                                                uint64_t* workspaceSize,
                                                aclOpExecutor** executor);
/**
 * @brief aclnnAtan's second interface is used to perform calculations.
 * Function description: Output after performing arctangent operation on each element of the input matrix..
 * Calculation formula:
 * out_{i}=tan^{-1}(input_{i})
 * Implementation instructions:
 * Basic path for API calculation:
```mermaid
graph LR
    A[(Self)] -->B([l0op::Contiguous])
    B --> C([l0op::Atan])
    C --> G([l0op::Cast])
    G --> E([l0op::ViewCopy])
    E --> S[(Out)]
```
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAtan(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor,
                                const aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceAtan calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Function description: Perform inverse cosine operation on each element of the input matrix and output it.
 * Calculation formula:
 * out_{i}=tan^{-1}(input_{i})
 * Parameter description:
 * @param [in]   input
 * Input Tensor, data type supports INT8, INT16, INT32, INT64, UINT8, BOOL, FLOAT, FLOAT16, DOUBLE. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in]   out
 * Output Tensor, data type supports FLOAT, FLOAT16, DOUBLE. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor returns the op executor, including the operator calculation process.
 * @return aclnnStatus return status code
 */
ACLNN_API aclnnStatus aclnnInplaceAtanGetWorkspaceSize(
    aclTensor* inputRef, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnInplaceAtan is used to perform calculations
 *
 * Operator function: Complete the atan operation on the input Tensor in place
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAtanGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceAtan(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       const aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif
