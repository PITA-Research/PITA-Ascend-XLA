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

#ifndef OP_API_INC_ROUND_H_
#define OP_API_INC_ROUND_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnRound calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Round the input value to the nearest integer. If the value is the same distance from two integers, it will be rounded to an even number.
 * @param [in] self: aclTensor on the NPU device side, data type supports BFLOAT16, FLOAT16, FLOAT32, DOUBLE, INT32, INT64,
 * The data format supports ND, but the shape does not support 9D and above.
 * @param [in] out: aclTensor on the NPU device side, the data type and shape are consistent with self, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnRoundGetWorkspaceSize(const aclTensor* self,
                                                 aclTensor* out,
                                                 uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnRound, used to perform calculations
 *
 * Operator function: Round the input value to the nearest integer. If the value is the same distance from two integers, it will be rounded to an even number.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnRoundGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRound(void* workspace, uint64_t workspaceSize,
                                 aclOpExecutor* executor,
                                 const aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceRound calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Round the input value to the nearest integer. If the value is the same distance from two integers, it will be rounded to an even number.
 * @param [in] selfRef: aclTensor on the NPU device side, data type supports FLOAT16, FLOAT32, DOUBLE, INT32, INT64,
 * The data format supports ND, but the shape does not support 9D and above.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnInplaceRoundGetWorkspaceSize(
    const aclTensor* selfRef, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnInplaceRound, used to perform calculations
 *
 * Operator function: Round the input value to the nearest integer. If the value is the same distance from two integers, it will be rounded to an even number.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSinGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceRound(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        const aclrtStream stream);

/**
 * The first interface of @brief aclnnRoundDecimals calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Function description: Rounds the elements of the input Tensor to the specified number of digits.
 * Parameter description:
 * @param [in]   self
 * Input Tensor, the data type supports FLOAT, BFLOAT16, FLOAT16, DOUBLE, INT32, INT64. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] decimals: The number of digits to be rounded to. The data type supports INT.
 * @param [in]   out
 * Output Tensor, data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor returns the op executor, including the operator calculation process.
 * @return aclnnStatus return status code
 */
ACLNN_API aclnnStatus aclnnRoundDecimalsGetWorkspaceSize(
    const aclTensor* self, int64_t decimals, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief The second interface of aclnnRoundDecimals is used to perform calculations.
 * Function description: Rounds the elements of the input Tensor to the specified number of digits.
 * Implementation instructions:
 * Basic path for API calculation:
```mermaid
graph LR
    A[(Self)] -->B([l0op::Contiguous])
    B --> D([l0op::Round])

    C((decimals)) --> D([l0op::Cast])
    D --> E([l0op::ViewCopy])
    E --> F[(Out)]
```
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnRoundDecimalsGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRoundDecimals(void* workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceRoundDecimals calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Function description: Rounds the elements of the input Tensor to the specified number of digits.
 * Parameter description:
 * @param [in]   selfRef
 * Input Tensor, the data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] decimals: The number of digits to be rounded to. The data type supports INT.
 * @param [out] workspaceSize returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor returns the op executor, including the operator calculation process.
 * @return aclnnStatus return status code
 */
ACLNN_API aclnnStatus aclnnInplaceRoundDecimalsGetWorkspaceSize(
    aclTensor* selfRef, int64_t decimals, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnInplaceRoundDecimals is used to perform calculations
 * Operator function: Rounds the elements of the input Tensor to the specified number of digits.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnInplaceRoundDecimalsGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceRoundDecimals(void* workspace,
                                                uint64_t workspaceSize,
                                                aclOpExecutor* executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ROUND_H_