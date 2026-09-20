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
#ifndef OP_API_INC_SQRT_H_
#define OP_API_INC_SQRT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnSqrt calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: completes the calculation of the square root of a non-negative number, and returns nan in the case of a negative number.
 * $$
 * sqrt(x)=\begin{cases}
 * \sqrt x, & x\ge 0 \\
 * nan, &  x\lt 0
 * \end{cases}
 * $$
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(self)] -->B([L0::Contiguous])
 * B --> C([L0::Sqrt])
 * C --> D([L0::ViewCopy])
 * D --> E[(out)]
 * ```
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports FLOAT16, FLOAT32, FLOAT64, COMPLEX64, COMPLEX128,
 * INT32, INT64, INT16, INT8, BOOL, BF16, support non-continuous Tensor.
 * Supports non-continuous Tensor, and the data format supports ND
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports FLOAT16, FLOAT32, FLOAT64, COMPLEX64, COMPLEX128,
 * INT32, INT64, INT16, INT8, BOOL, BF16, support non-continuous Tensor
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] opExecutor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSqrtGetWorkspaceSize(const aclTensor* self,
                                                aclTensor* out,
                                                uint64_t* workspaceSize,
                                                aclOpExecutor** opExecutor);
/**
 * @brief The second interface of aclnnSqrt is used to perform calculations.
 * Operator function: completes the calculation of the square root of a non-negative number, and returns nan in the case of a negative number.
 * $$
 * sqrt(x)=\begin{cases}
 * \sqrt x, & x\ge 0 \\
 * nan, &  x\lt 0
 * \end{cases}
 * $$
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(self)] -->B([L0::Contiguous])
 * B --> C([L0::Sqrt])
 * C --> D([L0::ViewCopy])
 * D --> E[(out)]
 * ```
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSqrtGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] opExecutor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSqrt(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* opExecutor, aclrtStream stream);
/**
 * The first interface of @brief aclnnInplaceSqrt calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: completes the calculation of the square root of a non-negative number, and returns nan in the case of a negative number.
 * $$
 * sqrt(x)=\begin{cases}
 * \sqrt x, & x\ge 0 \\
 * nan, &  x\lt 0
 * \end{cases}
 * $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(self)] -->B([L0::Contiguous])
 * B --> C([L0::Sqrt])
 * C --> D([L0::ViewCopy])
 * D --> E[(out)]
 * ```
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports FLOAT16, FLOAT32, FLOAT64, COMPLEX64, COMPLEX128,
 * INT32, INT64, INT16, INT8, BOOL, BF16, support non-continuous Tensor
 * Supports non-continuous Tensor, and the data format supports ND
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceSqrtGetWorkspaceSize(
    aclTensor* self, uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief The second interface of aclnnInplaceSqrt is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceReluGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceSqrt(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SQRT_H_
