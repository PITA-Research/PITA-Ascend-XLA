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

#ifndef OP_API_INC_LEVEL2_ACLNN_CUMSUM_H_
#define OP_API_INC_LEVEL2_ACLNN_CUMSUM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: Accumulate the elements of the input tensor self in sequence according to the specified dimension dim, and save the result to the output tensor out.
 * Calculation formula:
 * $x_{i}$ is an element in the input tensor self from the perspective of dimension dim (the subscripts of other dimensions remain unchanged, only the subscripts of dim dimension increase in sequence),
 * $y_{i}$ is the element at the corresponding position in the output tensor out, then
 *
 * $$
 * y_{i} = x_{1} + x_{2} + x_{3} + ...... + x_{i}
 * $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(self)] --> B([l0op::Contiguous])
 *     B --> C([l0op::Cast])
 *     C --> D([l0op::Cumsum])
 *     E((dim)) --> D
 *     F((dtype)) --> C
 *     D --> G([l0op::ViewCopy])
 *     G --> H[(out)]
 * ```
 */

/**
 * The first interface of @brief aclnnCumsum calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports FLOAT, DOUBLE, COMPLEX64, COMPLEX128, UINT8, INT8, INT16, INT32,
 * INT64, FLOAT16, BFLOAT16, BOOL, the data type needs to be convertible to out data type, supports non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in] dim: Integer on the host side, the data type supports INT64.
 * @param [in]
 * dtype: Data type enumeration on the host side, indicating the data type required for output tensor, supporting FLOAT, FLOAT16, INT32, DOUBLE, UINT8, INT8, INT16,
 * INT64, COMPLEX64, COMPLEX128, BFLOAT16 (supported by 910B), and must be consistent with the data type of the output tensor out.
 * @param [in] out: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, INT32, DOUBLE, UINT8, INT8, INT16, INT64,
 * COMPLEX64, COMPLEX128, BFLOAT16 (supported by 910B), and the data type needs to be consistent with the incoming dtype, and the shape needs to be consistent with self, supporting non-continuous Tensor, data
 * The format supports ND, but the data dimension does not support more than 8 dimensions.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCumsumGetWorkspaceSize(
    const aclTensor* self, int64_t dim, aclDataType dtype, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnCumsum is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnCumsumGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCumsum(void* workspace, uint64_t workspaceSize,
                                  aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief The first interface of aclnnCumsumV2 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnCumsumV2GetWorkspaceSize(
    const aclTensor* self, int64_t dim, bool exclusive, bool reverse,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnCumsumV2(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_CUMSUM_H_
