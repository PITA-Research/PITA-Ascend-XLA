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

#ifndef OP_API_INC_LEVEL2_ACLNN_SQUARE_H_
#define OP_API_INC_LEVEL2_ACLNN_SQUARE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnSquare calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Returns the result of the square of each element in the input Tensor
 * Calculation formula:
 * $$ out_{i} = self_{i} \times self_{i} $$
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *   A[(self)]--->B([l0op::Contiguous])
 *   B--->C([l0op::Square])
 *   C--->D([l0op::ViewCopy])
 *   D--->E[(out)]
 * ```
 *
 * @param [in] self: input parameter to be calculated for square. aclTensor on the NPU device side,
 * The data types support FLOAT, FLOAT16, INT64, INT32, and BF16, the data format supports ND, and non-continuous Tensors are supported.
 * @param [in] out: The output parameter of square calculation. aclTensor on the NPU device side,
 * The data types support FLOAT, FLOAT16, DOUBLE, INT32, and BF16, the data format supports ND, and non-continuous Tensors are supported.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnSquareGetWorkspaceSize(const aclTensor* self,
                                                  aclTensor* out,
                                                  uint64_t* workspaceSize,
                                                  aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnSquare is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSquareGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnSquare(void* workspace, uint64_t workspaceSize,
                                  aclOpExecutor* executor,
                                  const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_SQUARE_H_