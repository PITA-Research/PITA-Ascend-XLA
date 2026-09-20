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

#ifndef OP_API_INC_LEVEL2_ACLNN_ALL_H_
#define OP_API_INC_LEVEL2_ACLNN_ALL_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACLNN_MAX_SHAPE_RANK 8
#define DIM_BITS_LEN 64

/**
 * The first interface of @brief aclnnAll calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: perform AND operation on all elements in Tensor.

 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *   A[(self)] -->B([l0::Contiguous])
 *   B --> E([l0::ReduceAll])
 *   C[(dim)] --> E
 *   D[(keepdim)] --> E
 *   E --> G([l0::ViewCopy])
 *   G --> H[(out)]
 * ```
 *
 * @param [in] self: input parameter to be calculated for all. aclTensor on the NPU device side,
 * The data type supports BOOL, the data format supports ND, and non-continuous Tensors are supported.
 * @param [in] dim: Dimension that needs to be compressed, the value needs to be within the range of the input Tensor, supports negative numbers, and the data type supports INT
 * @param [in] keepdim: Whether the output tensor `dim` is consistent with the input, the data type supports BOOL
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAllGetWorkspaceSize(const aclTensor* self,
                                               const aclIntArray* dim,
                                               bool keepdim, aclTensor* out,
                                               uint64_t* workspaceSize,
                                               aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnAll is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAllGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAll(void* workspace, uint64_t workspaceSize,
                               aclOpExecutor* executor,
                               const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ALL_H_