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

#ifndef OP_API_INC_LEVEL2_ACLNN_ABS_H_
#define OP_API_INC_LEVEL2_ACLNN_ABS_H_

#include "aclnn_util.h"
#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACLNN_MAX_SHAPE_RANK 8

/**
 * The first interface of @brief aclnnAbs calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Returns the result of the absolute value of each element in the input Tensor
 * Calculation formula:
 * $$ out_{i} = abs(self_{i}) $$
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *   A[(self)]--->B([l0op::Contiguous])
 *   B--->C([l0op::Abs])
 *   C--->D([l0op::ViewCopy])
 *   D--->E[(out)]
 * ```
 *
 * @param [in] self: input parameter to be used for abs calculation. aclTensor on the NPU device side,
 * The data type supports BFLOAT16 (only supported by Ascend 910B AI processor, Ascend 910_93 AI processor, and Ascend 910_95 AI processor) FLOAT, FLOAT16, DOUBLE,
 * INT8, INT16, INT32, INT64, UINT8, BOOL, the data format supports ND, and supports non-continuous Tensor.
 * @param [in] out: The output parameter of abs calculation. aclTensor on the NPU device side,
 * The data type supports BFLOAT16 (only supported by Ascend 910B AI processor, Ascend 910_93 AI processor, and Ascend 910_95 AI processor) FLOAT, FLOAT16, DOUBLE,
 * INT8, INT16, INT32, INT64, UINT8, BOOL, the data format supports ND, and supports non-continuous Tensor.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnAbsGetWorkspaceSize(const aclTensor* self,
                                               aclTensor* out,
                                               uint64_t* workspaceSize,
                                               aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnAbs is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAbsGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnAbs(void* workspace, uint64_t workspaceSize,
                               aclOpExecutor* executor,
                               const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ABS_H_