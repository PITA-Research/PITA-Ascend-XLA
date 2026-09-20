/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_TAKE_H_
#define OP_API_INC_LEVEL2_ACLNN_TAKE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnTake calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Treat the input self tensor as a one-dimensional array, use the value of index as the index, take the value from self, and output a Tensor whose shape is consistent with index.
 * Calculation formula: The subscript i represents traversing from 0 to the number of index elements -1
 * $$ out_{i} = self_{index[i]}  $$
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 * A[(self)] --> B([l0op::Contiguous]) --> D([l0op::Gather])
 * I[(index)] --> IC([l0op::Contiguous]) --> D --> F1([l0op::ViewCopy]) --> J[(out)]
 * ```
 *
 * @param [in] self: input parameter to be calculated for take. aclTensor on the NPU device side,
 * Data types support UINT64, INT64, UINT32, INT32, FLOAT32, UINT16, INT16, FLOAT16, INT8, UINT8, DOUBLE, COMPLEX64, COMPLEX128, BOOL,
 * The data format supports ND, non-continuous Tensor, and scenarios with dimensions higher than 8.
 * @param [in] index: input parameter for take calculation. The data type of aclTensor on the NPU device side supports INT32 and INT64. The data format is ND.
 * @param [in] out: take the calculated output parameter. The aclTensor on the NPU device side has the same data type as self, the same shape and index, and the data format is ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTakeGetWorkspaceSize(const aclTensor* self,
                                                const aclTensor* index,
                                                aclTensor* out,
                                                uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnTake is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnTakeGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTake(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor,
                                const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_TAKE_H_
