/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_TRIU_H_
#define OP_API_INC_LEVEL2_ACLNN_TRIU_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnTriu calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Set the last two dimensions of the input self tensor (numbered from left to right according to shape) to zero along the lower left part of the main diagonal. The parameter diagonal can be positive or negative,
 * The default is zero, a positive number means the main diagonal moves to the upper right, a negative number means the main diagonal moves to the lower left.
 * Calculation formula: i is used below to represent the sequence number of traversing the penultimate dimensional element (i is the row index), and j is used to represent the sequence number of traversing the last dimensional element (j is the column index).
 * Use d to represent diagonal. In the two-dimensional coordinate diagram corresponding to (i, j), i+d==j means on the diagonal,
 * $$
 * The diagonal and its upper right corner, that is, i+d<=j, keep the original value: out_{i,j} = self_{i,j}\\
 * In the case of the lower left corner of the diagonal line, that is, i+d>j, set to zero: out_{i,j} = 0
 * $$
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0::Contiguous]) --> C([l0op::Triu])
 *   C --> D([l0op::ViewCopy]) --> E[(out)]
 * ```
 *
 * @param [in] self: input parameter to be used for triu calculation. aclTensor on the NPU device side,
 * Data types support DOUBLE, FLOAT, FLOAT16, BFLOAT16 (910B only), INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8, BOOL.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] diagonal: offset of the main diagonal, the data type supports INT.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTriuGetWorkspaceSize(const aclTensor* self,
                                                int64_t diagonal,
                                                aclTensor* out,
                                                uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief aclnnTriu's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAllGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTriu(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor, aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceTriu calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] selfRef: aclTensor on NPU device side,
 * Data types support DOUBLE, FLOAT, FLOAT16, BFLOAT16 (910B only), INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8, BOOL.
 * Supports non-continuous Tensor, and the data format supports ND
 * @param [in] diagonal: offset of the main diagonal, the data type supports INT.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceTriuGetWorkspaceSize(
    aclTensor* selfRef, int64_t diagonal, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceTriu is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceTriuGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceTriu(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_TRIU_H_
