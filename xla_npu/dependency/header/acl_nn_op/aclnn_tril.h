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
#ifndef OP_API_INC_LEVEL2_ACLNN_TRIL_H_
#define OP_API_INC_LEVEL2_ACLNN_TRIL_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnTril calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Returns the lower triangular part of the input matrix (2-D `Tensor`) or a batch of matrices, with other elements of the result `Tensor`out set to 0.

 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *   A[(self)] -->B([l0::Contiguous])
 *   B --> E([l0::Tril])
 *   C[(diagonal)] --> E
 *   E --> G([l0::ViewCopy])
 *   G --> H[(out)]
 * ```
 *
 * @param [in] self: input parameter to be tril calculated. aclTensor on the NPU device side,
 * Data types support DOUBLE, FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8, BOOL.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] diagonal: The position of the diagonal, the data type supports INT.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTrilGetWorkspaceSize(const aclTensor* self,
                                                int64_t diagonal,
                                                aclTensor* out,
                                                uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnTril is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAllGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTril(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor,
                                const aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceTril calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Returns the lower triangular part of the input matrix (2-D `Tensor`) or a batch of matrices, with other elements of the result `Tensor`out set to 0.
 *
 * @param [in] selfRef: input parameter to be used for tril calculation. aclTensor on the NPU device side,
 * Data types support DOUBLE, FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8, BOOL.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] diagonal: The position of the diagonal, the data type supports INT.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceTrilGetWorkspaceSize(
    const aclTensor* selfRef, int64_t diagonal, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceTril is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAllGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceTril(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_TRIL_H_