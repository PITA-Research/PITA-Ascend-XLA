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
#ifndef OP_API_INC_LOGDET_H_
#define OP_API_INC_LOGDET_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnLogdet calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Calculate the natural logarithm of the determinant of the input Tensor
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(self)]-->B([Contiguous])-->C([LogMatrixDeterminant])
 * D([Cast])-->E([ViewCopy])-->F([out])
 * ```
 *
 * @param [in] self: aclTensor on the NPU device side, the data type supports FLOAT, DOUBLE, COMPLEX64, COMPLEX128. The shape satisfies the form (*, n, n),
 * Where `*` represents a batch of 0 or more dimensions. Supports non-continuous Tensor. The data format supports ND.
 * @param [in] out: aclTensor on the NPU device side, the data type supports FLOAT, DOUBLE, COMPLEX64, COMPLEX128. The shape is consistent with the batch of self.
 * Supports non-continuous Tensor. The data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Returns the status code. ACLNN_SUCCESS is returned on success, and the corresponding error code is returned on failure.
 */
ACLNN_API aclnnStatus aclnnLogdetGetWorkspaceSize(const aclTensor* self,
                                                  aclTensor* out,
                                                  uint64_t* workspaceSize,
                                                  aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnLogdet is used to perform calculations.
 *
 * Operator function: complete the inverse calculation for each element of the input
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnLogdetGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLogdet(void* workspace, uint64_t workspaceSize,
                                  aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LOGDET_H_
