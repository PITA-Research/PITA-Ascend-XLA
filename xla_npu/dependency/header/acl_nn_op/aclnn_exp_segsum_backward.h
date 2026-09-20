/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef OP_API_INC_EXP_SEGSUM_BACKWARD_H_
#define OP_API_INC_EXP_SEGSUM_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnExpSegsumBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: reverse calculation of aclnnExpSegsum.
 * Calculation formula: as shown below (taking 5D input as an example)
 * 1. Input gradOutput(N1,N2,N3,N4,N4) is multiplied by input gradSelf (forward output).
 * $$
 * out\_mul = gradOutput * gradSelf
 * $$
 * 2. Generate a triangular matrix A of (N4, N4) type bool, with the upper triangle as True, the lower triangle as False, and the diagonal as True.
 * 3. Fill the elements in the input $out\_mul$ with 0 that correspond to the position in the matrix A where the value is True.
 * $$
 * out\_mul_i=
 * \begin{cases}out\_mul_i,\quad A_i==False
 * \\0, \quad A_i==True
 * \end{cases}
 * $$
 * 4. Reverse the penultimate dimension of out_mul to generate out_flip.
 * 5. Perform cumsum accumulation using the penultimate dimension of $out\_flip$. From a dimensional perspective, $out\_cumsum\_{i}$ is the element at the corresponding position in the output tensor.
 * $$
 * out\_cumsum_{i} = out\_flip_{1} + out\_flip_{2} + out\_flip_{3} + ...... + out\_flip_{i}
 * $$
 * 6. Reverse the -2 dimensions of $out\_cumsum$ to generate out_flip2.
 * 7. Generate (N4, N4) triangular matrix B of type bool, with the upper triangle as True, the lower triangle as False, and the diagonal as True.
 * 8. Fill the elements in $out\_flip2$ with 0 that correspond to the position in matrix B where the value is True.
 * $$
 * out\_flip2_i=
 * \begin{cases}out\_flip2_i,\quad B_i==False
 * \\0, \quad B_i==True
 * \end{cases}
 * $$
 * 9. Return the sum of each row in the last dimension of out\_flip2.
 *
 * @param [in] gradOutput: aclTensor on the NPU device side, the data type supports FLOAT, BFLOAT16, and FLOAT16.
 * Supports non-continuous Tensor and ND format.
 * @param [in] gradSelf: aclTensor on the NPU device side, the data type supports FLOAT, BFLOAT16, and FLOAT16.
 * The data type is consistent with gradOutput. Supports non-continuous Tensor and ND format.
 * @param [out] gradInput: aclTensor on the NPU device side, data type supports FLOAT, BFLOAT16, FLOAT16,
 * And the data type is consistent with the data type of self and supports ND format.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnExpSegsumBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* gradSelf,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnExpSegsumBackward is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: the workspace size applied for on the NPU device side,
 * Obtained by the first interface aclnnExpSegsumBackwardGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: Specifies the AscendCL Stream to execute the task.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnExpSegsumBackward(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_EXP_SEGSUM_BACKWARD_H_