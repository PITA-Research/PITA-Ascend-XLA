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

#ifndef OP_API_INC_SEGSUM_H_
#define OP_API_INC_SEGSUM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnExpSegsum calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: perform segmentation and calculation. Generate a semi-separable matrix with diagonal 0 and upper triangle -inf.
 * Calculation formula: as shown below (taking 4D input as an example)
 * 1. The input self is raised from (N1, N2, N3, N4) to (N1, N2, N3, N4, 1).
 * 2. Broadcast to get (N1, N2, N3, N4, N4)
 * 3. Generate a triangular matrix A of type (N4, N4) bool, with the upper triangle as True, the lower triangle as False, and the diagonal as True.
 * 4. Fill the element in the input self with 0 that corresponds to the position in the matrix A where the value is True.
 * $$
 * self_i=
 * \begin{cases}self_i,\quad A_i==False
 * \\0, \quad A_i==True
 * \end{cases}
 * $$
 * 5. Perform cumsum accumulation using the penultimate dimension of self. An element viewed from a dimensional perspective (the subscripts of other dimensions remain unchanged, and the subscripts of the current dimension increase in sequence),
 * $selfTemp\_{i}$ is the element at the corresponding position in the output tensor.
 * $$
 * selfTemp_{i} = self_{1} + self_{2} + self_{3} + ...... + self_{i}
 * $$
 * 6. Generate (N4, N4) triangular matrix B of type bool. The upper triangle is True, the lower triangle is False, and the diagonal is False.
 * 7. Use -inf to fill in the elements in selfTemp corresponding to the position in matrix B where the value is True.
 * $$
 * out_i=
 * \begin{cases}selfTemp_i,\quad B_i==False
 * \\-inf, \quad BD_i==True
 * \end{cases}
 * $$
 * 8. Calculate the index of each element in selfTemp.
 * $$
 * out_i=e^{selfTemp_i}
 * $$
 *
 * @param [in] self: aclTensor on the NPU device side. The data type supports FLOAT, BFLOAT16, and FLOAT16.
 * Supports non-contiguous Tensors.
 * @param [out] out: aclTensor on the NPU device side, data type supports FLOAT, BFLOAT16, FLOAT16,
 * And the data type is consistent with the data type of self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnExpSegsumGetWorkspaceSize(const aclTensor* self,
                                                     aclTensor* out,
                                                     uint64_t* workspaceSize,
                                                     aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnExpSegsum is used to perform calculations.
 *
 * Operator function: perform segmentation and calculation. Generate a semi-separable matrix with diagonal 0 and upper triangle -inf.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: the workspace size applied for on the NPU device side,
 * Obtained by the first interface aclnnExpSegsumGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: Specifies the AscendCL Stream to execute the task.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnExpSegsum(void* workspace, uint64_t workspaceSize,
                                     aclOpExecutor* executor,
                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SEGSUM_H_