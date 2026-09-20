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
#ifndef OP_API_INC_TRIANGULAR_SOLVE_H_
#define OP_API_INC_TRIANGULAR_SOLVE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnTriangularSolve calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Solve AX=b for input Tensor
 * @param [in] self: b in the formula, the data type supports FLOAT, DOUBLE, COMPLEX64, COMPLEX128, the data dimension is at least 2 and not greater than 8,
 * Supports non-continuous Tensor, the data format supports ND, and shape(*,m,n) needs to satisfy the broadcast relationship with A(*,m,m) except for the last two dimensions.
 * @param [in] A: A in the formula, the data type supports FLOAT, DOUBLE, COMPLEX64, COMPLEX128, and the data dimension is at least 2 and not greater than 8.
 * Supports non-continuous Tensors, the data format supports ND, and shape(*,m,n) needs to satisfy the broadcast relationship with self(*,m,n) except for the last two dimensions.
 * @param [in] upper: calculated attribute, the default is true, A is an upper triangular square matrix, when upper is false, A is a lower triangular square matrix
 * @param [in] transpose: calculated attribute, the default is false, when transpose is true, calculate ATX=B, AT is the transpose of A
 * @param [in] unitriangular: computed property, defaults to false. When unitriangular is true, the diagonal elements of A are treated as 1 instead of being referenced from A
 * @param [out] xOut: X in the formula, the data type supports FLOAT, DOUBLE, COMPLEX64, COMPLEX128, the data dimension is at least 2 and not greater than 8,
 * Supports non-continuous Tensor, the data format supports ND, and the shape needs to satisfy the AX=b constraint with A and b after broadcasting
 * @param [out] mOut: copy of the upper triangle (lower triangle) of A after broadcast. The data type supports FLOAT, DOUBLE, COMPLEX64, COMPLEX128,
 * The data dimension is at least 2 and not greater than 8. Non-continuous Tensors are supported. The data format supports ND, and the shape needs to be consistent with A after broadcast.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnTriangularSolveGetWorkspaceSize(
    const aclTensor* self, const aclTensor* A, bool upper, bool transpose,
    bool unitriangular, aclTensor* xOut, aclTensor* mOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnTriangularSolve, used to perform calculations
 *
 * Operator function: Complete the solution operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnTriangularSolveGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTriangularSolve(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_TRIANGULAR_SOLVE_H_