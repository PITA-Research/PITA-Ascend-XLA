/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_INDEX_SELECT_H_
#define OP_API_INC_LEVEL2_ACLNN_INDEX_SELECT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnIndexSelect calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: Returns the result of rounding down each element in the input Tensor
 * Calculation formula:
 * Take the input as a three-dimensional tensor as an example:
 *   x=$\begin{bmatrix}[[1,&2],&[3,&4]], \\ [[5,&6],&[7,&8]], \\ [[9,&10],&[11,&12]]\end{bmatrix}$
 *   idx=[1, 0],
 * dim is 0, index_select(0, idx): I=index[i]; y$[i][m][n]$ = x$[I][m][n]$
 * dim is 1, index_select(1, idx): J=index[j]; y$[l][j][n]$ = x$[l][J][n]$
 * dim is 2, index_select(2, idx): K=index[k]; y$[l][m][k]$ = x$[l][m][K]$
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *   A[(Self)] -->B([l0op::Contiguous])
 *   B --> In_0([l0op::cast]) --> Op([GatherV2])
 *   In_1[(index)] --> con([l0op::Contiguous])--> Op
 *   In_2(dim) --> a(dimVec) --> Op
 *   Op --> C([l0op::cast]) --> D([l0op::ViewCopy])  --> Out[(out)]
 * ```

 *
 * @param [in] self: Original tensor. aclTensor on the NPU device side,
 *
 The data types support FLOAT, FLOAT16, BFLOAT16, INT64, INT32, INT16, INT8, UINT8, BOOL, DOUBLE, COMPLEX64, and COMPLEX128. The data format supports ND and non-continuous Tensor.
 * @param [in] dim: int64 on the host side.
 * @param [in] index: indecies, used for slicing, aclTensor on the NPU device side.
 * The data type supports INT64 and INT32, the data format supports ND, and the dimension is 0D or 1D. Supports non-contiguous Tensors.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnIndexSelectGetWorkspaceSize(
    const aclTensor* self, int64_t dim, const aclTensor* index, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnIndexSelect is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnIndexSelectGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnIndexSelect(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_INDEX_SELECT_H_
