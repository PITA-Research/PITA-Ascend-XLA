/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_GATHER_V3_H_
#define OP_API_INC_LEVEL2_ACLNN_GATHER_V3_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: From the specified dimension 0 of the input Tensor, extract elements according to the subscript number in the index and save them in the out Tensor.
 * Calculation formula:
 * Take the input as a three-dimensional tensor as an example:
 *   x=$\begin{bmatrix}[[1,&2],&[3,&4]], \\ [[5,&6],&[7,&8]], \\ [[9,&10],&[11,&12]]\end{bmatrix}$
 *   idx=[1, 0],
 * I=index[i];  &nbsp;&nbsp;   y$[i][m][n]$ = x$[I][m][n]$
 * Calculation graph:
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0op::Contiguous])
 *   B --> In_0([l0op::cast]) --> Op([GatherV2])
 *   In_1[(index)] --> con([l0op::Contiguous])--> Op
 *   In_2(0) --> a(dimVec) --> Op
 *   Op --> C([l0op::cast]) --> D([l0op::ViewCopy]) --> Out[(out)]
 * ```
 */

/**
 * Operator function: From the specified dimension dim of the input Tensor, extract elements according to the subscript number in the index and save them to the out Tensor.
 * Calculation formula:
 * When batchDims is 0, taking the input as a three-dimensional tensor as an example:
 *   x=$\begin{bmatrix}[[1,&2],&[3,&4]], \\ [[5,&6],&[7,&8]], \\ [[9,&10],&[11,&12]]\end{bmatrix}$
 *   idx=[1, 0],
 * dim is 0: I=index[i]; y$[i][m][n]$ = x$[I][m][n]$
 * dim is 1: J=index[j]; y$[l][j][n]$ = x$[l][J][n]$
 * dim is 2: K=index[k]; y$[l][m][k]$ = x$[l][m][K]$
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0op::Contiguous])
 *   B --> In_0([l0op::cast]) --> Op([GatherV2])
 *   In_1[(index)] --> con([l0op::Contiguous])--> Op
 *   In_2(dim) --> a(dimVec) --> Op
 *   Op --> C([l0op::cast]) --> D([l0op::ViewCopy]) --> Out[(out)]
 * ```
 * batchDims is not 0:
 * batchDims is n, dim is m (m >= n), the shape size before batchDims is k, which is equivalent to batchDims=0 k times,
 * dim=m-n gather operation.
 */

/**
 * @brief The first interface of aclnnGatherV3 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, INT64, INT32, INT16, INT8, UINT8, BOOL, DOUBLE,
 * COMPLEX64, COMPLEX128, and BFLOAT16 support non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in] dim: Integer on the host side, the data type supports INT64.
 * @param [in] index: npu
 * The aclTensor on the device side supports INT64 and INT32 as data type and non-continuous Tensor. The data format supports ND. The data dimension does not support more than 8 dimensions.
 * @param [in] batchDims: Integer on the host side, the data type supports INT64.
 * @param [in] mode: Integer on the host side, the data type supports INT64.
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, INT64, INT32, INT16, INT8, UINT8, BOOL, DOUBLE,
 * COMPLEX64, COMPLEX128, BFLOAT16, the data type needs to be consistent with self, the dimension is equal to the sum of self dimension and index dimension minus one, except that the dim dimension is expanded to the shape of index
 * In addition, the length of other dimensions is consistent with the corresponding dimension of self, and the data format supports ND.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGatherV3GetWorkspaceSize(
    const aclTensor* self, int64_t dim, const aclTensor* index,
    int64_t batchDims, int64_t mode, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);
/**
 * @brief The second interface of aclnnGatherV3 is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGatherV2GetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGatherV3(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_GATHER_V3_H_
