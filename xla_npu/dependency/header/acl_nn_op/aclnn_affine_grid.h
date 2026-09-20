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

#ifndef OP_API_INC_AFFINE_GRID_H_
#define OP_API_INC_AFFINE_GRID_H_
#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnAffineGrid calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Given a set of affine matrices (theta), generate a 2D or 3D network to represent the coordinate position of the image after affine transformation on the original image.
 *
 * @param [in] theta: aclTensor on the NPU device side, indicating the affine parameters of affine transformation.
 * The data type supports FLOAT and FLOAT16. The shape is (N, 2, 3) or (N, 3, 4). Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] size: aclIntArray on the host side, indicating the size of the image to be output. The size is 4(N, C, H, W) or 5(N, C, D, H, W).
 * @param [in] alignCorners: bool on the host side.
 * @param [in] out: aclTensor on the NPU device side, the data type supports FLOAT and FLOAT16, and the data type is consistent with theta.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAffineGridGetWorkspaceSize(
    const aclTensor* theta, const aclIntArray* size, bool alignCorners,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnAffineGrid is used to perform calculations.
 *
 * Operator function: Given a set of affine matrices (theta), generate a 2D or 3D network to represent the coordinate position of the image after affine transformation on the original image.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAffineGridGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAffineGrid(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_AFFINE_GRID_H_
