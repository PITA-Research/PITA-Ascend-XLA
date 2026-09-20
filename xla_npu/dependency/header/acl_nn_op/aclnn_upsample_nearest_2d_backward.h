/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 *Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at
*    http://www.apache.org/licenses/LICENSE-2.0
*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an &quot;AS IS&quot; BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
 */

#ifndef OP_API_INC_UNAMPLE_NEAREST_2D_BACKWARD_H_
#define OP_API_INC_UNAMPLE_NEAREST_2D_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnUpsampleNearest2dBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: reverse calculation of upsample_nearest2d.
 *
 * @param [in] gradOut: aclTensor on the NPU device side. The data type supports FLOAT, BFLOAT16, and FLOAT16.
 * Supports [non-continuous Tensor](#non-continuous Tensor description), and the data format supports NCHW and NHWC ([reference](#reference)).
 * @param [in] outputSize: Input IntArray, size is 2. Indicates the spatial size of the input gradOut in the H and W dimensions.
 * @param [in] inputSize: Input IntArray, size is 4. Indicates the spatial size of the output out in the N, C, H and W dimensions respectively.
 * @param [in] scalesH: double constant, indicating the height dimension multiplier of the output out.
 * @param [in] scalesW: double constant, indicating the width dimension multiplier of the output out.
 * @param [out] gradInput: npu
 * The data type of aclTensor on the device side supports FLOAT, BFLOAT16, and FLOAT16, and the data type is consistent with the data type of gradOut.
 * Supports [non-continuous Tensor](#non-continuous Tensor description), and the data format supports NCHW and NHWC ([reference](#reference)).
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnUpsampleNearest2dBackwardGetWorkspaceSize(
    const aclTensor* gradOut, const aclIntArray* outputSize,
    const aclIntArray* inputSize, double scalesH, double scalesW,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnUpsampleNearest2dBackward is used to perform calculations.
 *
 * Operator function: reverse calculation of upsample_nearest2d.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: the workspace size applied for on the NPU device side,
 * Obtained by the first interface aclnnUpsampleNearest2dBackwardGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnUpsampleNearest2dBackward(void* workspace,
                                                     uint64_t workspaceSize,
                                                     aclOpExecutor* executor,
                                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNAMPLE_NEAREST_2D_BACKWARD_H_
