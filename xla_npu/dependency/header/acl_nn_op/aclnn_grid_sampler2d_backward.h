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
#ifndef OP_API_INC_GRID_SAMPLER2D_BACKWARD_H_
#define OP_API_INC_GRID_SAMPLER2D_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGridSampler2DBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: reverse calculation of aclnnGridSampler2D.
 *
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(gradOutput)] --> B([l0op::Contiguous]) --> C([l0op::GridSampler2DGrad])
 *     D[(input)] --> E([l0op::Contiguous]) --> C
 *     F[(grid)] --> G([l0op::Contiguous]) --> C
 *     H((interpolationMode)) --> C
 *     I((paddingMode)) --> C
 *     J((alignCorners)) --> C
 *     C --> K([l0op::ViewCopy]) --> Out1[("inputGrad")]
 *     C --> L([l0op::ViewCopy]) --> Out2[("gridGrad")]
 * ```
 *
 * @param [in] gradOutput: npu
 * The aclTensor on the device side supports FLOAT, FLOAT16, and DOUBLE data types, supports non-continuous Tensor, and supports ND in the data format.
 * @param [in] input: npu
 * The aclTensor on the device side supports FLOAT, FLOAT16, and DOUBLE data types, supports non-continuous Tensor, and supports ND in the data format.
 * @param [in] grid: npu
 * The aclTensor on the device side supports FLOAT, FLOAT16, and DOUBLE data types, supports non-continuous Tensor, and supports ND in the data format.
 * @param [in] interpolationMode: int64_t type, indicating the interpolation mode, 0: bilinear (bilinear interpolation), 1: nearest (nearest neighbor interpolation).
 * @param [in] paddingMode: int64_t type, indicating the padding mode, that is, when the (x, y) value exceeds the input feature map sampling range, a specific value is returned.
 * There are three modes: 0: zeros, 1: border, and 2: reflection.
 * @param [in] alignCorners: bool type, indicating how to set the correspondence between feature map coordinates and feature values. When set to true, the feature value is located at the center of the pixel.
 * @param [in] outputMask: aclBoolArray type, output mask.
 * @param [in] inputGrad: npu
 * The aclTensor on the device side supports FLOAT, FLOAT16, and DOUBLE data types, supports non-continuous Tensor, and supports ND in the data format.
 * @param [in] gridGrad: npu
 * The aclTensor on the device side supports FLOAT, FLOAT16, and DOUBLE data types, supports non-continuous Tensor, and supports ND in the data format.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGridSampler2DBackwardGetWorkspaceSize(
    const aclTensor *gradOutput, const aclTensor *input, const aclTensor *grid,
    int64_t interpolationMode, int64_t paddingMode, bool alignCorners,
    const aclBoolArray *outputMask, aclTensor *inputGrad, aclTensor *gridGrad,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * The second interface of @brief aclnnGridSampler2DBackward is used to perform calculations.
 *
 * Operator function: reverse calculation of aclnnGridSampler2D.
 *
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(gradOutput)] --> B([l0op::Contiguous]) --> C([l0op::GridSampler2DGrad])
 *     D[(input)] --> E([l0op::Contiguous]) --> C
 *     F[(grid)] --> G([l0op::Contiguous]) --> C
 *     H((interpolationMode)) --> C
 *     I((paddingMode)) --> C
 *     J((alignCorners)) --> C
 *     C --> K([l0op::ViewCopy]) --> Out1[("inputGrad")]
 *     C --> L([l0op::ViewCopy]) --> Out2[("gridGrad")]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, determined by the first interface aclnnGridSampler2DBackwardGetWorkspaceSize
 * Get.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGridSampler2DBackward(void *workspace,
                                                 uint64_t workspaceSize,
                                                 aclOpExecutor *executor,
                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GRID_SAMPLER2D_BACKWARD_H_