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

#ifndef OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_V2_BACKWARD_H_
#define OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_V2_BACKWARD_H_

#include <cstring>
#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnRoiAlignV2Backward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: aclnnRoiAlignV2Backward is the reverse propagation of aclnnRoiAlign. ROIAlign is a pooling layer that uses non-uniform input size feature maps and outputs fixed size feature maps.
 *
 * @param [in] gradOutput: aclTensor on the NPU device side, the data type supports FLOAT32, supports non-continuous Tensor, and the data format supports NCHW.
 * @param [in] boxes: aclTensor on the NPU device side, the data type supports FLOAT32, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] inputShape: aclIntArray, specify the output shape.
 * @param [in] pooledHeight: int type on the host side, H of the output image after forward RoiAlign pooling.
 * @param [in] pooledWidth: int type on the host side, W of the output image after forward RoiAlign pooling.
 * @param [in] spatialScale: float type on the host side, scaling factor, used to convert ROI coordinates into input feature maps.
 * @param [in] samplingRatio: int type on the host side, used to calculate the number of bins above the sum W of each output element.
 * @param [in] aligned: Bool type on the host side, used to determine whether the pixel frame is offset by -0.5 for better alignment.
 * @param [out] gradInput: npu
 * The data type of aclTensor on the device side supports FLOAT32, which is consistent with gradOutput. It supports non-continuous Tensor and the data format supports NCHW.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRoiAlignV2BackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* boxes,
    const aclIntArray* inputShape, int64_t pooledHeight, int64_t pooledWidth,
    float spatialScale, int64_t samplingRatio, bool aligned,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnRoiAlignV2Backward's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnRoiAlignV2BackwardGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnRoiAlignV2Backward(void* workspace,
                                              uint64_t workspaceSize,
                                              aclOpExecutor* executor,
                                              aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_BACKWARD_H_