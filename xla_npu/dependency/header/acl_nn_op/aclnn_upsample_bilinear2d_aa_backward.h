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

#ifndef ACLNN_UPSAMPLE_BILINEAR2D_AABACKWARD_H_
#define ACLNN_UPSAMPLE_BILINEAR2D_AABACKWARD_H_

#include "acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnUpsampleBilinear2dAABackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: back propagation of aclnnUpsampleBilinear2dAA.
 *
 * @param [in] gradOutput: aclTensor on the Device side. The data type supports FLOAT, FLOAT16, and BFLOAT16.
 * Supports non-continuous Tensors, the data format supports NCHW, and shape only supports four-dimensional Tensors. The data type is consistent with the data type of the outgoing parameter.
 * @param [in] outputSize:
 * aclIntArray on the Host side, the data type supports INT64, and the size is 2. Indicates the spatial size of the input gradOutput in the H and W dimensions.
 * @param [in] inputSize:
 * The aclIntArray on the Host side supports INT64 as the data type and the size is 4. Indicates the spatial size of the output out in the N, C, H and W dimensions respectively.
 * @param [in] scalesH: Floating point type on the Host side, indicating the height dimension multiplier of the output out.
 * @param [in] scalesW: Floating point type on the Host side, indicating the width dimension multiplier of the output out.
 * @param [out] out: aclTensor on the Device side. The data type supports FLOAT, FLOAT16, and BFLOAT16.
 * Supports non-continuous Tensors, the data format supports NCHW, and shape only supports four-dimensional Tensors. The data type is consistent with the data type of the input parameter gradOutput.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the Device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnUpsampleBilinear2dAABackwardGetWorkspaceSize(
    const aclTensor *gradOutput, const aclIntArray *outputSize,
    const aclIntArray *inputSize, bool alignCorners, double scalesH,
    double scalesW, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnUpsampleBilinear2dAABackward's second interface is used to perform calculations.
 *
 * Operator function: back propagation of aclnnUpsampleBilinear2dAA.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the Device side.
 * @param [in] workspaceSize: The workspace size applied for on the Device side.
 * Obtained by the first interface aclnnUpsampleBilinear2dAABackwardGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: Specifies the AscendCL Stream to execute the task.
 * @return aclnnStatus: Return status code.
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnUpsampleBilinear2dAABackward(void *workspace, uint64_t workspaceSize,
                                  aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
