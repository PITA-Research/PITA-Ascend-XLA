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
#ifndef OP_API_INC_UNAMPLE_NEAREST_1D_V2_H_
#define OP_API_INC_UNAMPLE_NEAREST_1D_V2_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUpsampleNearest1dV2's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Apply the nearest neighbor interpolation algorithm to upsample an input signal consisting of multiple input channels.
 *
 * @param [in] self: aclTensor on the NPU device side. The data type supports FLOAT, BFLOAT16, FLOAT16, and UINT8.
 * Supports non-contiguous Tensors.
 * @param [in] outputSize: aclIntArray on the NPU device side, specifying the output space size.
 * @param [in] scaleL: float constant, indicating the L-dimensional multiplier of output out.
 * @param [out] out: aclTensor on the NPU device side, data type supports FLOAT, BFLOAT16, FLOAT16, UINT8,
 * And the data type is consistent with the data type of self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnUpsampleNearest1dV2GetWorkspaceSize(
    const aclTensor *self, const aclIntArray *outputSize, float scaleL,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnUpsampleNearest1dV2's second interface is used to perform calculations.
 *
 * Operator function: Apply the nearest neighbor interpolation algorithm to upsample an input signal consisting of multiple input channels.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: the workspace size applied for on the NPU device side,
 * Obtained by the first interface aclnnUpsampleNearest1dV2GetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: Specifies the AscendCL Stream to execute the task.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnUpsampleNearest1dV2(void *workspace,
                                               uint64_t workspaceSize,
                                               aclOpExecutor *executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNAMPLE_NEAREST_1D_V2_H_
