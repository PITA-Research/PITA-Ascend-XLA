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

#ifndef OP_API_INC_UNAMPLE_NEAREST_EXACT1D_H_
#define OP_API_INC_UNAMPLE_NEAREST_EXACT1D_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnUpsampleNearestExact1d calculates the workspace size according to the specific calculation process.
 * Function description: Apply the nearest neighbor exact interpolation algorithm to the input signal consisting of three input channels for upsampling interpolation
 * Calculation formula:
 * out(N, C, l) = self(N, C, min(floor((l + 0.5) * scales),  L- 1))
 * @domain aclnn_ops_train
 * Parameter description:
 * @param [in]   self
 * Input Tensor, the data type supports FLOAT, FLOAT16, and BFLOAT16. Supports non-continuous Tensor, and the data format supports ND and NCL.
 * @param [in]   outputSize
 * The output size, the data type supports INT32 and INT64.
 * @param [in]   scales
 * The output scaling factor, the data type supports DOUBLE.
 * @param [in]   out
 * Output Tensor, data type supports FLOAT, FLOAT16, BFLOAT16. Supports non-continuous Tensor, and the data format supports ND and NCL.
 * @param [out] workspaceSize returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor returns the op executor, including the operator calculation process.
 * @return aclnnStatus return status code
 */
ACLNN_API aclnnStatus aclnnUpsampleNearestExact1dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* outputSize, double scales,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnUpsampleNearestExact1d's second interface is used to perform calculations.
 *
 * Function description: Apply the nearest neighbor exact interpolation algorithm to the input signal consisting of three input channels for upsampling interpolation.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnUpsampleNearestExact1dGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnUpsampleNearestExact1d(void* workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNAMPLE_NEAREST_EXACT1D_H_
