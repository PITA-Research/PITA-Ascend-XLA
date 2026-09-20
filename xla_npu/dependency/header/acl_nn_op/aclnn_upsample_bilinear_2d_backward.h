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
#ifndef OP_API_INC_UNAMPLE_BILINEAR_2D_BACKWARD_H_
#define OP_API_INC_UNAMPLE_BILINEAR_2D_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The second interface of @brief aclnnUpsampleBilinear2dBackward is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnUpsampleBilinear2dBackward(void *workspace,
                                                      uint64_t workspace_size,
                                                      aclOpExecutor *executor,
                                                      aclrtStream stream);

/**
 * The first interface of @brief aclnnUpsampleBilinear2dBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 */
ACLNN_API aclnnStatus aclnnUpsampleBilinear2dBackwardGetWorkspaceSize(
    const aclTensor *gradOut, const aclIntArray *outputSize,
    const aclIntArray *inputSize, bool alignCorners, double scalesH,
    double scalesW, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNAMPLE_BILINEAR_2D_BACKWARD_H_