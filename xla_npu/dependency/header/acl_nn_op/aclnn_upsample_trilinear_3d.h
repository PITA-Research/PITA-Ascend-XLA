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

#ifndef OP_API_INC_UPSAMPLE_TRILINEAR_3D_H_
#define OP_API_INC_UPSAMPLE_TRILINEAR_3D_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUpsampleTrilinear3d's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnUpsampleTrilinear3dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* outputSize, bool alignCorners,
    double scalesD, double scalesH, double scalesW, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnUpsampleTrilinear3d's second interface is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnUpsampleTrilinear3d(void* workspace,
                                               uint64_t workspaceSize,
                                               aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UPSAMPLE_TRILINEAR_3D_H_