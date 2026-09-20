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

#ifndef OP_API_INC_NON_MAX_SUPPRESSION_H_
#define OP_API_INC_NON_MAX_SUPPRESSION_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnNonMaxSuppressionGetWorkspaceSize(
    const aclTensor* boxes, const aclTensor* scores,
    aclIntArray* maxOutputBoxesPerClass, aclFloatArray* iouThreshold,
    aclFloatArray* scoreThreshold, int32_t centerPointBox,
    aclTensor* selectedIndices, uint64_t* workspaceSize,
    aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnNonMaxSuppression(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_NON_MAX_SUPPRESSION_H_