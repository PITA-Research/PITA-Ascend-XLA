/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_MAX_POOL_H_
#define OP_API_INC_LEVEL2_ACLNN_MAX_POOL_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnMaxPool calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnMaxPoolGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* kernelShape,
    const aclIntArray* strides, const int64_t autoPad, const aclIntArray* pads,
    const aclIntArray* dilations, const int64_t ceilMode, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnMaxPool is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnMaxPool(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_MAX_POOL_H_