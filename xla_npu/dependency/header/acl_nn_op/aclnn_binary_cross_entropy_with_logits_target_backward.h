/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/license/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_BINARY_CROSS_ENTROPY_WITH_LOGITS_TARGET_BACKEARD_H_
#define OP_API_INC_BINARY_CROSS_ENTROPY_WITH_LOGITS_TARGET_BACKEARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize calculates the workspace size according to the specific calculation process
 * @domain aclnn_ops_train
 * Operator function: Find the gradient value of binary cross entropy with respect to target backpropagation
 */
ACLNN_API aclnnStatus
aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
    const aclTensor* weightOptional, const aclTensor* posWeightOptional,
    int64_t reduction, aclTensor* gradTarget, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/*
 * The second interface of @brief aclnnBinaryCrossEntropyWithLogitsTargetBackward is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnBinaryCrossEntropyWithLogitsTargetBackward(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_BINARY_CROSS_ENTROPY_WITH_LOGITS_TARGET_BACKEARD_H_
