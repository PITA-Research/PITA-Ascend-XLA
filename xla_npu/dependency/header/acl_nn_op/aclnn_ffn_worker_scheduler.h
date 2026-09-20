/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_FFN_WORKER_SCHEDULER_H_
#define OP_API_INC_FFN_WORKER_SCHEDULER_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnFfnWorkerScheduler calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize(
    aclTensor* scheduleContextRef, int32_t syncGroupSize, int32_t executeMode,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnFfnWorkerScheduler is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnInplaceFfnWorkerScheduler(void* workspace,
                                                     uint64_t workspaceSize,
                                                     aclOpExecutor* executor,
                                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_FFN_WORKER_SCHEDULER_H_