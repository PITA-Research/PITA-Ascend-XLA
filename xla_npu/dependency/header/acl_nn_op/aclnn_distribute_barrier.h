/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_DISTRIBUTE_BARRIER_H_
#define OP_API_INC_DISTRIBUTE_BARRIER_H_

#include <string>

#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: realize full card synchronization
 * @brief
 * The first interface of aclnnDistributeBarrier calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] xRef: calculation input, placeholder is not used, Tensor, supports bfloat16, float16,
 * float32, bool, int8, int16, int32, int64, uint8, uint16, uint32, uint64。
 * @param [in] group: calculation input, str. Communication domain name, expert parallel communication domain.
 * @param [in] worldSize: calculation input, int64_t. Communication domain size.
 * @param [out] workspaceSize: Output parameter, returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Out parameter, returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return value, return status code.
 *
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnDistributeBarrierGetWorkspaceSize(aclTensor* xRef, const char* group,
                                       int64_t worldSize,
                                       uint64_t* workspaceSize,
                                       aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnDistributeBarrier is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnDistributeBarrierGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDistributeBarrier(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_DISTRIBUTE_BARRIER_H_