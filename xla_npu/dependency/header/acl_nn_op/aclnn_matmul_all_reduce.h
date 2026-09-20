/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_matmul_all_reduce.h
 * \brief
 */
#ifndef OP_API_INC_MATMUL_ALL_REDUCE_
#define OP_API_INC_MATMUL_ALL_REDUCE_

#include <string>

#include "aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnMatmulAllReduce calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: realize mm+AllReduce fusion calculation
 * @param [in] x1: matmul left matrix, data type support: float16, BF16.
 * @param [in] x2: matmul right matrix, data type support: float16, BF16.
 * @param [in] bias: bias, data type support: float16, BF16.
 * @param [in] group: String identifying the column group.
 * @param [in] reduceOp: reduce operation type, default value: sum.
 * @param [in] commTurn: communication data cut fraction, that is, total data volume/single communication volume, default value: 0.
 * @param [in] streamMode: enumeration of ACL stream mode, type support: 0/1.
 * @param [out] output: the result of calculation + communication, data type: same as input.
 * @param [out] workspaceSize: Returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnMatmulAllReduceGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
    const char* group, const char* reduceOp, int64_t commTurn,
    int64_t streamMode, const aclTensor* output, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnMatmulAllReduce's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMatmulAllReduceGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnMatmulAllReduce(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MATMUL_ALL_REDUCE_