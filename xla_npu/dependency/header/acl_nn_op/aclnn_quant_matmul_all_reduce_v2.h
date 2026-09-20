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
 * \file aclnn_quant_matmul_all_reduce_v2.h
 * \brief
 */
#ifndef OP_API_INC_QUANT_MATMUL_ALL_REDUCE_V2_
#define OP_API_INC_QUANT_MATMUL_ALL_REDUCE_V2_

#include <string>

#include "aclnn_base.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnQuantMatmulAllReduceV2's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: realize mm+AllReduce fusion calculation
 * @param [in] x1: matmul left matrix, data type support: int8.
 * @param [in] x2: matmul right matrix, data type support: int8.
 * @param [in] biasOptional: bias, data type support: int32.
 * @param [in] x3Optional: add operation parameter, data type support: float16, bfloat16.
 * @param [in] dequantScale: Dequantization coefficient, data type support: int64, uint64, bfloat16, float32.
 * @param [in] pertokenScaleOptional: per-token quantization coefficient, data type support: float32.
 * @param [in] group: String identifying the column group.
 * @param [in] reduceOp: reduce operation type, default value: sum.
 * @param [in] commTurn: communication data cut fraction, that is, total data volume/single communication volume, default value: 0.
 * @param [in] streamMode: enumeration of ACL stream mode, type support: 1.
 * @param [out] output: The result of calculation + communication, data type: float16, bfloat16.
 * @param [out] workspaceSize: Returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnQuantMatmulAllReduceV2GetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional,
    const aclTensor* x3Optional, const aclTensor* dequantScale,
    const aclTensor* pertokenScaleOptional, const char* group,
    const char* reduceOp, int64_t commTurn, int64_t streamMode,
    const aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnQuantMatmulAllReduceV2's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnQuantMatmulAllReduceV2GetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnQuantMatmulAllReduceV2(void* workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor* executor,
                                                  const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANT_MATMUL_ALL_REDUCE_V2_