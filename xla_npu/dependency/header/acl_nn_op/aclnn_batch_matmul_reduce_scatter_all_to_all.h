/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_
#define OP_API_INC_BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_

#include <string>

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: implement bmm + reduceScatter + alltoall fusion calculation
 * The first interface of @brief aclnnBatchMatMulReduceScatterAlltoAll calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] x: Calculation input, Tensor, data type float16, bfloat16, must be 3 dimensions. Left matrix calculated by BatchMatMul
 * @param [in] weight: Calculation input, Tensor, data type float16, bfloat16, must be 3-dimensional, and the type must be consistent with x. Right matrix calculated by BatchMatMul
 * @param [in] biasOptional: Calculation input, Tensor, data type float16, float32. When x is float16, bias needs to be float16; when x is bfloat16, bias needs to be float32. Supports two or three dimensions. The bias calculated by BatchMatMul. (Since ReduceScatter communication is required, Add needs to be added after communication)
 * @param [in] groupEp: calculation input, str. ep communication domain name, expert parallel communication domain
 * @param [in] groupTp: calculation input, str. tp communication domain name, Tensor parallel communication domain
 * @param [in] epWorldSize: calculation input, int. ep communication domain size, supports 2/4/8/16/32
 * @param [in] tpWorldSize: calculation input, int. tp communication domain size, supports 2/4/8/16/32
 * @param [in] yShardType: Calculation input, int, the default value is 0. 0 means that the output is sharded by tp in the H dimension, and 1 means that the output is sharded by tp in the C dimension.
 * @param [out] out: Calculation output, Tensor, data type float16, bfloat16, must be 3 dimensions. The final calculation result has the same type as the input x
 * @param [out] workspaceSize: Output parameter, returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Out parameter, returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return value, return status code
 * Because required for collective communication and BatchMatMul calculations, the input and output shapes need to satisfy the following mathematical relationship: (where ep=epWorldSize, tp=tpWorldSize)
 * Perform the ReduceScatter scenario according to the H axis, that is, the scenario where yShardType is 0:
 * x: (E/ep, ep * C, M/tp)
 * weight：(E/ep, M/tp, H)
 * bias: (E/ep, 1, H/tp) in two dimensions: (E/ep, H/tp)
 * y：(E, C, H/tp)
 * Perform the ReduceScatter scenario according to the C axis, that is, the scenario where yShardType is 1:
 * x: (E/ep, ep * tp * C/tp, M/tp)
 * weight：(E/ep, M/tp, H)
 * bias: (E/ep, 1, H) in two dimensions: (E/ep, H)
 * y：(E, C/tp, H)
 * Data relationship description:
 * For example, x.size(0) is equal to E/tp and y.size(0) is equal to E, which means that y.size(0) = ep * x.size(0), y.size(0) is an integer multiple of ep; other relationships are similar
 * The value range of E is [2, 512], and E is an integer multiple of ep;
 * The value range of H is: [1, 65535]. When yShardType is 0, H is an integer multiple of tp;
 * The value range of M/tp is: [1, 65535];
 * The value range of E/ep is: [1, 32];
 * Both ep and tp only support 2, 4, 8, 16, and 32;
 * The names of groupEp and groupTp cannot be the same;
 * C is greater than 0, and the upper limit is the upper limit of the operator device memory. When yShardType is 1, C is an integer multiple of tp;
 * Cross-supernodes are not supported, only within supernodes;
 */
ACLNN_API aclnnStatus aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(
    const aclTensor* x, const aclTensor* weight, const aclTensor* biasOptional,
    const char* groupEp, const char* groupTp, int64_t epWorldSize,
    int64_t tpWorldSize, int64_t yShardType, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnBatchMatMulReduceScatterAlltoAll's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnBatchMatMulReduceScatterAlltoAll(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_