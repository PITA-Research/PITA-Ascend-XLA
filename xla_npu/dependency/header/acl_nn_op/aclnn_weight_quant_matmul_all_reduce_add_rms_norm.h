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
 * \file aclnn_weight_quant_matmul_all_reduce_add_rms_norm.h
 * \brief
 */
#ifndef OP_API_INC_WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_
#define OP_API_INC_WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_

#include <string>

#include "aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnWeightQuantMatmulAllReduceAddRmsNorm calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: implement MatmulAllReduce+AddRmsNorm fusion calculation
 * @param [in] x1: matmul left matrix, data type support: float16, bfloat16.
 * @param [in] x2: matmul right matrix, data type support: int8, int4.
 * @param [in] bias: bias, data type support: float16, bfloat16.
 * @param [in] antiquantScale: Perform pseudo-quantization calculation parameters on x2, data type support: float16, bfloat16.
 * @param [in] antiquantOffset: pseudo-quantization calculation parameters for x2, data type support: float16, bfloat16.
 * @param [in] residual: residual, data type support: float16, bfloat16.
 * @param [in] gamma: RmsNorm normalization parameter, data type support: float16, bfloat16.
 * @param [in] epsilon: Prevent division by 0 errors, data type support: double.
 * @param [in] group: String identifying the column group.
 * @param [in] reduceOp: reduce operation type, default value: sum.
 * @param [in] commTurn: communication data cut fraction, that is, total data volume/single communication volume, default value: 0.
 * @param [in] streamMode: enumeration of ACL stream mode, type support: 1.
 * @param [in] antiquantGroupSize: groupSize input in per_group mode.
 * @param [out] y: Result of MatmulAllReduce+Add(residual), data type: same as input.
 * @param [out] normOut: The result of MatmulAllReduce+AddRmsNorm, data type: same as input.
 * @param [out] workspaceSize: Returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */

ACLNN_API aclnnStatus aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
    const aclTensor* antiquantScale, const aclTensor* antiquantOffset,
    const aclTensor* residual, const aclTensor* gamma, double epsilon,
    const char* group, const char* reduceOp, int64_t commTurn,
    int64_t streamMode, int64_t antiquantGroupSize, const aclTensor* y,
    const aclTensor* normOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnWeightQuantMatmulAllReduceAddRmsNorm's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: the workspace size applied for on the NPU device side,
 * Obtained by the first interface aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnWeightQuantMatmulAllReduceAddRmsNorm(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_