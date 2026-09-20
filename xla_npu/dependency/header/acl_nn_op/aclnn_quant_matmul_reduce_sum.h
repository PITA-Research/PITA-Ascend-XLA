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
 * \file aclnn_quant_matmul_reduce_sum.h
 * \brief
 */
#ifndef OP_API_ACLNN_QUANT_MATMUL_REDUCE_SUM_H_
#define OP_API_ACLNN_QUANT_MATMUL_REDUCE_SUM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnQuantMatmulReduceSumWeightNz calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Complete the quantified grouping matrix calculation, and then output the matrix calculation results of all groups after reduceSum.
 *
 * @param [in] x1: left matrix, data type support: int8.
 * @param [in] x2: right matrix, data type support: int8.
 * @param [in] x1Scale: [pertoken_scale] Inverse quantization parameter of matrix calculation, data type support: float32.
 * @param [in] x2Scale: [scale] The scaling factor in the quantization parameter, data type support: bfloat16.
 * @param [in] yScale: reserved parameters
 * @param [in] x1Offset: reserved parameter
 * @param [in] x2Offset: reserved parameter
 * @param [in] yOffset: reserved parameters
 * @param [in] bias: reserved parameter
 *
 * @param [in] transposeX1: reserved parameters
 * @param [in] transposeX2: reserved parameters
 * @param [in] groupSize: reserved parameter
 * @param [in] dims: one-dimensional array, representing the dimensions of cumulative summation, currently only supports [0]
 * @param [in] keepDims: reserved parameters
 *
 * @param [out] out: calculation result, data type: bfloat16.
 * @param [out] workspaceSize: Returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale,
    const aclTensor* x2Scale, const aclTensor* yScale,
    const aclTensor* x1Offset, const aclTensor* x2Offset,
    const aclTensor* yOffset, const aclTensor* bias, bool transposeX1,
    bool transposeX2, int64_t groupSize, const aclIntArray* dims, bool keepDims,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnQuantMatmulReduceSumWeightNz's second interface is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnQuantMatmulReduceSumWeightNz(void* workspace,
                                                        uint64_t workspaceSize,
                                                        aclOpExecutor* executor,
                                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_ACLNN_QUANT_MATMUL_REDUCE_SUM_H_
