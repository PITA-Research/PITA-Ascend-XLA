/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_TRANSPOSE_BATCH_MAT_MUL
#define OP_API_INC_TRANSPOSE_BATCH_MAT_MUL

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnTransposeBatchMatmul calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: implements the fusion operator of TransposeBatchMatMul operator and Quant. TransposeBatchMatMul can support parameters of int8.
 * @param [in] x1: matmul left matrix, data type support: float16, float32, bfloat16.
 * @param [in] x2: matmul right matrix, data type support: float16, float32, bfloat16.
 * @param [in] bias: bias, currently not supported.
 * @param [in] scale: scaling factor in quantization parameters, data type support: int64, uint64.
 * @param [in] permX1: Represents the shape of input x1.
 * @param [in] permX2: Represents the shape of input x2.
 * @param [in] permY: Represents the shape of input y.
 * @param [in] cubeMathType: used to specify the calculation logic of the Cube unit, an integer on the Host side. Data type support: int8.
 * @param [in] batchSplitFactor: Whether to re-split the shape. Data type support: int32.
 * @param [out] out: calculation result, data type: float32, float16, bfloat16, int8.
 * @param [out] workspaceSize: Returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnTransposeBatchMatMulGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
    const aclTensor* scale, const aclIntArray* permX1,
    const aclIntArray* permX2, const aclIntArray* permY, int8_t cubeMathType,
    const int32_t batchSplitFactor, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnBatchMatmulQuant's second interface is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnTransposeBatchMatMul(void* workspace,
                                                uint64_t workspaceSize,
                                                aclOpExecutor* executor,
                                                const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_TRANSPOSE_BATCH_MAT_MUL
