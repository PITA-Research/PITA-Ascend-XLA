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
 * \file aclnn_matmul_reduce_scatter_v2.h
 * \brief
 */
#ifndef OP_API_INC_MATMUL_REDUCE_SCATTER_V2_
#define OP_API_INC_MATMUL_REDUCE_SCATTER_V2_

#include <string>

#include "aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: implement mm + reduceScatter fusion calculation
 * The first interface of @brief aclnnMatmulReduceScatter calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] x1: matmul left matrix, data type support: float16, BF16, float8_E5M2, float_E4M3FN, HiF8.
 * @param [in] x2: matmul right matrix, data type support: float16, BF16, float8_E5M2, float_E4M3FN, HiF8.
 * @param [in] bias: bias, data type support: when the left matrix is ​​float16/BF16, the input type is the same as the left matrix; when the left matrix is ​​float8_E5M2/float_E4M3FN/HiF8, the input type is float.
 * @param [in] x1Scale: matmul left matrix inverse quantization scale, data type support: float.
 * @param [in] x2Scale: matmul right matrix inverse quantization scale, data type support: float.
 * @param [in] quantScale: Output matrix quantization scale, data type support: float.
 * @param [in] blockSize: The number of values ​​corresponding to different axes of a quantization coefficient in the output. Default value: 0.
 * @param [in] group: String identifying the column group.
 * @param [in] reduceOp: reduce operation type, default value: sum.
 * @param [in] commTurn: communication data cut fraction, that is, total data volume/single communication volume, default value: 0.
 * @param [in] streamMode: enumeration of ACL stream mode, type support: 0/1.
 * @param [in] groupSize: The number of values ​​corresponding to different axes of x1/x2 for an inverse quantization coefficient. Default value: 0.
 * @param [in] commMode: Communication mode. Currently two modes are supported: aicpu/aiv, default value: aicpu.
 * @param [out] output: The result of calculation + communication, data type: when the left matrix is ​​float16/BF16, the output is the same as the left matrix; when the left matrix is ​​float8_E5M2/float_E4M3FN/HiF8, the output is float16, BF16, float.
 * @param [out] amaxOutOptional: The maximum value of the output matrix, data type: float32.
 * @param [out] workspaceSize: Returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnMatmulReduceScatterV2GetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
    const aclTensor* x1Scale, const aclTensor* x2Scale,
    const aclTensor* quantScale, int64_t blockSize, const char* group,
    const char* reduceOp, int64_t commTurn, int64_t streamMode,
    int64_t groupSize, const char* commMode, aclTensor* output,
    aclTensor* amaxOutOptional, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnMatmulReduceScatter's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMatmulReduceScatterV2GetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnMatmulReduceScatterV2(void* workspace,
                                                 uint64_t workspaceSize,
                                                 aclOpExecutor* executor,
                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MATMUL_REDUCE_SCATTER_V2_
