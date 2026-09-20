/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_WEIGHT_QUANT_BATCH_MATMUL_V2_H_
#define OP_API_INC_LEVEL2_ACLNN_WEIGHT_QUANT_BATCH_MATMUL_V2_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnWeightQuantBatchMatmulV2's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: realizes matrix multiplication of pseudo-quantization scenarios.
 * @param [in] x: matmul left matrix, data type support: float16, bfloat16.
 * @param [in] weight: matmul right matrix, data type support: int8, int4, float8_e5m2, float8_e4m3fn, hifloat8, int32, float, float4_e2m1.
 * @param [in] antiquantScale: Antiquantization scale parameter, data type support: float16, bfloat16, float8_e8m0.
 * @param [in] antiquantOffsetOptional: Antiquantization offset parameter, data type support: required to be consistent with x data type.
 * @param [in] quantScaleOptional: Reserved parameter, fixed to pass in null pointer.
 * @param [in] quantOffsetOptional: Reserved parameter, fixed to pass in null pointer.
 * @param [in] biasOptional: bias, data type support: float16, float, bfloat16.
 * @param [in] antiquantGroupSize: Under pseudo-quantization pergroup and mx, the groupSize input for inverse quantization calculation of the input weight describes the size of the amount of data to be inverse quantization corresponding to a set of inverse quantization parameters in the Reduce direction.
 * @param [in] y: calculation result, data type support: float16, bfloat16.
 * @param [in] workspaceSize: Returns the workspace size that needs to be applied for on the NPU device side.
 * @param [in] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* weight,
    const aclTensor* antiquantScale, const aclTensor* antiquantOffsetOptional,
    const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional,
    const aclTensor* biasOptional, int antiquantGroupSize, const aclTensor* y,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnWeightQuantBatchMatmulV2's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnWeightQuantBatchMatmulV2GetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnWeightQuantBatchMatmulV2(void* workspace,
                                                    uint64_t workspaceSize,
                                                    aclOpExecutor* executor,
                                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_WEIGHT_QUANT_BATCH_MATMUL_V2_H_