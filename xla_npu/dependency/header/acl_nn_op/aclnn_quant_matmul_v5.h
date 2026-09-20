/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_QUANT_MATMUL_V5_
#define OP_API_INC_QUANT_MATMUL_V5_

#include <string>

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnQuantMatmulV5 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: realizes matrix multiplication of quantized scenes.
 * @param [in] x1: matmul left matrix, data type support: float4_e2m1, float4_e1m2, int8, float8_e4m3fn, float8_e5m2, hifloat8.
 * @param [in] x2: matmul right matrix, data type support: float4_e2m1, float4_e1m2, int8, float8_e4m3fn, float8_e5m2, hifloat8.
 * @param [in] x1Scale: x1 quantization parameter, data type support: float8_e8m0, float32.
 * @param [in] x2Scale: x2 quantization parameter, data type support: float8_e8m0, bfloat16, float32, int64, uint64.
 * @param [in] yScale: y quantification parameter, data type support: int64, uint64.
 * @param [in] x1Offset: Reserved parameter, the current interface does not support this parameter.
 * @param [in] x2Offset: quantization parameter, data type support: float32.
 * @param [in] yOffset: Reserved parameter, the current interface does not support this parameter.
 * @param [in] bias: bias, supported data types: int32, bfloat16, float16, float32.
 * @param [in] transposeX1: Whether the x1 matrix is ​​transposed.
 * @param [in] transposeX2: Whether the x2 matrix is ​​transposed.
 * @param [in] groupSize: Quantitative parameter, data type support: int64.
 * @param [out] out: calculation result, data type: int8, float16, bfloat16, float32, float8_e4m3fn, hifloat8.
 * @param [out] workspaceSize: Returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnQuantMatmulV5GetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale,
    const aclTensor *x2Scale, const aclTensor *yScale,
    const aclTensor *x1Offset, const aclTensor *x2Offset,
    const aclTensor *yOffset, const aclTensor *bias, bool transposeX1,
    bool transposeX2, int64_t groupSize, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnQuantMatmulV5's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnQuantMatmulV5GetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnQuantMatmulV5(void *workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor *executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANT_MATMUL_V5_
