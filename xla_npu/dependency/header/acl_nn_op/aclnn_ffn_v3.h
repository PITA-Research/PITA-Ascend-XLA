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
 * \file aclnn_ffn_v3.h
 * \brief
 */

#ifndef OP_API_INC_FFN_V3_H
#define OP_API_INC_FFN_V3_H
#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnFFNV3 calculates the workspace size according to the specific calculation process.
 * Function description: This FFN operator provides the calculation functions of MoeFFN and FFN
 * Calculation formula: y=activation(xW1+b1)W2+b2
 * @domain aclnn_ops_infer
 * @param [in]
 * x: required parameter, aclTensor on the Device side, input x in the formula, data type supports FLOAT16, BFLOAT16, INT8, data format supports ND, supported input dimensions are at least 2 dimensions [M,
 * K1], up to 8 dimensions.
 * @param [in]
 * weight1: required parameter, aclTensor on the Device side, expert weight data, W1 in the formula, data type supports FLOAT16, BFLOAT16, INT8, INT4, data format supports ND, input with/without experts is [E,
 * K1, N1]/[K1, N1]。
 * @param [in]
 * weight2: required parameter, aclTensor on the Device side, expert weight data, W2 in the formula, the data type supports FLOAT16, BFLOAT16, INT8, INT4, the data format supports ND, the input is [E,
 * K2, N2]/[K2, N2]。
 * @param [in]
 * expertTokensOptional: Optional parameter. The aclTensor type on the Device side represents the number of tokens for each expert. The data type supports INT64 and the data format supports ND. If it is not empty, the maximum length supported is 256.
 * @param [in]
 * bias1Optional: Optional parameter, aclTensor on the Device side, weight data correction value, b1 in the formula, data type supports FLOAT16, FLOAT32, INT32, data format supports ND, input with or without experts is [E,
 * N1]/[N1]。
 * @param [in]
 * bias2Optional: Optional parameters, aclTensor on the Device side, weight data correction value, b2 in the formula, data type supports FLOAT16, FLOAT32, INT32, data format supports ND, input with or without experts is [E,
 * N2]/[N2]。
 * @param [in]
 * scaleOptional: Optional parameter, aclTensor on the Device side, quantization parameter, quantification scaling coefficient, data type supports FLOAT32, data format supports ND, input under per-tensor is a one-dimensional vector with/without experts, and the number of input elements is [E]/[1] with/without experts respectively; input under per-channel is a two-dimensional vector/one-dimensional vector with/without experts, and the number of input elements is [E,
 * N1]/[N1]。
 * @param [in]
 * offsetOptional: Optional parameter, aclTensor on the Device side, quantization parameter, quantization offset, data type supports FLOAT32, data format supports ND, one-dimensional vector, the number of input elements is [E]/[1] when there is/without experts respectively.
 * @param [in]
 * deqScale1Optional: Optional parameter, aclTensor on the Device side, quantization parameter, the inverse quantization scaling coefficient of the first matmul, the data type supports UINT64, INT64, FLOAT32, BFLOAT16, the data format supports ND, the input is [E,
 * N1]/[N1]。
 * @param [in]
 * deqScale2Optional: Optional parameter, aclTensor on the Device side, quantization parameter, inverse quantization scaling coefficient of the second matmul, data type supports UINT64, INT64, FLOAT32, BFLOAT16, data format supports ND, input with/without experts is [E,
 * N2]/[N2]。
 * @param [in]
 * antiquantScale1Optional: Optional parameter, aclTensor on the Device side, pseudo quantization parameter, scaling coefficient of the first matmul, data type supports FLOAT16, BFLOAT16, data format supports ND, per-channel input with/without experts is [E,
 * N1]/[N1], per-in-group input with or without experts is [E, G, N1]/[G, N1] respectively.
 * @param [in]
 * antiquantScale2Optional: Optional parameters, aclTensor on the Device side, pseudo quantization parameter, scaling coefficient of the second matmul, data type supports FLOAT16, BFLOAT16, data format supports ND, per-channel input with/without experts is [E,
 * N2]/[N2], per-in-group input with/without experts is [E, G, N2]/[G, N2] respectively.
 * @param [in]
 * antiquantOffset1Optional: Optional parameter, aclTensor on the Device side, pseudo quantization parameter, offset of the first matmul, data type supports FLOAT16, BFLOAT16, data format supports ND, per-channel input with/without experts is [E,
 * N1]/[N1], per-in-group input with or without experts is [E, G, N1]/[G, N1] respectively.
 * @param [in]
 * antiquantOffset2Optional: Optional parameter, aclTensor on the Device side, pseudo quantization parameter, offset of the second matmul, data type supports FLOAT16, BFLOAT16, data format supports ND, per-channel input with/without experts is [E,
 * N2]/[N2], per-in-group input with/without experts is [E, G, N2]/[G, N2] respectively.
 * @param [in]
 * activation: required parameter, attribute value on the Host side, representing the activation function used, activation in the formula, currently supports fastgelu/gelu/relu/silu and geglu/swiglu/reglu.
 * @param [in]
 * innerPrecise: optional parameter, int on the Host side, indicating high-precision or high-performance selection. Data type support: INT64. This parameter only takes effect for FLOAT16. BFLOAT16 and INT8 do not distinguish between high precision and high performance.
 * @param [in] tokensIndexFlag: Optional parameter, bool on the Host side, indicating whether expertTokens is an index value. Data type support: bool.
 * @param [out] y: Output Tensor, the output y in the formula, the data type supports FLOAT16, BFLOAT16, the data format supports ND, and the output dimension is consistent with x.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the Device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFFNV3GetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2,
    const aclTensor *expertTokensOptional, const aclTensor *bias1Optional,
    const aclTensor *bias2Optional, const aclTensor *scaleOptional,
    const aclTensor *offsetOptional, const aclTensor *deqScale1Optional,
    const aclTensor *deqScale2Optional,
    const aclTensor *antiquantScale1Optional,
    const aclTensor *antiquantScale2Optional,
    const aclTensor *antiquantOffset1Optional,
    const aclTensor *antiquantOffset2Optional, const char *activation,
    int64_t innerPrecise, bool tokensIndexFlag, const aclTensor *y,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnFFNV3 is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the Device side.
 * @param [in] workspaceSize: The workspace size applied for on the Device side, obtained by the first interface aclnnFFNV3GetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: Specifies the AscendCL stream to execute the task.
 * @return aclnnStatus: return status code
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFFNV3(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_FFN_V3_H