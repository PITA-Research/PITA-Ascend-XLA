/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ADD_LAYER_NORM_QUANT_H_
#define OP_API_INC_LEVEL2_ADD_LAYER_NORM_QUANT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnAddLayerNormQuant calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Add + LayerNorm + fusion operator for quantification calculation. The addition calculation result is quantized after layer normalization calculation.
 * And return the addition calculation result, the quantized normalization calculation result and the quantization parameter.
 * Calculation formula:
 *     x = x1 + x2 + bias
 *     y = LayerNorm(x, gamma, beta)
 *     y1, out_scales1 = quant(y, scales1, zero_points1)
 *     y2, out_scales2 = quant(y, scales2, zero_points2)
 *
 * @param [in] x1:
 * Input `x1` in the formula, the data type supports BFLOAT16, FLOAT, FLOAT16 and needs to be consistent with x2, and the shape needs to be consistent with x2.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] x2:
 * Input `x2` in the formula, the data type supports BFLOAT16, FLOAT, FLOAT16 and needs to be consistent with x1, and the shape needs to be consistent with x1.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] gamma:
 * Input `gamma` in the formula, the data type supports BFLOAT16, FLOAT, FLOAT16 and needs to be consistent with x1, the shape needs to be consistent with beta, and can be broadcast to x1.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] beta:
 * Input `beta` in the formula, the data type supports BFLOAT16, FLOAT, FLOAT16 and needs to be consistent with x1, and the shape needs to be consistent with gamma.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] biasOptional:
 * Enter `bias` in the formula. The data type supports BFLOAT16, FLOAT, and FLOAT16 and needs to be consistent with x1. The shape needs to be consistent with gamma or x1.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] scales1Optional:
 * Enter `scales1` in the formula. The data type supports BFLOAT16, FLOAT, and FLOAT16 and needs to be consistent with scales2Optional. The shape needs to be consistent with gamma.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] scales2Optional:
 * Enter `scales2` in the formula. The data type supports BFLOAT16, FLOAT, and FLOAT16 and needs to be consistent with scales1Optional. The shape needs to be consistent with gamma.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] zeroPoints1Optional:
 * Enter `zero_points1` in the formula. The data type supports BFLOAT16, FLOAT, and FLOAT16 and needs to be consistent with scales1Optional. The shape needs to be consistent with gamma.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] zeroPoints2Optional:
 * Input `zero_points2` in the formula. The data type supports BFLOAT16, FLOAT, and FLOAT16 and needs to be consistent with scales1Optional. The shape needs to be consistent with gamma.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] quantMode: const char* type, specifies the quantization mode to be applied,
 * Supports "dynamic" or "static", indicating dynamic quantization/static quantization respectively.
 * @param [in] epsilon: double type, a parameter used in layer normalization to prevent division by zero.
 * @param [in] additionalOutput: bool
Type, used to specify whether the calculation result x of Add is output. When it is Fasle, the operator will not output xOut, and xOut is dirty data at this time.
 * @param [in] divMode: bool
Type, used to specify the algorithm for static quantization calculation. When False, static quantization calculations use multiplication and vice versa. Only takes effect when quantMode is "dynamic".
 * @param [in] y1Out:
 * The output `y1` in the formula, the data type supports INT8, and the shape needs to be consistent with x1.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] y2Out:
 * The output `y2` in the formula, the data type supports INT8, and the shape needs to be consistent with x1.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] xOut:
 * For the output `x` in the formula, the data type supports BFLOAT16, FLOAT, and FLOAT16 and needs to be consistent with x1, and the shape needs to be consistent with x1.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] outScales1Out:
 * The input `out_scales1` in the formula, the data type supports FLOAT, and the shape needs to be consistent with x1 excluding the last dimension.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] outScales2Out:
 * The input `out_scales2` in the formula, the data type supports FLOAT, and the shape needs to be consistent with x1 excluding the last dimension.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
* @param [out] executor: Returns the op executor, including the operator calculation process.
* @return aclnnStatus: Return status code.

 */
ACLNN_API aclnnStatus aclnnAddLayerNormQuantGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma,
    const aclTensor* beta, const aclTensor* biasOptional,
    const aclTensor* scales1Optional, const aclTensor* scales2Optional,
    const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional,
    const char* quantMode, double epsilon, bool additionalOutput, bool divMode,
    aclTensor* y1Out, aclTensor* y2Out, aclTensor* xOut,
    aclTensor* outScales1Out, aclTensor* outScales2Out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnAddLayerNormQuant is used to perform calculations.
 *
 * Operator function: Convert the input tensor to the specified dtype.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddLayerNormQuantGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAddLayerNormQuant(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ADD_LAYER_NORM_QUANT_H_
