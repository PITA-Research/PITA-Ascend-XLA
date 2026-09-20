/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ADD_RMS_NORM_QUANT_H_
#define OP_API_INC_LEVEL2_ADD_RMS_NORM_QUANT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnAddRmsNormQuant calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Add + RmsNorm + fusion operator for quantitative calculation. The addition calculation result is quantized after layer normalization calculation.
 * And return the addition calculation result, the quantized normalization calculation result and the quantization parameter.
 * Calculation formula:
 *     x = x1 + x2
 *     y = RmsNorm(x, gamma)
 *     y1 = quant(y, scales1, zero_points1)
 *     y2 = quant(y, scales2, zero_points2)
 *
 * @param [in] x1:
 * Input `x1` in the formula, the data type supports BFLOAT16, FLOAT, and FLOAT16, and the shape dimension needs to be less than or equal to 8 dimensions.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] x2:
 * Input `x2` in the formula, the data type supports BFLOAT16, FLOAT, and FLOAT16, and the shape dimension needs to be less than or equal to 8 dimensions.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] gamma:
 * Input `gamma` in the formula, the data type supports BFLOAT16, FLOAT, and FLOAT16, and the shape dimension needs to be less than or equal to 8 dimensions.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] scales1:
 * Input `scales1` in the formula, the data type supports BFLOAT16, FLOAT, and FLOAT16, and the shape dimension needs to be less than or equal to 8 dimensions.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] scales2Optional:
 * Input `scales2` in the formula, the data type supports BFLOAT16, FLOAT, and FLOAT16, and the shape dimension needs to be less than or equal to 8 dimensions.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] zeroPoints1Optional:
 * The input `zero_points1` in the formula is an optional parameter. The data type supports INT32, BFLOAT16, FLOAT, and FLOAT16. The shape dimension needs to be less than or equal to 8 dimensions.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] zeroPoints2Optional:
 * The input `zero_points2` in the formula is an optional parameter. The data type supports INT32, BFLOAT16, FLOAT, and FLOAT16. The shape dimension needs to be less than or equal to 8 dimensions.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] axis: int64 type, specifies the quantization axis to be applied, currently only -1 is supported.
 * @param [in] epsilon: double type, a parameter used in layer normalization to prevent division by zero.
 * @param [in] divMode: bool type, used to specify the algorithm for static quantization calculation. When False, static quantization calculations use multiplication and vice versa.
 * @param [in] y1Out:
 * The output `y1` in the formula supports INT8, HiFLOAT8, FLOAT8_E5M2, and FLOAT8_E4M3FN data types, and the shape needs to be consistent with x1.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] y2Out:
 * The output `y2` in the formula supports INT8, HiFLOAT8, FLOAT8_E5M2, and FLOAT8_E4M3FN data types, and the shape needs to be consistent with x1.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] xOut:
 * For the output `x` in the formula, the data type supports BFLOAT16, FLOAT, and FLOAT16 and needs to be consistent with x1, and the shape needs to be consistent with x1.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAddRmsNormQuantGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma,
    const aclTensor* scales1, const aclTensor* scales2Optional,
    const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional,
    int64_t axis, double epsilon, bool divMode, aclTensor* y1Out,
    aclTensor* y2Out, aclTensor* xOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnAddRmsNormQuant's second interface is used to perform calculations.
 *
 * Operator function: Convert the input tensor to the specified dtype.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddRmsNormQuantGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAddRmsNormQuant(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ADD_RMS_NORM_QUANT_H_
