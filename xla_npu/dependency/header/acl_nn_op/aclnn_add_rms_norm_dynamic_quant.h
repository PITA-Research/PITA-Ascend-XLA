/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ADD_RMS_NORM_DYNAMIC_QUANT_H_
#define OP_API_INC_LEVEL2_ADD_RMS_NORM_DYNAMIC_QUANT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnAddRmsNormDynamicQuant calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Add + RmsNorm + fusion operator for quantitative calculation. The addition calculation result is quantized after layer normalization calculation.
 * And return the addition calculation result, the quantized normalization calculation result and the quantization parameter.
 * Calculation formula:
 *     x = x1 + x2
 *     y = RmsNorm(x, gamma)
 *     y1 = dynamicQuant(y, smoothScale1Optional, smoothScale2Optional)
 *     y2 = dynamicQuant(y, smoothScale1Optional, smoothScale2Optional)
 *
 * @param [in] x1:
 * Input `x1` in the formula, the data type supports BFLOAT16, FLOAT16, and the shape dimension supports 2-8 dimensions.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] x2:
 * Input `x2` in the formula, the data type supports BFLOAT16, FLOAT16, and the shape dimension supports 2-8 dimensions.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] gamma:
 * Input `gamma` in the formula, the data type supports BFLOAT16, FLOAT16
 * , the shape dimension needs to be less than or equal to 8 dimensions.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] smoothScale1Optional:
 * The input in the formula is `smoothScale1`, the data type supports BFLOAT16 and FLOAT16, and the shape dimension needs to be equal to the last dimension of x1.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] smoothScale2Optional:
 * The input in the formula is `smoothScale2Optional`, the data type supports BFLOAT16 and FLOAT16, and the shape dimension needs to be equal to the last dimension of x1.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] epsilon: double type, a parameter used in layer normalization to prevent division by zero.
 * @param [in] y1Out:
 * The output `y1` in the formula supports INT8, HiFLOAT8, FLOAT8_E5M2, and FLOAT8_E4M3FN data types, and the shape needs to be consistent with x1.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] y2Out:
 * The output `y2` in the formula supports INT8, HiFLOAT8, FLOAT8_E5M2, and FLOAT8_E4M3FN data types, and the shape needs to be consistent with x1.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] xOut:
 * For the output `x` in the formula, the data type supports BFLOAT and FLOAT16 and needs to be consistent with x1, and the shape needs to be consistent with x1.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] scale1Out:
 * The output `scale1` in the formula, the data type supports FLOAT, and the shape needs to be consistent with x1 except for the last dimension.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] scale2Out:
 * The output `scale2` in the formula, the data type supports FLOAT, and the shape needs to be consistent with x1 except for the last dimension.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAddRmsNormDynamicQuantGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma,
    const aclTensor* smoothScale1Optional,
    const aclTensor* smoothScale2Optional, double epsilon, aclTensor* y1Out,
    aclTensor* y2Out, aclTensor* xOut, aclTensor* scale1Out,
    aclTensor* scale2Out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAddRmsNormQuant's second interface is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddRmsNormDynamicQuantGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAddRmsNormDynamicQuant(void* workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ADD_RMS_NORM_QUANT_H_
