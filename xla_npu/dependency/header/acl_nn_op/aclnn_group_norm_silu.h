/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUP_NORM_SILU_H_
#define OP_API_INC_GROUP_NORM_SILU_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGroupNormSilu calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Complete the GroupNorm and Silu fusion operator function.
 * The calculation formula is:
 * y=((x-Ex) / (sqrt(Var(x)+ϵ​x))​)∗ γ + β
 * Divide the channel direction into groups, then normalize within each group, and calculate the mean value of (C//G) HW
 *
 * Attachment: The calculation formula of Silu is:
 * f(x) = x * sigmoid(x)
 * When x is greater than 0, the Silu activation function will amplify x, and when x is less than 0, the Silu activation function will reduce x, which can suppress overfitting.
 * ```
 *
 * @param [in] self: calculation input, shape is greater than or equal to 2 dimensions
 * aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, and BFLOAT16 types.
 * The data format supports ND and non-continuous Tensor.
 * @param [in] gamma: calculation input, shape is 1 dimension and equal to c dimension.
 * aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, and BFLOAT16 types.
 * The data format supports ND and non-continuous Tensor.
 * @param [in] beta: shape is 1 dimension and equal to c dimension.
 * aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, and BFLOAT16 types.
 * The data format supports ND and non-continuous Tensor.
 * @param [in] group: calculated attribute, integer on host side, data type supports INT64, group information
 * @param [in] esp: calculated attribute, floating point number on the host side, data type supports double, default le-5
 * @param [out] out: the output of y.
 * aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, and BFLOAT16 types.
 * The data format supports ND.
 * @param [out] meanOut: The output of the mean.
 * aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, and BFLOAT16 types.
 * The data format supports ND.
 * @param [out] rstdOut:1/sqrt(Var(x)+ϵ​x) result output
 * aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, and BFLOAT16 types.
 * The data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGroupNormSiluGetWorkspaceSize(
    const aclTensor* self, const aclTensor* gamma, const aclTensor* beta,
    int64_t group, double eps, aclTensor* out, aclTensor* meanOut,
    aclTensor* rstdOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGroupNormSilu's second interface, used to perform calculations
 */
ACLNN_API aclnnStatus aclnnGroupNormSilu(void* workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         aclrtStream stream);

/**
 * @brief The first interface of aclnnGroupNormSiluV2 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnGroupNormSiluV2GetWorkspaceSize(
    const aclTensor* self, const aclTensor* gamma, const aclTensor* beta,
    int64_t group, double eps, bool activateSilu, aclTensor* out,
    aclTensor* meanOut, aclTensor* rstdOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnGroupNormSiluV2(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GROUP_NORM_SILU_H_
