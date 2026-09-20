/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_SMOOTH_L1_LOSS_BACKWARD_H_
#define OP_API_INC_SMOOTH_L1_LOSS_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnSmoothL1LossBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: Complete the SmoothL1Loss backward operation on the input Tensor
 * For the first case of smoothL1Loss, that is, $|x-y|<1$, the derivative is: $$\frac{\partial SmoothL1Loss(x,y)}{\partial x} = x - y $$
 * For the first case of smoothL1Loss, that is $|x-y|\geq 1$, its derivative is: $$\frac{\partial SmoothL1Loss(x,y)}{\partial x} = sign(x-y)$$
 * @param [in] gradOut: aclTensor on the NPU device side, the data type supports FLOAT and FLOAT16, the data format supports ND, and non-continuous Tensor is supported.
 * @param [in] self: aclTensor on the NPU device side, the data type supports FLOAT and FLOAT16, the data format supports ND, and non-continuous Tensor is supported.
 * @param [in] target: aclTensor on the NPU device side, the data type supports FLOAT and FLOAT16, the data format supports ND, and non-continuous Tensor is supported.
 * @param [in] reduction: parameter on the host side, data type is int64_t, specifies the reduction to be applied to the output, supports 0("none"), 1("mean"), 2"sum").
 * @param [in] beta: Specifies the threshold that changes between L1 and L2 losses. The data type is double. The value must be non-negative.
 * @param [in] gradInput: aclTensor on the NPU device side, the data type supports FLOAT and FLOAT16, the data format supports ND, and non-continuous Tensor is supported.
 * When reduction is "none", shape is the same as self, otherwise shape is [1]
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Returns the status code. ACLNN_SUCCESS is returned on success, and the corresponding error code is returned on failure.
 */
ACLNN_API aclnnStatus aclnnSmoothL1LossBackwardGetWorkspaceSize(
    const aclTensor* gradOut, const aclTensor* self, const aclTensor* target,
    int64_t reduction, float beta, aclTensor* gradInput,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnSmoothL1LossBackward is used to perform calculations
 *
 * Operator function: Complete the SmoothL1Loss backward operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSigmoidBackwardGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Returns the status code, returns ACLNN_SUCCESS if successful, and returns the corresponding error code if failed.
 */
ACLNN_API aclnnStatus aclnnSmoothL1LossBackward(void* workspace,
                                                uint64_t workspaceSize,
                                                aclOpExecutor* executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SMOOTH_L1_LOSS_BACKWARD_H_