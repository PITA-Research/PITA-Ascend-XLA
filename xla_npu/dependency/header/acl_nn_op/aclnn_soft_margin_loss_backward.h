/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_SOFT_MARGIN_LOSS_BACKWARD_H_
#define OP_API_INC_SOFT_MARGIN_LOSS_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnSoftMarginLossBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: Backpropagation of the mean square error function.
 *
 * @param [in] gradOutput：npu
 * aclTensor on the device side, the data type supports FLOAT and FLOAT16, and the shape needs to be the same as self,
 * The target satisfies the broadcast relationship. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] self：npu
 * The data type of aclTensor on the device side supports FLOAT and FLOAT16, and the shape needs to satisfy the broadcast relationship with gradOutput and target.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] target：npu
 * aclTensor on the device side, the data type supports FLOAT and FLOAT16, and the shape needs to be consistent with gradOutput and
 * self satisfies the broadcast relationship. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] reduction: int64 on the host side, specifies the reduction to be applied to the output, supports 0('none') | 1('mean') | 2('sum'). 'none'
 * means no reduction will be applied, 'mean' means the sum of the output will be divided by the number of elements in the output, 'sum' means the output will be summed.
 * @param [in] out：npu
 * aclTensor on the device side, the data type supports FLOAT and FLOAT16, and the shape needs to be target, self, and gradOutput to meet broadcast
 * shape afterwards. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSoftMarginLossBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
    int64_t reduction, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnSoftMarginLossBackward is used to perform calculations.
 *
 * Operator function: Backpropagation of the mean square error function.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, determined by the first interface
 * aclnnSoftMarginLossBackwardGetWorkspaceSize gets.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSoftMarginLossBackward(void* workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SOFT_MARGIN_LOSS_BACKWARD_H_
