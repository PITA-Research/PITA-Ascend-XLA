/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_NLL_LOSS_BACKWARD_H_
#define OP_API_INC_NLL_LOSS_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnNLLLossBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: Backpropagation of negative log-likelihood loss function.
 *
 * @param [in] gradOutput: npu
 * The aclTensor on the device side has shape (N,) or (1,), and the data type supports FLOAT. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] self: aclTensor on the NPU device side, shape is (N,C) or (C,), where N represents batch
 * size, C represents the number of categories. The data type supports FLOAT. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] target: aclTensor on the NPU device side, which represents the real label. The shape is (N,) or (1,), and the value range of each element is [0, C - 1].
 * The data type supports INT64, UINT8, non-continuous Tensor, and the data format supports ND.
 * @param [in] weight: npu
 * The aclTensor on the device side represents the weight of each category, and the shape is (C,). The data type supports FLOAT. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] reduction: int64_t on the host side, specifies the calculation method of the loss function, supports 0('none') | 1('mean') | 2('sum').
 * 'none' means no reduction will be applied, 'mean' means the sum of the output will be divided by the number of elements in the output, 'sum' means the output will be summed.
 * @param [in] ignoreIndex: int64_t on the host side, specifies a target value that is ignored and does not affect the input gradient.
 * @param [in] totalWeight: npu
 * The aclTensor on the device side has a shape of (1,) and the data type supports FLOAT. And the data type is the same as weight. The data format supports ND.
 * @param [in] out: aclTensor on the NPU device side. shape needs to be consistent with self. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnNLLLossBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
    const aclTensor* weight, int64_t reduction, int64_t ignoreIndex,
    const aclTensor* totalWeight, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnNLLLossBackward is used to perform calculations.
 *
 * Operator function: Backpropagation of negative log-likelihood loss function.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnNLLLossBackwardGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnNLLLossBackward(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ADD_H_
