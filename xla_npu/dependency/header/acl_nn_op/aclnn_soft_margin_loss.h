/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_SOFT_MARGIN_LOSS_H_
#define OP_API_INC_SOFT_MARGIN_LOSS_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnSoftMarginLoss calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: Calculate the binary logical loss function of the input self and target target.
 *
 * @param [in] self: aclTensor on the NPU device side. The data type supports FLOAT and FLOAT16. The shape needs to satisfy the broadcast relationship with the target.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] target: aclTensor on the NPU device side. The data type supports FLOAT and FLOAT16. The shape needs to satisfy the broadcast relationship with self.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] reduction: int64 on the host side, specifying the reduction to be applied to the output, supporting 0('none') | 1('mean') | 2('sum').
 * 'none' means no reduction will be applied, 'mean' means the sum of the output will be divided by the number of elements in the output, 'sum' means the output will be summed.
 * @param [in] out: aclTensor on the NPU device side, the data type supports FLOAT and FLOAT16. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSoftMarginLossGetWorkspaceSize(
    const aclTensor* self, const aclTensor* target, int64_t reduction,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnSoftMarginLoss is used to perform calculations.
 *
 * Operator function: Calculate the binary logical loss function of the input self and target target.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSoftMarginLossGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSoftMarginLoss(void* workspace,
                                          uint64_t workspaceSize,
                                          aclOpExecutor* executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SOFT_MARGIN_LOSS_H_
