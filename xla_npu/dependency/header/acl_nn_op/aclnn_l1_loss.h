/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_L1_LOSS_H_
#define OP_API_INC_L1_LOSS_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnL1Loss calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: Calculate the mean square error between each element in input x and target y.
 *
 * @param [in] self:
 * The input `self` in the formula must meet the data type derivation rules with the target. After derivation, the data type supports INT64, FLOAT, and FLOAT16, and the shape needs to meet the broadcast rules with the target.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] target:
 * The input `target` in the formula must meet the data type derivation rules with self. After derivation, the data type supports INT64, FLOAT, and FLOAT16, and shape needs to meet the broadcast rules with self.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] reduction: input `reduction` in the formula, specifying the reduction to be applied to the output,
 * Supports 0('none') | 1('mean') | 2('sum').
 * 'none' means no reduction will be applied, 'mean' means the sum of the output will be divided by the number of elements in the output, 'sum' means the output will be summed.
 * @param [in] out:
 * The data type of `out` in the formula supports INT64, FLOAT, and FLOAT16. When reduction is 0, the shape needs to be consistent with the shape after broadcasting self and target.
 * Other scenarios are rank1de Tensor. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnL1LossGetWorkspaceSize(
    const aclTensor* self, const aclTensor* target, int64_t reduction,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnL1Loss is used to perform calculations.
 *
 * Operator function: Calculate the mean square error between each element in input x and target y.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnL1LossGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnL1Loss(void* workspace, uint64_t workspaceSize,
                                  aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_L1_LOSS_H_
