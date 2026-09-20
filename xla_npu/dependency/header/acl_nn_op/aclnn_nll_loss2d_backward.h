/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_NLL_LOSS2D_BACKWARD_H_
#define OP_API_INC_NLL_LOSS2D_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnNLLLoss2dBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: the inverse of negative log-likelihood loss.
 *
 * @param [in] gradOutput：npu
 * The shape of aclTensor on the device side is three-dimensional (the first dimension is N) or one-dimensional (and the number of elements is 1), and the data type supports FLOAT, FLOAT16, and BFLOAT16.
 * Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [in] self: aclTensor on the NPU device side, the shape is four-dimensional, the first dimension is N, which means batch
 * size, the second dimension is C representing the category, and the data type supports FLOAT and FLOAT16. Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [in] target: aclTensor on the NPU device side, indicating the real label, the shape is three-dimensional, the first dimension is N, and the value range of each element is [0, C -
 * 1], the data type supports INT64 and UINT8. Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [in] weight: aclTensor on the NPU device side, indicating the weight of each category, the shape is (C,), and the data type supports FLOAT and FLOAT16.
 * Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [in] reduction: int64_t on the host side, specifying the reduction to be applied to the output. Supports 0('none') | 1('mean') | 2('sum').
 * 'none' means no reduction will be applied, 'mean' means the sum of the output will be divided by the number of elements in the output, 'sum' means the output will be summed.
 * @param [in] ignoreIndex: int64_t on the host side, specifies a target value that is ignored and does not affect the input gradient.
 * @param [in] totalWeight：npu
 * The data type of aclTensor on the device side supports FLOAT and FLOAT16, and the data type is the same as weight, the shape is (1,), and the data format supports ND ([Reference] (#)).
 * @param [out] out: aclTensor on the NPU device side, the shape is the same as self. Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnNLLLoss2dBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target,
    const aclTensor* weight, int64_t reduction, int64_t ignoreIndex,
    aclTensor* totalWeight, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnNLLLoss2dBackward is used to perform calculations.
 *
 * Operator function: the inverse of negative log-likelihood loss.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnNLLLoss2dBackwardGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnNLLLoss2dBackward(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_NLL_LOSS2_DBACKWARD_H_
