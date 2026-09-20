/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_SIGMOID_BACKWARD_H_
#define OP_API_INC_SIGMOID_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnSigmoidgBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: Complete the sigmoid backward operation on the input Tensor
 * @param [in] gradOutput: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128,
 * The shape is non-empty, the data format supports ND, and non-continuous Tensors are supported.
 * @param [in] output: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128,
 * The shape remains the same as gradOutput. The data format supports ND and non-continuous Tensor.
 * @param [in] gradInput: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128,
 * The shape remains the same as gradOutput. The data format supports ND and non-continuous Tensor.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnSigmoidBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* output, aclTensor* gradInput,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnSigmoidBackward is used to perform calculations
 *
 * Operator function: Complete the sigmoid backward operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSigmoidBackwardGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSigmoidBackward(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SIGMOID_BACKWARD_H_