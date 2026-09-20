/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEAKY_RELU_BACKWARD_H_
#define OP_API_INC_LEAKY_RELU_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * The first interface of @brief aclnnLeakyReluBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: reverse activation function.
 *
 * @param [in] gradOutput: npu
 * The data type of aclTensor on the device side supports FLOAT, BFLOAT16, FLOAT16, and DOUBLE, and the data type is consistent with self, and the shape is the same as self.
 * Supports [non-continuous Tensor](#non-continuous Tensor description), and the data format supports ND[(reference)](#reference).
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports FLOAT, BFLOAT16, FLOAT16, and DOUBLE. Supports [non-continuous Tensor](#non-continuous Tensor description), and the data format supports ND[(reference)](#reference).
 * @param [in] negativeSlope: aclScalar on the host side, indicating the slope when self<0.
 * @param [in] selfIsResult: bool on the host side, indicating whether self is used as output.
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports FLOAT, BFLOAT16, FLOAT16, and DOUBLE. Supports [non-continuous Tensor](#non-continuous Tensor description), and the data format supports ND[(reference)](#reference).
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLeakyReluBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self,
    const aclScalar* negativeSlope, bool selfIsResult, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * The second interface of @brief aclnnLeakyReluBackward is used to perform calculations.
 *
 * Operator function: reverse activation function.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnLeakyReluGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLeakyReluBackward(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEAKY_RELU_BACKWARD_H_