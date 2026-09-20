/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_PRELU_BACKWARD_H_
#define OP_API_INC_PRELU_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnPreluBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: Complete the reverse of PRelu.
 *
 * @param [in] gradOutput: The gradient value of backpropagation, that is, the output gradient of the previous layer.
 * The data type of aclTensor on the NPU device side supports FLOAT16 and FLOAT32 types.
 * @param [in] self: The output value of PRelu.
 * The data type of aclTensor on the NPU device side supports FLOAT16 and FLOAT32 types.
 * @param [in] weight: the weight of PRelu.
 * The data type of aclTensor on the NPU device side supports FLOAT16 and FLOAT32 types.
 * @param [out] gradInput: The output of backward is the gradient value of the input, that is, the result of derivation of the input.
 * The data type of aclTensor on the NPU device side supports FLOAT16 and FLOAT32 types.
 * @param [out] gradWeight: The output of backward is the gradient value of weight, that is, the result of derivation of weight.
 * The data type of aclTensor on the NPU device side supports FLOAT16 and FLOAT32 types.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnPreluBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* weight,
    aclTensor* gradInput, aclTensor* gradWeight, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnPreluBackward is used to perform calculations
 */
ACLNN_API aclnnStatus aclnnPreluBackward(void* workspace,
                                         uint64_t workspace_size,
                                         aclOpExecutor* executor,
                                         aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_PRELU_BACKWARD_H_
