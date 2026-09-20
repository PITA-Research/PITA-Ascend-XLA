/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GELU_BACKWARD_V2_H_
#define OP_API_INC_GELU_BACKWARD_V2_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGeluBackwardV2's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * Operator function: Complete the reverse of Gelu.
 * @param [in] gradOutput: The gradient value of backpropagation, that is, the output gradient of the previous layer, is consistent with the shape of the forward output.
 * aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, and BFLOAT16 types.
 * The data format supports FRACTAL_NZ, NC1HWC0, ND, and non-continuous Tensor.
 * @param [in] self: output value of gelu.
 * aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, and BFLOAT16 types.
 * The data format supports FRACTAL_NZ, NC1HWC0, ND, and non-continuous Tensor.
 * @param [in] approximate: input parameter for gelu_backward calculation, specifies Gaussian approximation algorithm, configurable to "none" or "tanh".
 * @param [out] gradInput: The output of gelu_backward is the gradient value of the input, that is, the result of derivation of the input.
 * aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, and BFLOAT16 types.
 * The data format supports FRACTAL_NZ, NC1HWC0, ND, and non-continuous Tensor.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGeluBackwardV2GetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, char* approximate,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGeluBackwardV2's second interface, used to perform calculations
 */
ACLNN_API aclnnStatus aclnnGeluBackwardV2(void* workspace,
                                          uint64_t workspace_size,
                                          aclOpExecutor* executor,
                                          const aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GELU_BACKWARD_V2_H_
