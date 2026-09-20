/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GELU_BACKWARD_H_
#define OP_API_INC_GELU_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGeluBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: Complete the reverse of Gelu.
 * Gelu forward (where x can be a scalar or Tensor):
 * $$
 * Gelu(x)=x \cdot \Phi(x)=x/2 \cdot [1+erf(x/\sqrt{2})]
 * $$
 * The calculation formula of erf is:
 * $$
 * erf(x)=\frac{2}{\sqrt \pi}\sum^{\infty}_{n=0}{\frac{(-1)^n \cdot x^{2n+1}}{n! \cdot (2n+1)}}
 * $$
 * The relationship between gradInput and gradOutput can be expressed as:
 * $$
 * gradInput = gradOutput \cdot (\frac{1}{2}+\frac{1}{2} \cdot \\
 * erf(\frac{x}{\sqrt2})+\frac{x}{\sqrt{2\pi}} \cdot e^{-\frac{x^2}{2}})
 * $$
 * Attachment: The approximate calculation formula of Gelu is:
 * $$
 * Gelu(x)=0.5x(1+tanh(\sqrt{2/\pi}(x+0.044715x^3)))
 * $$
 *
 * Implementation instructions
 * API calculation base path:
 * ```mermaid
 * graph LR
 * A[(gradOutput)] -->B([l0op::Contiguous])
 * B --> C([l0op::GeluGrad])
 * D[(self)] --> E([l0op::Contiguous])
 * H[(gradInput)] --> I([l0op::Contiguous])
 * E --> C
 * I --> C
 * C --> F([l0op::ViewCopy])
 * F --> G[(gradInput)]
 * ```
 *
 * @param [in] gradOutput: The gradient value of backpropagation, that is, the output gradient of the previous layer, is consistent with the shape of the forward output.
 * aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, and BFLOAT16 types.
 * The data format supports FRACTAL_NZ, NC1HWC0, ND, and non-continuous Tensor.
 * @param [in] self: Gelu's output value.
 * aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, and BFLOAT16 types.
 * The data format supports FRACTAL_NZ, NC1HWC0, ND, and non-continuous Tensor.
 * @param [out] gradInput: The output of backward is the gradient value of the input, that is, the result of derivation of the input.
 * aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, and BFLOAT16 types.
 * The data format supports FRACTAL_NZ, NC1HWC0, ND.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGeluBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self,
    const aclTensor* gradInput, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnGeluBackward is used to perform calculations
 */
ACLNN_API aclnnStatus aclnnGeluBackward(void* workspace,
                                        uint64_t workspace_size,
                                        aclOpExecutor* executor,
                                        const aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GELU_BACKWARD_H_
