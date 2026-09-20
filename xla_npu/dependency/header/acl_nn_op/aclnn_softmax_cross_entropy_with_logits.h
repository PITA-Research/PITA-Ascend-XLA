/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_SOFTMAX_CROSS_ENTROPY_WITH_H_
#define OP_API_INC_LEVEL2_ACLNN_SOFTMAX_CROSS_ENTROPY_WITH_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnSoftmaxCrossEntropyWithLogits calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Function description: Calculate the cross entropy loss of softmax and cross entropy and give the reverse gradient of the input logits.
 * Calculation formula:
 * loss = -Σ (y_i * log(softmax(x_i)))
 * backprop = softmax(x_i) - y_i
 * Among them, $x_i$ corresponds to the input features, and $y_i$ corresponds to the input labels.
 * @param [in] features: Input Tensor, data type supports FLOAT, FLOAT16, BFLOAT16. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] labels: Input Tensor, data type supports FLOAT, FLOAT16, BFLOAT16. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] loss: Output Tensor, data type supports FLOAT, FLOAT16, BFLOAT16. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] backprop: Output Tensor, data type supports FLOAT, FLOAT16, BFLOAT16. Supports non-continuous Tensor, and the data format supports ND.
 */
ACLNN_API aclnnStatus aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize(
    const aclTensor* features, aclTensor* labels, aclTensor* loss,
    aclTensor* backprop, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnSoftmaxCrossEntropyWithLogits is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGeluGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSoftmaxCrossEntropyWithLogits(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_SOFTMAX_CROSS_ENTROPY_WITH_H_