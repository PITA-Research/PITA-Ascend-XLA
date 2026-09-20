/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
#ifndef OP_API_INC_DROPOUT_V3_H_
#define OP_API_INC_DROPOUT_V3_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnDropoutV3 calculates the workspace size according to the specific calculation process.
 * Operator function: During the training process, the mask is generated according to probability P, the elements in the input are set to zero, and the output is scaled according to the ratio of 1/(1-p).
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *     A1[(input)]  --> B1([l0op::Contiguous])]--> C([l0op::DropoutV3])]
 *     A2[(noiseShape)]  --> B1([l0op::Contiguous])]--> C([l0op::DropoutV3])]
 *     A3((p)) --> C([l0op::DropoutV3])]
 *     A4((seed)) --> C([l0op::DropoutV3])]
 *     A5((offset)) --> C([l0op::DropoutV3])]
 *     C([l0op::DropoutV3])] --> D([l0op::ViewCopy]) -->E1[(out)]
 *     D --> E2[(maskOut)]
 * ```
 *
 * @param [in] input: aclTensor on the NPU device side.
 * The data type supports FLOAT16, FLOAT32, and BFLOAT16, and the data type must be the same as out. The data format supports ND. The shape must be the same as out. Non-continuous Tensors are supported.
 * @param [in] optionalNoiseShape: aclTensor on the NPU device side.
 * The data type supports INT64, the data format supports ND, and the shape is 1D, which is used to control the number of random numbers generated. The number of elements will be aligned to 128 numbers.
 * @param [in] p: The probability of discarding, the data type supports DOUBLE.
 * @param [in] seed: The seed for generating random numbers. The data type supports INT64.
 * @param [in] offset: offset to generate random numbers, the data type supports INT64.
 * @param [in] out: aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, BFLOAT16, and the data type must be the same as self, the data format supports ND, and the shape must be the same as self.
 * @param [in] maskOut: The mask of the input data that needs to be discarded, aclTensor on the NPU device side, the data type supports UINT8, the data format supports ND, and the shape is 1D.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnDropoutV3GetWorkspaceSize(
    const aclTensor* input, const aclTensor* optionalNoiseShape, double p,
    int64_t seed, int64_t offset, aclTensor* out, aclTensor* maskOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnDropoutV3 is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnDropoutV3GetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnDropoutV3(void* workspace, uint64_t workspaceSize,
                                     aclOpExecutor* executor,
                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_DROPOUT_V3_H_
