/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef OP_API_INC_UNIFORM_H_
#define OP_API_INC_UNIFORM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnInplaceUniform calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 *
 * Operator function: Generate discrete uniformly distributed random numbers in the interval [from, to) and fill them into the self tensor
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *  A[(selfRef)] -.->B([l0op::Contiguous])
 *  B --> D([l0op::StatelessRandomUniformV2])
 *  E[(seed)] -->D
 *  X[(offset)] -->D
 *  P[(alg)] -->D
 *  D --> F([l0op::Muls])
 *  G[(to)] --> F
 *  D --> H([l0op::Muls])
 *  I[(from)] --> H
 *  H --> J([l0op::Sub])
 *  K[(from)] --> J
 *  F --> L([l0op::Sub])
 *  J --> L
 *  L --> Q([l0op::Cast])
 *  Q -.-> N([l0op::ViewCopy])
 *  N --> O[(Out)]
 * ```
 *
 * @param [in] selfRef: aclTensor on NPU device side,
 * Data types support FLOAT, INT32, INT64, FLOAT16, INT16, INT8, UINT8, BOOL, and DOUBLE. The data format supports ND. Supports non-contiguous Tensors.
 * @param [in] from: Floating point type on the host side, the left boundary of discrete uniformly distributed values, and the input is DOUBLE data type.
 * @param [in] to: The floating point type on the host side, the right boundary of the discrete uniform distribution value, the input is DOUBLE data type.
 * @param [in] seed: The seed of the random number generator, which affects the generated random number sequence. The input is the UINT64_T data type.
 * @param [in] offset:
 * The offset of the random number generator, which affects the position of the generated random number sequence. After setting the offset, the generated random number sequence will start from the specified position. The input is of UINT64_T data type.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceUniformGetWorkspaceSize(
    const aclTensor* selfRef, double from, double to, uint64_t seed,
    uint64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceUniform is used to perform calculations.
 *
 * Operator function: Generate discrete uniformly distributed random numbers in the interval [from, to) and fill them into the self tensor
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *  A[(selfRef)] -.->B([l0op::Contiguous])
 *  B --> D([l0op::StatelessRandomUniformV2])
 *  E[(seed)] -->D
 *  X[(offset)] -->D
 *  P[(alg)] -->D
 *  D --> F([l0op::Muls])
 *  G[(to)] --> F
 *  D --> H([l0op::Muls])
 *  I[(from)] --> H
 *  H --> J([l0op::Sub])
 *  K[(from)] --> J
 *  F --> L([l0op::Sub])
 *  J --> L
 *  L --> Q([l0op::Cast])
 *  Q -.-> N([l0op::ViewCopy])
 *  N --> O[(Out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnUniformGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceUniform(void* workspace,
                                          uint64_t workspace_size,
                                          aclOpExecutor* executor,
                                          const aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceUniformTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 *
 * Operator function: Generate discrete uniformly distributed random numbers in the interval [from, to) and fill them into the self tensor
 * @param [in] selfRef: aclTensor on NPU device side,
 * Data types support FLOAT, INT32, INT64, FLOAT16, INT16, INT8, UINT8, BOOL, and DOUBLE. The data format supports ND. Supports non-contiguous Tensors.
 * @param [in] from: Floating point type on the host side, the left boundary of discrete uniformly distributed values, and the input is DOUBLE data type.
 * @param [in] to: The floating point type on the host side, the right boundary of the discrete uniform distribution value, the input is DOUBLE data type.
 * @param [in] seedTensor: aclTensor on the NPU device side, the data type supports UINT64. The data format supports ND.
 * The seed of the random number generator, which affects the sequence of random numbers generated.
 * @param [in] offsetTensor: aclTensor on the NPU device side, the data type supports UINT64. The data format supports ND.
 * The offset of the random number generator, which affects the position of the generated random number sequence. After setting the offset, the generated random number sequence will start from the specified position.
 * @param [in] offset: an integer on the host side, the offset of the random number generator, which affects the position of the generated random number sequence. The input is of UINT64_T data type.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceUniformTensorGetWorkspaceSize(
    const aclTensor* selfRef, double from, double to,
    const aclTensor* seedTensor, const aclTensor* offsetTensor, uint64_t offset,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceUniformTensor is used to perform calculations.
 *
 * Operator function: Generate discrete uniformly distributed random numbers in the interval [from, to) and fill them into the self tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceUniformTensorGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceUniformTensor(void* workspace,
                                                uint64_t workspace_size,
                                                aclOpExecutor* executor,
                                                const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNIFORM_H_
