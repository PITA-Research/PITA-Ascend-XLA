/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
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
#ifndef OP_API_INC_BERNOULLI_H_
#define OP_API_INC_BERNOULLI_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBernoulli's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 *
 * Operator function: extract binary random numbers from Bernoulli distribution
 * Calculation formula:
 * $$ out_i∼Bernoulli(self_i) $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(Self)]  --> B([l0::Contiguous]) -->D([l0op::StatelessBernoulli]) --> I([l0op::ViewCopy]) --> J[(Out)]
 * K((p)) --> K0([ConvertToTensor]) --> D
 * E((seed)) --> D
 * F((offset)) --> D
 * ```
 *
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports integer and floating point types, supports non-continuous Tensor, and the data format supports ND
 * @param [in] prob: aclScalar on the host side, floating point type, needs to satisfy $ 0≤p≤1 $
 * @param [in] seed: aclScalar on the host side
 * @param [in] offset: aclScalar on the host side
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports integer and floating point types
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBernoulliGetWorkspaceSize(
    const aclTensor* self, const aclScalar* prob, int64_t seed, int64_t offset,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnBernoulli's second interface is used to perform calculations.
 *
 * Operator function: extract binary random numbers from Bernoulli distribution
 * Calculation formula:
 * $$ out_i∼Bernoulli(input_i) $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(Self)]  --> B([l0::Contiguous]) -->D([l0op::StatelessBernoulli]) --> I([l0op::ViewCopy]) --> J[(Out)]
 * K((p)) --> K0([ConvertToTensor]) --> D
 * E((seed)) --> D
 * F((offset)) --> D
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBernoulli(void* workspace, uint64_t workspaceSize,
                                     aclOpExecutor* executor,
                                     aclrtStream stream);

/**
 * The first interface of @brief aclnnBernoulliTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 *
 * Operator function: extract binary random numbers from Bernoulli distribution
 * Calculation formula:
 * $$ out_i∼Bernoulli(self_i) $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(Self)]  --> B([l0::Contiguous]) -->D([l0op::StatelessBernoulli]) --> I([l0op::ViewCopy]) --> J[(Out)]
 * K((p)) --> K0([ConvertToTensor]) --> D
 * E((seed)) --> D
 * F((offset)) --> D
 * ```
 *
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports integer and floating point types, supports non-continuous Tensor, and the data format supports ND
 * @param [in] prob: npu
 * aclTensor on the device side, the data type supports floating point type, supports non-continuous Tensor, and the data format supports ND
 * @param [in] seed: aclScalar on the host side
 * @param [in] offset: aclScalar on the host side
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports integer and floating point types
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBernoulliTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* prob, int64_t seed, int64_t offset,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnBernoulliTensor's second interface is used to perform calculations.
 *
 * Operator function: extract binary random numbers from Bernoulli distribution
 * Calculation formula:
 * $$ out_i∼Bernoulli(input_i) $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(Self)]  --> B([l0::Contiguous]) -->D([l0op::StatelessBernoulli]) --> I([l0op::ViewCopy]) --> J[(Out)]
 * K((p)) --> K0([ConvertToTensor]) --> D
 * E((seed)) --> D
 * F((offset)) --> D
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBernoulliTensor(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

/**
 * @brief aclnnInplaceBernoulli's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 *
 * Operator function: extract binary random numbers from Bernoulli distribution
 * Calculation formula:
 * $$ out_i∼Bernoulli(selfRef_i) $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(Self)]  --> B([l0::Contiguous]) -->D([l0op::StatelessBernoulli]) --> I([l0op::ViewCopy]) --> J[(Self)]
 * K((p)) --> K0([ConvertToTensor]) --> D
 * E((seed)) --> D
 * F((offset)) --> D
 * ```
 *
 * @param [in] selfRef: npu
 * aclTensor on the device side, the data type supports integer and floating point types, supports non-continuous Tensor, and the data format supports ND
 * @param [in] prob: aclScalar on the host side, floating point type, needs to satisfy $ 0≤p≤1 $
 * @param [in] seed: aclScalar on the host side
 * @param [in] offset: aclScalar on the host side
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceBernoulliGetWorkspaceSize(
    const aclTensor* selfRef, const aclScalar* prob, int64_t seed,
    int64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceBernoulli's second interface is used to perform calculations.
 *
 * Operator function: extract binary random numbers from Bernoulli distribution
 * Calculation formula:
 * $$ out_i∼Bernoulli(input_i) $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(Self)]  --> B([l0::Contiguous]) -->D([l0op::StatelessBernoulli]) --> I([l0op::ViewCopy]) --> J[(Self)]
 * K((p)) --> K0([ConvertToTensor]) --> D
 * E((seed)) --> D
 * F((offset)) --> D
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceBernoulli(void* workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceBernoulliTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 *
 * Operator function: extract binary random numbers from Bernoulli distribution
 * Calculation formula:
 * $$ out_i∼Bernoulli(selfRef_i) $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(Self)]  --> B([l0::Contiguous]) -->D([l0op::StatelessBernoulli]) --> I([l0op::ViewCopy]) --> J[(Self)]
 * K((p)) --> K0([ConvertToTensor]) --> D
 * E((seed)) --> D
 * F((offset)) --> D
 * ```
 *
 * @param [in] selfRef: npu
 * aclTensor on the device side, the data type supports integer and floating point types, supports non-continuous Tensor, and the data format supports ND
 * @param [in] prob: npu
 * aclTensor on the device side, the data type supports floating point type, supports non-continuous Tensor, and the data format supports ND
 * @param [in] seed: aclScalar on the host side
 * @param [in] offset: aclScalar on the host side
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceBernoulliTensorGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* prob, int64_t seed,
    int64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceBernoulliTensor's second interface is used to perform calculations.
 *
 * Operator function: extract binary random numbers from Bernoulli distribution
 * Calculation formula:
 * $$ out_i∼Bernoulli(input_i) $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(Self)]  --> B([l0::Contiguous]) -->D([l0op::StatelessBernoulli]) --> I([l0op::ViewCopy]) --> J[(Self)]
 * K((p)) --> K0([ConvertToTensor]) --> D
 * E((seed)) --> D
 * F((offset)) --> D
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceBernoulliTensor(void* workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_BERNOULLI_H_
