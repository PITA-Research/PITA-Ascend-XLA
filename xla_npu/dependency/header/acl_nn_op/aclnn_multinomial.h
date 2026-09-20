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
#ifndef OP_API_INC_LEVEL2_ACLNN_MULTINOMIAL_H_
#define OP_API_INC_LEVEL2_ACLNN_MULTINOMIAL_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnMultinomial calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 */
ACLNN_API aclnnStatus aclnnMultinomialGetWorkspaceSize(
    const aclTensor* self, int64_t numsamples, bool replacement, int64_t seed,
    int64_t offset, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnMultinomial is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnMultinomial(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream);

/**
 * The first interface of @brief aclnnMultinomialTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 *
 * Operator function: Extract numsamples samples according to the probability of each object distribution in the input tensor, and store the indices of these samples in the output tensor.
 * @param [in] self: aclTensor on the NPU device side, the shape is (N, C) or (C), the value range of self needs to be greater than or equal to 0 and the dimensions of self and out must be consistent.
 * The data type supports BFLOAT16, FLOAT16, FLOAT, and DOUBLE. The data format supports ND. Supports non-contiguous Tensors.
 * @param [in] numsamples: Integer on the host side, the number of samples drawn from each multinomial distribution.
 * numsamples is a non-negative integer. When replacement is false, numsamples is not greater than C. The input is of INT64_t data type.
 * @param [in] replacement: Boolean type on the host side, which determines whether elements are replaced during sampling. The input is of BOOL data type.
 * @param [in] seedTensor: aclTensor on the NPU device side, the data type supports INT64_t. The data format supports ND.
 * The seed of the random number generator, which affects the sequence of random numbers generated.
 * @param [in] offsetTensor: aclTensor on the NPU device side, the data type supports INT64_t. The data format supports ND.
 * The offset of the random number generator, which affects the position of the generated random number sequence. After setting the offset, the generated random number sequence will start from the specified position.
 * @param [in] offset: an integer on the host side, the offset of the random number generator, which affects the position of the generated random number sequence. The input is of INT64_T data type.
 * @param [in] out: aclTensor on the NPU device side, the data type supports INT64. The shape is (N, numsamples) or (numsamples), and the dimensions of self and out are consistent.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMultinomialTensorGetWorkspaceSize(
    const aclTensor* self, int64_t numsamples, bool replacement,
    const aclTensor* seedTensor, const aclTensor* offsetTensor, int64_t offset,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnMultinomialTensor is used to perform calculations.
 *
 * Operator function: Extract numsamples samples according to the probability of each object distribution in the input tensor, and store the indices of these samples in the output tensor.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMultinomialTensorGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMultinomialTensor(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_MULTINOMIAL_H_
