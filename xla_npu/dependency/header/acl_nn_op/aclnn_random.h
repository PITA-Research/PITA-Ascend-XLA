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
#ifndef OP_API_INC_LEVEL2_ACLNN_RANDOM_H_
#define OP_API_INC_LEVEL2_ACLNN_RANDOM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnInplaceRandom calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 * @param [in] self: aclTensor on NPU device side, data type supports FLOAT, INT32, INT64, BFLOAT16, FLOAT16,
 * INT16, INT8, UINT8, BOOL, DOUBLE. The data format supports ND. Supports non-contiguous Tensors.
 * @param [in] from: Floating point type on the host side, the left boundary of discrete uniformly distributed values, and the input is DOUBLE data type.
 * @param [in] to: The floating point type on the host side, the right boundary of the discrete uniform distribution value, the input is DOUBLE data type.
 * @param [in] seed: The seed of the random number generator, which affects the generated random number sequence. The input is the UINT64_T data type.
 * @param [in] offset: The offset of the random number generator, which affects the position of the generated random number sequence. After setting the offset,
 * The generated random number sequence will start from the specified position. The input is of UINT64_T data type.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceRandomGetWorkspaceSize(
    const aclTensor* selfRef, int64_t from, int64_t to, int64_t seed,
    int64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceRandom is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceRandom.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */

ACLNN_API aclnnStatus aclnnInplaceRandom(void* workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceRandomTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 *
 * Operator function: Fill the selfRef tensor with numbers sampled from the discrete uniform distribution of [from, to-1].
 * @param [in] selfRef: aclTensor on the NPU device side, data type supports FLOAT, INT32, INT64, BFLOAT16, FLOAT16,
 * INT16, INT8, UINT8, BOOL, DOUBLE. The data format supports ND. Supports non-contiguous Tensors.
 * @param [in] from: Floating point type on the host side, the left boundary of discrete uniformly distributed values, and the input is DOUBLE data type.
 * @param [in] to: The floating point type on the host side, the right boundary of the discrete uniform distribution value, the input is DOUBLE data type.
 * @param [in] seedTensor: aclTensor on the NPU device side, the data type supports INT64. The data format supports ND.
 * The seed of the random number generator, which affects the sequence of random numbers generated.
 * @param [in] offsetTensor: aclTensor on the NPU device side, the data type supports INT64. The data format supports ND.
 * The offset of the random number generator, which affects the position of the generated random number sequence. After setting the offset, the generated random number sequence will start from the specified position.
 * @param [in] offset: Integer on the host side, offset of the random number generator. The input is of INT64_T data type.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceRandomTensorGetWorkspaceSize(
    const aclTensor* selfRef, int64_t from, int64_t to,
    const aclTensor* seedTensor, const aclTensor* offsetTensor, int64_t offset,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceRandomTensor is used to perform calculations.
 *
 * Operator function: Fill the selfRef tensor with numbers sampled from the discrete uniform distribution of [from, to-1].
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceRandom.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */

ACLNN_API aclnnStatus aclnnInplaceRandomTensor(void* workspace,
                                               uint64_t workspaceSize,
                                               aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_RANDOM_H_
