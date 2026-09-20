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

/*!
 * \file aclnn_dropout_gen_mask.h
 * \brief
 */
#ifndef OP_API_INC_DROPOUT_GEN_MASK_H_
#define OP_API_INC_DROPOUT_GEN_MASK_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnDropoutGenMask calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 */
ACLNN_API aclnnStatus aclnnDropoutGenMaskGetWorkspaceSize(
    const aclIntArray* shape, double prob, int64_t seed, int64_t offset,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnDropoutGenMask(void* workspace,
                                          uint64_t workspaceSize,
                                          aclOpExecutor* executor,
                                          aclrtStream stream);

/**
 * @brief The first interface of aclnnDropoutGenMaskV2 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 */
ACLNN_API aclnnStatus aclnnDropoutGenMaskV2GetWorkspaceSize(
    const aclIntArray* shape, double prob, int64_t seed, int64_t offset,
    aclDataType probDataType, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnDropoutGenMaskV2(void* workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);

/**
 * The first interface of @brief aclnnDropoutGenMaskV2Tensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 * 
 * Operator function: During the training process, a mask is generated according to the probability prob, which is used to set the elements to zero.
 * @param [in] shape: aclIntArray on the NPU device side,
 * The data type supports INT32 or INT64. The data format supports ND. Supports non-contiguous Tensors.
 * @param [in] prob: Floating point type on the host side, probability of setting to zero, input is DOUBLE data type.
 * @param [in] seedTensor: aclTensor on the NPU device side, the data type supports INT64. The data format supports ND.
 * The seed of the random number generator, which affects the sequence of random numbers generated.
 * @param [in] offsetTensor: aclTensor on the NPU device side, the data type supports INT64. The data format supports ND.
 * The offset of the random number generator, which affects the position of the generated random number sequence. After setting the offset, the generated random number sequence will start from the specified position.
 * @param [in] offset: an integer on the host side, the offset of the random number generator, which affects the position of the generated random number sequence. The input is of INT64_T data type.
 * @param [in] probDataType: Data type enumeration on the Host side, indicating the data type of prob. The data type supports FLOAT, FLOAT16, and BFLOAT16.
 * @param [out] out: Operation result, aclTensor on the NPU device side, bit type and mask data stored in UINT8 type.
 * The data type supports UINT8, and the shape needs to be (align(number of input elements, 128)/8). The data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnDropoutGenMaskV2TensorGetWorkspaceSize(
    const aclIntArray* shape, double prob, const aclTensor* seedTensor,
    const aclTensor* offsetTensor, int64_t offset, aclDataType probDataType,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnDropoutGenMaskV2Tensor's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnDropoutGenMaskV2TensorGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnDropoutGenMaskV2Tensor(void* workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_DROPOUT_GEN_MASK_H_
