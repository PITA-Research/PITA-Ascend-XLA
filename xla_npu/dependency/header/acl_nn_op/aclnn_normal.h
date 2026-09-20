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

#ifndef OP_API_INC_NORMAL_H_
#define OP_API_INC_NORMAL_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * The first interface of @brief aclnnInplaceNormal calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports integer, floating point, and complex number data types, and the data type needs to form a mutual derivation relationship with other, and the shape needs to satisfy the broadcast relationship with other.
 * Supports non-continuous Tensor, and the data format supports ND, NCHW, NHWC, HWCN, NDHWC, NCDHW, and the data format needs to be consistent with other.
 * @param [in] other: npu
 * The data type of aclTensor on the device side supports integer, floating point, and complex data types, and the data type needs to form a mutual derivation relationship with self, and the shape needs to satisfy the broadcast relationship with self.
 * Supports non-continuous Tensor, the data format supports ND, NCHW, NHWC, HWCN, NDHWC, NCDHW, and the data format needs to be consistent with self.
 * @param [in] alpha: aclScalar on the host side, the data type needs to be convertible into the data type derived from self and other.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceNormalGetWorkspaceSize(
    const aclTensor* selfRef, float mean, float std, int64_t seed,
    int64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceNormal is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceAddGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceNormal(void* workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceNormalTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_rand
 * 
 * Operator function: Generate random numbers from a discrete normal distribution with mean mean and standard deviation std, and fill them into the self tensor
 * @param [in] selfRef: aclTensor on NPU device side,
 * Data types support FLOAT, INT32, INT64, FLOAT16, BFLOAT16, INT16, INT8, UINT8, BOOL, and DOUBLE. The data format supports ND. Supports non-contiguous Tensors.
 * @param [in] mean: The floating point type on the host side, the left boundary of the discrete uniform distribution value, the input is the FLOAT data type.
 * @param [in] std: The floating point type on the host side, the right boundary of the discrete uniform distribution value, the input is the FLOAT data type.
 * @param [in] seedTensor: aclTensor on the NPU device side, the data type supports INT64. The data format supports ND.
 * The seed of the random number generator, which affects the sequence of random numbers generated.
 * @param [in] offsetTensor: aclTensor on the NPU device side, the data type supports INT64. The data format supports ND.
 * The offset of the random number generator, which affects the position of the generated random number sequence. After setting the offset, the generated random number sequence will start from the specified position.
 * @param [in] offset: an integer on the host side, the offset in the graph of the random number generator, which affects the position of the generated random number sequence. The input is of INT64_T data type.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceNormalTensorGetWorkspaceSize(
    const aclTensor* selfRef, float mean, float std,
    const aclTensor* seedTensor, const aclTensor* offsetTensor, int64_t offset,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceNormalTensor is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceNormalTensorGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceNormalTensor(void* workspace,
                                               uint64_t workspaceSize,
                                               aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_NORMAL_H_
