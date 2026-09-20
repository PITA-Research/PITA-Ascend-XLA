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
#ifndef OP_API_INC_ADD_RELU_H_
#define OP_API_INC_ADD_RELU_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnAddRelu calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: After completing the addition calculation, the tensor result is processed: if it is greater than 0, it will remain unchanged, if it is less than 0, it will be modified to 0.
 *
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to form a mutual derivation relationship with other, and the shape needs to satisfy the broadcast relationship with other.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with other.
 * @param [in] other: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to form a mutual derivation relationship with self, and the shape needs to satisfy the broadcast relationship with self.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with self.
 * @param [in] alpha: aclScalar on the host side, the data type needs to be convertible into the data type derived from self and other.
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to be a convertible data type after self and other are deduced, and the shape needs to be self and other.
 * The data format of the shape after broadcast supports ND, and the data format needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAddReluGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclScalar* alpha,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnAdd is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAddRelu(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor, aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceAddRelu calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: After completing the addition calculation, the tensor result is processed: if it is greater than 0, it will remain unchanged, if it is less than 0, it will be modified to 0.
 *
 * @param [in] selfRef: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to form a mutual derivation relationship with other, and the shape needs to satisfy the broadcast relationship with other.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with other.
 * @param [in] other: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to form a mutual derivation relationship with self, and the shape needs to satisfy the broadcast relationship with self.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with self.
 * @param [in] alpha: aclScalar on the host side, the data type needs to be convertible into the data type derived from self and other.
 * The data type of aclTensor on the device side supports integer and floating point types, and the data type needs to be a convertible data type after self and other are deduced, and the shape needs to be self and other.
 * The data format of the shape after broadcast supports ND, and the data format needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceAddReluGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, aclScalar* alpha,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnAdd is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceAddRelu(void* workspace,
                                          uint64_t workspaceSize,
                                          aclOpExecutor* executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ADD_RELU_H_
