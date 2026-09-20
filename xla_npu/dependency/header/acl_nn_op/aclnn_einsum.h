/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OP_API_INC_EINSUM_H_
#define OP_API_INC_EINSUM_H_

#include <string>
#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnEinsum calculates the workspace size according to the specific calculation process.
 * Function description: Compute algebraic tensor operations on sequences of tensors using the Einstein summation convention.
 * @domain aclnn_ops_infer
 * @param [in]   tensors:
 * Enter TensorList, the data type supports FLOAT16, FLOAT, INT16, UINT16, INT32, UINT32, INT64, and UINT64. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] equation: input string, the data type supports const char *.
 * @param [out]  output:
 * Output Tensor, data type supports FLOAT16, FLOAT, INT16, UINT16, INT32, UINT32, INT64, UINT64. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnEinsumGetWorkspaceSize(const aclTensorList* tensors,
                                                  const char* equation,
                                                  aclTensor* output,
                                                  uint64_t* workspaceSize,
                                                  aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnEinsum is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnEinsumGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEinsum(void* workspace, uint64_t workspaceSize,
                                  aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_EINSUM_H_