/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#ifndef OP_API_INC_SCATTER_ADD_H_
#define OP_API_INC_SCATTER_ADD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnTfScatterAdd calculates the workspace size according to the specific calculation process.
 * @domain aclnnop_ops_infer
 * @domain aclnnop_ops_train
 * Operator function: Accumulate the values ​​in tensor updates to the slice of tensor varRef according to the specified index tensor indices.
 * @param [in] varRef: aclTensor on NPU device side, data type supports FLOAT16, FLOAT32, BFLOAT16, INT32, INT8, UINT8
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] indices: aclTensor on the NPU device side. The data type supports INT32 and INT64 types. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] updates: aclTensor on NPU device side, data type supports FLOAT16, FLOAT32, BFLOAT16, INT32, INT8, UINT8
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnTfScatterAddGetWorkspaceSize(
    aclTensor* varRef, const aclTensor* indices, const aclTensor* updates,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnTfScatterAdd is used to perform calculations
 * @domain aclnnop_ops_infer
 * @domain aclnnop_ops_train
 * Operator function: Accumulate the values ​​in tensor updates to the slice of tensor varRef according to the specified index tensor indices.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnTfScatterAddGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTfScatterAdd(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SCATTER_ADD_H_