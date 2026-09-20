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

#ifndef OP_API_INC_LEVEL2_ACLNN_SEARCHSORTED_H_
#define OP_API_INC_LEVEL2_ACLNN_SEARCHSORTED_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnSearchSorted calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Find the position where a given value (self) should be inserted in a sorted tensor (sortedSequence).
 * @param [in] sortedSequence: aclTensor on the NPU device side, data type supports FLOAT16, FLOAT32, INT32, INT8, UINT8, INT16, INT64,
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] self: aclTensor on the NPU device side, data type supports FLOAT16, FLOAT32, INT32, INT8, UINT8, INT16, INT64,
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] outInt32: BOOL type on the host side, specifying whether the output Tensor is of INT32 type.
 * @param [in] right: BOOL type on the host side. If the same value is found, whether to return the right position.
 * @param [in] sorter: aclTensor on the NPU device side, the data type supports INT64, specifies the order of elements in sortedSequence.
 * @param [in] out: aclTensor on the NPU device side, data type supports FLOAT16, FLOAT32, INT32, INT8, UINT8, INT16, INT64,
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnSearchSortedGetWorkspaceSize(
    const aclTensor* sortedSequence, const aclTensor* self, const bool outInt32,
    const bool right, const aclTensor* sorter, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnSearchSorted calculates the workspace size according to the specific calculation process.
 *
 * Operator function: Find the position where a given value (self) should be inserted in a sorted tensor (sortedSequence).
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSearchSortedGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSearchSorted(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        aclrtStream stream);

/**
 * The first interface of @brief aclnnSearchSorteds calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Find the position where a given value (self) should be inserted in a sorted tensor (sortedSequence).
 * @param [in] sortedSequence: aclTensor on the NPU device side, data type supports FLOAT16, FLOAT32, INT32, INT8, UINT8, INT16, INT64,
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] self: aclScalar on the NPU device side, the data type supports FLOAT16, FLOAT32, INT32, INT8, UINT8, INT16, INT64.
 * @param [in] outInt32: BOOL type on the host side, specifying whether the output Tensor is of INT32 type.
 * @param [in] right: BOOL type on the host side. If the same value is found, whether to return the right position.
 * @param [in] sorter: aclTensor on the NPU device side, the data type supports INT64, specifies the order of elements in sortedSequence.
 * @param [in] out: aclTensor on the NPU device side, data type supports FLOAT16, FLOAT32, INT32, INT8, UINT8, INT16, INT64,
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnSearchSortedsGetWorkspaceSize(
    const aclTensor* sortedSequence, const aclScalar* self, const bool outInt32,
    const bool right, const aclTensor* sorter, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnSearchSorteds is used to perform calculations.
 *
 * Operator function: Find the position where a given value (self) should be inserted in a sorted tensor (sortedSequence).
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSearchSortedsGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSearchSorteds(void* workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_SEARCHSORTED_H_
