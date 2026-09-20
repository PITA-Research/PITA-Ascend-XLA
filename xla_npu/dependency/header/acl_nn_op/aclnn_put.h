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
#ifndef OP_API_INC_PUT_H_
#define OP_API_INC_PUT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnInplacePut calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: treat selfRef as a one-dimensional tensor, use the element value in the index tensor as the index, if accumulate is true, perform an accumulation operation on the elements in the source and the elements at the positions corresponding to selfRef;
 * If accumulate is false, replace the element in source with the element at the corresponding position of selfRef.
 *
 * @param [in] selfRef: npu
 * aclTensor on the device side, the data type supports FLOAT, BFLOAT16, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL,
 * Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [in] index: npu
 * The data type of aclTensor on the device side supports INT32 and INT64, and the number of elements must be consistent with the source. Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with self.
 * @param [in] source: npu
 * The aclTensor on the device side has the same data type as selfRef, and the number of elements is the same as index. Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplacePutGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* index, const aclTensor* source,
    bool accumulate, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplacePut is used to perform calculations.
 *
 * Operator function: treat selfRef as a one-dimensional tensor, use the element value in the index tensor as the index, if accumulate is true, perform an accumulation operation on the elements in the source and the elements at the positions corresponding to selfRef;
 * If accumulate is false, replace the element in source with the element at the corresponding position of selfRef.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplacePutGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplacePut(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_PUT_H_
