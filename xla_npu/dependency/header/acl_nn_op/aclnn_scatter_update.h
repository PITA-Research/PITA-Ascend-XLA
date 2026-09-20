/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef OP_API_INC_SCATTER_UPDATE_H_
#define OP_API_INC_SCATTER_UPDATE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnScatterUpdate calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Update the values ​​in tensor data one by one according to the specified axis axis and index indices. This operator is a custom operator semantics,
 * There is no corresponding tensorflow or pytorch interface.
 * @param [in] data: aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, INT8, BFLOAT16, data only supports four dimensions.
 * And the dimension size and number of dimensions need to be consistent with updates. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] indices: aclTensor on the NPU device side, the data type supports INT32,
 * The int64 type currently only supports one dimension. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] updates: aclTensor on NPU device side, data type supports FLOAT16, FLOAT32, INT8,
 * BFLOAT16, and the dimension size and number of dimensions need to be consistent with data. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] axis: axis on the host side, the data type supports INT64.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnInplaceScatterUpdateGetWorkspaceSize(
    aclTensor* data, const aclTensor* indices, const aclTensor* updates,
    int64_t axis, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnScatterUpdate is used to perform calculations
 *
 * Operator function: Update the values ​​in tensor data one by one according to the specified axis axis and index indices. This operator is a custom operator semantics,
 * There is no corresponding tensorflow or pytorch interface.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnInplaceScatterUpdateGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceScatterUpdate(void* workspace,
                                                uint64_t workspaceSize,
                                                aclOpExecutor* executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SCATTER_UPDATE_H_