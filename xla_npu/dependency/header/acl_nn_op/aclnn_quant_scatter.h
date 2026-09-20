/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef OP_API_INC_LEVEL2_ACLNN_QUANT_SCATTER_H_
#define OP_API_INC_LEVEL2_ACLNN_QUANT_SCATTER_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnInplaceQuantScatter calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function:
 * First quantize the updates, and then update the values in selfRef one by one according to the specified axis axis and index indices. This operator is a custom operator semantics,
 * There is no corresponding tensorflow or pytorch interface.
 * @param [in] selfRef: aclTensor on the NPU device side, the data type supports INT8. The number of dimensions needs to be consistent with updates.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] indices: aclTensor on the NPU device side, the data type supports INT32, INT64. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] updates: aclTensor on the NPU device side, the data type supports FLOAT16, BFLOAT16 (only supported by Ascend 910B AI processor).
 * The number of dimensions needs to be consistent with selfRef. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] quantScales: aclTensor on the NPU device side, the data type supports FLOAT32, BFLOAT16 (only supported by Ascend 910B AI processor) types.
 * Supports non-continuous tensors. The data format supports ND.
 * @param [in] quantZeroPoints: aclTensor on the NPU device side, the data type supports INT32, BFLOAT16 (only supported by Ascend 910B AI processor) type.
 * Supports non-continuous tensors. The data format supports ND.
 * @param [in] axis: The dimension used for scatter, the data type is INT64.
 * @param [in] quantAxis: Dimension used for quantification, data type is INT64.
 * @param [in] reduction: Specifies the reduction to be applied to the output, the data type is INT64. Currently only 1('update') is supported.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnInplaceQuantScatterGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* indices, const aclTensor* updates,
    const aclTensor* quantScales, const aclTensor* quantZeroPoints,
    int64_t axis, int64_t quantAxis, int64_t reduction, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnInplaceQuantScatter is used to perform calculations
 * @domain aclnn_ops_infer
 * Operator function:
 * First quantize the updates, and then update the values in selfRef one by one according to the specified axis axis and index indices. This operator is a custom operator semantics,
 * There is no corresponding tensorflow or pytorch interface.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnInplaceQuantScatterGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceQuantScatter(void* workspace,
                                               uint64_t workspaceSize,
                                               aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_QUANT_SCATTER_H_