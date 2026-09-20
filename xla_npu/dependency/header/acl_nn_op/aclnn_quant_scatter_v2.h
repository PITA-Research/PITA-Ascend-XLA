/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#ifndef OP_API_INC_LEVEL2_ACLNN_QUANT_SCATTER_V2_H_
#define OP_API_INC_LEVEL2_ACLNN_QUANT_SCATTER_V2_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnInplaceQuantScatterV2's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function:
 * First quantize the updates, and then update the values in selfRef one by one according to the specified axis axis and index indices. This operator is a custom operator semantics,
 * There is no corresponding tensorflow or pytorch interface.
 * @param [in] selfRef: aclTensor on the NPU device side, the data type supports INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8. The number of dimensions needs to be consistent with updates.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] indices: aclTensor on the NPU device side, the data type supports INT32, INT64. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] updates: aclTensor on the NPU device side, the data type supports FLOAT16, BFLOAT16.
 * The number of dimensions needs to be consistent with selfRef. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] quantScales: aclTensor on the NPU device side, the data type supports FLOAT32, BFLOAT16.
 * Supports non-continuous tensors. The data format supports ND.
 * @param [in] quantZeroPoints: aclTensor on the NPU device side, the data type supports INT32, BFLOAT16.
 * Supports non-continuous tensors. The data format supports ND.
 * @param [in] axis: The dimension used for scatter, the data type is INT64.
 * @param [in] quantAxis: Dimension used for quantification, data type is INT64.
 * @param [in] reduction: Specifies the reduction to be applied to the output, the data type is INT64. Currently only 1('update') is supported.
 * @param [in] roundMode: Specifies the data conversion mode, supporting {"rint", "round", "hybrid"} modes. When the Dtype of selfRef is 2/35/36 and the corresponding data type is INT8/FLOAT8_E4M3FN/FLOAT8_E5M2, only {"rint"} is supported. Dtype is 34, the corresponding data type is HIFLOAT8, and {"round", "hybrid"} is supported.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnInplaceQuantScatterV2GetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* indices, const aclTensor* updates,
    const aclTensor* quantScales, const aclTensor* quantZeroPoints,
    int64_t axis, int64_t quantAxis, int64_t reduction, const char* roundMode,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnInplaceQuantScatterV2 is used to perform calculations
 * @domain aclnn_ops_infer
 * Operator function:
 * First quantize the updates, and then update the values in selfRef one by one according to the specified axis axis and index indices. This operator is a custom operator semantics,
 * There is no corresponding tensorflow or pytorch interface.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnInplaceQuantScatterV2GetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceQuantScatterV2(void* workspace,
                                                 uint64_t workspaceSize,
                                                 aclOpExecutor* executor,
                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_QUANT_SCATTER_V2_H_