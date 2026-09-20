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
#ifndef ACLNN_GROUPED_DYNAMIC_MX_QUANT_H_
#define ACLNN_GROUPED_DYNAMIC_MX_QUANT_H_

#include "acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGroupedDynamicMxQuant calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * 
 * @param [in] x: input parameter to be calculated by GroupedDynamicMxQuant. aclTensor on the NPU device side,
 * The data type supports float16, bfloat16, the data format supports ND, and non-continuous Tensors are supported.
 * @param [in] groupIndex: aclTensor on the NPU device side, the data type supports int32
 * @param [in] roundMode: aclScalar on the host side, data type string, only supports "rint"
 * @param [in] dstType: aclScalar on the host side, data type int, input range is {35, 36}, corresponding data types of output y are {35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}
 * @param [in] blocksize: aclScalar on the host side, data type int, only supports "32"
 * @param [in] y: Output parameter calculated by GroupedDynamicMxQuant. aclTensor on the NPU device side,
 * The data type supports float8_e4m3fn, float8_e5m2, the data format supports ND, and non-continuous Tensor is supported.
 * @param [in] mxscale: Output parameter calculated by GroupedDynamicMxQuant. aclTensor on the NPU device side,
 * The data type supports float8_e8m0, the data format supports ND, and non-continuous Tensors are not supported.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnGroupedDynamicMxQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* groupIndex, const char* roundMode,
    int64_t dstType, int64_t blocksize, const aclTensor* y,
    const aclTensor* mxscale, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnGroupedDynamicMxQuant's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGroupQuantGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedDynamicMxQuant(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_GROUPED_DYNAMIC_MX_QUANT_H_