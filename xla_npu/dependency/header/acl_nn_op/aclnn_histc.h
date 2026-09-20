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
#ifndef OP_API_INC_HISTC_H_
#define OP_API_INC_HISTC_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnHistc calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] self: aclTensor on the NPU device side. The data type supports FLOAT, FLOAT16, and DOUBLE. It supports non-continuous Tensor. The data format supports ND.
 * @param [in] bins: number of histogram bins, data type INT32
 * @param [in] min: Histogram statistical lower limit (inclusive), aclScalar on the host side, the data type needs to be convertible to a FLOAT data type.
 * @param [in] max: The upper limit of histogram statistics (inclusive), aclScalar on the host side, the data type needs to be convertible to a FLOAT data type.
 * @param [in] out: aclTensor on the NPU device side, the data type supports FLOAT and INT32. And the data type is a data type that can be converted by self. Data format supports ND
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnHistcGetWorkspaceSize(
    const aclTensor* self, int64_t bins, const aclScalar* min,
    const aclScalar* max, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnHistc is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnHistcGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnHistc(void* workspace, uint64_t workspaceSize,
                                 aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_HISTC_H_
