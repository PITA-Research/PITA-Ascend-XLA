/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/license/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_DIGAMMA_H_
#define OP_API_INC_LEVEL2_ACLNN_DIGAMMA_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACLNN_MAX_SHAPE_RANK 8

/**
 * The first interface of @brief aclnnDigamma calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Returns the result of the absolute value of each element in the input Tensor
 * Calculation formula:
 * $$ out_{i} = digamma(self_{i}) $$
 * @param [in] self: input parameter to be calculated for digamma. aclTensor on the NPU device side,
 * The data type supports FLOAT, FLOAT16, and DOUBLE, the data format supports ND, and non-continuous Tensors are supported.
 * @param [in] out: The output parameter of digamma calculation. aclTensor on the NPU device side,
 * The data type supports FLOAT, FLOAT16, and DOUBLE, the data format supports ND, and non-continuous Tensors are supported.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnDigammaGetWorkspaceSize(const aclTensor* self,
                                                   aclTensor* out,
                                                   uint64_t* workspaceSize,
                                                   aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnDigamma is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnDigammaGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnDigamma(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_DIGAMMA_H_