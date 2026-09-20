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
#ifndef OP_API_INC_TENSOREQUAL_H_
#define OP_API_INC_TENSOREQUAL_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnEqual calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Calculate whether two Tensors have the same size and elements, returning a Bool type:
 *
 * $$ out = (self == other)  ?  True : False $$
 *
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([l0op::Contiguous])
 *     B -->D([l0op::TensorEqual])
 *     E[(other)] -->F([l0op::Contiguous])
 *     F -->D --> F1([l0op::ViewCopy])
 *     F1 --> J[(out)]
 * ```
 *
 * @param [in] self: aclTensor on NPU device side,
 * The data type supports FLOAT16, FLOAT, INT32, INT8, UINT8, BOOL, DOUBLE, INT64, INT16, UINT16, UINT32, UINT64 data types,
 * The data types of self and other are consistent, support non-continuous Tensor, and the data format supports ND.
 * @param [in] other: aclTensor on NPU device side,
 * The data type supports FLOAT16, FLOAT, INT32, INT8, UINT8, BOOL, DOUBLE, INT64, INT16, UINT16, UINT32, UINT64 data types,
 * The data types of self and other are consistent, support non-continuous Tensor, and the data format supports ND.
 * @param [in] out: Output a Tensor whose data type is BOOL.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEqualGetWorkspaceSize(const aclTensor* self,
                                                 const aclTensor* other,
                                                 aclTensor* out,
                                                 uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnEqual is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnEqualGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEqual(void* workspace, uint64_t workspaceSize,
                                 aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_TENSOREQUAL_H_
