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
#ifndef OP_API_INC_LEVEL2_ACLNN_FLIP_H_
#define OP_API_INC_LEVEL2_ACLNN_FLIP_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnFlip calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Invert (reverse order) the specified dimensions of an n-dimensional tensor, and the calculation formula for each axis specified in dims:
 *
 * $$
 * \operatorname{out}(i_0, i_1,
 * \ldots, i_{n-1}) = \operatorname{input}(j_0, j_1, \ldots, j_{n-1})
 * $$
 *
 * Among them, $n$ is the dimension of the input tensor, $j_k$ = $\operatorname{dimSize}(k)$ -1 - $i_k$, $\operatorname{dimSize}(k)$
 * Indicates the length of the $k$th axis.
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 * 	A[(Self)] -->B([l0op::Contiguous])
 *     B -->D([l0op::Flip])
 *     G((dims)) --> D([l0op::ReverseV2])
 *     D -->R([l0op::ViewCopy])
 *     R --> J[(out)]
 * ```
 * @param [in] self: input parameter to be calculated for flip. aclTensor on the NPU device side,
 * The data type supports float16, float, int32, int16, int64, bool, int8, uint8, float64, complex64, complex128, and the data format supports ND.
 * Supports non-contiguous Tensors.
 * @param [in] dims: Indicates the axis that needs to be flipped.
 * @param [in] out: The output parameter of flip calculation. aclTensor on the NPU device side,
 * The data type supports float16, float, int32, int16, int64, bool, int8, uint8, float64, complex64, complex128, and the data format supports ND.
 * Supports non-contiguous Tensors.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFlipGetWorkspaceSize(const aclTensor* self,
                                                const aclIntArray* dims,
                                                aclTensor* out,
                                                uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnFlip is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnFlipGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFlip(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_FLIP_H_