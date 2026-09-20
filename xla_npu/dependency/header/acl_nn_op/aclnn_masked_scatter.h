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
#ifndef OP_API_INC_MASKED_SCATTER_H_
#define OP_API_INC_MASKED_SCATTER_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnMaskedScatter calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: According to the position where the element in the mask tensor is True, copy the element in the source to the position corresponding to selfRef.
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *   A[(selfRef)] -->B([l0op::Contiguous])
 *   B  -->D([l0op::MaskedScatter])
 *   D  -->J([l0op::ViewCopy])
 *   J   --> K[(selfRef)]
 *   A2[(mask)] -->B2([l0op::Contiguous])
 *   B2 --> C2([l0op::Cast])
 *   C2  -->D
 *   A1[(source)]-->B1([l0op::Contiguous])
 *   B1-->D
 * ```
 *
 * @param [in] selfRef: aclTensor on the NPU device side. The data type supports FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, and UINT8.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] mask: aclTensor on the NPU device side, the data type supports BOOL and UINT8. shape cannot be larger than selfRef, and needs to satisfy selfRef
 * broadcast relationship. The data format supports ND.
 * @param [in] source: aclTensor on the NPU device side, the data type needs to be the same as selfRef.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceMaskedScatterGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* mask, const aclTensor* source,
    uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief The second interface of aclnnMaskedScatter is used to perform calculations.
 *
 * Operator function: According to the position where the element in the mask tensor is True, copy the element in source to the position corresponding to self.
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *   A[(selfRef)] -->B([l0op::Contiguous])
 *   B  -->D([l0op::MaskedScatter])
 *   D  -->J([l0op::ViewCopy])
 *   J   --> K[(selfRef)]
 *   A2[(mask)] -->B2([l0op::Contiguous])
 *   B2 --> C2([l0op::Cast])
 *   C2  -->D
 *   A1[(source)]-->B1([l0op::Contiguous])
 *   B1-->D
 * ```
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMaskedScatterGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceMaskedScatter(void* workspace,
                                                uint64_t workspaceSize,
                                                aclOpExecutor* executor,
                                                aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MASKED_SCATTER_H_
