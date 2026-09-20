/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LogicalXor_H_
#define OP_API_INC_LogicalXor_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnLogicalXor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the logical XOR operation of the given input tensor elements. When self and other are non-bool types, 0 is considered False and non-0 is considered True.
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A1[(self)] -->B1([Contiguous])-->C1([Cast])-->D([LogicalXor])
 * A2[(other)]-->B2([Contiguous])-->C2([Cast])-->D([LogicalXor])
 * D([LogicalXor])-->E([Cast])-->F([ViewCopy])-->G[(out)]
 * ```
 *
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the shape needs to satisfy the broadcast relationship with other.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with other.
 * @param [in] other: npu
 * The data type of aclTensor on the device side supports integer and floating point types, and the shape needs to satisfy the broadcast relationship with self.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with self.
 * @param [in] out: npu
 * For aclTensor on the device side, the data type supports integer and floating point types, and the shape needs to be the shape after self and other broadcast. The data format supports ND.
 * And the data format needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLogicalXorGetWorkspaceSize(const aclTensor* self,
                                                      const aclTensor* other,
                                                      aclTensor* out,
                                                      uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnLogicalXor is used to perform calculations.
 *
 * Operator function: Complete the logical XOR operation of the given input tensor elements. When self and other are non-bool types, 0 is considered False and non-0 is considered True.
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A1[(self)] -->B1([Contiguous])-->C1([Cast])-->D([LogicalXor])
 * A2[(other)]-->B2([Contiguous])-->C2([Cast])-->D([LogicalXor])
 * D([LogicalXor])-->E([Cast])-->F([ViewCopy])-->G[(out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnLogicalXorGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLogicalXor(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LogicalXor_H_