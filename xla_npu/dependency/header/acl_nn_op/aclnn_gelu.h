/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_GELU_H_
#define OP_API_INC_LEVEL2_ACLNN_GELU_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGelu calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: Gaussian error linear unit activation function
 * Calculation formula:
 * $$ out_{i}=Gelu(self_{i})=self_{i}×Φ(self_{i}) $$
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *     A[(Self)]  --> B{l0op::Contiguous}
 *     B -->C([l0op::Gelu])
 *     C --> D{l0op::ViewCopy}
 *     D --> E[(out)]
 * ```
 *
 * @param [in] self: input parameter to be calculated for gelu. aclTensor on NPU device side.
 * The data type supports FLOAT16, FLOAT32, and BFLOAT16, and the data type must be the same as out. The data format supports ND. The shape must be the same as out. Non-continuous Tensors are supported.
 * @param [in] out: The output parameter calculated by gelu.
 * npu
 * The data type of aclTensor on the device side supports FLOAT16, FLOAT32, and BFLOAT16, and the data type must be the same as self, the data format supports ND, and the shape must be the same as self.
 * Supports non-contiguous Tensors.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGeluGetWorkspaceSize(const aclTensor* self,
                                                aclTensor* out,
                                                uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnGelu is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGeluGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGelu(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor,
                                const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_GELU_H_