/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_GELUV2_H_
#define OP_API_INC_LEVEL2_ACLNN_GELUV2_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief The first interface of aclnnGeluV2 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: Gaussian error linear unit activation function
 * Calculation formula:
 * $$ y_{i}=Gelu(x_{i})=x_{i}×Φ(x_{i}) $$
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *     A[(x)]  --> B{l0op::Contiguous}
 *     B -->C([l0op::GeluV2])
 *     C --> D{l0op::ViewCopy}
 *     D --> E[(y)]
 * F[approximate(optional)]-->G[getApproximateStr]-->C
 * ```
 *
 * @param [in] x: input parameters to be calculated by gelu_v2. aclTensor on NPU device side.
 * The data type supports FLOAT16, FLOAT32, and BFLOAT16, and the data type must be the same as y. The data format supports ND. The shape must be the same as y. Non-continuous Tensors are supported.
 * @param [in] approximate: input parameter for gelu_v2 calculation, specifying the Gaussian approximation algorithm, the default value is 0 (meaning: 0: "none", 1: "tanh").
 * @param [in] y: The output parameter calculated by gelu_v2.
 * npu
 * The data type of aclTensor on the device side supports FLOAT16, FLOAT32, and BFLOAT16, and the data type must be the same as x, the data format supports ND, and the shape must be the same as x.
 * Supports non-contiguous Tensors.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGeluV2GetWorkspaceSize(const aclTensor* x,
                                                  int64_t approximate,
                                                  aclTensor* y,
                                                  uint64_t* workspaceSize,
                                                  aclOpExecutor** executor);
/**
 * @brief The second interface of aclnnGeluV2 is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGeluv2GetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGeluV2(void* workspace, uint64_t workspaceSize,
                                  aclOpExecutor* executor,
                                  const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_GELUV2_H_