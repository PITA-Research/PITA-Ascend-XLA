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
#ifndef OP_API_INC_MAX_DIM_H_
#define OP_API_INC_MAX_DIM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnMaxDim calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Returns the index of the maximum value of the tensor in the specified dimension.
 *
 * Implementation description: Basic path for API calculation:
 * ```mermaid
 *  graph LR
 *  A[(self)] -.->B([l0op::Contiguous])
 *  B --> C([l0op::ArgMaxWithValue])
 *  C --> F([l0op::Cast])
 *  D([dim]) --> C
 *  F -.-> E([l0op::ViewCopy])
 *  E --> O[(Out)]
 * ```
 *
 * @param [in] self: aclTensor on the NPU device side. The data type supports FLOAT, FLOAT16, and BFLOAT16. The data format supports ND. Supports non-contiguous Tensors.
 * @param [in] dim: Host side int64 type, specifies the dimension to be calculated for the maximum value.
 * @param [in] keepdim: Boolean type on the host side, whether to retain the dimensions of the input tensor in the output tensor.
 * @param [in] indices: aclTensor on the NPU device side. The data type supports INT32 and INT64. The data format supports ND. Supports non-contiguous Tensors.
 * @param [in] out: aclTensor on the NPU device side, the data type supports FLOAT and FLOAT16. The data format supports ND. Supports non-contiguous Tensors.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMaxDimGetWorkspaceSize(
    const aclTensor* self, int64_t dim, bool keepdim, aclTensor* out,
    aclTensor* indices, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnArgMax calculates the workspace size according to the specific calculation process.
 *
 * Operator function: Returns the index of the maximum value of the tensor in the specified dimension.
 *
 * Implementation description: Basic path for API calculation:
 * ```mermaid
 *  graph LR
 *  A[(self)] -.->B([l0op::Contiguous])
 *  B --> C([l0op::ArgMaxWithValue])
 *  C --> F([l0op::Cast])
 *  D([dim]) --> C
 *  F -.-> E([l0op::ViewCopy])
 *  E --> O[(Out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnArgMaxGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMaxDim(void* workspace, uint64_t workspaceSize,
                                  aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MAX_DIM_H_