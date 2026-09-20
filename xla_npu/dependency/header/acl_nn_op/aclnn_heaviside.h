/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_HEAVISIDE_H_
#define OP_API_INC_HEAVISIDE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnHeavisideGetWorkspaceSize calculates the workspace size according to the specific calculation process.
 * @domain aclnnop_ops_infer
 * Function description: Calculate the Heaviside step function of each element in the input input.
 * Calculation formula: as shown below
 * $$
 * \text{heaviside}(\text{input}, \text{values}) =
 * \begin{cases}
 * 0, & \text{if input} < 0 \\
 * \text{values}, & \text{if input} = 0 \\
 * 1, & \text{if input} > 0
 * \end{cases}
 * $$
 *
 * @param [in] input:  npu
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, and BFLOAT16. The shape needs to satisfy the broadcast relationship with values.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with values.
 * @param [in] values: npu
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, and BFLOAT16. The shape needs to satisfy the broadcast relationship with the input.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with the input.
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16, and the shape needs to be input and values.
 * The data format of the shape after broadcast supports ND, and the data format needs to be consistent with input. Supports non-continuous Tensors.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnHeavisideGetWorkspaceSize(const aclTensor* input,
                                                     const aclTensor* values,
                                                     aclTensor* out,
                                                     uint64_t* workspaceSize,
                                                     aclOpExecutor** executor);

/**
 * @brief aclnnHeaviside's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnForeachAbsGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnHeaviside(void* workspace, uint64_t workspaceSize,
                                     aclOpExecutor* executor,
                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_HEAVISIDE_H_