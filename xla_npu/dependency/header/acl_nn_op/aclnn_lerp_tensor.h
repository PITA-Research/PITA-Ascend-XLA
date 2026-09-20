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

/*!
 * \file aclnn_lerp_tensor.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_LERP_TENSOR_H_
#define OP_API_INC_LEVEL2_ACLNN_LERP_TENSOR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnLerp calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Perform linear interpolation between the starting and ending Tensor based on the given weight, and return the interpolated Tensor.

 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 * A1[(self)] --> B1([l0op::Contiguous])
 * A2[(end)] --> B2([l0op::Contiguous])
 * A3[(weight)] --> B3([l0op::Contiguous])
 * C([l0op::Lerp])
 * B1 --> C
 * B2 --> C
 * B3 --> C
 * C --> D([l0op::Cast])
 * D --> E([l0op::ViewCopy]) --> F[(out)]
 * ```
 *
 * @param [in] self: The input start in the formula, the data type supports FLOAT16, FLOAT, and the shape needs to satisfy the broadcast relationship with end and weight.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] end: The input end in the formula, the data type supports FLOAT16, FLOAT and is consistent with the data type of self. The shape needs to satisfy the broadcast relationship with self and weight.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] weight: The input weight in the formula, the data type supports FLOAT16, FLOAT and is consistent with the data type of `self`. The shape needs to satisfy the broadcast relationship with `self` and `end`.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] out: out in the formula, the data type supports FLOAT16, FLOAT and is consistent with the data type of self, shape is the same as self, end and weight
 * The shape after broadcast is consistent. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLerpGetWorkspaceSize(
    const aclTensor* self, const aclTensor* end, const aclTensor* weight,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnLerp is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnLerpGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLerp(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor,
                                const aclrtStream stream);

/**
 * @brief The first interface of aclnnInplaceLerp calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Perform linear interpolation between the starting and ending Tensor based on the given weight, and return the interpolated Tensor.

 *
 * @param [in] selfRef: The input start in the formula, the data type supports FLOAT16, FLOAT, the shape needs to satisfy the broadcast relationship with end and weight, and the shape after broadcast is consistent with selfRef.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] end: The input end in the formula, the data type supports FLOAT16, FLOAT and is consistent with the data type of self. The shape needs to satisfy the broadcast relationship with selfRef and weight.
 * And the shape after broadcast is consistent with selfRef. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] weight: The input weight in the formula, the data type supports FLOAT16, FLOAT, and the shape needs to satisfy the broadcast relationship with selfRef and end. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceLerpGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* end, const aclTensor* weight,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceLerp is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceLerpGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceLerp(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_LERP_TENSOR_H_