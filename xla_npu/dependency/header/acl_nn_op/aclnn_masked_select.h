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
#ifndef OP_API_INC_MASKED_SELECT_H_
#define OP_API_INC_MASKED_SELECT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnMaskedSelect calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Select elements in the input tensor (self) as output based on the value in a Boolean mask tensor (mask) to form a new one-dimensional tensor.
 *
 * @param [in] self: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL
 * Shape needs to satisfy [broadcast relationship]() with mask. Supports [non-continuous Tensor](), and the data format supports ND.
 * @param [in] mask: npu
 * The aclTensor on the device side only supports bool or uint8. If it is uint8, its value must be 0 or 1, and the shape needs to satisfy the [broadcast relationship]() with self.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMaskedSelectGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mask, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief The second interface of aclnnMaskedSelect is used to perform calculations.
 *
 * Operator function: For the input Tensor self, perform a bitwise mask selection operation according to the mask, select the element corresponding to the non-zero position in the mask mask, and form a new
 * Tensor, and return. Implementation instructions: Basic path for API calculation:
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMaskedSelect(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MASKED_SELECT_H_
