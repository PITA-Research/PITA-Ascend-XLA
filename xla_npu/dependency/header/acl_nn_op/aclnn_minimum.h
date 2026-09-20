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
#ifndef OP_API_INC_MINIMUM_H_
#define OP_API_INC_MINIMUM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnMinimum calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Calculate the minimum value of each element in two tensors and return a new tensor.
 *
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL,
 * And the data type needs to form a mutual derivation relationship with other, and the shape needs to satisfy the broadcast relationship with other. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] other: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL,
 * And the data type needs to form a mutual derivation relationship with other, and the shape needs to satisfy the broadcast relationship with other. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL,
 * And the data type needs to be a convertible data type after derivation of self and other, and the shape needs to be self and other.
 * The shape after broadcast. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMinimumGetWorkspaceSize(const aclTensor* self,
                                                   const aclTensor* other,
                                                   aclTensor* out,
                                                   uint64_t* workspaceSize,
                                                   aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnMinimum is used to perform calculations.
 *
 * Operator function: Calculate the minimum value of each element in two tensors and return a new tensor.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMinimumGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMinimum(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MINIMUM_H_
