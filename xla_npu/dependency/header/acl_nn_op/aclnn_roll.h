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
#ifndef OP_API_ACLNN_ROLL_H
#define OP_API_ACLNN_ROLL_H

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnRoll calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the roll operation on the input Tensor
 * @param [in] x: aclTensor on the device side, the data type supports BFLOAT16, FLOAT16, FLOAT32, INT8, UINT8, INT32, UINT32, BOOL. support
 * [Discontinuous Tensor](#), the data format supports ND ([Reference](#)).
 * @param [in] shifts: array of int64, the length of the array is consistent with dims.
 * @param [in] dims: array of int64, the array length is consistent with shifts, the value range is [-x.dim(), x.dim() -
 * 1], for example: if the dimension of x is 4, the value range is [-4, 3].
 * @param [in] out: aclTensor on the device side, the data type and data format are consistent with the input x
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnRollGetWorkspaceSize(
    const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnRoll, used to perform calculations
 *
 * Operator function: Complete the roll operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnRollGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRoll(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_ACLNN_ROLL_H