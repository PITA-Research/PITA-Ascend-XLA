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
#ifndef OP_API_INC_ADD_H_
#define OP_API_INC_ADD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnInplaceFillDiagonal calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: fill tensor diagonal with fillValue
 *
 * @param [in] selfRef: aclTensor on the NPU device side, input selfRef, data type supports FLOAT, FLOAT16, INT32, INT64.
 * Supports [non-continuous Tensor](#non-continuous Tensor description), and the data format supports ND ([reference](#reference)).
 * @param [in] fillValue:
 * For aclScalar on the host side, enter fillValue. The data type supports FLOAT, FLOAT16, DOUBLE, UINT8, INT8, INT16, INT32, INT64, and BOOL.
 * @param [in] wrap: Input wrap, the data type supports BOOL, whether to form a new diagonal after each row of `min(col, row)`, only valid for two-dimensional input Tensor.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceFillDiagonalGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* fillValue, bool wrap,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceFillDiagonal is used to perform calculations.
 *
 * Operator function: fill tensor diagonal with fillValue
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnInplaceFillDiagonalGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceFillDiagonal(void* workspace,
                                               uint64_t workspaceSize,
                                               aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ADD_H_
