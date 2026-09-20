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

#ifndef OP_API_INC_BINCOUNT_H_
#define OP_API_INC_BINCOUNT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnBincount calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Calculate the frequency of each number in an array of non-negative integers.
 * Calculation formula:
 * If n is the value of self at position i, if weights is specified, then
 * out[n] = out[n] + weights[i]
 * Otherwise
 * out[n] = out[n] + 1
 *
 * @param [in] self: aclTensor on NPU device side, data type supports INT8, INT16, INT32, INT64, UINT8,
 * It must be a non-negative integer, and the data format supports 1-dimensional ND. Supports non-contiguous Tensors.
 * @param [in] weights: aclTensor on the NPU device side, the weight of each value of self, which can be a null pointer.
 * Data types support FLOAT, FLOAT16, FLOAT64, INT8, INT16, INT32, INT64, UINT8, BOOL,
 * The data format supports 1-dimensional ND, and the shape must be consistent with self. Supports non-contiguous Tensors.
 * @param [in] minlength: int type on the host side, specifying the minimum length of the output tensor. If the calculated maximum value of size is less than minlength,
 * The output length is minlength, otherwise it is size.
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports INT32, INT64, FLOAT, and DOUBLE. The data format supports 1D ND. Supports non-contiguous Tensors.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBincountGetWorkspaceSize(
    const aclTensor* self, const aclTensor* weights, int64_t minlength,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnBincount calculates the workspace size according to the specific calculation process.
 *
 * Operator function: Calculate the frequency of each number in an array of non-negative integers.
 * Calculation formula:
 * If n is the value of self at position i, if weights is specified, then
 * out[n] = out[n] + weights[i]
 * Otherwise
 * out[n] = out[n] + 1

 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnBincountGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBincount(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_BINCOUNT_H_