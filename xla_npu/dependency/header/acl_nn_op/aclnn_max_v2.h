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
#ifndef OP_API_INC_MAX_V2_H_
#define OP_API_INC_MAX_V2_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnMaxV2 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Find the maximum element value of the input tensor according to the specified dimension.
 *
 * @param [in] self: aclTensor on the device side, the data type supports integer and floating point types. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] dims: aclIntArray on the host side, specifies the dimension to be calculated for the maximum value. The data type supports INT32 and INT64.
 * @param [in] keepDims: Boolean type on the host side, whether to retain the dimensions of the input tensor in the output tensor.
 * @param [in] noopWithEmptyDims: Boolean type on the host side, defining the behavior when dims is empty.
 * @param [in] out: aclTensor on the device side, the data type supports integer and floating point types. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMaxV2GetWorkspaceSize(
    const aclTensor* self, const aclIntArray* dims, bool keepDims,
    bool noopWithEmptyDims, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnMaxV2 is used to perform calculations.
 *
 * Operator function: Find the maximum element value of the input tensor according to the specified dimension.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMaxV2GetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMaxV2(void* workspace, uint64_t workspaceSize,
                                 aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MAX_V2_H_
