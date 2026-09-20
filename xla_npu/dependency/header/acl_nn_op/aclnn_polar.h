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

#ifndef OP_API_INC_POLAR_H_
#define OP_API_INC_POLAR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnPolar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * 
 * Operator function: Construct a tensor whose elements are Cartesian coordinates corresponding to polar coordinates with absolute value abs and angle angle.
 *
 * @param [in] input: aclTensor on the NPU device side, the data type supports FLOAT,
 * Supports non-continuous Tensor, and the data format supports ND
 * @param [in] angle: aclTensor on the NPU device side, the data type must be consistent with input.
 * Supports non-continuous Tensor, and the data format supports ND
 * @param [out] out: The data type supports complex64.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 */
ACLNN_API aclnnStatus aclnnPolarGetWorkspaceSize(const aclTensor* input,
                                                 const aclTensor* angle,
                                                 aclTensor* out,
                                                 uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnPolar is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnPolarGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */

ACLNN_API aclnnStatus aclnnPolar(void* workspace, uint64_t workspaceSize,
                                 aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_POLAR_H_