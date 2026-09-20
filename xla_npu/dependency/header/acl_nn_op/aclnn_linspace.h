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

#ifndef OP_API_INC_LINSPACE_H_
#define OP_API_INC_LINSPACE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnLinspace calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Function description: Create a 1-dimensional vector of size steps, whose values are linearly uniformly distributed from start to end (inclusive).
 *
 * Calculation formula: $$ out = (start, start + \frac{end - start}{steps - 1},...,
 *                    tart + (steps -2) * \frac{end - start}{steps - 1}, end) $$
 *
 * Parameter description:
 * @param [in]   start
 * Get the starting position of the value range: aclScalar on the host side. The data type supports integer and floating-point data types. The data format supports ND.
 * @param [in]   end
 * Get the end position of the value range: aclScalar on the host side. The data type supports integer and floating-point data types. The data format supports ND.
 * @param [in]   steps
 * Tensor size: aclScalar on the host side, the data type supports integer. The data format supports ND. It is required that steps is greater than or equal to 0.
 * @param [in] out specified output tensor: NPU
 * The data type of aclTensor on the device side supports integer and floating-point data types, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLinspaceGetWorkspaceSize(
    const aclScalar* start, const aclScalar* end, int64_t steps, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief The second interface of aclnnLinspace is used to perform calculations.
 *
 *
 * Function description: Create a 1-dimensional vector of size steps, whose values are linearly uniformly distributed from start to end (inclusive).
 *
 * Calculation formula: $$ out = (start, start + \frac{end - start}{steps - 1},...,
 *                    start + (steps -2) * \frac{end - start}{steps - 1}, end) $$
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnLinspaceGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLinspace(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif