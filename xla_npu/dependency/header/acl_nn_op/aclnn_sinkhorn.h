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

#ifndef OP_API_INC_SINKHORN_H_
#define OP_API_INC_SINKHORN_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSinkhorn's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: calculate Sinkhorn distance
 *
 * @param [in] cost: aclTensor on the NPU device side, the data type supports FLOAT, FLOAT16, BFLOAT16, and the data format supports ND.
 * @param [in] tol: error, supports FLOAT type. If a null pointer is passed in, tol will be 0.0001.
 * @param [in] p: aclTensor on the NPU device side, the data type supports FLOAT, FLOAT16, BFLOAT16, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSinkhornGetWorkspaceSize(const aclTensor* cost,
                                                    const aclScalar* tol,
                                                    aclTensor* p,
                                                    uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);
/**
 * @brief aclnnSinkhorn's second interface is used to perform calculations.
 *
 * Operator function: Calculate the Sinkhorn distance for the input Tensor cost, form a new Tensor, and return it.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSinkhornGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSinkhorn(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SINKHORN_H_
