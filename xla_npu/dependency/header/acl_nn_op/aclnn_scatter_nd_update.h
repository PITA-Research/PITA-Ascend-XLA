/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_SCATTER_ND_UPDATE_H_
#define OP_API_INC_SCATTER_ND_UPDATE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnScatterNdUpdate calculates the workspace size according to the specific calculation process.
 * @domain aclnnop_ops_infer
 * @domain aclnnop_ops_train
 * Operator function: Update the values ​​in tensor var according to the specified index indices one by one.
 * @param [in] varRef: aclTensor on the NPU device side, data type supports FLOAT16, FLOAT32, BOOL
 * INT64, BFLOAT16, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] indices: aclTensor on the NPU device side. The data type supports INT32 and INT64 types. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] updates: aclTensor on NPU device side, data type supports FLOAT16, FLOAT32, BOOL
 * INT64, BFLOAT16, support non-continuous Tensor, data format supports ND,
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnScatterNdUpdateGetWorkspaceSize(
    aclTensor* varRef, const aclTensor* indices, const aclTensor* updates,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnScatterNdUpdate is used to perform calculations
 * @domain aclnnop_ops_infer
 * @domain aclnnop_ops_train
 * Operator function: Update the values ​​in tensor var according to the specified index indices one by one.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnScatterNdUpdateGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnScatterNdUpdate(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SCATTER_ND_UPDATE_H_