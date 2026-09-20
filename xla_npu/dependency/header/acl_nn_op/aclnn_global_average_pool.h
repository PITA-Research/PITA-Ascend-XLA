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
#ifndef OP_API_INC_GLOBAL_AVERAGE_POOL_H_
#define OP_API_INC_GLOBAL_AVERAGE_POOL_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGlobalAveragePool calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: GlobalAveragePool consumes an input tensor X and applies average pooling across the values in the same
 * channel.
 *
 * @param [in] self: npu
 * The data type of aclTensor on the NPU device side supports FLOAT16, FLOAT32, and DOUBLE.
 * Supports continuous and non-continuous Tensor, and the data format supports ND.
 * @param [in] out:
 * The data type of aclTensor on the NPU device side supports FLOAT16, FLOAT32, and DOUBLE. Supports continuous and non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGlobalAveragePoolGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnGlobalAveragePool is used to perform calculations.
 *
 * Operator function: GlobalAveragePool consumes an input tensor X and applies average pooling across the values in the same
 * channel.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnGlobalAveragePoolGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGlobalAveragePool(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GLOBAL_AVERAGE_POOL_H_