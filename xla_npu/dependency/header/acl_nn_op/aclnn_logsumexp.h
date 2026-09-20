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
#ifndef OP_API_INC_LOGSUMEXP_H_
#define OP_API_INC_LOGSUMEXP_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnLogSumExp calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: npu
 * aclTensor on the NPU device side, the data type supports FLOAT16, BFLOAT16, FLOAT32
 * Supports non-continuous Tensor, and the data format supports ND, NCHW, NHWC, HWCN, NDHWC, and NCDHW.
 * @param [in] dim: The dimension to be used for logsumexp, data type INT64
 * @param [in] keepDim: whether to reduce the dimension
 * @param [in] out: npu
 * aclTensor on the NPU device side, the data type supports FLOAT16, BFLOAT16, FLOAT32, and the data format supports ND, NCHW, NHWC, HWCN, NDHWC, NCDHW.
 * And the data format needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */

ACLNN_API aclnnStatus aclnnLogSumExpGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* dim, bool keepDim, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnLogSumExp is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnLogSoftmaxGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLogSumExp(void* workspace, uint64_t workspaceSize,
                                     aclOpExecutor* executor,
                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LOGSUMEXP_H_
