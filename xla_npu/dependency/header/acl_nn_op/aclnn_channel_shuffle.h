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

#ifndef OP_API_INC_LEVEL2_ACLNN_CHANNEL_SHUFFLE_H_
#define OP_API_INC_LEVEL2_ACLNN_CHANNEL_SHUFFLE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnChannelShuffle calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Divide the channels of the $(*, C, H, W)$ tensor into $g$ groups, and then rearrange them into $(*, C_{,}^{\underline{g}}g, H, W)$,
 * At the same time, keep the shape of the final output tensor consistent with the input tensor.
 *
 * @param [in] self: aclTensor on the NPU device side, data type supports BFLOAT16, FLOAT16, FLOAT, DOUBLE, UINT8, INT8, INT16, INT32,
 * LONG, COMPLEX64, COMPLEX128, BOOL, supports non-continuous Tensor, the data format supports ND, the data dimension is greater than 2 and does not support more than 7 dimensions.
 * @param [in] groups: int64_t on the host side, indicating how many groups the channels input to self are divided into. The value needs to be greater than 0 and must be divisible by the channels of self.
 * @param [in] out: aclTensor on the NPU device side, the data type and data dimension need to be consistent with self, support non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnChannelShuffleGetWorkspaceSize(
    const aclTensor* self, int64_t groups, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnChannelShuffle is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnChannelShuffleGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnChannelShuffle(void* workspace,
                                          uint64_t workspaceSize,
                                          aclOpExecutor* executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_CHANNEL_SHUFFLE_H_