/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ROTARY_POSITION_EMBEDDING_H_
#define OP_API_INC_ROTARY_POSITION_EMBEDDING_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnRotaryPositionEmbedding calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * @param [in] x: npu
 * The aclTensor on the device side supports FLOAT32, FLOAT16, BFLOAT16 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [in] sin: npu
 * aclTensor on the device side, the data type supports FLOAT32, FLOAT16, BFLOAT16, the data type is consistent with x, and the shape and x satisfy the broadcast relationship
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] cos: npu
 * aclTensor on the device side, the data type supports FLOAT32, FLOAT16, BFLOAT16, the data type is consistent with x, and the shape is consistent with sin
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] mode: supports int64_t type, value range [0, 3], representing four modes of half, quarter, interleave, half-interleave respectively.
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports FLOAT32, FLOAT16, BFLOAT16, and the data type is consistent with x.
 * The shape is the same as x, the data format supports ND, and the data format needs to be consistent with x.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRotaryPositionEmbeddingGetWorkspaceSize(
    const aclTensor* x, const aclTensor* cos, const aclTensor* sin,
    int64_t mode, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);
/* The second interface of @brief aclnnRotaryPositionEmbedding, used to perform calculations.
 * @param [in] workspace: the starting address of the workspace memory applied for on the NPU device side
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnBitwiseNotGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRotaryPositionEmbedding(void* workspace,
                                                   uint64_t workspaceSize,
                                                   aclOpExecutor* executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
