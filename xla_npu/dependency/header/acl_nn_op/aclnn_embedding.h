/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_EMBEDDING_H_
#define OP_API_INC_EMBEDDING_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnEmbedding calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: map the data set to vector space, and then quantify the data
 *
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(weight)] --> B([l0op::Contiguous])
 *     B --> C([l0op::GatherV2])
 *     D[(indices)] --> E([l0op::Contiguous]) --> C
 *     F((dim=0)) --> G[(dimTensor)] --> C
 *     C --> H([l0op::ViewCopy]) --> Out[(out)]
 * ```
 *
 * @param [in] weight: npu
 * aclTensor on the device side, the data type supports BFLOAT16,
 * FLOAT、FLOAT16、INT64、INT32、INT16、INT8、UINT8、BOOL、DOUBLE、COMPLEX64、
 * COMPLEX128, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] indices: npu
 * The aclTensor on the device side supports INT64 and INT32 data types, non-continuous Tensor, and ND data format.
 * @param [out] out: npu
 * The aclTensor on the device side has the same data type as weight, and the data format supports ND, which has one more dimension than indices.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEmbeddingGetWorkspaceSize(const aclTensor* weight,
                                                     const aclTensor* indices,
                                                     const aclTensor* out,
                                                     uint64_t* workspaceSize,
                                                     aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnEmbedding is used to perform calculations.
 *
 * * Operator function: map the data set to vector space, and then quantify the data
 *
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(weight)] --> B([l0op::Contiguous])
 *     B --> C([l0op::GatherV2])
 *     D[(indices)] --> E([l0op::Contiguous]) --> C
 *     F((dim=0)) --> G[(dimTensor)] --> C
 *     C --> H([l0op::ViewCopy]) --> Out[(out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnEmbeddingGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEmbedding(void* workspace, uint64_t workspaceSize,
                                     aclOpExecutor* executor,
                                     const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_EMBEDDING_H_