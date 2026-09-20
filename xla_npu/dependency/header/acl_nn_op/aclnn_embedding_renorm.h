/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_EMBEDDING_RENORM_H_
#define OP_API_INC_EMBEDDING_RENORM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnEmbeddingRenorm calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Returns the correction result of tensor under specified indices according to the given max_norm and norm_type
 *
 * Calculation formula:
 * The formula for calculating the norm of a vector is as follows, where n is the norm value specified by norm_type:
 * $$ ||X||_{p}=\sqrt[p]{\sum_{i=1}^nx_{i}^p} $$
 * $$ where X = (x_{1},x_{2},..., x_{n}) $$
 * For scenarios where the calculated norm is greater than the given max_norm, normalization processing needs to be performed, and the 0-dimensional elements specified by indices are multiplied by the coefficient:
 * $$ scaler = \frac{max\_norm}{current\_norm + 1e^{-7}} $$
 *
 * Implementation instructions
 * API calculation base path:
 *```mermaid
 * graph LR
 * A[(selfRef)]  --> B([l0op::contiguous])
 * B --> C([l0op::GatherV2])
 * D((axis=0)) --> C
 * E[(Indices)] --> F([l0op::contiguous])
 * F --> R([l0op::Reshape])
 * R --> C
 * C --> G([l0op::Renorm])
 * H((maxNorm)) --> G
 * I((normType)) --> G
 * J((dim=0)) --> G
 * U[(indicesGatherV2)] --> V([l0op::Contiguous])
 * V --> K([l0op::GatherV2])
 * L((axis=0)) --> K
 * G --> K
 * C --> M([l0op::Mul])
 * K --> M
 * B --> N([l0op::ScatterUpdate])
 * M --> N
 * R --> N
 * N --> O([l0op::ViewCopy])
 * O --> P[(selfRef)]
 * ```
 *
 * @param [in] selfRef: npu
 * The data type of aclTensor on the device side supports BFLOAT16, FLOAT16, and FLOAT32 types.
 * Supports non-continuous Tensor, the data format supports ND, and the dimension must be 2.
 * @param [in] indices: npu
 * The data type of aclTensor on the device side supports INT64.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] maxNorm: The data type supports double and float.
 * @param [in] normType: The data type supports double and float.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEmbeddingRenormGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* indices, double maxNorm,
    double normType, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnEmbeddingRenorm is used to perform calculations
 *
 * Operator function: Returns the correction result of tensor under specified indices according to the given max_norm and norm_type
 *
 * Calculation formula:
 * The formula for calculating the norm of a vector is as follows, where n is the norm value specified by norm_type:
 * $$ ||X||_{p}=\sqrt[p]{\sum_{i=1}^nx_{i}^p} $$
 * $$ where X = (x_{1},x_{2},..., x_{n}) $$
 * For scenarios where the calculated norm is greater than the given max_norm, normalization processing needs to be performed, and the 0-dimensional elements specified by indices are multiplied by the coefficient:
 * $$ scaler = \frac{max\_norm}{current\_norm + 1e^{-7}} $$
 *
 * Implementation instructions
 * API calculation base path:
 *```mermaid
 * graph LR
 * A[(selfRef)]  --> B([l0op::contiguous])
 * B --> C([l0op::GatherV2])
 * D[(axis=0)] --> C
 * E[(Indices)] --> F([l0op::contiguous])
 * F --> C
 * C --> G([l0op::Renorm])
 * H((maxNorm)) --> G
 * I((normType)) --> G
 * J((dim=0)) --> G
 * F --> K([l0op::GatherV2])
 * L[(axis=0)] --> K
 * G --> K
 * C --> M([l0op::Mul])
 * K --> M
 * B --> N([l0op::ScatterUpdate])
 * M --> N
 * F --> N
 * N --> O([l0op::ViewCopy])
 * O --> P[(selfRef)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEmbeddingRenorm(void* workspace,
                                           uint64_t workspace_size,
                                           aclOpExecutor* executor,
                                           const aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_EMBEDDING_RENORM_H_
