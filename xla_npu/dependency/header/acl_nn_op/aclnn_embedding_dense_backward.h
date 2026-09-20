/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_EMBEDDING_DENSE_BACKWARD_H_
#define OP_API_INC_EMBEDDING_DENSE_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnEmbeddingDenseBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: Complete the reverse calculation of Embedding
 *
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(grad)] -->B([l0op::Contiguous])
 *     B --> L([l0op::Cast])
 *     L --> C([l0op::EmbeddingDenseGrad])
 *     D[(indices)] -->E([l0op::Contiguous])
 *     E --> F([l0op::Cast])
 *     F --> C
 *     C --> M([l0op::Cast])
 *     M --> G([l0op::ViewCopy])
 *     G --> K[(out)]
 *     H((numWeights)) --> C
 *     I((paddingIdx)) --> C
 *     J((scaleGradByFreq)) --> C
 * ```
 *
 * @param [in] grad: npu
 * The aclTensor and gradient Tensor on the device side are consistent with the forward output shape, NPU
 * The data type of aclTensor on the device side supports float, float16, and bfloat16 types, and the data format supports ND, which has one more dimension than indices.
 * @param [in] indices: npu
 * The aclTensor on the device side needs to be mapped to the index tensor of the vector space in the forward direction. The data type supports int32, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] numWeights: The size of the vector space.
 * @param [in] paddingIdx:
 * Padding ID, defaults to None. If specified, all vector elements at the specified position will be set to 0, and the parameters corresponding to paddingIdx will not affect the gradient.
 * @param [in] scaleGradByFreq: Scale the gradient according to the frequency of word occurrence. The default is False.
 * @param [out] out: npu
 * Reverse output Tensor, the data type supports float32 type, and the data format only supports 2D.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEmbeddingDenseBackwardGetWorkspaceSize(
    const aclTensor* grad, const aclTensor* indices, uint64_t numWeights,
    uint64_t paddingIdx, bool scaleGradByFreq, const aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnEmbeddingDenseBackward is used to perform calculations.
 *
 * Operator function: Complete the reverse calculation of Embedding
 *
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(grad)] -->B([l0op::Contiguous])
 *     B --> L([l0op::Cast])
 *     L --> C([l0op::EmbeddingDenseGrad])
 *     D[(indices)] -->E([l0op::Contiguous])
 *     E --> F([l0op::Cast])
 *     F --> C
 *     C --> M([l0op::Cast])
 *     M --> G([l0op::ViewCopy])
 *     G --> K[(out)]
 *     H((numWeights)) --> C
 *     I((paddingIdx)) --> C
 *     J((scaleGradByFreq)) --> C
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnEmbeddingDenseBackwardGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEmbeddingDenseBackward(void* workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor* executor,
                                                  const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_EMBEDDING_DENSE_BACKWARD_H_
