/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*!
 * \file aclnn_embedding_bag.h
 * \brief
 */
#ifndef OP_API_INC_FUSED_CROSS_ENTROPY_LOSS_WITH_MAX_SUM_H_
#define OP_API_INC_FUSED_CROSS_ENTROPY_LOSS_WITH_MAX_SUM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnFusedCrossEntropyLossWithMaxSum The first section of the interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: This operator is part of the cross-entropy calculation module in vocabulary parallel scenarios. It solves the problem of video memory and computing efficiency in extremely large-scale vocabulary. The current part is the result of calculating loss and softMax.
 * Calculation formula: lossOut = log(sum_exp_logits) - predicted_logits
 *           softMaxOutOptional = exp(vocab_parallel_logits -logits_max.unsqueeze(dim = -1)) \ sum_exp_logits.unsqueeze(dim = -1)
 * @param [in] logitsMax: aclTensor on the NPU device side. The data type supports float32 and must be 1D.
 * @param [in] sumExpLogits: aclTensor on the NPU device side, the data type supports float32. It must be 1Dtensor, and the shape is consistent with logitsMax.
 * @param [in] predictedLogits: aclTensor on the NPU device side, the data type supports float32, must be 1Dtensor, and the shape is consistent with logitsMax.
 * @param [in] labelSmoothing: float type, smoothing coefficient, currently supports 0.
 * @param [in] inputOptional: aclTensor on the NPU device side, the data type supports float16, bfloat6, and must be 1Dtensor.
 * @param [in] weightOptional: aclTensor on the NPU device side, the data type supports float16, bfloat6, and must be 1Dtensor.
 * @param [in] vocabParallelLogitsOptional: aclTensor on the NPU device side, the data type supports float32, float16, bfloat6, and must be 2Dtensor.
 * @param [out] lossOut: aclTensor on the NPU device side, the data type supports float32, must be 1Dtensor, and the shape is consistent with logitsMax.
 * @param [out] softMaxOutOptional: aclTensor on the NPU device side, the data type supports float32, must be 2Dtensor, and the shape is consistent with vocabParallelLogitsOptional.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU side.
 * @param [out] executor: Returns the op executor.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFusedCrossEntropyLossWithMaxSumGetWorkspaceSize(
    const aclTensor* logitsMax, const aclTensor* sumExpLogits,
    const aclTensor* predictedLogits, float labelSmoothing,
    const aclTensor* inputOptional, const aclTensor* weightOptional,
    const aclTensor* vocabParallelLogitsOptional, aclTensor* lossOut,
    aclTensor* softMaxOutOptional, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnFusedCrossEntropyLossWithMaxSum is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceIndexCopyGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFusedCrossEntropyLossWithMaxSum(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_INDEX_COPY_H_
