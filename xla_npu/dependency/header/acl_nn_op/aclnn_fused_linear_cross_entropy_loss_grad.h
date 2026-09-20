/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef OP_API_INC_FUSED_LINEAR_CROSS_ENTROPY_LOSS_GRAD_H_
#define OP_API_INC_FUSED_LINEAR_CROSS_ENTROPY_LOSS_GRAD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnFusedLinearCrossEntropyLossGrad calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * @param [in] grad: aclTensor on the NPU device side, the data type supports FLOAT, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] input: aclTensor on the NPU device side, the data type supports FLOAT16, BFLOAT16, non-continuous Tensor, and the data format supports ND.
 * @param [in] weight: aclTensor on the NPU device side, the data type supports FLOAT16, BFLOAT16, non-continuous Tensor, and the data format supports ND.
 * @param [in] targetMask: aclTensor on the NPU device side. The data type supports BOOL and UINT8, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] maskedTarget: aclTensor on the NPU device side, the data type supports INT64, INT32, non-continuous Tensor, and the data format supports ND.
 * @param [in] labelSmoothing: float type on the host side, label smoothing coefficient.
 * @param [in] logitsMaxOptional: aclTensor on the NPU device side, the data type supports FLOAT, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] sumExpLogitsOptional: aclTensor on the NPU device side, the data type supports FLOAT, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] softmaxOptional: aclTensor on the NPU device side, the data type supports FLOAT, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] inputGradOut: aclTensor on the NPU device side, the data type supports FLOAT16, BFLOAT16, non-continuous Tensor, and the data format supports ND.
 * @param [in] weightGradOut: aclTensor on the NPU device side, the data type supports FLOAT16, BFLOAT16, non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize(
    const aclTensor *grad, const aclTensor *input, const aclTensor *weight,
    const aclTensor *targetMask, const aclTensor *maskedTarget,
    float labelSmoothing, const aclTensor *logitsMaxOptional,
    const aclTensor *sumExpLogitsOptional, const aclTensor *softmaxOptional,
    aclTensor *inputGradOut, aclTensor *weightGradOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnFusedLinearCrossEntropyLossGrad's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aaclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFusedLinearCrossEntropyLossGrad(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_FUSED_LINEAR_CROSS_ENTROPY_LOSS_GRAD_H_
