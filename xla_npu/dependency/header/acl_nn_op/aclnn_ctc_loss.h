/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_CTCLOSS_H_
#define OP_API_INC_CTCLOSS_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnCtcLoss calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * Calculate the connection timing classification loss value.
 *
 * Define $y_{k}^{t}$ to represent the probability that the real character is $k$ at time $t$. (Generally, $y_{k}^{t}$ is an element in the output matrix after softmax).
 * The set of all sequences that can be formed by the character set $L^{'}$ is called $L^{'T}$, and any sequence in $L^{'T}$ is called a path and marked as $π$. The distribution of $π$ is formula (1):
 *
 * $$
 * p(π|x)=\prod_{t=1}^{T}y^{t}_{π_{t}} , \forall π \in L'^{T}. \tag{1}
 * $$
 *
 * Define many to one mapping B: $L^{'T} \to L^{\leq T}$, and calculate the conditional probability of $l \in L^{\leq T}$ through mapping B,
 * Equal to the sum of the probabilities of all possible paths corresponding to $l$, formula (2):
 *
 * $$
 * p(l|x)=\sum_{π \in B^{-1}(l)}p(π|x).\tag{2}
 * $$
 *
 * The task of finding the path of $l$ that maximizes $p(l|x)$ is called decoding, formula (3):
 *
 * $$
 * h(x)=^{arg \  max}_{l \in L^{ \leq T}} \ p(l|x).\tag{3}
 * $$
 *
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 * A[(logProbs)] -->B([l0op::Contiguous])-->D([l0op::CTCLossV2])
 * A1[(targets)] -->B1([l0op::Contiguous])-->D
 * A2[(targetLengths)] -->B2([ConvertToTensor])-->D
 * A3[(inputLengths)] -->B3([ConvertToTensor])-->D
 * A4((blank)) -->D
 * A6((zeroInfinity)) -->D
 * D--negLogLikelihood--> F1([l0op::ViewCopy])--> J[(negLogLikelihoodOut)]
 * D--logAlpha-->H([l0op::ViewCopy])-->J1[(logAlphaOut)]
 * ```
 *
 * @param [in] logProbs(aclTensor*): The data type supports FLOAT and DOUBLE data types, and the shape is ($T,N,C$),
 * $T$ is the input length, $N$ is the batch size, $C$ is the number of categories, which must be greater than 0, including blank flags. This Tensor represents the logarithmic probability of the output,
 * Supports [non-continuous Tensor](https://), and the data format supports ND.
 * @param [in] targets(aclTensor*): The data type supports INT64, INT32, BOOL, FLOAT, and FLOAT16 data types. When the shape is ($N, S$),
 * $S$ is a value that is not less than the maximum value in $targetLengths$; or the shape is (SUM($targetLengths$)), assuming that $targets$ is unfilled and cascaded in 1 dimension;
 * Supports [non-continuous Tensor](https://), and the data format supports ND.
 * @param [in] inputLengths(aclIntArray*): The data type supports UINT8, INT8, INT16, INT32, INT64, and the array length is $N$.
 * Each value in the array must be less than or equal to $T$.
 * @param [in] targetLengths(aclIntArray*): The data type supports UINT8, INT8, INT16, INT32, INT64, and the array length is $N$.
 * When the shape of targets is ($N,S$), each value in the array must be less than or equal to $S$.
 * @param [in] blank(int): int integer, blank flag, default is 0, the value must be less than $C$ and greater than or equal to 0.
 * @param [in] zeroInfinity(bool): bool type, indicating whether to zero the infinite loss and related gradients. The default value is $False$.
 * @param [out] negLogLikelihoodOut(aclTensor*): output loss value, data type FLOAT, DOUBLE;
 * Outputs a Tensor of size ($N$), supports [non-continuous Tensor](https://), and the data format supports ND.
 * @param [out] logAlphaOut(aclTensor*): The data type supports FLOAT and DOUBLE data types (the data type must be consistent with logProbs),
 * Represents the probability of possible tracking input to the target. The Tensor is 3-dimensional and supports [non-continuous Tensor](https://). The data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCtcLossGetWorkspaceSize(
    const aclTensor* logProbs, const aclTensor* targets,
    const aclIntArray* inputLengths, const aclIntArray* targetlengths,
    int64_t blank, bool zeroInfinity, aclTensor* negLogLikelihoodOut,
    aclTensor* logAlphaOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnCtcLoss is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnCtcLossGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCtcLoss(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_CTCLOSS_H_
