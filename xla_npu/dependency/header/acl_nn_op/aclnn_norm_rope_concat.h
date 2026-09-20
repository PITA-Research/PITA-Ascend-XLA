/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_norm_rope_concat.h
 * \brief
 */
#ifndef ACLNN_NORM_ROPE_CONCAT_H_
#define ACLNN_NORM_ROPE_CONCAT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnInterleaveRope calculates the workspace size according to the specific calculation process.
 *
 * @param [in] query: npu
 * The aclTensor on the device side supports FLOAT32, FLOAT16, BFLOAT16 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [in] key: npu
 * The aclTensor on the device side supports FLOAT32, FLOAT16, BFLOAT16 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [in] value: npu
 * The aclTensor on the device side supports FLOAT32, FLOAT16, BFLOAT16 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [in] encoderQuery: npu
 * The aclTensor on the device side supports FLOAT32, FLOAT16, BFLOAT16 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [in] encoderKey: npu
 * The aclTensor on the device side supports FLOAT32, FLOAT16, BFLOAT16 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [in] encoderValue: npu
 * The aclTensor on the device side supports FLOAT32, FLOAT16, BFLOAT16 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [in] normQueryWeight: npu
 * The aclTensor on the device side supports FLOAT32, FLOAT16, BFLOAT16 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [in] normQueryBias: npu
 * The aclTensor on the device side supports FLOAT32, FLOAT16, BFLOAT16 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [in] normKeyWeight: npu
 * The aclTensor on the device side supports FLOAT32, FLOAT16, BFLOAT16 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [in] normKeyBias: npu
 * The aclTensor on the device side supports FLOAT32, FLOAT16, BFLOAT16 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [in] normAddedQueryWeight: npu
 * The aclTensor on the device side supports FLOAT32, FLOAT16, BFLOAT16 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [in] normAddedQueryBias: npu
 * The aclTensor on the device side supports FLOAT32, FLOAT16, BFLOAT16 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [in] normAddedKeyWeight: npu
 * The aclTensor on the device side supports FLOAT32, FLOAT16, BFLOAT16 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [in] normAddedKeyBias: npu
 * The aclTensor on the device side supports FLOAT32, FLOAT16, BFLOAT16 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [in] ropeSin: npu
 * The aclTensor on the device side supports FLOAT32, FLOAT16, BFLOAT16 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [in] ropeCos: npu
 * The aclTensor on the device side supports FLOAT32, FLOAT16, BFLOAT16 as data type, supports non-continuous Tensor, and supports ND as the data format.
 * @param [in] normType: normalization type, value range [0, 1], 0 represents LayerNorm, 1 represents AFFINE LayerNorm.
 * @param [in] normAddedType: Normalization type with affine transformation parameters, value range [0, 1], 0 represents LayerNorm, 1 represents AFFINE LayerNorm.
 * @param [in] ropeType: Rotation position encoding type, value range [0, 1], 0 means Interleave, 1 means Half.
 * @param [in] concatOrder: concatenation order, value range [0, 1], 0 means query first, 1 means query last.
 * @param [in] eps: epsilon value during normalization.
 * @param [in] isTraining: Whether it is the training phase.
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports FLOAT32, FLOAT16, BFLOAT16, and the data type is consistent with x.
 * The shape is the same as x, the data format supports ND, and the data format needs to be consistent with x.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnNormRopeConcatGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const aclTensor *encoderQuery, const aclTensor *encoderKey,
    const aclTensor *encoderValue, const aclTensor *normQueryWeight,
    const aclTensor *normQueryBias, const aclTensor *normKeyWeight,
    const aclTensor *normKeyBias, const aclTensor *normAddedQueryWeight,
    const aclTensor *normAddedQueryBias, const aclTensor *normAddedKeyWeight,
    const aclTensor *normAddedKeyBias, const aclTensor *ropeSin,
    const aclTensor *ropeCos, int64_t normType, int64_t normAddedType,
    int64_t ropeType, int64_t concatOrder, double eps, bool isTraining,
    const aclTensor *queryOutput, const aclTensor *keyOutput,
    const aclTensor *valueOutput, const aclTensor *normQueryMean,
    const aclTensor *normQueryRstd, const aclTensor *normKeyMean,
    const aclTensor *normKeyRstd, const aclTensor *normAddedQueryMean,
    const aclTensor *normAddedQueryRstd, const aclTensor *normAddedKeyMean,
    const aclTensor *normAddedKeyRstd, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* The second interface of @brief aclnnInterleaveRope, used to perform calculations.
 * @param [in] workspace: the starting address of the workspace memory applied for on the NPU device side
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnBitwiseNotGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnNormRopeConcat(void *workspace,
                                          uint64_t workspaceSize,
                                          aclOpExecutor *executor,
                                          aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif
