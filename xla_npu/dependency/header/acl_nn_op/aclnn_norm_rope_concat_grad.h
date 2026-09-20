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
 * \file aclnn_norm_rope_concat_grad.h
 * \brief
 */

#ifndef OP_API_INC_NORM_ROPE_CONCAT_GRAD_H_
#define OP_API_INC_NORM_ROPE_CONCAT_GRAD_H_

#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnNormRopeConcatBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 */
aclnnStatus aclnnNormRopeConcatBackwardGetWorkspaceSize(
    const aclTensor *gradQueryOutput, const aclTensor *gradKeyOutput,
    const aclTensor *gradValueOutput, const aclTensor *query,
    const aclTensor *key, const aclTensor *encoderQuery,
    const aclTensor *encoderKey, const aclTensor *normQueryWeight,
    const aclTensor *normQueryMean, const aclTensor *normQueryRstd,
    const aclTensor *normKeyWeight, const aclTensor *normKeyMean,
    const aclTensor *normKeyRstd, const aclTensor *normAddedQueryWeight,
    const aclTensor *normAddedQueryMean, const aclTensor *normAddedQueryRstd,
    const aclTensor *normAddedKeyWeight, const aclTensor *normAddedKeyMean,
    const aclTensor *normAddedKeyRstd, const aclTensor *ropeSin,
    const aclTensor *ropeCos, int64_t normType, int64_t normAddedType,
    int64_t ropeType, int64_t concatOrder, const aclTensor *gradQuery,
    const aclTensor *gradKey, const aclTensor *gradValue,
    const aclTensor *gradEncoderQuery, const aclTensor *gradEncoderKey,
    const aclTensor *gradEncoderValue, const aclTensor *gradNormQueryWeight,
    const aclTensor *gradNormQueryBias, const aclTensor *gradNormKeyWeight,
    const aclTensor *gradNormKeyBias, const aclTensor *gradNormAddedQueryWeight,
    const aclTensor *gradNormAddedQueryBias,
    const aclTensor *gradNormAddedKeyWeight,
    const aclTensor *gradNormAddedKeyBias, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * The second interface of @brief aclnnNormRopeConcatBackward is used to perform calculations.
 */
aclnnStatus aclnnNormRopeConcatBackward(void *workspace, uint64_t workspaceSize,
                                        aclOpExecutor *executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_NORM_ROPE_CONCAT_GRAD_H_
