/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/license/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_ADA_LAYER_NORM_QUANT_H
#define OP_API_INC_LEVEL2_ACLNN_ADA_LAYER_NORM_QUANT_H

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAdaLayerNormQuantGetWorkspaceSize
 */
ACLNN_API aclnnStatus aclnnAdaLayerNormQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* scale, const aclTensor* shift,
    const aclTensor* weightOptional, const aclTensor* biasOptional,
    const aclTensor* smoothScalesOptional, double epsilon,
    const char* quantMode, aclTensor* out, aclTensor* quantScale,
    aclTensor* quantOffsetOptional, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/*
 * @brief aclnnAdaLayerNormQuant
 */
ACLNN_API aclnnStatus aclnnAdaLayerNormQuant(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ADA_LAYER_NORM_QUANT_H