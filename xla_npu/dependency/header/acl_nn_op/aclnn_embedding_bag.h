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
 * \file aclnn_embedding_bag.h
 * \brief
 */
#ifndef OP_API_INC_EMBEDDING_BAG_H_
#define OP_API_INC_EMBEDDING_BAG_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * The first interface of @brief aclnnEmbeddingBag calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Get a set of aggregated numbers from weight based on indices, and then perform aggregation such as max, sum, mean, etc. based on the value of offsets and the aggregation mode specified by mode.
 *
 * @param [in] weight: aclTensor on the NPU device side. The data type supports float32 float16 bfloat16 and must be 2D.
 * @param [in] indices: aclTensor on the NPU device side, data type supports UINT8, INT8, INT16, INT32, INT64. Must be a 1D or 2D tensor.
 * @param [in] offsets: npu
 * The data type of aclTensor on the device side supports UINT8, INT8, INT16, INT32, and INT64, but one of the indices must be INT32 or INT64.
 * When indices is 1D, offsets must be 1D tensor.
 * @param [in] scaleGradByFreq: bool data type, used to control the scaling gradient.
 * @param [in] mode: int64 type, used to control aggregation mode.
 * @param [in] sparse: bool data type, used to control sparse mode.
 * @param [in] perSampleWeights: aclTensor on the NPU device side, specifying the sample weight. Must be 1D
 * tensor, the data type is the same as weight, only sum can not be null.
 * @param [in] includeLastOffset: bool data type, controls whether to include the last offset.
 * @param [in] paddingIdx: int64 type, the value range is [-n,n-1], where n is the number of elements in the first dimension of weigit.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU side.
 * @param [out] executor: Returns the op executor.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEmbeddingBagGetWorkspaceSize(
    const aclTensor* weight, const aclTensor* indices, const aclTensor* offsets,
    bool scaleGradByFreq, int64_t mode, bool sparse,
    const aclTensor* perSampleWeights, bool includeLastOffset,
    int64_t paddingIdx, aclTensor* output, aclTensor* offset2bag,
    aclTensor* bagSize, aclTensor* maxIndices, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnEmbeddingBag is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceIndexCopyGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEmbeddingBag(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_INDEX_COPY_H_
