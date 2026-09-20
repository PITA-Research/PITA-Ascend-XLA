/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_NZ
#define OP_API_INC_GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_NZ

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGroupedMatmulFinalizeRoutingWeightNz calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: Implements the fusion operator of GroupedMatmul and MoeFinalizeRouting. The output calculated by GroupedMatmul is combined according to the index. Currently, only w is in nz format.
 * @param [in] x1: matmul left matrix, data type support: int8.
 * @param [in] x2: matmul right matrix, data type support: int8.
 * @param [in] scale: scaling factor in quantization parameters, data type support: float32.
 * @param [in] bias: bias, data type support: float32.
 * @param [in] pertokenScaleOptional: dequantization parameter, data type support: float32.
 * @param [in] groupList: represents the matmul size distribution in the input and output grouping axis directions, data type support: int64.
 * @param [in] sharedInput: 
 * The output of the shared expert in moe calculation needs to be combined with the output of the moe expert. The data type supports: bfloat16, float16.
 * @param [in] logit: 
 * Moe experts multiply the logit size of each token and the calculated output of the matrix multiplication with the logit, and then combine the indexes. Data type support: float32.
 * @param [in] rowIndex: 
 * The moe expert output is combined according to the rowIndex, and the value is the index of combine for scatter add. The data type supports: int64, int32.
 * @param [in] sharedInputWeight: 
 * The coefficient of combine between sharing expert and moe expert. shareInput multiplies this parameter first, and then accumulates the result with moe expert. Data type support: float32.
 * @param [in] sharedInputOffset: The offset of the shared expert output in the total output, data type support: int64.
 * @param [in] transposeX1: Whether the left matrix is ​​transposed, default value: false.
 * @param [in] transposeX2: Whether the right matrix is ​​transposed, default value: false.
 * @param [in] groupListType: GroupedMatmul grouping type, default value: 1, count mode, data type support: int64..
 * @param [out] out: calculation result, data type: float32, float16, bfloat16.
 * @param [out] workspaceSize: Returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNzGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *scale,
    const aclTensor *bias, const aclTensor *pertokenScaleOptional,
    const aclTensor *groupList, const aclTensor *sharedInput,
    const aclTensor *logit, const aclTensor *rowIndex, int64_t dtype,
    float sharedInputWeight, int64_t sharedInputOffset, bool transposeX1,
    bool transposeX2, int64_t groupListType, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnGroupedMatmulFinalizeRoutingWeightNz's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGroupedMatmulFinalizeRoutingNZGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNz(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANT_MATMUL_NZ