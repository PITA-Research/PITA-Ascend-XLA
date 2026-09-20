/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_WEIGHT_NZ_H
#define OP_API_INC_GROUPED_MATMUL_WEIGHT_NZ_H
#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGroupedMatmulWeightNz calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * @param [in] x: Indicates the input x in the formula. The data type supports FLOAT16, BFLOAT16, INT8, and FLOAT8_E4M3FN. The data format supports ND. The maximum length supported is 128.
 * @param [in] weight: Indicates the weight in the formula. The data type supports FLOAT16, BFLOAT16, INT8, INT4, and FLOAT4_E2M1. The data format supports NZ. The maximum supported length is 128.
 * @param [in] biasOptional: Indicates the bias in the formula. The data type supports FLOAT16, BFLOAT16, FLOAT32, and INT32. The data format supports ND. The length is the same as weight.
 * @param [in] scaleOptional: represents the scaling factor in the quantization parameter. The data type supports BFLOAT16, FLOAT32, and UINT64. The data format supports ND. The length is the same as weight.
 * @param [in] offsetOptional: represents the offset in the quantization parameter, the data type supports FLOAT32, the data format supports ND, and the length is the same as weight.
 * @param [in] antiquantScaleOptional: represents the scaling factor in the pseudo quantization parameter. The data type supports FLOAT16, BFLOAT16, and FLOAT8_E8M0. The data format supports ND. The length is the same as weight.
 * @param [in] antiquantOffsetOptional: represents the offset in the pseudo quantization parameter, the data type supports FLOAT16, BFLOAT16, the data format supports ND, and the length is the same as weight.
 * @param [in] perTokenScaleOptional: represents the scaling factor introduced by the quantization of input x in the quantization parameter. The data type supports FLOAT32 and FLOAT8_E8M0. The data format supports ND. The length is the same as x.
 * @param [in] groupListOptional: represents the matmul size distribution of the input and output grouping axes. The data type supports INT64, the data format supports ND, and the length is the same as weight.
 * @param [in] activationInputOptional: Optional parameter, reverse input of activation function.
 * @param [in] activationQuantScaleOptional: Optional parameter, the quantization coefficient of the output of the activation function.
 * @param [in] activationQuantOffsetOptional: Optional parameter, the quantization offset of the output of the activation function.
 * @param [in] splitItem: Integer parameter, representing whether the output needs to be split into tensors. 0/1 represents the output as multiple tensors; 2/3 represents the output as a single tensor. The default value is 0.
 * @param [in] groupType：
 * Integer parameter, representing the axis that needs to be grouped. For example, the matrix multiplication is C[m,n]=A[m,k]xB[k,n], then groupType takes the value -1: no grouping, 0: m-axis grouping, 1: n-axis grouping, 2: k-axis grouping. The default value is -1. Currently, n-axis and k-axis grouping are not supported.
 * @param [in] groupListType：
 * Integer parameter, which can take the value 0 or 1. 0 means that the value in groupListOptional is the cumsum result (cumulative sum) of the grouping axis size. 1 means that the value in groupListOptional is the size of each grouping axis. The default value is 0.
 * @param [in] actType: integer parameter, representing the activation function type. Currently, only 0 is supported, and there is no activation function.
 * @param [in] tuningConfigOptional：
 * Tuning parameters. The first value in the array represents the expected value of the number of tokens processed by each expert. When tiling the operator, optimal tiling will be performed based on this expected value.
 * @param [in] quantGroupSize: Integer parameter, representing the group size of group quantization (per-group). When group quantization is not involved, fill in 0.
 * @param [out] out: Indicates out in the formula. The data type supports FLOAT16, BFLOAT16, and INT32 data types. The data format supports ND. The maximum supported length is 128.
 * @param [out] activationFeatureOutOptional: input data of activation function.
 * @param [out] dynQuantScaleOutOptional: When there is an activation function, the quantization coefficient output is dynamically quantized on the activated output.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process
 * @return aclnnStatus: Return status code.
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnGroupedMatmulWeightNzGetWorkspaceSize(
    const aclTensorList *x, const aclTensorList *weight,
    const aclTensorList *biasOptional, const aclTensorList *scaleOptional,
    const aclTensorList *offsetOptional,
    const aclTensorList *antiquantScaleOptional,
    const aclTensorList *antiquantOffsetOptional,
    const aclTensorList *perTokenScaleOptional,
    const aclTensor *groupListOptional,
    const aclTensorList *activationInputOptional,
    const aclTensorList *activationQuantScaleOptional,
    const aclTensorList *activationQuantOffsetOptional, int64_t splitItem,
    int64_t groupType, int64_t groupListType, int64_t actType,
    aclIntArray *tuningConfigOptional, int64_t quantGroupSize,
    aclTensorList *out, aclTensorList *activationFeatureOutOptional,
    aclTensorList *dynQuantScaleOutOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * The second interface of @brief aclnnGroupedMatmulWeightNz, used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGroupedMatmulWeightNzGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulWeightNz(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif