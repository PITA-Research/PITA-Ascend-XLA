/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_V4_H
#define OP_API_INC_GROUPED_MATMUL_V4_H
#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  GMM_ACT_TYPE_NONE = 0L,
  GMM_ACT_TYPE_RELU = 1L,
  GMM_ACT_TYPE_GELU_TANH = 2L,
  GMM_ACT_TYPE_GELU_ERR_FUNC = 3L,
  GMM_ACT_TYPE_FAST_GELU = 4L,
  GMM_ACT_TYPE_SILU = 5L,
} GMMActType;

/**
 * @brief The first interface of aclnnGroupedMatmulV4 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * @param [in] x:
 * Represents x in the formula. The data type supports FLOAT16, BFLOAT16, INT8, FLOAT32, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E1M2, and FLOAT4_E2M1 data types. The data format supports ND, and the maximum supported length is 128.
 * @param [in] weight:
 * Indicates the weight in the formula. The data type supports FLOAT16, BFLOAT16, INT8, FLOAT32, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E1M2, and FLOAT4_E2M1 data types. The data format supports ND, and the maximum supported length is 128.
 * @param [in] biasOptional:
 * Indicates the bias in the formula. The data type supports FLOAT16, FLOAT32, INT32, and BFLOAT16 data types. The data format supports ND. The maximum supported length is 128.
 * @param [in] scaleOptional:
 * Indicates quantization parameters. The data type supports UINT64, INT64, BFLOAT16, FLOAT32, and FLOAT8_E8M0 data types. The data format supports ND, and the maximum supported length is 128.
 * @param [in] offsetOptional: represents the quantization parameter, the data type supports FLOAT32 data type, the data format supports ND, and the maximum supported length is 128.
 * @param [in] antiquantScaleOptional:
 * Indicates pseudo quantization parameters. The data type supports FLOAT16 and BFLOAT16 data types. The data format supports ND. The maximum supported length is 128.
 * @param [in] antiquantOffsetOptional:
 * Indicates pseudo quantization parameters. The data type supports FLOAT16 and BFLOAT16 data types. The data format supports ND. The maximum supported length is 128.
 * @param [in] perTokenScaleOptional:
 * Indicates the per token quantification parameter. The data type supports FLOAT32 and FLOAT8_E8M0 data types. The data format supports ND. The maximum length supported is the same as the length of x.
 * @param [in] groupListOptional: optional parameter, representing the index status on the input and output grouping axes, the data type supports INT64,
 * The maximum length supported by some scenarios is 1024 (see the interface document constraint description for details), and the maximum length supported by other scenarios is 128.
 * @param [in] activationInputOptional: optional parameter, representing the reverse input of the activation function.
 * @param [in] activationQuantScaleOptional: Optional parameter, reserved parameter.
 * @param [in] activationQuantOffsetOptional: optional parameter, reserved parameter.
 * @param [in] splitItem:
 * An integer parameter that represents whether the output needs to be divided into tensors. 0/1 represents the output as multiple tensors; 2/3 represents the output as a single tensor. The default value is 0.
 * @param [in] groupType:
 * The integer parameter represents the axis that needs to be segmented. -1 represents that no segmentation is required; 0 represents that the M axis needs to be segmented; 1 represents that the N axis needs to be segmented; 2 represents that the K axis needs to be segmented.
 * @param [in] groupListType:
 * Integer parameter, which can take the value 0 or 1. 0 means that the value in groupListOptional is the cumsum result (cumulative sum) of the group axis size.
 * 1 means that the value in groupListOptional is the size of each group on the grouping axis.
 * @param [in] actType: Integer parameter, representing the activation function type. The enumeration value of each activation function refers to the enumeration class GMMActType.
 * @param [out] out: Represents out in the formula. The data type supports FLOAT16, BFLOAT16, INT8, and FLOAT32 data types. The data format supports ND. The maximum supported length is 128.
 * @param [out] activationFeatureOutOptional: input data of activation function.
 * @param [out] dynQuantScaleOutOptional: reserved parameter.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnGroupedMatmulV4GetWorkspaceSize(
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
    aclTensorList *out, aclTensorList *activationFeatureOutOptional,
    aclTensorList *dynQuantScaleOutOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnGroupedMatmulV4's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGroupedMatmulV4GetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulV4(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif