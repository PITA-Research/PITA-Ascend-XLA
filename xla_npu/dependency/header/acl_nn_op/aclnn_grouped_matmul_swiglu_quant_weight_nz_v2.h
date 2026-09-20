/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_V2_H
#define OP_API_INC_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_V2_H
#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGroupedMatmulSwigluQuantWeightNzV2 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * @param [in] x: represents x in the formula, the data type supports INT8 data type, and the data format supports ND.
 * @param [in] weight:
 * Represents the weight in the formula, the data type supports INT8 data type, and the data format supports NZ.
 * @param [in] weightScale:
 * Represents quantization parameters, the data type supports FLOAT32 data type, and the data format supports ND.
 * @param [in] weightAssistMatrix:
 * Represents the weight auxiliary matrix, and the data type supports the FLOAT32 data type.
  * @param [in] weightScale:
 * Represents offset, the data type supports FLOAT32 data type, and the data format supports ND.
 * @param [in] xScale:
 * Represents perToken quantification parameters, the data type supports FLOAT8_E8M0 data type, and the data format supports ND.
 * @param [in] smoothScale:
 * The data type supports FLOAT32 data type.
 * @param [in] groupList: Required parameter, representing the index status on the input and output grouping axes, the data type supports INT64.
 * @param [in] dequantMode: Indicates dequantization mode.
 * @param [in] dequantDtype: Represents the dequantization data format.
 * @param [in] quantMode: Indicates the quantization mode.
 * @param [in] quantDtype: Represents the quantified data format.
 * @param [out] quantOutput: Indicates out in the formula. The data type supports FLOAT8_E4M3FN and FLOAT8_E5M2 data types. The data format supports ND.
 * @param [out] quantScaleOutput: Represents outQuantScale in the formula, and the data type supports the FLOAT8_E8M0 data type.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(
    const aclTensor *x, const aclTensorList *weight,
    const aclTensorList *weightScale,
    [[maybe_unused]] const aclTensorList *weightAssistMatrix,
    [[maybe_unused]] const aclTensor *bias,
    [[maybe_unused]] const aclTensor *xScale,
    [[maybe_unused]] const aclTensor *smoothScale, const aclTensor *groupList,
    [[maybe_unused]] int64_t dequantMode, [[maybe_unused]] int64_t dequantDtype,
    [[maybe_unused]] int64_t quantMode, int64_t groupListType,
    const aclIntArray *tuningConfigOptional, aclTensor *output,
    aclTensor *outputScale, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnGroupedMatmulSwigluQuantV2's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnGroupedMatmulSwigluQuantWeightNzV2(void *workspace, uint64_t workspaceSize,
                                        aclOpExecutor *executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif