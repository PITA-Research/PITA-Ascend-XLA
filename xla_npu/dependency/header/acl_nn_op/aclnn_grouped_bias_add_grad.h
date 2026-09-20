/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GROUPED_BIAS_ADD_GRAD_H_
#define OP_API_INC_GROUPED_BIAS_ADD_GRAD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGroupedBiasAddGrad calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * Operator function: realizes the reverse calculation of groupBiasAdd.
 * @param [in] gradY: Backpropagation gradient, gradY in the formula, aclTensor on the Device side, data type supports FLOAT, FLOAT16, BFLOAT16.
 * When there is optional input groupIdxOptional, shape only supports 2 dimensions. When there is no optional input groupIdxOptional, shape only supports 3 dimensions.
 * @param [in] groupIdxOptional:
 * The end position of each group, groupIdxOptional in the formula, aclTensor on the Device side, the data type supports INT32, INT64, and shape only supports 1 dimension.
 * @param [out] out:
 * The gradient of bias, out in the formula, aclTensor on the Device side, the data type supports FLOAT, FLOAT16, BFLOAT16, the data type must be consistent with the data type of gradY, and shape only supports 2 dimensions.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGroupedBiasAddGradGetWorkspaceSize(
    const aclTensor* gradY, const aclTensor* groupIdxOptional, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGroupedBiasAddGrad's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnGroupedBiasAddGradGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGroupedBiasAddGrad(void* workspace,
                                              uint64_t workspaceSize,
                                              aclOpExecutor* executor,
                                              aclrtStream stream);

/**
 * The first interface of @brief aclnnGroupedBiasAddGrad calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * Operator function: realizes the reverse calculation of groupBiasAdd.
 * @param [in] gradY: Backpropagation gradient, gradY in the formula, aclTensor on the Device side, data type supports FLOAT, FLOAT16, BFLOAT16.
 * When there is optional input groupIdxOptional, shape only supports 2 dimensions. When there is no optional input groupIdxOptional, shape only supports 3 dimensions.
 * @param [in] groupIdxOptional:
 * The end position of each group, groupIdxOptional in the formula, aclTensor on the Device side, the data type supports INT32, INT64, and shape only supports 1 dimension.
 * @param [in] groupIdxType: Indicates the type of groupIdx. Supported values are 0 and 1,
 * 0: indicates that the value in groupIdxOptional is the end index of each group, 1: indicates that the value in groupIdxOptional is the size of each group. The data type supports Int64.
 * @param [out] out:
 * The gradient of bias, out in the formula, aclTensor on the Device side, the data type supports FLOAT, FLOAT16, BFLOAT16, the data type must be consistent with the data type of gradY, and shape only supports 2 dimensions.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGroupedBiasAddGradV2GetWorkspaceSize(
    const aclTensor* gradY, const aclTensor* groupIdxOptional,
    int64_t groupIdxType, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnGroupedBiasAddGrad's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnGroupedBiasAddGradV2GetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGroupedBiasAddGradV2(void* workspace,
                                                uint64_t workspaceSize,
                                                aclOpExecutor* executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
