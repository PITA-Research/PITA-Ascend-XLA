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
#ifndef OP_API_INC_LEVEL2_ACLNN_SILENT_CHECK_H_
#define OP_API_INC_LEVEL2_ACLNN_SILENT_CHECK_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnSilentCheck calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * Operator function: Determine whether to trigger a silent detection error based on the input val.
 * This operator has custom operator semantics and has no corresponding tensorflow or pytorch interface.
 * @param [in] val: aclTensor on the NPU device side, the data type supports FLOAT32, FLOAT16, BFLOAT16. The shape is [1].
 * @param [in] inputGrad: aclTensor on the NPU device side, the data type supports FLOAT32, FLOAT16, BFLOAT16.
 * @param [in] sfda: aclTensor on the NPU device side, the data type supports FLOAT32. The shape is [3].
 * @param [in] step: aclTensor on the NPU device side, the data type supports INT64. The shape is [1].
 * @param [in] cMinSteps: The number of steps to be accumulated, the data type is INT32.
 * @param [in] cThreshL1: Numerical L1 threshold, data type is FLOAT.
 * @param [in] cCoeffL1: L1 threshold on transition, data type is FLOAT.
 * @param [in] cThreshL2: Numerical L2 threshold, data type is FLOAT.
 * @param [in] cCoeffL2: L2 threshold on transition, data type is FLOAT.
 * @param [in] npuAsdDetect: Environment variable NPU_ASD_DETECT, data type is INT32.
 * @param [in] result: aclTensor to determine whether to trigger the silent detection failure result, the data type is INT32.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnSilentCheckGetWorkspaceSize(
    const aclTensor *val, aclTensor *inputGradRef, aclTensor *sfdaRef,
    aclTensor *stepRef, const int32_t cMinSteps, const float cThreshL1,
    const float cCoeffL1, const float cThreshL2, const float cCoeffL2,
    const int32_t npuAsdDetect, aclTensor *result, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief: The second interface of aclnnSilentCheck is used to perform calculations
 * @domain aclnn_ops_train
 * Operator function: Determine whether to trigger a silent detection error based on the input val.
 * This operator has custom operator semantics and has no corresponding tensorflow or pytorch interface.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSilentCheckGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSilentCheck(void *workspace, uint64_t workspaceSize,
                                       aclOpExecutor *executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_SILENT_CHECK_H_