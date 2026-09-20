/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_SILENT_CHECK_V2_H_
#define OP_API_INC_LEVEL2_ACLNN_SILENT_CHECK_V2_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnSilentCheckV2 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * Operator function: Determine whether to trigger a silent detection error based on the input val.
 * This operator has custom operator semantics and has no corresponding tensorflow or pytorch interface.
 * @param [in] val: aclTensor on the NPU device side, the data type supports FLOAT32, FLOAT16, BFLOAT16. shape is [0].
 * @param [in] max: aclTensor on the NPU device side, the data type supports FLOAT32, FLOAT16, BFLOAT16. shape is [0].
 * @param [in] avg: aclTensor on the NPU device side, the data type supports FLOAT32, FLOAT16, BFLOAT16. shape is [0].
 * @param [in] inputGrad: aclTensor on the NPU device side, the data type supports FLOAT32, FLOAT16, BFLOAT16.
 * @param [in] step: aclTensor on the NPU device side, the data type supports INT64. The shape is [1].
 * @param [in] dstSize: aclIntArray on the NPU device side, the data type supports INT64.
 * @param [in] dstStride: aclIntArray on the NPU device side, the data type supports INT64.
 * @param [in] dstOffset: aclIntArray on the NPU device side, the data type supports INT64.
 * @param [in] cThreshL1: Numerical L1 threshold, data type is FLOAT.
 * @param [in] cThreshL2: Numerical L2 threshold, data type is FLOAT.
 * @param [in] beta1: The calculation parameter of the non-first step, the data type is FLOAT.
 * @param [in] npuAsdDetect: Environment variable NPU_ASD_DETECT, data type is INT32.
 * @param [in] result: aclTensor to determine whether to trigger the silent detection failure result, the data type is INT32.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnSilentCheckV2GetWorkspaceSize(
    const aclTensor *val, const aclTensor *max, aclTensor *avgRef,
    aclTensor *inputGradRef, aclTensor *stepRef, aclIntArray *dstSize,
    aclIntArray *dstStride, aclIntArray *dstOffset, float cThreshL1,
    float cThreshL2, float beta1, int32_t npuAsdDetect, aclTensor *result,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief: The second interface of aclnnSilentCheckV2, used to perform calculations
 * @domain aclnn_ops_train
 * Operator function: Determine whether to trigger a silent detection error based on the input val.
 * This operator has custom operator semantics and has no corresponding tensorflow or pytorch interface.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSilentCheckV2GetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSilentCheckV2(void *workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor *executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_SILENT_CHECK_V2_H_