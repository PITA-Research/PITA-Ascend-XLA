/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
#ifndef OP_API_INC_ALL_GATHER_MATMUL_V2_
#define OP_API_INC_ALL_GATHER_MATMUL_V2_

#include <string>
#include "aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * Operator function: realize allGather + mm fusion calculation
 * @brief aclnnAllGatherMatmulV2's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] x1: matmul left matrix, data type support: float16, BF16, fp8_e4m3, fp8_e5m2, hif8.
 * @param [in] x2: matmul right matrix, data type support: float16, BF16, fp8_e4m3, fp8_e5m2, hif8.
 * @param [in] bias: bias, supported data types: float16, BF16, float32.
 * @param [in] x1Scale: matmul left matrix quantization parameter, data type support: float32.
 * @param [in] x2Scale: matmul right matrix quantization parameter, data type support: float32.
 * @param [in] quantScale: quantization parameter, data type support: float32.
 * @param [in] blockSize: The number of values ​​corresponding to different axes of a quantization coefficient in the output, default value: 0.
 * @param [in] group: String identifying the column group.
 * @param [in] gatherIndex: identifies the gather target, 0: left matrix, 1: right matrix.
 * @param [in] commTurn: communication data cut fraction, that is, total data volume/single communication volume.
 * @param [in] streamMode: enumeration of ACL stream mode, type support: 0/1.
 * @param [in] groupSize: The number of values ​​corresponding to different axes of an inverse quantization coefficient in x1/x2, default value: 0.
 * @param [out] output: The result of calculation + communication, data type support: float16, BF16, float32.
 * @param [out] gatherOut: Only the result of gather communication operation, data type: same as input.
 * @param [out] amaxOut: The maximum value output by matmul, data type: float32.
 * @param [out] workspaceSize: Returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnAllGatherMatmulV2GetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
    const aclTensor* x1Scale, const aclTensor* x2Scale,
    const aclTensor* quantScale, int64_t blockSize, const char* group,
    int64_t gatherIndex, int64_t commTurn, int64_t streamMode,
    int64_t groupSize, aclTensor* output, aclTensor* gatherOut,
    aclTensor* amaxOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAllGatherMatmulV2's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAllGatherMatmulV2GetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnAllGatherMatmulV2(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ALL_GATHER_MATMUL_V2_