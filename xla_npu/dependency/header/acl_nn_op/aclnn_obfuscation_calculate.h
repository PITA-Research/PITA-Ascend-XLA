/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#ifndef OP_API_INC_OBFUSCATION_CALCULATE_H_
#define OP_API_INC_OBFUSCATION_CALCULATE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnObfuscationCalculate calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * 
 * Operator function: PMCC project calculation implementation.
 * @param [in] fd: aclTensor on the NPU device side, indicating the socket connector, with shape (1). The data type supports INT32. The data format supports ND.
 * @param [in] x: aclTensor on the NPU device side, shape is (, *,..., hiddenSize), that is, the size of the last dimension is hiddenSize, and the data format supports ND.
 * Ascend 310P AI processor: Tensor data type supports ACL_FLOAT, ACL_FLOAT16, and ACL_INT8.
 * Ascend 910B AI processor: Tensor data type supports ACL_FLOAT, ACL_FLOAT16, ACL_INT8, and ACL_BF16.
 * @param [in] param: int32_t on the host side, reserved parameter field.
 * @param [in] cmd: int32_t on the host side, obfuscation operator instruction number. Currently only 1 is supported.
 * @param [out] y: aclTensor on the NPU device side, the obfuscated tensor, the data type and Shape are the same as x. The data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnObfuscationCalculateGetWorkspaceSize(
    int32_t fd, const aclTensor* x, int32_t param, int32_t cmd, aclTensor* y,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnObfuscationCalculate, used to perform calculations
 * @param [in] workspace: the workspace memory address applied for on the NPU device side
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnObfuscationCalculateGetWorkspaceSize
 * @param [in] executor: Returns the op executor, including the operator calculation process
 * @param [in] stream: ACL stream
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnObfuscationCalculate(void* workspace,
                                                uint64_t workspaceSize,
                                                aclOpExecutor* executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif  // OP_API_INC_OBFUSCATION_CALCULATE_H_
