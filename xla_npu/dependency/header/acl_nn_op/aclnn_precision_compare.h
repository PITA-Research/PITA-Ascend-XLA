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
#ifndef OP_API_INC_PRECISION_COMPARE_H_
#define OP_API_INC_PRECISION_COMPARE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * The first interface of @brief aclnnPrecisionCompare calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Compare and calculate whether two Tensors are the same and return the status code of operator execution:
 *
 * @param [in] golden: aclTensor on NPU device side,
 * The data type supports FLOAT16, FLOAT data type,
 * golden is the same data type as realdata
 * @param [in] realdata: aclTensor on NPU device side,
 * The data type supports FLOAT16, FLOAT data type,
 * realdata has the same data type as golden
 * @param [in] out: Output a Tensor whose data type is UINT32.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */

ACLNN_API aclnnStatus aclnnPrecisionCompareGetWorkspaceSize(
    const aclTensor *golden, const aclTensor *realdata, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);
/**
 * @brief The second interface of aclnnPrecisionCompare is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnPrecisionCompareGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */

ACLNN_API aclnnStatus aclnnPrecisionCompare(void *workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor *executor,
                                            aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif  // OP_API_INC_PRECISION_COMPARE_H_