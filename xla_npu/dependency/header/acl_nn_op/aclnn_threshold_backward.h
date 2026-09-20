/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_THRESHOLD_BACKWARG_H_
#define OP_API_INC_THRESHOLD_BACKWARG_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnThresholdBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: complete threshold_forward corresponding to reverse direction
 * Calculation formula:
 * res(i) = gradOutput(i) if self(i) > threshold else 0
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(self)] --> B([l0op::Contiguous])
 * B --> C([l0op::ThresholdGradV2D or l0op::ReluGrad])
 * D[(grad_output)] -->  E([l0op::Contiguous])
 * E  --> C
 * F[(threshold)] --> C
 * C--> G([l0op::ViewCopy])
 * G --> H[(out)]
 * ```
 *
 * @param [in] gradOutput: aclTensor on the NPU device side. The data type supports FLOAT, BFLOAT16, FLOAT16, INT32, INT8, and UINT8. The shape needs to be consistent with self.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with self.
 * @param [in] self: aclTensor on the NPU device side. The data type supports FLOAT, BFLOAT16, FLOAT16, INT32, INT8, and UINT8.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] threshold: aclScalar on the host side, the data type needs to be convertible into the data type derived from self and other.
 * @param [in] out: aclTensor on the NPU device side. The data type supports FLOAT, BFLOAT16, FLOAT16, INT32, INT8, and UINT8. The shape needs to be consistent with self.
 * Supports non-continuous Tensor, the data format supports ND, and the data format needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnThresholdBackwardGetWorkspaceSize(
    const aclTensor *gradOutput, const aclTensor *self,
    const aclScalar *threshold, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor);
/**
 * The second interface of @brief aclnnAdd is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnThresholdBackward(void *workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor *executor,
                                             const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_THRESHOLD_BACKWARG_H_
