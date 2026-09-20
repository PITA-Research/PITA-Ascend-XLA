/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_UNAMPLE_NEAREST_2D_H_
#define OP_API_INC_UNAMPLE_NEAREST_2D_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUpsampleNearest2D's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Apply the nearest neighbor interpolation algorithm to upsample an input signal consisting of multiple input channels.
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *  A[(self)] -.->B([l0op::Contiguous])
 *  B --> D([l0op::TransData])
 *  D --> I([l0op::ResizeNearestNeighborV2])
 *  C[(outputSize)] --> I
 *  I --> J([l0op::TransData])
 *  J -.-> P([l0op::ViewCopy])
 *  P --> G[(out)]
 * ```
 *
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports FLOAT, BFLOAT16, FLOAT16, DOUBLE, and UIN8. The data format supports NCHW and NHWC. Supports non-contiguous Tensors.
 * @param [in] outputSize: aclIntArray on the NPU device side, specifying the output space size.
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports FLOAT, BFLOAT16, FLOAT16, DOUBLE, and UIN8. The data format supports NCHW and NHWC. Supports non-contiguous Tensors.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnUpsampleNearest2dGetWorkspaceSize(
    const aclTensor *self, const aclIntArray *outputSize, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnUpsampleNearest2D's second interface is used to perform calculations.
 *
 * Operator function: Apply the nearest neighbor interpolation algorithm to upsample an input signal consisting of multiple input channels.
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *  A[(self)] -.->B([l0op::Contiguous])
 *  B --> D([l0op::TransData])
 *  D --> I([l0op::ResizeNearestNeighborV2])
 *  C[(outputSize)] --> I
 *  I --> J([l0op::TransData])
 *  J -.-> P([l0op::ViewCopy])
 *  P --> G[(out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnUpsampleNearest2dGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnUpsampleNearest2d(void *workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor *executor,
                                             const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNAMPLE_NEAREST_2D_H_
