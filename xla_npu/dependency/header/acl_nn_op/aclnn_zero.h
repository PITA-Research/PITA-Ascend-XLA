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

#ifndef OP_API_INC_Zero_H_
#define OP_API_INC_Zero_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnInplaceZero calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Fill the self tensor with all zeros.
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(selfRef)]-->B([l0op::Contiguous])
 *     B-->C([l0op::Zeros_Like])
 *     C-->D([l0op::ViewCopy])
 *     D-->E[(selfRef)]
 * ```
 *
 * @param [in] selfRef: npu
 * aclTensor on the device side, the data type supports INT8, INT16, INT32, INT64, QINT8, QINT16, QINT32, UINT8, UINT16,
 * UINT32, UINT64, QUINT8, QUINT16, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128. Support [non-continuous Tensor](#), the data format supports ND
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceZeroGetWorkspaceSize(
    aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceZero is used to perform calculations.
 *
 * Operator function: Fill the self tensor with all zeros.
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(selfRef)]-->B([l0op::Contiguous])
 *     B-->C([l0op::Zeros_Like])
 *     C-->D([l0op::ViewCopy])
 *     D-->E[(selfRef)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceZeroGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceZero(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_Zero_H_
