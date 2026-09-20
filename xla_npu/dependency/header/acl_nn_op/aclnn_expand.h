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
#ifndef OP_API_INC_EXPAND_H_
#define OP_API_INC_EXPAND_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnExpand calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Broadcast the input tensor self into a tensor of specified size.
 *
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(self)] --> B([l0op::Contiguous]) --> C([l0op::Expand])
 *     D((size)) --> E([sizeTensor]) --> C
 *     C --> F([l0op::ViewCopy]) --> Out[(out)]
 * ```
 *
 * @param [in] self: npu
 * The aclTensor on the device side supports BFLOAT16, FLOAT16, FLOAT, UINT8, INT8, INT32, INT64, and BOOL data types. It supports non-continuous Tensor and the data format supports ND.
 * @param [in] size: aclIntArray type, the data type supports INT.
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports BFLOAT16, FLOAT16, FLOAT, UINT8, INT8, INT32, INT64, and BOOL, and needs to be consistent with self.
 * Supports [non-continuous Tensor](#), the data format supports ND ([Reference](#)), and shape needs to satisfy the derivation result of self's shape based on size.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnExpandGetWorkspaceSize(const aclTensor* self,
                                                  const aclIntArray* size,
                                                  aclTensor* out,
                                                  uint64_t* workspaceSize,
                                                  aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnExpand is used to perform calculations.
 *
 * Operator function: Broadcast the input tensor self into a tensor of specified size.
 *
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(self)] --> B([l0op::Contiguous]) --> C([l0op::Expand])
 *     D((size)) --> E([sizeTensor]) --> C
 *     C --> F([l0op::ViewCopy]) --> Out[(out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnExpandGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnExpand(void* workspace, uint64_t workspaceSize,
                                  aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_EXPAND_H_