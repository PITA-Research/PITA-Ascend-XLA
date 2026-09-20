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

#ifndef OP_API_INC_LEVEL2_ACLNN_FILL_TENSOR_H_
#define OP_API_INC_LEVEL2_ACLNN_FILL_TENSOR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: assign value to each position of Tensor
 * Calculation formula:
 * Not involved
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 * A[(selfRef)] --> B(Generate AclTensor: dims) -->C([l0op::Fill])
 *    D[(value)] --> C
 *    C --> E([l0op::ViewCopy])--> F[(out)]
 * ```
 */

/**
 * The first interface of @brief aclnnInplaceFillTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] selfRef: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64, DOUBLE,
 * COMPLEX64, COMPLEX128, BOOL, BFLOAT16, support non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in] value: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64, DOUBLE,
 * COMPLEX64, COMPLEX128, BOOL, BFLOAT16, and the data type needs to be convertible to the selfRef data type. The data format supports ND, and the data dimension can only be 0D or
 * 1D with size=1.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceFillTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* value, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceFillTensor is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspacSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnFillTensorGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceFillTensor(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_FILL_TENSOR_H_