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
#ifndef OP_API_INC_IS_IN_H_
#define OP_API_INC_IS_IN_H_
#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnIsInScalarTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Check whether element is in tensor testElements.
 *
 * @param [in] element: aclScalar on the NPU device side. The data type supports integer and floating point types.
 * @param [in] testElements: aclTensor on the host side, the data type supports integer and floating point types. Supports non-continuous Tensor, and the data format supports ND
 * @param [in] assumeUnique: bool on the host side, indicating whether the value in testElements is unique.
 * @param [in] invert: bool on the host side, indicating whether the output result is inverted.
 * @param [in] out: aclTensor on the NPU device side, the data type supports Boolean. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnIsInScalarTensorGetWorkspaceSize(
    const aclScalar* element, const aclTensor* testElements, bool assumeUnique,
    bool invert, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnIsInScalarTensor is used to perform calculations.
 *
 * Operator function: Check whether element is in tensor testElements.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnIsInScalarTensorGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnIsInScalarTensor(void* workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_IS_IN_H_
