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

#ifndef OP_API_INC_FILL_H_
#define OP_API_INC_FILL_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnInplaceFillScalar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Complete filling the tensor with the specified scalar
 *
 *
 * @param [in] selfRef: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, UINT8, INT8,
 * INT16, INT32, INT64, DOUBLE, COMPLEX64, COMPLEX128, BOOL, BFLOAT16. Support non-continuous Tensor,
 * Data format ND, data dimension does not support more than 8 dimensions.
 * @param [in] value: aclScalar on the host side, the data type needs to be converted to the data type of self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceFillScalarGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* value, uint64_t* workspaceSize,
    aclOpExecutor** executor);
/**
 * @brief The second interface of aclnnInplaceFillScalar is used to perform calculations.
 *
 * Operator function: Complete filling the tensor with the specified scalar
 *
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnInplaceFillScalarGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceFillScalar(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_FILL_H_
