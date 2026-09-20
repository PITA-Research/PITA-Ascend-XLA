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
#ifndef OP_API_INC_REFLECTION_PAD1D_H_
#define OP_API_INC_REFLECTION_PAD1D_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnReflectionPad1d calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: fills the input tensor with reflection from the input boundary.
 * @param [in] self: aclTensor on the NPU device side, data type supports BFLOAT16, FLOAT16, FLOAT32, DOUBLE, INT8, INT16,
 * INT32, INT64, UINT8, BOOL, the data format supports ND, and the dimension supports two or three dimensions.
 * @param [in] padding: aclIntArray array on the NPU device side,
 * The data type is INT64 and the length is 2. The two values represent the values that need to be filled on the left and right sides in turn, and both of them need to be smaller than the value of the last dimension of self.
 * @param [in] out: aclTensor on NPU device side,
 * The data type, data format, and dimensions are consistent with self. The value of the last dimension is equal to the value of the last dimension of self plus the first two padding values.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnReflectionPad1dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* padding, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnReflectionPad1d, used to perform calculations
 *
 * Operator function: Fills the input tensor with reflection from the input boundary.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnReflectionPad1dGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnReflectionPad1d(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_REFLECTION_PAD1D_H_