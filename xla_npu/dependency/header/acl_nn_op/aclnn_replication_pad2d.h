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
#ifndef OP_API_INC_REPLICATION_PAD2D_H_
#define OP_API_INC_REPLICATION_PAD2D_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnReplicationPad2d calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: fills the input tensor with the input boundary.
 * @param [in] self: data type supports FLOAT16, FLOAT32, DOUBLE, INT8, INT16, INT32, INT64, UINT8,
 * COMPLEX64, COMPLEX128, support non-continuous Tensor, the data format supports ND, the dimension supports three or four dimensions, and pad is used in the last two dimensions.
 * @param [in] padding: The data type is INT64, the length is 4, and the values ​​represent the left, right, upper and lower values ​​that need to be padded.
 * @param [in] out: The data type, data format, and dimensions are consistent with self. The value of the penultimate dimension is equal to the penultimate dimension of self.
 * The value plus the last two values ​​of padding, the value of the last dimension is equal to the value of the last dimension of self plus the first two values ​​of padding. Supports non-contiguous Tensors.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnReplicationPad2dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* padding, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnReplicationPad2d, used to perform calculations
 *
 * Operator function: Fill the input tensor with the input boundary.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnReplicationPad2dGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnReplicationPad2d(void* workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_REPLICATION_PAD2D_H_