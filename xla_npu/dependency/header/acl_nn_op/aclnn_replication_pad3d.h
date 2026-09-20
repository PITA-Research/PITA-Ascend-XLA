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
#ifndef OP_API_INC_REPLICATION_PAD3D_H_
#define OP_API_INC_REPLICATION_PAD3D_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnReplicationPad3d calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: 3D boundary filling.
 * @param [in] self: data type supports FLOAT16, FLOAT32, DOUBLE, INT8, INT16, INT32, INT64, UINT8,
 * COMPLEX64, COMPLEX128, support non-continuous Tensor, the data format supports ND, the dimension supports four or five dimensions, and pad is used in the last three dimensions.
 * @param [in] padding: The data type is INT64, the length is 6, and the values ​​represent the values ​​to be padded left, right, up, down, and back.
 * @param [in] out: The data type, data format, and dimensions are consistent with self. The value of the third-to-last dimension is equal to the third-to-last dimension of self.
 * The value plus the last two values of padding, the value of the penultimate dimension is equal to the value of the penultimate dimension of self plus the two middle values of padding, the value of the last dimension is equal to
 * The value of the last dimension of self plus the first two values ​​of padding. Supports non-contiguous Tensors.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnReplicationPad3dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* padding, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnReplicationPad3d is used to perform calculations
 *
 * Operator function: Fill the input tensor with the input boundary.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnReplicationPad3dGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnReplicationPad3d(void* workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_REPLICATION_PAD3D_H_