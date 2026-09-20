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
#ifndef OP_API_INC_CIRCULAR_PAD3D_H_
#define OP_API_INC_CIRCULAR_PAD3D_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnCircularPad3d calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @domain aclnn_ops_train
 *
 * Operator function: Fill the input tensor using a loop of input boundaries.
 * @param [in] self: aclTensor on the NPU device side, data type supports BFLOAT16, FLOAT16, FLOAT32, INT8,
 * INT32, the data format supports ND, and the dimension supports four or five dimensions.
 * @param [in] padding: aclIntArray array on the NPU device side. The data type is INT64 and the length is 6. The values ​​represent the values ​​that need to be filled in the left, right, up, down, and back.
 * The first two values need to be smaller than the value of the last dimension of self, the middle two values need to be smaller than the value of the second to last dimension of self, and the last two values need to be smaller than the value of the third to last dimension of self.
 * @param [in] out: aclTensor on NPU device side,
 * The data type, data format, and dimensions are consistent with self. The value of the third-to-last dimension is equal to the third-to-last dimension of self.
 * The two values after adding padding to the value. The value in the second to last dimension is equal to the second to last dimension of self.
 * The value plus the two middle values of padding, the value of the last dimension is equal to the value of the last dimension of self plus the first two values of padding.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCircularPad3dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* padding, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnCircularPad3d, used to perform calculations
 *
 * Operator function: Fills the input tensor using a loop of input boundaries.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnCircularPad3dGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCircularPad3d(void* workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_CIRCULAR_PAD_H_