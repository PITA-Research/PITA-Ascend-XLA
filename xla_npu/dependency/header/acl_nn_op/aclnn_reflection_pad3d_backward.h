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
#ifndef OP_API_INC_REFLECTION_PAD3D_BACKWARD_H_
#define OP_API_INC_REFLECTION_PAD3D_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnReflectionPad3dBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: Backpropagation of reflection filling.
 * @param [in] gradOutput: aclTensor on the NPU device side, data type supports FLOAT16, FLOAT32, DOUBLE, COMPLEX64,
 * COMPLEX128, data format supports ND,
 * The dimension supports four or five dimensions and is consistent with self and gradInput. The shape needs to be consistent with the output of reflection_pad3d forward propagation.
 * @param [in] self: aclTensor on NPU device side,
 * The data type is consistent with gradOutput, the data format supports ND, the dimension supports four or five dimensions and is consistent with gradOutput and gradInput, and the shape is consistent with gradInput.
 * @param [in] padding: aclIntArray array on the NPU device side. The data type is INT64 and the length is 6. The values ​​represent the values ​​that need to be filled in the left, right, up, down, and back.
 * The first two values must be smaller than the value of the last dimension of self, and the middle two values must be smaller than the value of the penultimate dimension of self.
 * The last two values ​​​​need to be smaller than the value of the third dimension from the last to self.
 * @param [in] gradInput: aclTensor on the NPU device side, the data type is consistent with gradOutput, the shape is consistent with self, and the data format supports ND.
 * The dimension supports four or five dimensions and is consistent with gradOutput and self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnReflectionPad3dBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self,
    const aclIntArray* padding, aclTensor* gradInput, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnReflectionPad3dBackward is used to perform calculations
 *
 * Operator function: Backpropagation of reflection filling.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnReflectionPad3dBackwardGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnReflectionPad3dBackward(void* workspace,
                                                   uint64_t workspaceSize,
                                                   aclOpExecutor* executor,
                                                   const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_REFLECTION_PAD3D_BACKWARD_H_