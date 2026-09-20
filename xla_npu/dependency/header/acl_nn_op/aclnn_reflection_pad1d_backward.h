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
#ifndef OP_API_INC_REFLECTION_PAD1D_BACKWARD_H_
#define OP_API_INC_REFLECTION_PAD1D_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnReflectionPad1dBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: Backpropagation of reflection_pad1d.
 * @param [in] gradOutput: data type supports FLOAT16, FLOAT32, DOUBLE, COMPLEX64,
 * COMPLEX128, supports [non-continuous Tensor](#), the data format supports ND([reference](#)),
 * The dimension supports two or three dimensions and is consistent with self and gradInput. The shape needs to be consistent with the output of reflection_pad1d forward propagation.
 * @param [in] self:
 * The data type is consistent with gradOutput, supports [non-continuous Tensor] (#), the data format supports ND ([reference] (#)), and the dimension supports two or three dimensions and is consistent with gradOutput and
 * gradInput is consistent, and shape is consistent with gradInput.
 * @param [in] padding: The data type is INT64, the length is 2, and the values ​​represent the left and right values ​​that need to be padded in turn, which must be smaller than the value of the last dimension of self.
 * @param [in] gradInput:
 * The data type is consistent with gradOutput, the shape is consistent with self, supports [non-continuous Tensor](#), and the data format supports ND([Reference](#)).
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnReflectionPad1dBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self,
    const aclIntArray* padding, aclTensor* gradInput, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnReflectionPad1dBackward is used to perform calculations
 *
 * Operator function: Backpropagation of reflection_pad1d.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnReflectionPad1dBackwardGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnReflectionPad1dBackward(void* workspace,
                                                   uint64_t workspaceSize,
                                                   aclOpExecutor* executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_REFLECTION_PAD1D_BACKWARD_H_