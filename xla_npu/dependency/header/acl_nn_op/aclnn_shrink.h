/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_SHRINK_H_
#define OP_API_INC_SHRINK_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnShrink calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer、aclnn_ops_train
 * Function description: Perform nonlinear transformation on the input tensor, and scale and offset the input through the offset bias according to the relationship between the input value self and the threshold lambd.
 * Calculation formula: as follows
 * $$
 * out=
 * \begin{cases}
 * x-bias, if x > lambd \\
 * x+bias, if x < -lambd \\
 * 0, otherwise \\
 * \end{cases}
 * $$
 * Parameter description:
 * @param [in]   self
 * Input Tensor, the data type supports FLOAT and FLOAT16. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in]   lambd
 * Enter Scalar, the data type supports FLOAT.
 * @param [in]   bias
 * Enter Scalar, the data type supports FLOAT.
 * @param [in]   out
 * Output Tensor, data type supports FLOAT, FLOAT16. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor returns the op executor, including the operator calculation process.
 * @return aclnnStatus return status code
 */
ACLNN_API aclnnStatus aclnnShrinkGetWorkspaceSize(
    const aclTensor* self, const aclScalar* lambd, const aclScalar* bias,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnShrink is used to perform calculations.
 * Function description: Perform nonlinear transformation on the input tensor, and scale and offset the input through the offset bias according to the relationship between the input value self and the threshold lambd.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnShrinkGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnShrink(void* workspace, uint64_t workspaceSize,
                                  aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  //OP_API_INC_SHRINK_H_