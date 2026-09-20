/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_SOFTPLUS_H_
#define OP_API_INC_SOFTPLUS_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnSoftplus calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: activation function softplus
 * Calculation formula:
 * $$
 * Softplus(x) = \begin{cases}
 * \frac{1}{\beta} \log(1+\exp(\beta x)), \beta *x \le threshold \\
 * x, \beta *x > threshold
 * \end{cases}
 * $$
 *
 * @param [in] self: npu
 * For aclTensor on the device side, the data type supports floating point type and non-continuous Tensor, and the data format supports ND, NCHW, NHWC, HWCN, NDHWC, and NCDHW.
 * @param [in] beta: aclScalar on the host side, the data type needs to be convertible to the data type of self.
 * @param [in] threshold: aclScalar on the host side, the data type needs to be convertible to the data type of self.
 * @param [in] out: npu
 * For aclTensor on the device side, the data type supports floating point type and non-continuous Tensor, and the data format supports ND, NCHW, NHWC, HWCN, NDHWC, and NCDHW.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSoftplusGetWorkspaceSize(
    const aclTensor* self, const aclScalar* beta, const aclScalar* threshold,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnSoftplus is used to perform calculations.
 *
 * Operator function: activation function softplus
 * Calculation formula:
 * $$
 * Softplus(x) = \begin{cases}
 * \frac{1}{\beta} \log(1+\exp(\beta x)), \beta *x \le threshold \\
 * x, \beta *x > threshold
 * \end{cases}
 * $$
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSoftplusGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSoftplus(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SOFTPLUS_H_