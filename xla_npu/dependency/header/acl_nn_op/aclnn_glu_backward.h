/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GLU_BACKWARD_H_
#define OP_API_INC_GLU_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGluBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * Operator function: reverse of GLU.
 *
 * $$
 * \frac{\partial GLU(a,b)}{\partial(a,b)}=cat(\sigma(b),\sigma(b) \otimes a \otimes (1-\sigma(b)))
 * $$
 *
 * Mathematical calculation expression:
 * Assume that the output GLUGrad consists of two parts: out=[a_grad, b_grad], then:
 * sig_b = sigmoid(b)
 * **a_grad** = y_grad * sig_b
 * **b_grad** = a_grad * (a - a * sig_b)
 * Among them: y_grad is gradOut, a represents the first part of the tensor after the input tensor is evenly divided according to the specified dim, and b represents the second half of the tensor.
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *    A0[(gradOut)] -->B0([l0op::Contiguous])-->C1([l0op::Mul])-->C2([l0op::Mul])
 *    A1[(self)] -->B1([l0op::Contiguous])
 *    B1 -->D0([l0op::SplitV])--a--> C0-->G0([l0op::Sub])
 *    D0--a-->G0-->C2--b_grad-->H0([l0op::ConcatD])
 *    E0((dim)) -->D0--b-->D1([l0op::Sigmoid])-->C0([l0op::Mul])
 *    D1-->C1--a_grad-->H0
 *    E0-->H0
 *    H0 -->F0([l0op::ViewCopy])--> J0[(out)]
 * ```
 *
 * @param [in] gradOut: represents the gradient update coefficient. The data type supports DOUBLE, FLOAT, and FLOAT16 data types. The data type must be consistent with the data type of self.
 * The shape is $(*_1,M,*_2)$, where $*$ represents the corresponding dimension in self, $M = N /2$, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] self:
 * The data type supports DOUBLE, FLOAT, and FLOAT16 data types. The dimension of the tensor must be greater than 0, and the shape must be divisible by 2 in the dimension corresponding to the input parameter dim.
 * Shape is expressed as $(*_1,N,*_2)$, where $*$ represents any number of additional dimensions, $N$ represents the dimension size specified by dim, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] dim: Indicates the dimension of the input self to be split. The data type supports INT64, and the value range is [-self.dim, self.dim-1].
 * @param [out] out: The data type supports DOUBLE, FLOAT, and FLOAT16 data types. The data type must be consistent with the data type of self.
 * The shape must be consistent with the shape of self, supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGluBackwardGetWorkspaceSize(
    const aclTensor* gradOut, const aclTensor* self, int64_t dim,
    const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnGluBackward is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGtTensorGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGluBackward(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GLU_BACKWARD_H_
