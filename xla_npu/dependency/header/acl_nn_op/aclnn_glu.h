/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GLU_H_
#define OP_API_INC_GLU_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGlu calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: GLU is a gated linear unit function that divides the input tensor equally into two tensors along the specified dimension dim,
 * And multiply the results of the Sigmoid function output of the front part tensor and the back part tensor element-wise.
 *
 * $$
 * GLU(a,b)=a \otimes \sigma(b)
 * $$
 *
 * a represents the first part of the tensor after the input tensor is equally divided according to the specified dim, and b represents the second half of the tensor.
 *
 * Calculation graph:
 * 1. When the dtype of self is equal to the dtype of out:
 * ```mermaid
 * graph LR
 *    A[(self)] -->B([l0op::Contiguous])
 *    B -->D([l0op::SplitV])--a--> C3
 *    E((dim)) -->D--b-->D1([l0op::Sigmoid])-->C3([l0op::Mul])
 *    C3 -->F1([l0op::ViewCopy])--> J[(out)]
 * ```
 *
 * 2. When the dtype of self is not equal to the dtype of out:
 * ```mermaid
 * graph LR
 *    A[(self)] -->B([l0op::Contiguous])
 *    B -->D([l0op::SplitV])--a--> C3
 *    E((dim)) -->D--b-->D1([l0op::Sigmoid])-->C3([l0op::Mul])
 *    C3 -->F0([l0op::Cast])-->F1([l0op::ViewCopy])--> J[(out)]
 * ```
 *
 * @param [in] self: The data type supports DOUBLE, FLOAT, BFLOAT16, and FLOAT16 data types. The dimension of tensor must be greater than 0.
 * And the shape must be divisible by 2 in the dimension corresponding to the input parameter dim. The shape is expressed as $(*_1,N,*_2)$, where $*$ represents any number of additional dimensions, and $N$ represents the dimension size specified by dim.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] dim: Indicates the dimension of the input self to be split. The data type supports INT64, and the value range is [-self.dim, self.dim-1].
 * @param [out] out: The data type supports DOUBLE, FLOAT, BFLOAT16, and FLOAT16 data types. The data type must be obtainable by self cast.
 * The shape is $(*_1,M,*_2)$, where $*$ represents the corresponding dimension in self, $M = N /2$, supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGluGetWorkspaceSize(const aclTensor* self,
                                               int64_t dim,
                                               const aclTensor* out,
                                               uint64_t* workspaceSize,
                                               aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnGlu is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGtTensorGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGlu(void* workspace, uint64_t workspaceSize,
                               aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GLU_H_
