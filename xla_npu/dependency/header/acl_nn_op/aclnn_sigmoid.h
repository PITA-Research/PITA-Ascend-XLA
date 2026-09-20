/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_SIGMOID_H_
#define OP_API_INC_SIGMOID_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnSigmoid calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Complete the sigmoid operation on the input Tensor
 * @param [in] self: aclTensor on the NPU device side, the data type supports floating point type, the shape is non-empty, supports non-continuous Tensor, and the data format supports ND.
 * Supports non-contiguous Tensors.
 * @param [in] out: aclTensor on the NPU device side, the data type supports floating point type, the shape is the same as self, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnSigmoidGetWorkspaceSize(const aclTensor* self,
                                                   aclTensor* out,
                                                   uint64_t* workspaceSize,
                                                   aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnSigmoid is used to perform calculations
 *
 * Operator function: Complete the sigmoid operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSigmoidGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSigmoid(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor,
                                   const aclrtStream stream);

/**
 * @brief The first interface of aclnnInplaceSigmoid calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Complete the sigmoid operation on the input Tensor
 * @param [in] self: aclTensor on NPU device side,
 * The data type supports floating point type, the shape is non-empty, supports non-continuous Tensor, and the data format supports ND, NCHW,
 * NHWC, supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnInplaceSigmoidGetWorkspaceSize(
    aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnInplaceSigmoid is used to perform calculations
 *
 * Operator function: Complete the sigmoid operation on the input Tensor in place
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSigmoidGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceSigmoid(void* workspace,
                                          uint64_t workspaceSize,
                                          aclOpExecutor* executor,
                                          const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SIGMOID_H_