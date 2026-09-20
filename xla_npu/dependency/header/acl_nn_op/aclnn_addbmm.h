/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ADDBMM_H_
#define OP_API_INC_ADDBMM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnAddbmm calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: First perform the matrix multiplication calculation of batch1 and batch2, then add the results according to the first dimension (batch dimension) in batches, and add the three-dimensional vector
 * Compress it into a two-dimensional vector (the shape size is the shape of the last two dimensions), then multiply the result with α, and then sum it with the product of β and self to get the result
 * Calculation formula:
 * $$ out = βself+α(\sum_{i=0}^{b-1}batch1_{i}@batch2_{i}) $$
 *
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, and BFLOAT16 types, and the data type of batch1@batch2 must meet the data type derivation rules.
 * Shape needs to satisfy the broadcast relationship with the last two dimensions of batch1@batch2. Supports non-continuous Tensor, supports empty Tensor input, and supports ND data format.
 * @param [in] batch1: npu
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, and BFLOAT16 types, and the data type needs to meet the data type derivation rules with the data type of batch2.
 * Shape needs to satisfy the bmm input constraint relationship with batch2. Supports non-continuous Tensor, supports empty Tensor input, and supports ND data format.
 * @param [in] batch2: npu
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, and BFLOAT16 types, and the data type needs to meet the data type derivation rules of batch1.
 * Shape needs to satisfy the bmm input constraint relationship with batch1. Supports non-continuous Tensor, supports empty Tensor input, and supports ND data format.
 * @param [in] beta: aclScalar on the host side, the data type needs to be convertible into the data type deduced by self and batch1@batch2.
 * @param [in] alpha: aclScalar on the host side, the data type needs to be convertible into the data type deduced by self and batch1@batch2.
 * @param [in] cubeMathType:
 * An enumeration value of type INT8, used to determine which calculation logic the Cube unit should use for calculations. Functions such as HFLOAT32 can be enabled through this switch.
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, and BFLOAT16 types, and the data type needs to be consistent with self, and the shape needs to be consistent with the last two dimensions of batch1@batch2.
 * Supports non-continuous Tensor, supports empty Tensor input, and supports ND data format.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAddbmmGetWorkspaceSize(
    const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2,
    const aclScalar* beta, const aclScalar* alpha, aclTensor* out,
    int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnAddbmm is used to perform calculations.
 *
 * Operator function: First perform the matrix multiplication calculation of batch1 and batch2, then add the results according to the first dimension (batch dimension) in batches, and add the three-dimensional vector
 * Compress it into a two-dimensional vector (the shape size is the shape of the last two dimensions), then multiply the result with α, and then sum it with the product of β and self to get the result
 * Calculation formula:
 * $$ out = βself+α(\sum_{i=0}^{b-1}batch1_{i}@batch2_{i}) $$
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddbmmGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAddbmm(void* workspace, uint64_t workspaceSize,
                                  aclOpExecutor* executor, aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceAddbmm calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: First perform the matrix multiplication calculation of batch1 and batch2, then add the results according to the first dimension (batch dimension) in batches, and add the three-dimensional vector
 * Compress it into a two-dimensional vector (the shape size is the shape of the last two dimensions), then multiply the result with α, and then sum it with the product of β and selfRef to get the result
 * Calculation formula:
 * $$ selfRef = βselfRef+α(\sum_{i=0}^{b-1}batch1_{i}@batch2_{i}) $$
 *
 * @param [in] selfRef: npu
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, and BFLOAT16 types, and the data type of batch1@batch2 must meet the data type derivation rules.
 * Shape needs to satisfy the broadcast relationship with the last two dimensions of batch1@batch2. Supports non-continuous Tensor, supports empty Tensor input, and supports ND data format.
 * @param [in] batch1: npu
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, and BFLOAT16 types, and the data type needs to meet the data type derivation rules with the data type of batch2.
 * Shape needs to satisfy the bmm input constraint relationship with batch2. Supports non-continuous Tensor, supports empty Tensor input, and supports ND data format.
 * @param [in] batch2: npu
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, and BFLOAT16 types, and the data type needs to meet the data type derivation rules of batch1.
 * Shape needs to satisfy the bmm input constraint relationship with batch1. Supports non-continuous Tensor, supports empty Tensor input, and supports ND data format.
 * @param [in] beta: aclScalar on the host side, the data type needs to be convertible into the data type deduced by self and batch1@batch2.
 * @param [in] alpha: aclScalar on the host side, the data type needs to be convertible into the data type deduced by self and batch1@batch2.
 * @param [in] cubeMathType:
 * An enumeration value of type INT8, used to determine which calculation logic the Cube unit should use for calculations. Functions such as HFLOAT32 can be enabled through this switch.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceAddbmmGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* batch1, const aclTensor* batch2,
    const aclScalar* beta, const aclScalar* alpha, int8_t cubeMathType,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceAddbmm is used to perform calculations.
 *
 * Operator function: First perform the matrix multiplication calculation of batch1 and batch2, then add the results according to the first dimension (batch dimension) in batches, and add the three-dimensional vector
 * Compress it into a two-dimensional vector (the shape size is the shape of the last two dimensions), then multiply the result with α, and then sum it with the product of β and selfRef to get the result
 * Calculation formula:
 * $$ selfRef = βselfRef+α(\sum_{i=0}^{b-1}batch1_{i}@batch2_{i}) $$
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceAddbmmGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceAddbmm(void* workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_BADDBMM_H_
