/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_BADDBMM_H_
#define OP_API_INC_BADDBMM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnBaddbmm calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: complete addition calculation
 * Calculation formula: Calculate the product of α and the matrix multiplication results of batch1 and batch2, and then sum it with the product of β and self
 * $$ out = βself+α(batch1@batch2) $$
 *
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, and BFLOAT16 types, and needs to be consistent with batch1@batch2; the shape needs to satisfy the broadcast relationship with batch1@batch2.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] batch1: npu
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, and BFLOAT16 types, and the data type needs to be consistent with batch2, and the shape needs to meet the bmm input constraint relationship with batch2.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] batch2: npu
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, and BFLOAT16 types, and the data type needs to be consistent with batch1, and the shape needs to meet the bmm input constraint relationship with batch1.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] beta: aclScalar on the host side, default is 1
 * @param [in] alpha: aclScalar on the host side, default is 1
 * @param [in] cubeMathType:
 * An enumeration value of type INT8, used to determine which calculation logic the Cube unit should use for calculations. Functions such as HFLOAT32 can be enabled through this switch.
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, and BFLOAT16 types. Both dtype and format need to be consistent with self and batch1@batch2.
 * Supports non-continuous Tensor, and the data format supports ND. The output shape must be consistent with the shape of batch1@batch2.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBaddbmmGetWorkspaceSize(
    const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2,
    const aclScalar* beta, const aclScalar* alpha, aclTensor* out,
    int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnBaddbmm is used to perform calculations.
 *
 * Operator function: complete addition calculation
 * Calculation formula: Calculate the product of α and the matrix multiplication results of batch1 and batch2, and then sum it with the product of β and self
 * $$ out = βself+α(batch1@batch2) $$
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnBaddbmmGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBaddbmm(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor, aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceBaddbmm calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: complete addition calculation
 * Calculation formula: Calculate the product of α and the matrix multiplication results of batch1 and batch2, and then sum it with the product of β and self
 * $$ out = βself+α(batch1@batch2) $$
 *
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, and BFLOAT16 types, and needs to be consistent with batch1@batch2; the shape needs to satisfy the broadcast relationship with batch1@batch2.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] batch1: npu
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, and BFLOAT16 types, and the data type needs to be consistent with batch2, and the shape needs to meet the bmm input constraint relationship with batch2.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] batch2: npu
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, and BFLOAT16 types, and the data type needs to be consistent with batch1, and the shape needs to meet the bmm input constraint relationship with batch1.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] beta: aclScalar on the host side, default is 1
 * @param [in] alpha: aclScalar on the host side, default is 1
 * @param [in] cubeMathType:
 * An enumeration value of type INT8, used to determine which calculation logic the Cube unit should use for calculations. Functions such as HFLOAT32 can be enabled through this switch.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceBaddbmmGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* batch1, const aclTensor* batch2,
    const aclScalar* beta, const aclScalar* alpha, int8_t cubeMathType,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceBaddbmm's second interface is used to perform calculations.
 *
 * Operator function: complete addition calculation
 * Calculation formula: Calculate the product of α and the matrix multiplication results of batch1 and batch2, and then sum it with the product of β and self
 * $$ out = βself+α(batch1@batch2) $$
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceBaddbmmGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceBaddbmm(void* workspace,
                                          uint64_t workspaceSize,
                                          aclOpExecutor* executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_BADDBMM_H_
