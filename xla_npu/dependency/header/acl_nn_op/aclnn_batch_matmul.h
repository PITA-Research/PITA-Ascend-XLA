/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_ACLNN_BATCHMATMUL_H
#define OP_API_ACLNN_BATCHMATMUL_H

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnBatchMatMul calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnBatchMatMulGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat2, aclTensor* out,
    int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnBatchMatMul is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnBatchMatMul(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream);

/**
 * The first interface of @brief aclnnBatchMatMulWeightNz calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Compared to aclnnBatchMatMul, mat2 is in NZ format.
 *
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports FLOAT16 and BFLOAT16 types, and the data type of mat2 needs to meet the data type derivation rules.
 * Shape needs to satisfy the bmm input constraint relationship with mat2. Supports non-continuous Tensor, supports empty Tensor input, and supports ND data format.
 * @param [in] mat2: npu
 * The data type of aclTensor on the device side supports FLOAT16 and BFLOAT16 types, and the data type needs to meet the data type derivation rules of self.
 * Shape needs to satisfy the bmm input constraint relationship with self. Supports non-continuous Tensor, supports empty Tensor input, and supports NZ data format.
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports FLOAT16 and BFLOAT16 types, and the data type needs to be consistent with self, and the shape needs to be consistent with the last two dimensions of self@mat2.
 * Supports non-continuous Tensor, supports empty Tensor input, and supports ND data format.
 * @param [in] cubeMathType:
 * An enumeration value of type INT8, used to determine which calculation logic the Cube unit should use for calculations. Functions such as HFLOAT32 can be enabled through this switch.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBatchMatMulWeightNzGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat2, aclTensor* out,
    int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * The second interface of @brief aclnnBatchMatMulWeightNz is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnBatchMatMulWeightNzGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnBatchMatMulWeightNz(void* workspace,
                                               uint64_t workspaceSize,
                                               aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_ACLNN_BATCHMATMUL_H