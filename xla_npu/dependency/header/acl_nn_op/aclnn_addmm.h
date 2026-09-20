/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ADDMM_H_
#define OP_API_INC_ADDMM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnAddmm calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnAddmmGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat1, const aclTensor* mat2,
    const aclScalar* beta, const aclScalar* alpha, aclTensor* out,
    int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnInplaceAddmm calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnInplaceAddmmGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* mat1, const aclTensor* mat2,
    const aclScalar* beta, const aclScalar* alpha, int8_t cubeMathType,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnAddmm is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnAddmm(void* workspace, uint64_t workspaceSize,
                                 aclOpExecutor* executor,
                                 const aclrtStream stream);

/**
 * The second interface of @brief aclnnInplaceAddmm is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnInplaceAddmm(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        const aclrtStream stream);

/**
 * The first interface of @brief aclnnAddmmWeightNz calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: Compared to aclnnAddmm, mat2 is in NZ format.
 * @param [in] self: bias matrix, data type support: float16, bfloat16, float32, format only supports ND.
 * @param [in] mat1: matmul left matrix, data type support: float16, bfloat16, format only supports ND.
 * @param [in] mat2: matmul right matrix, data type support: float16, bfloat16, supports AI processor-friendly data layout format (NZ).
 * @param [in] beta: self matrix coefficient, data type support: FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8.
 * @param [in] alpha: (mat1 @ mat2) matrix calculation result coefficient, data type support: FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8.
 * @param [in] cubeMathType: used to specify the calculation logic of the Cube unit, the type is the integer int8_t on the Host side.
 * @param [out] out: calculation result, data type: float16, bfloat16, float32.
 * @param [out] workspaceSize: Returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 */
ACLNN_API aclnnStatus aclnnAddmmWeightNzGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mat1, const aclTensor* mat2,
    const aclScalar* beta, const aclScalar* alpha, aclTensor* out,
    int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnAddmmWeightNz is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAddmmWeightNzGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnAddmmWeightNz(void* workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         const aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ADD_H_
