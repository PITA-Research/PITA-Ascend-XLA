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
#ifndef OP_API_INC_LEVEL2_ACLNN_CLAMP_H_
#define OP_API_INC_LEVEL2_ACLNN_CLAMP_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnClamp calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnClampGetWorkspaceSize(const aclTensor* self,
                                                 const aclScalar* clipValueMin,
                                                 const aclScalar* clipValueMax,
                                                 aclTensor* out,
                                                 uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnClamp is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnClamp(void* workspace, uint64_t workspaceSize,
                                 aclOpExecutor* executor,
                                 const aclrtStream stream);

/**
 * The first interface of @brief aclnnClampMin calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnClampMinGetWorkspaceSize(
    const aclTensor* self, const aclScalar* clipValueMin, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnClampMin is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnClampMin(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    const aclrtStream stream);

/**
 * The first interface of @brief aclnnClampTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnClampTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* clipValueMin,
    const aclTensor* clipValueMax, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnClampMinTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnClampMinTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* clipValueMin, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnClampMinTensor is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnClampMinTensor(void* workspace,
                                          uint64_t workspaceSize,
                                          aclOpExecutor* executor,
                                          const aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceClampMinTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnInplaceClampMinTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* clipValueMin, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnInplaceClampMinTensor's second interface is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnInplaceClampMinTensor(void* workspace,
                                                 uint64_t workspaceSize,
                                                 aclOpExecutor* executor,
                                                 const aclrtStream stream);

/**
 * @brief The second interface of aclnnClampTensor is used to perform calculations.
 */
ACLNN_API aclnnStatus aclnnClampTensor(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       const aclrtStream stream);

/**
 * The first interface of @brief aclnnClampMax calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Limit all input elements to the range [-inf,max].
 * Calculation formula:
 * $$ {y}_{i} = min({{x}_{i}},max) $$
 *
 * @param [in] self: Input tensor, data type supports FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, INT64.
 * Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [in] clipValueMax: upper bound, the data type needs to be convertible to self.
 * @param [in] out: output, data type supports FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, INT64,
 * The data type is consistent with self, the shape is consistent with self, and the data format supports ND ([Reference](#)).
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnClampMaxGetWorkspaceSize(
    const aclTensor* self, const aclScalar* clipValueMax, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnClampMax is used to perform calculations.
 *
 * Operator function: Limit all input elements to the range [-inf,max].
 * Calculation formula:
 * $$ {y}_{i} = min({{x}_{i}},max) $$
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnClampMaxGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnClampMax(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceClampMax calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Limit all input elements to the range [-inf,max].
 * Calculation formula:
 * $$ {y}_{i} = min({{x}_{i}},max) $$
 *
 * @param [in] selfRef: input tensor, data type supports FLOAT16, FLOAT, FLOAT64, INT8, UINT8, INT16, INT32, INT64.
 * Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [in] clipValueMax: upper bound, the data type needs to be convertible to self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceClampMaxGetWorkspaceSize(
    const aclTensor* selfRef, const aclScalar* clipValueMax,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceClampMax is used to perform calculations.
 *
 * Operator function: Limit all input elements to the range [-inf,max].
 * Calculation formula:
 * $$ {y}_{i} = min({{x}_{i}},max) $$
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceClampMaxGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceClampMax(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

/**
 * The first interface of @brief aclnnClampMaxTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Limit all input elements to the range [-inf, max].
 *
 * @param [in] self: input tensor, data type supports FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64,
 * The data type needs to meet the data type derivation rules of max, and the shape needs to meet the broadcast relationship with max. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] max: input the upper limit value tensor, the data type supports FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64
 * And the data type needs to satisfy the data type derivation rules with the data type of self, and the shape needs to satisfy the broadcast relationship with max. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] out: Output tensor, data type supports FLOAT16, FLOAT, DOUBLE, INT8, UINT8, INT16, INT32, INT64,
 * And the data type needs to be a convertible data type after the derivation of self and max, and the shape needs to be self and max.
 * The shape after broadcast. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnClampMaxTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* max, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnInplaceClampMaxTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnInplaceClampMaxTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* max, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnClampMaxTensor is used to perform calculations.
 *
 * Operator function: Limit all input elements to the range [-inf, max].
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnClampMaxTensorGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnClampMaxTensor(void* workspace,
                                          uint64_t workspaceSize,
                                          aclOpExecutor* executor,
                                          aclrtStream stream);

ACLNN_API aclnnStatus aclnnInplaceClampMaxTensor(void* workspace,
                                                 uint64_t workspaceSize,
                                                 aclOpExecutor* executor,
                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_CLAMP_H_