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

#ifndef OP_API_INC_GTSCALAR_H_
#define OP_API_INC_GTSCALAR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGt calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: determine whether each element in the input Tensor is greater than other
 * The value of Scalar returns a Bool type Tensor, corresponding to whether the greater than judgment of each position in the input Tensor is established. Calculation formula: $$ out_{i}= (self_i >
 * other) ? True : False $$
 *
 * @param [in] self: input parameter to be calculated by gt, aclTensor on NPU device side,
 * The data types support FLOAT16, FLOAT32, INT32, INT64, INT8, UINT8, DOUBLE, UINT16, UINT32, UINT64, BOOL, and BFLOAT16. The data format supports ND and non-continuous Tensor.
 * @param [in] other: input parameter to be calculated by gt, aclScalar.
 * @param [in] out: aclTensor on the NPU device side, outputs a Tensor whose data type is BOOL, and the data format supports ND.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGtScalarGetWorkspaceSize(const aclTensor* self,
                                                    const aclScalar* other,
                                                    aclTensor* out,
                                                    uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnGtScalar is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGtScalarGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGtScalar(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

/**
 * @brief The first interface of aclnnInplaceGtScalar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Determine whether each element in the input Tensor is greater than the value of other Scalar
 * Calculation formula: $$ selfRef_{i} = (selfRef_{i} > other_{i})? True: False $$
 *
 * @param [in] selfRef: input parameter to be performed for gt local calculation, aclTensor on NPU device side,
 * The data types support FLOAT16, FLOAT32, INT32, INT64, INT8, UINT8, DOUBLE, UINT16, UINT32, UINT64, BOOL, and BFLOAT16. The data format supports ND and non-continuous Tensor.
 * @param [in] other: input parameter to be calculated by gt, aclScalar.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceGtScalarGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceGtScalar is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceGtScalarGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceGtScalar(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GTTENSOR_H_
