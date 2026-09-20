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
#ifndef OP_API_INC_LEVEL2_ACLNN_GESCALAR_H_
#define OP_API_INC_LEVEL2_ACLNN_GESCALAR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGeScalar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Determine whether each element in the input Tensor is greater than or equal to the value of other Scalar, and return a Bool type Tensor.
 * Corresponding to the greater or equal judgment of each position in the input Tensor, whether it is true or not
 * Calculation formula: $$ out_{i}= (self_i >= other)? True: False $$
 *
 * @param [in] self: input parameter to be calculated for ge, aclTensor on NPU device side,
 * The data type supports FLOAT16, FLOAT32, INT32, INT64, INT8, UINT8, DOUBLE, UINT16, UINT32, UINT64, BFLOAT16, and the data format supports ND,
 * Supports non-contiguous Tensors.
 * @param [in] other: input parameter to be calculated by ge, aclScalar.
 * @param [in] out: The output parameter calculated by ge. aclTensor on the NPU device side,
 * Output a Tensor with data type BOOL. The data format supports ND.
 * Supports non-contiguous Tensors.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGeScalarGetWorkspaceSize(const aclTensor* self,
                                                    const aclScalar* other,
                                                    aclTensor* out,
                                                    uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnGeScalar is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGeScalarGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGeScalar(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    const aclrtStream stream);

/**
 * @brief The first interface of aclnnInplaceGeScalar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Determine whether each element in the input Tensor is greater than or equal to the value of other Scalar
 * Calculation formula: $$ selfRef_{i} = (selfRef_{i} >= other_{i})? True: False $$
 *
 * @param [in] selfRef: input parameter to be calculated locally, aclTensor on NPU device side,
 * The data types support FLOAT16, FLOAT32, INT32, INT64, INT8, UINT8, DOUBLE, UINT16, UINT32, UINT64, BOOL, and BFLOAT16. The data format supports ND and non-continuous Tensor.
 * @param [in] other: input parameter to be calculated by ge, aclScalar.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceGeScalarGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceGeScalar is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceGeScalarGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceGeScalar(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_GESCALAR_H_
