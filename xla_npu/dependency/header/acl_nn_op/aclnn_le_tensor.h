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

#ifndef OP_API_INC_LEVEL2_ACLNN_LE_TENSOR_H_
#define OP_API_INC_LEVEL2_ACLNN_LE_TENSOR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnLeTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Calculate whether the elements in two Tensors are less than or equal to the value of other, and write the comparison result of each element of self with the value of other into out
 * @param [in] self: aclTensor on NPU device side,
 * The data type supports INT8, UINT8, INT16, UINT16, INT32, INT64, FLOAT16, FLOAT, and DOUBLE data types.
 * The data type needs to form a mutual derivation relationship with other, the shape needs to satisfy the broadcast relationship with other, discontinuous Tensors are supported, and the data format supports ND.
 * @param [in] other: aclTensor on NPU device side,
 * The data type supports INT8, UINT8, INT16, UINT16, INT32, INT64, FLOAT16, FLOAT, and DOUBLE data types.
 * The data type needs to form a mutual derivation relationship with other, the shape needs to satisfy the broadcast relationship with other, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] out: aclTensor on NPU device side,
 * The data type supports INT8, UINT8, INT16, UINT16, INT32, INT64, FLOAT16, FLOAT, and DOUBLE data types, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLeTensorGetWorkspaceSize(const aclTensor* self,
                                                    const aclTensor* other,
                                                    aclTensor* out,
                                                    uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnLeTensor is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnLeTensorGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLeTensor(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceLeTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Determine whether each element in the input Tensor is less than or equal to the value of other Tensor
 * Calculation formula: $$ selfRef_{i} = (selfRef_{i} <= other_{i})? True: False $$
 *
 * @param [in] selfRef: input parameter to be calculated locally, aclTensor on NPU device side,
 * The data types support FLOAT16, FLOAT32, INT32, INT64, INT8, UINT8, DOUBLE, UINT16, UINT32, UINT64, BOOL, BFLOAT16, BFLOAT16. The data format supports ND and non-continuous Tensor.
 * @param [in] other: input parameter to be calculated by ge, aclTensor.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceLeTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnInplaceLeTensor's second interface is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceLeTensorGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceLeTensor(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_LE_TENSOR_H_
