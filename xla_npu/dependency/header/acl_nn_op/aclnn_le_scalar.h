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

#ifndef OP_API_INC_LEVEL2_ACLNN_LE_SCALAR_H_
#define OP_API_INC_LEVEL2_ACLNN_LE_SCALAR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnLeScalar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Calculate whether the value of the element in self is less than or equal to the value of other, and write the comparison result of each element of self with the value of other into out
 * @param [in] self: aclTensor on NPU device side,
 * The data type supports INT8, UINT8, INT16, UINT16, INT32, INT64, FLOAT16, FLOAT, and DOUBLE. The data format supports ND and non-continuous Tensor.
 * @param [in] other: aclScalar on the host side, data type supports INT8, UINT8, INT16, UINT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE
 * The data type needs to be convertible to a data type of self
 * @param [in] out: aclTensor on the NPU device side, data type supports INT8, UINT8, INT16, UINT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE
 * The shape is consistent with the shape of self, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLeScalarGetWorkspaceSize(const aclTensor* self,
                                                    const aclScalar* other,
                                                    aclTensor* out,
                                                    uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnLeScalar is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnLeScalarGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLeScalar(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceLeScalarGetWorkspaceSize calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Calculate whether the elements in selfRef Tensor are less than or equal to other
 * Elements in Scalar, return the modified selfRef, selfRef<=other is True, otherwise it is False
 * @param [in] selfRef: aclTensor on NPU device side,
 * Data types support FLOAT16, FLOAT, INT64, INT32, UINT8, BOOL, UINT64, UINT32, DOUBLE, UINT16, BFLOAT16 data types,
 * Shape needs to satisfy the broadcast relationship with other, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] other: aclScalar on the host side, the shape needs to satisfy the broadcast relationship with other.
 * @param [in] out: aclTensor on the NPU device side, outputs a Tensor whose data type is BOOL, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceLeScalarGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnLeScalar is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnLeScalarGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceLeScalar(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_LE_SCALAR_H_
