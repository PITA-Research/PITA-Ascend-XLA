/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_THRESHOLD_H_
#define OP_API_INC_THRESHOLD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnThreshold calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: perform threshold operation on input self. When the element in self is greater than threshold, return the element itself; otherwise, return value.
 * @param [in] self: aclTensor on the NPU device side, supporting data types FLOAT, FLOAT16, INT32, INT8, UINT8, INT16, INT64.
 * Support [non-continuous Tensor](), data format supports ND[reference](#)
 * @param [in] threshold: aclScalar on the NPU device side, supports data types FLOAT, FLOAT16, INT32, INT8, UINT8, INT16, INT64.
 * Represents the threshold for judging the size of the input self element.
 * @param [in] value: aclScalar on the NPU device side, supporting data types FLOAT, FLOAT16, INT32, INT8, UINT8, INT16, and INT64.
 * Indicates the return value when the elements of the input self are less than the threshold.
 * @param [out] out: aclTensor on the NPU device side, supports data types FLOAT, FLOAT16, INT32, INT8, UINT8, INT16, INT64.
 * Supports non-contiguous Tensors. And the data type, data format and shape need to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnThresholdGetWorkspaceSize(
    const aclTensor* self, const aclScalar* threshold, const aclScalar* value,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * The second interface of @brief aclnnThreshold is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnThresholdGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnThreshold(void* workspace, uint64_t workspaceSize,
                                     aclOpExecutor* executor,
                                     aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceThreshold calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: perform threshold operation on input self. When the element in self is greater than threshold, return the element itself; otherwise, return value.
 * @param [in] selfRef: aclTensor on the NPU device side, supports data types FLOAT, FLOAT16, INT32, INT8, UINT8, INT16, INT64.
 * Support [non-continuous Tensor](), data format supports ND[reference](#)
 * @param [in] threshold: aclScalar on the NPU device side, supports data types FLOAT, FLOAT16, INT32, INT8, UINT8, INT16, INT64.
 * Represents the threshold for judging the size of the input self element.
 * @param [in] value: aclScalar on the NPU device side, supporting data types FLOAT, FLOAT16, INT32, INT8, UINT8, INT16, and INT64.
 * Indicates the return value when the elements of the input self are less than the threshold.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceThresholdGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* threshold, const aclScalar* value,
    uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * The second interface of @brief aclnnInplaceThreshold is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceThresholdGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceThreshold(void* workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_THRESHOLD_H_
