/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_INDEX_PUT_H_
#define OP_API_INC_INDEX_PUT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnIndexPutImpl calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] selfRef: aclTensor on NPU device side,
 * The data types support FLOAT, FLOAT16, INT64, INT32, INT16, INT8, UINT8, BOOL, and DOUBLE, support non-continuous Tensor, and the data format supports ND.
 * @param [in] indices: aclTensorList on the NPU device side, indicating index slicing.
 * @param [in] values: aclTensor on the NPU device side, the value updated at the index.
 * @param [in] accumulate: bool type attribute, True means superimposing the value at the index, False means replacing the value at the index.
 * @param [in] unsafe: bool type attribute.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnIndexPutImplGetWorkspaceSize(
    aclTensor* selfRef, const aclTensorList* indices, const aclTensor* values,
    const bool accumulate, const bool unsafe, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnIndexPutImpl is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aaclnnIndexPutImplGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnIndexPutImpl(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ADD_H_
