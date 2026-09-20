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

#ifndef OP_API_INC_COPY_H_
#define OP_API_INC_COPY_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnInplaceCopy calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Copy the elements in src to the selfRef tensor and return selfRef.
 *
 * @param [in] selfRef: aclTensor on the NPU device side, data type supports int8, int16, int32, int64, uint8, float16, float32, bool,
 * double, complex64, complex128, uint16, uint32, uint64. Support [non-continuous Tensor](#), the data format supports ND
 * @param [in] src: aclTensor on NPU device side, data type supports int8, int16, int32, int64, uint8, float16, float32, bool,
 * double, complex64, complex128, uint16, uint32, uint64. Support [non-continuous Tensor](#), the data format supports ND
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceCopyGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* src, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceCopy is used to perform calculations.
 *
 * Operator function: Copy the elements in src to the selfRef tensor and return selfRef.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceZeroGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceCopy(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_CONVOLUTION_H_
