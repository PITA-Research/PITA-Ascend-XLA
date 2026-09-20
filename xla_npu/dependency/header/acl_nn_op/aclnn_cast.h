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

#ifndef OP_API_INC_CAST_H_
#define OP_API_INC_CAST_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnCast calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Convert the input tensor to the specified dtype.
 *
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports FLOAT16, FLOAT, FlOAT64, INT8, UINT8, INT16, INT32, INT64, and BOOL.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] dtype: aclDataType on the host side, enter the target dtype to be converted by tensor.
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports FLOAT16, FLOAT, FlOAT64, INT8, UINT8, INT16, INT32, INT64, BOOL,
 * COMPLEX64, COMPLEX128. The data type is dtype, the shape is the same as self, the data format supports ND, and the data format needs to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCastGetWorkspaceSize(const aclTensor* self,
                                                const aclDataType dtype,
                                                aclTensor* out,
                                                uint64_t* workspaceSize,
                                                aclOpExecutor** executor);
/**
 * @brief The second interface of aclnnCast is used to perform calculations.
 *
 * Operator function: Convert the input tensor to the specified dtype.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnCastGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCast(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_CAST_H_
