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

#ifndef OP_API_INC_LEVEL2_ACLNN_COMPLEX_TENSOR_H_
#define OP_API_INC_LEVEL2_ACLNN_COMPLEX_TENSOR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnComplex calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] real: aclTensor on NPU device side,
 * The data type supports FLOAT16, FLOAT data type,
 * Shape needs to satisfy the broadcast relationship with imag, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] imag: aclTensor on NPU device side,
 * The data type supports FLOAT16, FLOAT data type,
 * Shape needs to satisfy the broadcast relationship with real, supports non-continuous Tensor, the data format supports ND, and the data type needs to be the same as real.
 * @param [in] out: aclTensor on the NPU device side, the data format supports ND.
 * @param [in] Tout: int on the host side, indicating
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnComplexGetWorkspaceSize(const aclTensor* real,
                                                   const aclTensor* imag,
                                                   aclTensor* out,
                                                   uint64_t* workspaceSize,
                                                   aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnComplex is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnComplexGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnComplex(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_COMPLEX_TENSOR_H_