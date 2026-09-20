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
#ifndef OP_API_INC_AMIN_H_
#define OP_API_INC_AMIN_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnAmin calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Returns the minimum value of each slice of the tensor in the specified dimension.
 *
 * @param [in] self:
 * The data type of aclTensor on the device side supports FLOAT, BFLOAT16, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, and BOOL. The data format supports ND.
 * Supports non-contiguous Tensors.
 * @param [in] dim: aclIntArray on the host side, specifies the dimension to be used for minimum value calculation. The data type supports INT32 and INT64.
 * @param [in] keepDim: Boolean type on the host side, whether to retain the dimensions of the input tensor in the output tensor.
 * @param [in] out: aclTensor on the device side, the data type needs to be the same as self. The data format supports ND. Supports non-contiguous Tensors.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAminGetWorkspaceSize(const aclTensor* self,
                                                const aclIntArray* dim,
                                                bool keepDim, aclTensor* out,
                                                uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnAmin is used to perform calculations.
 *
 * Operator function: Returns the minimum value of each slice of the tensor in the specified dimension.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAminGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAmin(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_AMIN_H_