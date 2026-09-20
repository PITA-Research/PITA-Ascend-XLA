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

#ifndef OP_API_INC_LEVEL2_ACLNN_SPLIT_WITH_SIZE_H_
#define OP_API_INC_LEVEL2_ACLNN_SPLIT_WITH_SIZE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnSplitWithSize calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, BFLOAT16, INT32, INT64, INT16, INT8, UINT8,
 * BOOL, COMPLEX128 and COMPLEX64. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] splitSize: aclIntArray on the host side,
 * Indicates the size of each block that needs to be split. The data type supports INT64 and INT32. The sum of the sizes of all blocks needs to be equal to the shape size of self in the dim dimension.
 * @param [in] dim: Integer on the host side, the data type is INT64, indicating the dimension of the input tensor to be split.
 * @param [in] out: aclTensorList on the NPU device side, indicating the list of output tensors after splitting. The data type supports FLOAT, FLOAT16, DOUBLE,
 * BFLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL, COMPLEX128 and COMPLEX64. Yes, the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSplitWithSizeGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* splitSize, int64_t dim,
    aclTensorList* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnSplitWithSize is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSplitWithSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSplitWithSize(void* workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_SPLIT_WITH_SIZE_H_
