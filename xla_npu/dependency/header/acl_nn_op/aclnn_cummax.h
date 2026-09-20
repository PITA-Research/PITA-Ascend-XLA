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

#ifndef OP_API_INC_LEVEL2_ACLNN_CUMMAX_H_
#define OP_API_INC_LEVEL2_ACLNN_CUMMAX_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: Calculate the cumulative maximum value in self and return the value and its corresponding index
 * Calculation formula:
 * $self_{i}$ is an element in the input tensor self from the perspective of dimension dim (the subscripts of other dimensions remain unchanged, only the dim dimension subscripts increase sequentially),
 * $$
 * valuesOut_{i} = max(self_{1}, self_{2}, self_{3}, ......, self_{i})
 * $$
 * $$
 * indicesOut_{i} = argmax(self_{1}, self_{2}, self_{3}, ......, self_{i})
 * $$
 */

/**
 * The first interface of @brief aclnnCummax calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] self: aclTensor on the NPU device side, data type supports FLOAT, DOUBLE, UINT8, INT8, INT16, INT32, INT64,
 * FLOAT16, BFLOAT16, BOOL, the data type needs to be convertible to out data type, supports non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in] dim: Integer on the host side, the data type supports INT64.
 * @param [in] valuesOut: aclTensor on the NPU device side, data type supports FLOAT, DOUBLE, UINT8, INT8, INT16, INT32, INT64,
 * FLOAT16, BFLOAT16, BOOL, support non-continuous Tensor, the data format supports ND, the data dimension does not support more than 8 dimensions, and the shape must be consistent with self.
 * @param [in] indicesOut: aclTensor on the NPU device side, the data type supports INT32 and INT64. Supports non-continuous Tensor, and the data format supports ND.
 * The data dimension does not support more than 8 dimensions, and the shape must be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCummaxGetWorkspaceSize(
    const aclTensor* self, int64_t dim, aclTensor* valuesOut,
    aclTensor* indicesOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnCummax is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnCummaxGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCummax(void* workspace, uint64_t workspaceSize,
                                  aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_CUMMAX_H_
