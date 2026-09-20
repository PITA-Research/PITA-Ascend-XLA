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

#ifndef OP_API_INC_SORT_H_
#define OP_API_INC_SORT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnSort calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Complete the sort operation on the input Tensor
 * @param [in] self: npu
 * The data type of aclTensor on the device side supports FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, and UINT8. Support empty Tensor,
 * Support [non-consecutive Tensor](#).
 * @param [in] stable: Whether the sorting is stable, True means stable sorting, False means unstable sorting, the data type is BOOLEAN.
 * @param [in] dim: The dimension used as the sorting criterion, the data type is INT. The range is [-N, N-1].
 * @param [in] descending: Control the sorting order, True is descending order, False is ascending order, the data type is BOOLEAN.
 * @param [in] valuesOut: aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, and UINT8. support
 * Empty Tensor, supports [non-continuous Tensor](#). The data format supports ND([Reference](#)). The shape and data format need to be consistent with self.
 * @param [in] indicesOut: aclTensor on the NPU device side,
 * The data type supports INT64. Supports empty Tensor and [non-continuous Tensor](#). The data format supports ND([Reference](#)). The shape and data format need to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnSortGetWorkspaceSize(
    const aclTensor* self, bool stable, int64_t dim, bool descending,
    aclTensor* valuesOut, aclTensor* indicesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnSort is used to perform calculations
 *
 * Operator function: Complete the sort operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSortGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSort(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor,
                                const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
