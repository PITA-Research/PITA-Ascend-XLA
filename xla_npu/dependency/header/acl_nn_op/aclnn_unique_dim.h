/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file aclnn_unique_dim.h
 * \brief
 */
#ifndef OP_API_INC_UNIQUE_DIM_H_
#define OP_API_INC_UNIQUE_DIM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnUniqueDim calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: On a certain dim axis, perform a deduplication operation on the input tensor self..
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 *  graph LR
 *   A[(self)] ---> B([l0op::Contiguous])
 *   B ---> F([l0op::UniqueDim])
 *   J((sorted)) ---> F
 *   C((returnInverse)) --->F
 *   E((dim)) --->F
 *   F --> G[(valueOut)] --> F
 *   F --> H[(inverseOut)] -->F
 *   F --> I[(countsOut)] --> F
 * ```
 * @param [in]
 * self: aclTensor on the Device side, data type supports FLOAT, FLOAT16, UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64,
 * INT64, FLOAT64. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] sorted: Indicates whether the returned output result valueOut is sorted.
 * @param [in] returnInverse: Indicates whether to return the position subscript of the corresponding element in valueOut for each element of self on the dim axis. It will be returned when True and not returned when False.
 * @param [in] dim: Integer type on the Host side, specifying the dimension for deduplication operation. The data type supports INT64, and the value range is [-self.dim(), self.dim()).
 * @param [in] valueOut:
 * The first output tensor represents the deduplication result, aclTensor on the Device side, the data type supports FLOAT, FLOAT16, UINT8, INT8, UINT16,
 * INT16, UINT32, INT32, UINT64, INT64, FLOAT64. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] inverseOut: The second output tensor represents the position subscript of each element of self on the dim axis in the corresponding element in valueOut. The data type supports INT64.
 * @param [in] countsOut: The third output tensor represents the number of times each element in valueOut appears in self. The data type supports INT64.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnUniqueDimGetWorkspaceSize(
    const aclTensor* self, bool sorted, bool returnInverse, int64_t dim,
    aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnUniqueDim is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnUniqueDim.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnUniqueDim(void* workspace, uint64_t workspaceSize,
                                     aclOpExecutor* executor,
                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNIQUE_DIM_H_