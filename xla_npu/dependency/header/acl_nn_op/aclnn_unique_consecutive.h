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
 * \file aclnn_unique_consecutive.h
 * \brief
 */
#ifndef OP_API_INC_UNIQUE_CONSECUTIVE_H_
#define OP_API_INC_UNIQUE_CONSECUTIVE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnUniqueConsecutive calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Eliminate all elements except the first element from each consecutive group of identical elements.
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 *  graph LR
 *      A[(self)] ---> B([l0op::Contiguous])
 *      B ---> F([l0op::UniqueConsecutive])
 *      C((returnInverse)) --->F
 *      D((returnCounts)) --->F
 *      E((dim)) --->F
 *      F --> G[(valueOut)]
 *      F --> H[(inverseOut)]
 *      F --> I[(countsOut)]
 * ```
 * @param [in] self: data type supports FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32,
 * UINT64, COMPLEX64, COMPLEX128, BOOL, data format supports ND.
 * @param [in] returnInverse: Indicates whether to return the position subscript of the corresponding element in valueOut for each element in self. It will be returned when True and not returned when False.
 * @param [in] returnCounts: Indicates whether to return the number of consecutive repeated occurrences of each element in valueOut in self. It will be returned when True and not returned when False.
 * @param [in] dim: Indicates the dimension for deduplication.
 * @param [in] valueOut: The first output tensor, returns the result after eliminating consecutive repeated elements. The data type supports FLOAT, FLOAT16, DOUBLE, INT8, INT16,
 * INT32, INT64, UINT8, UINT16, UINT32, UINT64, COMPLEX64, COMPLEX128, BOOL, the data format supports ND.
 * @param [in] inverseOut:
 * The second output tensor is meaningful when returnInverse is True. It returns the position subscript of each element in self in the corresponding element in valueOut.
 * The data type supports INT64 and the data format supports ND.
 * @param [in] countsOut: The third output tensor, which is meaningful when returnCounts is True, returns the number of consecutive repetitions of each element in valueOut in self.
 * The data type supports INT64 and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnUniqueConsecutiveGetWorkspaceSize(
    const aclTensor* self, bool returnInverse, bool returnCounts, int64_t dim,
    aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnUniqueConsecutive is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnUniqueConsecutiveGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnUniqueConsecutive(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNIQUE_CONSECUTIVE_H_