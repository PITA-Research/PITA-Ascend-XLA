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
 * \file aclnn_unique2.h
 * \brief
 */
#ifndef OP_API_INC_UNIQUE2_H_
#define OP_API_INC_UNIQUE2_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnUnique2 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Returns the unique elements in the input tensor
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B ---> F([UniqueWithCountsAndSorting])
 *     C[(sorted)] --->F
 *     D[(returnInverse)] --->F
 *     E[(returnCounts)] --->F
 *     F --> G([valueOut])
 *     F --> H([inverseOut])
 *     F --> I([countsOut])
 * ```
 *
 * @param [in] self: aclTensor on the NPU device side, data type supports BOOL, FLOAT, FLOAT16, DOUBLE, UINT8, INT8, UINT16, INT16,
 * INT32, UINT32, UINT64, INT64, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] sorted: Optional parameter, default False, indicating whether to sort valueOut in ascending order.
 * @param [in] returnInverse: optional parameter, defaults to False, indicating whether to return the subscript of each element in the input data in valueOut.
 * @param [in] returnCounts: Optional parameter, default False, indicating whether to return the number of each unique element in valueOut in the original input Tensor.
 * @param [in] valueOut: aclTensor on the NPU device side, the first output tensor, the only element in the input tensor, the data type supports BOOL, FLOAT,
 * FLOAT16, DOUBLE, UINT8, INT8, UINT16, INT16, INT32, UINT32, UINT64, INT64, the data format supports ND.
 * @param [in] inverseOut: npu
 * aclTensor on the device side, the second output tensor, is meaningful when returnInversie is True, and returns the position where each element in self appears in valueOut
 * Standard, data type supports INT64, shape is consistent with self
 * @param [in] countsOut: npu
 * aclTensor on the device side, the third output tensor, is meaningful when returnCounts is True, returns the number of times each element in valueOut appears in self, data
 * The type supports INT64, and the shape is consistent with valueOut.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnUnique2GetWorkspaceSize(
    const aclTensor* self, bool sorted, bool returnInverse, bool returnCounts,
    aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnUnique2 is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnUnique2GetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnUnique2(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNIQUE2_H_