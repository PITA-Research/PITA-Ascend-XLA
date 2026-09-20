/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MEDIAN_H_
#define OP_API_INC_MEDIAN_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnMedian calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64,
 * And the data type is the same as valuesOut. Supports [non-continuous Tensor](#non-continuous Tensor description), and the data format supports ND ([reference](#reference)).
 * @param [in] valuesOut: aclTensor on the device side, data type supports FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64,
 * And the data type is the same as self, valuesOut is a one-dimensional Tensor, and the shape is (1,). Supports [non-continuous Tensor](#non-continuous Tensor description), and the data format supports ND ([reference](#reference)).
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMedianGetWorkspaceSize(const aclTensor* self,
                                                  aclTensor* valuesOut,
                                                  uint64_t* workspaceSize,
                                                  aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnMedian is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMedianGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMedian(void* workspace, uint64_t workspaceSize,
                                  aclOpExecutor* executor, aclrtStream stream);

/**
 * The first interface of @brief aclnnMedianDim calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: self: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64.
 * Supports [non-continuous Tensor] (#non-continuous Tensor description), and the data format supports ND.
 * @param [in] dim: the specified dimension, the data type supports INT64
 * @param [in] keepDim: Whether the dimension of the reduce axis is retained, the data type supports BOOL
 * @param [in] valuesOut: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64.
 * Supports [non-continuous Tensor] (#non-continuous Tensor description), and the data format supports ND.
 * @param [in] indicesOut：npu
 * The data type of aclTensor on the device side supports INT64. Supports [non-continuous Tensor] (#non-continuous Tensor description), and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMedianDimGetWorkspaceSize(
    const aclTensor* self, int64_t dim, bool keepDim, aclTensor* valuesOut,
    aclTensor* indicesOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMedianDim's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMedianDimGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMedianDim(void* workspace, uint64_t workspaceSize,
                                     aclOpExecutor* executor,
                                     aclrtStream stream);

/**
 * The first interface of @brief aclnnNanMedian calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64,
 * And the data type is the same as valuesOut. Supports [non-continuous Tensor](#non-continuous Tensor description), and the data format supports ND ([reference](#reference)).
 * @param [in] out: aclTensor on the device side, data type supports FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64,
 * And the data type is the same as self, out is a one-dimensional Tensor, and the shape is (1,). Supports [non-continuous Tensor](#non-continuous Tensor description), and the data format supports ND ([reference](#reference)).
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnNanMedianGetWorkspaceSize(const aclTensor* self,
                                                     aclTensor* out,
                                                     uint64_t* workspaceSize,
                                                     aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnNanMedian is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMedianGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnNanMedian(void* workspace, uint64_t workspaceSize,
                                     aclOpExecutor* executor,
                                     aclrtStream stream);

/**
 * The first interface of @brief aclnnNanMedianDim calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: self: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64.
 * Supports [non-continuous Tensor] (#non-continuous Tensor description), and the data format supports ND.
 * @param [in] dim: the specified dimension, the data type supports INT64
 * @param [in] keepDim: Whether the dimension of the reduce axis is retained, the data type supports BOOL
 * @param [in] valuesOut: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, UINT8, INT8, INT16, INT32, INT64.
 * Supports [non-continuous Tensor] (#non-continuous Tensor description), and the data format supports ND.
 * @param [in] indicesOut：npu
 * The data type of aclTensor on the device side supports INT64. Supports [non-continuous Tensor] (#non-continuous Tensor description), and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnNanMedianDimGetWorkspaceSize(
    const aclTensor* self, int64_t dim, bool keepDim, aclTensor* valuesOut,
    aclTensor* indicesOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnNanMedianDim is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnNanMedianDimGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnNanMedianDim(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MEDIAN_H_