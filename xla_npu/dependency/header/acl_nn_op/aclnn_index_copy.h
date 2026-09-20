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
 * \file aclnn_index_copy.h
 * \brief
 */
#ifndef OP_API_INC_INDEX_COPY_H_
#define OP_API_INC_INDEX_COPY_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief The first section of aclnnIndexCopy interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Use the element value in the index tensor as the index, and copy the elements in the source to the corresponding position of outRef for the specified axis dim. Other elements in outRef are copied from the corresponding positions of selfRef
 *
 * @param [in] selfRef: npu
 * aclTensor on the device side, the input data type supports FLOAT, BFLOAT16, FLOAT16, INT32, INT64, INT16, INT8, UINT8,
 * DOUBLE, BOOL, COMPLEX128, COMPLEX64, support [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [in] dim: int64 type on the host side.
 * @param [in] index: aclTensor on the NPU device side. The data type supports INT32 and INT64. The dimension cannot be greater than 1. The number of elements is required to be in the tensor source.
 * The dim axis is the same size. Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [in] source: npu
 * The data type of aclTensor on the device side is the same as selfRef. Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [in] outRef: aclTensor on the NPU device side, the data type and data format are the same as selfRef.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnIndexCopyGetWorkspaceSize(
    aclTensor* selfRef, int64_t dim, const aclTensor* index,
    const aclTensor* source, aclTensor* outRef, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnIndexCopy is used to perform calculations.
 *
 * Operator function: Use the element value in the index tensor as the index, and copy the elements in the source to the corresponding position of outRef for the specified axis dim. and copy the remaining elements from selfRef to outRef
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceIndexCopyGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnIndexCopy(void* workspace, uint64_t workspaceSize,
                                     aclOpExecutor* executor,
                                     aclrtStream stream);

/**
 * @brief aclnnInplaceIndexCopy The first section of the interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Use the element value in the index tensor as the index, and copy the elements in the source to the corresponding position of selfRef for the specified axis dim.
 *
 * @param [in] selfRef: npu
 aclTensor on the device side, the input data type supports FLOAT, BFLOAT16, FLOAT16, INT32, INT64, INT16, INT8, UINT8,
 * DOUBLE, BOOL, COMPLEX128, COMPLEX64, support [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [in] dim: int64 type on the host side.
 * @param [in] index: aclTensor on the NPU device side. The data type supports INT32 and INT64. The dimension cannot be greater than 1. The number of elements is required to be in the tensor source.
 * The dim axis is the same size. Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [in] source: npu
 The data type of aclTensor on the device side is the same as selfRef. Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).

 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceIndexCopyGetWorkspaceSize(
    aclTensor* selfRef, int64_t dim, const aclTensor* index,
    const aclTensor* source, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplacePut is used to perform calculations.
 *
 * Operator function: Use the element value in the index tensor as the index, and copy the elements in the source to the corresponding position of selfRef for the specified axis dim.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceIndexCopyGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceIndexCopy(void* workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_INDEX_COPY_H_