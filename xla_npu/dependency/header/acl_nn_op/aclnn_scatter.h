/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_SCATTER_H_
#define OP_API_INC_SCATTER_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnScatter calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Complete scatter operation on input Tensor
 * @param [in] self: aclTensor on NPU device side,
 * Data types support UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE,
 * COMPLEX64, COMPLEX128 types. The number of dimensions of self needs to be the same as index and src. The data type of self needs to be consistent with src. Support empty tensor, support
 * Non-continuous tensor. The data format supports ND.
 * @param [in] dim: The dimension used for scatter, the data type is INT64. The range is [-number of dimensions of self, number of dimensions of self-1].
 * @param [in] index: aclTensor on the NPU device side. The data type supports INT32,
 * INT64. The number of dimensions of index needs to be the same as self and src. Supports empty tensors and non-continuous tensors. The data format supports ND.
 * @param [in] src: npu
 * aclTensor on the device side, the data type supports UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE,
 * COMPLEX64, COMPLEX128 types. The number of dimensions of src needs to be the same as self and index. The data type of src needs to be consistent with self. Support empty tensor,
 * Supports non-contiguous tensors. The data format supports ND.
 * @param [in] reduce: Select the reduction operation to apply. The optional operation options and corresponding int values ​​are (add, 1), (mul, 2), (none, 0).
 * The specific operation meanings are as follows:
 * 0: Indicates a replacement operation, replacing the value of the corresponding position in src to the corresponding position in out according to index
 * 1: Indicates the accumulation operation. The value of the corresponding position in src is accumulated to the corresponding position in out according to the index.
 * 2: Indicates the cumulative multiplication operation, multiply the value of the corresponding position in src to the corresponding position in out according to the index.
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE,
 * COMPLEX64, COMPLEX128 types. The data format, data type, and shape need to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnScatterGetWorkspaceSize(
    const aclTensor* self, int64_t dim, const aclTensor* index,
    const aclTensor* src, int64_t reduce, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnScatter, used to perform calculations
 *
 * Operator function: Complete scatter operation on input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnScatterGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnScatter(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor,
                                   const aclrtStream stream);

/**
 * The first interface of @brief aclnnScatterValue calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Complete scatter operation on input Tensor
 * @param [in] self: aclTensor on NPU device side,
 * Data types support UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE,
 * COMPLEX64, COMPLEX128 types. The number of dimensions of self needs to be the same as index and src. The data type of self needs to be consistent with src. Support empty tensor, support
 * Non-continuous tensor. The data format supports ND.
 * @param [in] dim: The dimension used for scatter, the data type is INT64. The range is [-number of dimensions of self, number of dimensions of self-1].
 * @param [in] index: aclTensor on the NPU device side. The data type supports INT32,
 * INT64. The number of dimensions of index needs to be the same as self and src. Supports empty tensors and non-continuous tensors. The data format supports ND.
 * @param [in] value: aclScalar on the host side,
 * Data types support UINT8, INT8, INT16, INT32, INT64, FLOAT16, FLOAT32, DOUBLE, COMPLEX64,
 * COMPLEX128. When value is COMPLEX, self must also be COMPLEX tensor.
 * @param [in] reduce: Select the reduction operation to apply. The optional operation options and corresponding int values ​​are (add, 1), (mul, 2), (none, 0).
 * The specific operation meanings are as follows:
 * 0: Indicates a replacement operation, replacing the value of the corresponding position in src to the corresponding position in out according to index
 * 1: Indicates the accumulation operation. The value of the corresponding position in src is accumulated to the corresponding position in out according to the index.
 * 2: Indicates the cumulative multiplication operation, multiply the value of the corresponding position in src to the corresponding position in out according to the index.
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE,
 * COMPLEX64, COMPLEX128 types. The data format, data type, and shape need to be consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnScatterValueGetWorkspaceSize(
    const aclTensor* self, int64_t dim, const aclTensor* index,
    const aclScalar* value, int64_t reduce, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnScatter, used to perform calculations
 *
 * Operator function: Complete scatter operation on input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnScatterValueGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnScatterValue(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        const aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceScatter calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Complete scatter operation on input Tensor
 * @param [in] selfRef:
 * Data types support UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, and COMPLEX128 types.
 * The number of dimensions of selfRef needs to be the same as index and src. The data type of selfRef needs to be consistent with src. Support empty tensor,
 * Supports non-continuous tensors. The data format supports ND.
 * @param [in] dim: The dimension used for scatter, the data type is INT64. The range is [-the number of dimensions of selfRef, the number of dimensions of selfRef-1].
 * @param [in] index: aclTensor on the NPU device side. The data type supports INT32,
 * INT64. The number of dimensions of index needs to be the same as selfRef and src. Supports empty tensors and non-continuous tensors. The data format supports ND.
 * @param [in] src: npu
 * aclTensor on the device side, the data type supports UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE,
 * COMPLEX64, COMPLEX128 types. The number of dimensions of src needs to be the same as selfRef and index. The data type of src needs to be consistent with selfRef. Support empty tensor,
 * Supports non-contiguous tensors. The data format supports ND.
 * @param [in] reduce: Select the reduction operation to apply. The optional operation options and corresponding int values ​​are (add, 1), (mul, 2), (none, 0).
 * The specific operation meanings are as follows:
 * 0: Indicates a replacement operation, replacing the value of the corresponding position in src to the corresponding position in out according to index
 * 1: Indicates the accumulation operation. The value of the corresponding position in src is accumulated to the corresponding position in out according to the index.
 * 2: Indicates the cumulative multiplication operation, multiply the value of the corresponding position in src to the corresponding position in out according to the index.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnInplaceScatterGetWorkspaceSize(
    aclTensor* selfRef, int64_t dim, const aclTensor* index,
    const aclTensor* src, int64_t reduce, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnInplaceScatter is used to perform calculations
 *
 * Operator function: Complete scatter operation on input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceScatterGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceScatter(void* workspace,
                                          uint64_t workspaceSize,
                                          aclOpExecutor* executor,
                                          aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceScatterValue calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Complete scatter operation on input Tensor
 * @param [in] selfRef: aclTensor on the NPU device side, data type supports UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32,
 * DOUBLE, COMPLEX64, COMPLEX128 types. The number of dimensions of selfRef needs to be the same as index and src. The data type of selfRef needs to be consistent with src. Support empty tensor,
 * Supports non-continuous tensors. The data format supports ND.
 * @param [in] dim: The dimension used for scatter, the data type is INT64. The range is [-the number of dimensions of selfRef, the number of dimensions of selfRef-1].
 * @param [in] index: aclTensor on the NPU device side. The data type supports INT32,
 * INT64. The number of dimensions of index needs to be the same as selfRef and src. Supports empty tensors and non-continuous tensors. The data format supports ND.
 * @param [in] value: aclScalar on the host side,
 * Data types support UINT8, INT8, INT16, INT32, INT64, FLOAT16, FLOAT32, DOUBLE, COMPLEX64,
 * COMPLEX128. When value is COMPLEX, selfRef must also be COMPLEX tensor.
 * @param [in] reduce: Select the reduction operation to apply. The optional operation options and corresponding int values ​​are (add, 1), (mul, 2), (none, 0).
 * The specific operation meanings are as follows:
 * 0: Indicates a replacement operation, replacing the value of the corresponding position in src to the corresponding position in out according to index
 * 1: Indicates the accumulation operation. The value of the corresponding position in src is accumulated to the corresponding position in out according to the index.
 * 2: Indicates the cumulative multiplication operation, multiply the value of the corresponding position in src to the corresponding position in out according to the index.
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE,
 * COMPLEX64, COMPLEX128 types. The data format, data type, and shape need to be consistent with selfRef.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnInplaceScatterValueGetWorkspaceSize(
    aclTensor* selfRef, int64_t dim, const aclTensor* index,
    const aclScalar* value, int64_t reduce, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnInplaceScatterValue is used to perform calculations
 *
 * Operator function: Complete scatter operation on input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnInplaceScatterValueGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceScatterValue(void* workspace,
                                               uint64_t workspaceSize,
                                               aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
