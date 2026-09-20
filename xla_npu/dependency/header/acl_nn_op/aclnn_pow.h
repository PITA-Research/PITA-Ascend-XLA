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
#ifndef OP_API_INC_POW_TENSOR_SCALAR_H_
#define OP_API_INC_POW_TENSOR_SCALAR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnPowTensorScalar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8,
 * UINT8, BOOL, COMPLEX64, COMPLEX128, BFLOAT16 (this data type is not supported on Ascend910 and earlier chips),
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] exponent: npu
 * aclScalar on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8,
 * UINT8, BOOL, COMPLEX64, COMPLEX128, BFLOAT16 (this data type is not supported on Ascend910 and earlier chips).
 * @param [in] out: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8,
 * UINT8, COMPLEX64, COMPLEX128, BFLOAT16 (this data type is not supported on Ascend910 and earlier chips),
 * And the data type, data format and shape are consistent with self.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnPowTensorScalarGetWorkspaceSize(
    const aclTensor* self, const aclScalar* exponent, const aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief The second interface of aclnnPowTensorScalar is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnPowTensorScalarGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnPowTensorScalar(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           const aclrtStream stream);

/**
 * The first interface of @brief aclnnInplacePowTensorScalar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * @param [in] self: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8,
 * UINT8, BOOL, COMPLEX64, COMPLEX128, BFLOAT16 (this data type is not supported on Ascend910 and earlier chips),
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] exponent: npu
 * aclScalar on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16, DOUBLE, INT32, INT64, INT16, INT8,
 * UINT8, COMPLEX64, COMPLEX128, BFLOAT16 (this data type is not supported on Ascend910 and earlier chips).
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplacePowTensorScalarGetWorkspaceSize(
    const aclTensor* self, const aclScalar* exponent, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnInplacePowTensorScalar's second interface is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: the workspace size applied for on the NPU device side,
 * Obtained by the first interface aclnnInplacePowTensorScalarGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplacePowTensorScalar(void* workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor* executor,
                                                  aclrtStream stream);

/**
 * The first interface of @brief aclnnPowScalarTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnPowScalarTensorGetWorkspaceSize(
    const aclScalar* self, const aclTensor* exponent, const aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnPowScalarTensor(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_POW_TENSOR_SCALAR_H_
