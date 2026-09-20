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

#ifndef OP_API_INC_LEVEL2_ACLNN_REMAINDER_H_
#define OP_API_INC_LEVEL2_ACLNN_REMAINDER_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

// tensor self, tensor other
/**
 * @brief aclnnRemainderTensorTensor's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the remainder operation on the input Tensor
 * @param [in] self: npu
 * For aclTensor on the device side, the data type of self and the data type of other must meet [data type derivation rules] (#), and the derived data type
 * Must belong to one of the INT32, INT64, FLOAT16, FLOAT, and DOUBLE types. shape needs to satisfy [broadcast relationship](#) with other. Support [non-continuous Tensor](#),
 * The data format supports ND([Reference](#)).
 * @param [in] other: npu
 * aclTensor on the device side. The data type of self and the data type of other must satisfy [data type derivation rules] (#), and the derived data
 * The type must belong to one of the INT32, INT64, FLOAT16, FLOAT, and DOUBLE types. Shape needs to satisfy [broadcast relationship](#) with self. Support [non-continuous Tensor]
 * (#), the data format supports ND([Reference](#)).
 * @param [in] out: aclTensor on the NPU device side, data type supports UINT8, INT8, INT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE,
 * COMPLEX64, COMPLEX128 type, and the data type needs to be a convertible data type after derivation of self and other ([Reference](#)). shape needs to be self and other
 * The shape after broadcast. Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#).
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnRemainderTensorTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnRemainderTensorTensor, used to perform calculations
 *
 * Operator function: Complete the remainder operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnRemainderTensorTensorGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRemainderTensorTensor(void* workspace,
                                                 uint64_t workspaceSize,
                                                 aclOpExecutor* executor,
                                                 aclrtStream stream);

// tensor self, scalar other
/**
 * The first interface of @brief aclnnRemainderTensorScalar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the remainder operation on the input Tensor
 * @param [in] self: aclTensor on the NPU device side, the data type of self must be one of the INT32, INT64, FLOAT16, FLOAT, and DOUBLE types.
 * Supports [non-continuous Tensor](#), and the data format supports ND([Reference](#)).
 * @param [in] other:
 * aclScalar on the host side. The data type of self and the data type of other must satisfy [data type derivation rules](#), and the derived data type must
 * Belongs to one of the INT32, INT64, FLOAT16, FLOAT, and DOUBLE types. Shape needs to satisfy [broadcast relationship](#) with self. Support [non-continuous Tensor](#), data
 * The format supports ND([Reference](#)).
 * @param [in] out: aclTensor on the NPU device side, data type supports UINT8, INT8, INT16, INT32, INT64, FLOAT16, FLOAT, DOUBLE,
 * COMPLEX64, COMPLEX128 type, and the data type needs to be a self-convertible data type ([Reference](#)). shape needs to be consistent with self. Support [non-continuous Tensor](#),
 * The data format supports ND ([Reference](#).
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnRemainderTensorScalarGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnRemainderTensorScalar is used to perform calculations
 *
 * Operator function: Complete the remainder operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnRemainderTensorScalarGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRemainderTensorScalar(void* workspace,
                                                 uint64_t workspaceSize,
                                                 aclOpExecutor* executor,
                                                 aclrtStream stream);

// scalar self, tensor other
/**
 * The first interface of @brief aclnnRemainderScalarTensor calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the remainder operation on the input Tensor
 * @param [in] self: aclScalar on the host side. The data type of self needs to be convertible to the data type of other.
 * @param [in] other: npu
 * aclTensor on the device side. The data type of other must belong to one of the INT32, INT64, FLOAT16, FLOAT, and DOUBLE types. support
 * [Discontinuous Tensor](#), the data format supports ND([Reference](#)).
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports INT32, INT64, FLOAT16, FLOAT, and DOUBLE types, and the data type needs to be the same as the number of other
 * The data type is consistent ([Reference](#)). The shape needs to be consistent with other. Supports [non-continuous Tensor](#), and the data format supports ND([Reference](#)).
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnRemainderScalarTensorGetWorkspaceSize(
    const aclScalar* self, const aclTensor* other, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnRemainderScalarTensor, used to perform calculations
 *
 * Operator function: Complete the remainder operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnRemainderScalarTensorGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRemainderScalarTensor(void* workspace,
                                                 uint64_t workspaceSize,
                                                 aclOpExecutor* executor,
                                                 aclrtStream stream);

// inplace
// tensor self, tensor other
/**
 * @brief aclnnInplaceRemainderTensorTensor's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the remainder operation on the input Tensor
 * @param [in] selfRef: npu
 * aclTensor on the device side. The data type of selfRef and the data type of other must satisfy [data type derivation rules](#), and the derived
 * The data type must be one of INT32, INT64, FLOAT16, FLOAT, and DOUBLE types, and it needs to be a convertible data type after derivation ([Reference](#)). shape needs to be the same as
 * other satisfies [broadcast relationship](#), and the shape is consistent with the result of the final broadcast relationship. Supports [non-continuous Tensor](#), and the data format supports ND([Reference](#)).
 * @param [in] other: npu
 * aclTensor on the device side. The data type of selfRef and the data type of other must satisfy [data type derivation rules](#), and the derived data
 * The type must belong to one of the INT32, INT64, FLOAT16, FLOAT, and DOUBLE types. Shape needs to satisfy [broadcast relationship](#) with selfRef. support
 * [Discontinuous Tensor](#), the data format supports ND([Reference](#)).
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnInplaceRemainderTensorTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnInplaceRemainderTensorTensor is used to perform calculations
 *
 * Operator function: Complete the remainder operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, determined by the first interface
 * aclnnInplaceRemainderTensorTensorGetWorkspaceSize gets.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceRemainderTensorTensor(void* workspace,
                                                        uint64_t workspaceSize,
                                                        aclOpExecutor* executor,
                                                        aclrtStream stream);

// tensor self, scalar other
/**
 * The first interface of @brief aclnnInplaceRemainderTensorScalar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the remainder operation on the input Tensor
 * @param [in] selfRef: npu
 * aclTensor on the device side. The data type of selfRef and the data type of other must satisfy [data type derivation rules](#), and the derived number
 * The data type must belong to one of the INT32, INT64, FLOAT16, FLOAT, and DOUBLE types, and it needs to be a convertible data type after derivation ([Reference](#)). support
 * [Discontinuous Tensor](#), the data format supports ND([Reference](#)).
 * @param [in] other: aclScalar on the host side. The data type of other and the data type of selfRef must satisfy [data type derivation rules](#), and the derived data
 * The type must be convertible to the data type of selfRef.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnInplaceRemainderTensorScalarGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnInplaceRemainderTensorScalar is used to perform calculations
 *
 * Operator function: Complete the remainder operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, determined by the first interface
 * aclnnInplaceRemainderTensorScalarGetWorkspaceSize gets.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceRemainderTensorScalar(void* workspace,
                                                        uint64_t workspaceSize,
                                                        aclOpExecutor* executor,
                                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_REMAINDER_H_