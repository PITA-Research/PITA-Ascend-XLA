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
#ifndef OP_API_INC_LEVEL2_ACLNN_MUL_H_
#define OP_API_INC_LEVEL2_ACLNN_MUL_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnMuls calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL,
 * COMPLEX128, COMPLEX64. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] other: aclScalar on the host side, data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL,
 * COMPLEX128、COMPLEX64。
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL,
 * COMPLEX128, COMPLEX64. shape is equal to the shape of the input self. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMulsGetWorkspaceSize(const aclTensor* self,
                                                const aclScalar* other,
                                                aclTensor* out,
                                                uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnMul calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] self: nnpu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL,
 * COMPLEX128, COMPLEX64, and the data type needs to form a mutual derivation relationship with other, and the shape needs to satisfy the broadcast relationship with other. Support non-consecutive
 * Tensor, the data format supports ND.
 * @param [in] other: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL,
 * COMPLEX128, COMPLEX64, and the data type needs to form a mutual derivation relationship with self, and the shape needs to satisfy the broadcast relationship with self. Support non-continuous Tensor, data format
 * Support ND.
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL,
 * COMPLEX128, COMPLEX64. The shape needs to be equal to the shape after the input broadcast. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMulGetWorkspaceSize(const aclTensor* self,
                                               const aclTensor* other,
                                               aclTensor* out,
                                               uint64_t* workspaceSize,
                                               aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnInplaceMuls calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] selfRef：npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL,
 * COMPLEX128, COMPLEX64. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] other: aclScalar on the host side, data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL,
 * COMPLEX128、COMPLEX64。
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceMulsGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnInplaceMul calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] selfRef：npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL,
 * COMPLEX128, COMPLEX64, the data type needs to form a mutual derivation relationship with other, and the deduced data type needs to support conversion to the data type of selfRef.
 * The shape needs to satisfy the broadcast relationship with other, and the shape after broadcast needs to be equal to the shape of selfRef. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] other: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL,
 * COMPLEX128, COMPLEX64, and the data type needs to form a mutual derivation relationship with selfRef, and the shape needs to satisfy the broadcast relationship with selfRef. Support non-continuous Tensor,
 * The data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceMulGetWorkspaceSize(aclTensor* selfRef,
                                                      const aclTensor* other,
                                                      uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnMuls is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMulsGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMuls(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief The second interface of aclnnMul is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMulGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMul(void* workspace, uint64_t workspaceSize,
                               aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief The second interface of aclnnInplaceMuls is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceMulsGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceMuls(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream);

/**
 * @brief The second interface of aclnnInplaceMul is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceMulGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceMul(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_MUL_H_
