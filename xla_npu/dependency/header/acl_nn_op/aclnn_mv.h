/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_MV_H_
#define OP_API_INC_LEVEL2_ACLNN_MV_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnMv calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Calculate the product of matrix input and vector vec
 * @param [in] self: aclTensor on the NPU device side. The data type supports FLOAT16, FLOAT, DOUBLE, COMPLEX64, and COMPLEX128 types. support
 * [Discontinuous Tensor](#), the shape is a two-dimensional tensor of n*m, and the data format supports ND ([Reference](#)).
 * @param [in] vec: aclTensor on the NPU device side. The data type supports FLOAT16, FLOAT, DOUBLE, COMPLEX64, and COMPLEX128 types, and the data type is the same as
 * self remains consistent. Supports [non-continuous Tensor](#), shape is a one-dimensional tensor with length m, and the data format supports ND([Reference](#)).
 * @param [in] out: npu
 * aclTensor on the device side, the data type supports FLOAT16, FLOAT, DOUBLE, COMPLEX64, COMPLEX128 types, and the data type is the same as self
 * Be consistent. Supports [non-continuous Tensor](#), shape is a one-dimensional tensor with length n, and the data format supports ND([Reference](#)).
 * @param [in] cubeMathType(INT8,
 * Calculation input): INT8 type enumeration value, used to determine which calculation logic the Cube unit should use for calculations, 0: KEEP_DTYPE, keep
 * The input data type is maintained for calculation. If the input is FP32, Ascend910 series chips do not support it currently. When the value is 0, an error will be reported. 1: ALLOW_FP32_DOWN_PRECISION, allow conversion input
 * The data type reduces the precision calculation. If the input is FP32, it is converted to FP16 for calculation on the Ascend910 series chip, and converted to HF16 for calculation on the Ascend910B series chip.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnMvGetWorkspaceSize(
    const aclTensor* self, const aclTensor* vec, aclTensor* out,
    int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnMv is used to perform calculations
 *
 * Operator function: Calculate the product of matrix input and vector vec
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMvGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMv(void* workspace, uint64_t workspaceSize,
                              aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_MV_H_