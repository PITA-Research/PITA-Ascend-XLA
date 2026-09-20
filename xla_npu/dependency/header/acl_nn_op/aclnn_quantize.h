/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_QUANTIZE_H_
#define OP_API_INC_QUANTIZE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnQuantize calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: quantize the input vector
 *
 * @param [in] x: Input Tensor, aclTensor on the NPU device side, the data type supports FLOAT, FLOAT16, BFLOAT16, and the data format supports ND.
 * @param [in] scales: aclTensor on the NPU device side, the data type supports FLOAT, FLOAT16, BFLOAT16, and the data format supports ND.
 * @param [in] zeroPoints: aclTensor on the NPU device side, the data type supports INT32, INT8, UINT8, BFLOAT16, and the data format supports ND.
 * @param [in] axis: Integer type on the Host side, the axis that needs to be quantized. The specified axis cannot exceed the number of dimensions of the input x.
 * @param [in] dtype: Quantitative output data type, aclDataType type, data type supports INT8, UINT8, INT32, data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnQuantizeGetWorkspaceSize(
    const aclTensor* x, const aclTensor* scales, const aclTensor* zeroPoints,
    aclDataType dtype, int32_t axis, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnQuantize is used to perform calculations.
 *
 * Operator function: quantize the input Tensor.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnQuantizeGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnQuantize(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANTIZE_H_
