/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_FLAT_QUANT_H_
#define OP_API_INC_FLAT_QUANT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnFlatQuant calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: perform two small matrix multiplications for the matrix x in sequence, and then perform quantization processing on the result of the matrix multiplication.
 *
 * @param [in] x: aclTensor on the Device side, shape is [K, M, N], where K and N must be an integer multiple of 16.
 * Supports non-continuous Tensor, the data format supports ND, and the data type supports FLOAT16 and BFLOAT16.
 * @param [in] kroneckerP1: aclTensor on the Device side, shape is [M, M], M is consistent with the M dimension in x.
 * Supports non-continuous Tensors, the data format supports ND, the data type is consistent with the data type of the input parameter `x`, and the data type supports FLOAT16 and BFLOAT16.
 * @param [in] kroneckerP2: aclTensor on the Device side, shape is [N, N], N is consistent with the N dimension in x.
 * Supports non-continuous Tensors, the data format supports ND, the data type is consistent with the data type of the input parameter `x`, and the data type supports FLOAT16 and BFLOAT16.
 * @param [in] clipRatio: Double parameter on the Host side, used to control the quantized clipping ratio. The input data range is [0, 1], and the default value is 1.
 * @param [out] out: aclTensor on the Device side supports non-continuous Tensor, the data format supports ND, and the data type supports INT4 and INT32.
 * When the type is INT32, the last dimension of the shape is 1/8 of the last dimension of the input parameter `x`, and the remaining dimensions are consistent with x; when the type is INT4, the shape is consistent with the input parameter `x`.
 * @param [out] quantScale: aclTensor on the Device side, shape is [K], K is consistent with the K dimension in x.
 * Supports non-continuous Tensor, data format supports ND, and data type supports FLOAT32.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFlatQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* kroneckerP1,
    const aclTensor* kroneckerP2, double clipRatio, aclTensor* out,
    aclTensor* quantScale, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnFlatQuant is used to perform calculations.
 *
 * Operator function: perform two small matrix multiplications for the matrix x in sequence, and then perform quantization processing on the result of the matrix multiplication.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the Device side.
 * @param [in] workspaceSize: The workspace size applied for on the Device side. Obtained by the first interface aclnnFlatQuantGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: Specifies the Stream to perform the task.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFlatQuant(void* workspace, uint64_t workspaceSize,
                                     aclOpExecutor* executor,
                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_FLAT_QUANT_H_
