/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_GELU_QUANT_H_
#define OP_API_INC_LEVEL2_ACLNN_GELU_QUANT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGeluQuant calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * @param [in] self: Input parameters to be calculated by GeluQuant. aclTensor on the NPU device side,
 * The data type supports float16, bfloat16, float, the data format supports ND, and non-continuous Tensors are supported.
 * @param [in] inputScaleOptional: aclTensor on NPU device side, data type supports float16, bfloat16, float
 * @param [in] inputOffsetOptional: aclTensor on NPU device side, data type supports float16, bfloat16, float
 * @param [in] approximate: aclScalar on the host side, data type string, only supports "none", "tanh"
 * @param [in] quantMode: aclScalar on the host side, data type string, only supports "static", "dynamic"
 * @param [in] roundMode: aclScalar on the host side, data type string, supports "rint", "round", "hybrid", "floor", "ceil", "trunc"
 * @param [in] dstType: aclScalar on the host side, data type int, input range is {2, 34, 35, 36}, corresponding data type of output y is {2: INT8,
 * 34: HIFLOAT8, 35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}
 * @param [in] y: The output parameter calculated by GeluQuant. aclTensor on the NPU device side,
 * The data type supports int8, hifloat8, float8_e4m3fn, float8_e5m2, the data format supports ND, and non-continuous Tensors are not supported.
 * @param [in] outScaleOptional: Output parameter calculated by GeluQuant. aclTensor on the NPU device side,
 * The data type supports float32, the data format supports ND, and non-continuous Tensors are supported.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGeluQuantGetWorkspaceSize(
    const aclTensor* self, const aclTensor* inputScaleOptional,
    const aclTensor* inputOffsetOptional, const char* approximate,
    const char* quantMode, const char* roundMode, int64_t dstType,
    const aclTensor* y, const aclTensor* outScaleOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGeluQuant's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGeluQuant.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGeluQuant(void* workspace, uint64_t workspaceSize,
                                     aclOpExecutor* executor,
                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_GELU_QUANT_H_
