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
#ifndef OP_API_INC_LEVEL2_SCALE_H_
#define OP_API_INC_LEVEL2_SCALE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnScale calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * @param [in] x: input parameter to be calculated for Scale. aclTensor on the NPU device side,
 * The data type supports float16, bfloat16, float32, and the data format supports ND.
 * Supports non-contiguous Tensors.
 * @param [in] scale: aclTensor on NPU device side, data type supports float, BF16, float16
 * @param [in] bias: aclTensor on NPU device side, data type supports float, BF16, float16
 * @param [in] axis: aclScalar on the host side, data type int64_t
 * @param [in] numAxes: aclScalar on the host side, data type int64_t
 * @param [in] scaleFromBlob: aclScalar on the host side, data type bool
 * @param [in] y: Output parameter for Scale calculation. aclTensor on the NPU device side,
 * The data type supports int8, the data format supports ND,
 * Supports non-contiguous Tensors.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnScaleGetWorkspaceSize(
    const aclTensor* x, const aclTensor* scale, const aclTensor* bias,
    int64_t axis, int64_t numAxes, bool scaleFromBlob, aclTensor* y,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnScale is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnScaleGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnScale(void* workspace, uint64_t workspaceSize,
                                 aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_SCALE_H_
