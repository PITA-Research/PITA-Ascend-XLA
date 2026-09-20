/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_UNAMPLE_BICUBIC2D_AA_H_
#define OP_API_INC_UNAMPLE_BICUBIC2D_AA_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnUpsampleBicubic2dAA's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: anti-aliasing calculation of upsample_bicubic2d.
 *
 * @param [in] x: aclTensor on the NPU device side, the data type supports FLOAT, BFLOAT16, and FLOAT16.
 * Supports [non-continuous Tensor] (#non-continuous Tensor description), and the data format supports NCHW.
 * @param [in] outputSize: Input IntArray, size is 2. Indicates the spatial size of the output out in the H and W dimensions.
 * @param [in] alignCorners: bool constant, indicating whether to use corner alignment calculation.
 * @param [in] scalesH: double constant, indicating the height dimension multiplier of the output out.
 * @param [in] scalesW: double constant, indicating the width dimension multiplier of the output out.
 * @param [out] out: npu
 * The data type of aclTensor on the device side supports FLOAT, BFLOAT16, and FLOAT16, and the data type is consistent with the data type of x.
 * Supports [non-continuous Tensor] (#non-continuous Tensor description), and the data format supports NCHW.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnUpsampleBicubic2dAAGetWorkspaceSize(
    const aclTensor *x, const aclIntArray *outputSize, const bool alignCorners,
    const double scalesH, const double scalesW, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnUpsampleBicubic2dAA's second interface is used to perform calculations.
 *
 * Operator function: anti-aliasing calculation of upsample_bicubic2d.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: the workspace size applied for on the NPU device side,
 * Obtained by the first interface aclnnUpsampleBicubic2dAAGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnUpsampleBicubic2dAA(void *workspace,
                                               uint64_t workspaceSize,
                                               aclOpExecutor *executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UNAMPLE_BICUBIC2D_AA_H_
