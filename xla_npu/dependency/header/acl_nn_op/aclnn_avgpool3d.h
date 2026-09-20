/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_AVGPOOL3D_H_
#define OP_API_INC_AVGPOOL3D_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnAvgPool3d calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: perform average pooling operation on the input self.
 *
 * @param [in] self: npu
 * The aclTensor on the device side supports the empty tensor scenario, and the data type supports FLOAT16, BFLOAT16 and FLOAT. Supports 4D or 5D. Supports non-contiguous Tensors. The supported data format is ND.
 * @param [in] kernelSize: npu
 * The aclIntArray on the device side has a length of 1 (kD=kH=kW) or 3 (kD, kH, kW), indicating the pooling window size. The data type supports INT32 and INT64. The value must be greater than 0.
 * @param [in] stride: npu
 * The aclIntArray on the device side has a length of 0 (default is kernelSize) or 1 (sD=sH=sW) or 3 (sD, sH, sW), indicating the step size of the pooling operation.
 * The data type supports INT32 and INT64. The value must be greater than 0.
 * @param [in] padding: npu
 * The aclIntArray on the device side has a length of 1 (padD=padH=padW) or 3 (padD, padH, padW), indicating the number of padding layers in the input D, H, and W directions.
 * The data type supports INT32 and INT64. The value is in the range [0, kernelSize/2].
 * @param [in] ceilMode: The data type supports BOOL. Indicates that when calculating the output shape, round down (False), otherwise round up.
 * @param [in] countIncludePad: The data type supports BOOL. Indicates that zero padding is included in the average calculation (True), otherwise it is not.
 * @param [in] divisorOverride: The data type supports INT64. If specified, it will be used as the divisor in average calculations. When the value is 0, this property has no effect.
 * @param [out] output: npu
 * aclTensor on the device side, outputs Tensor, and the data type supports FLOAT16, BFLOAT16 and FLOAT. Supports 4D or 5D. Supports non-contiguous Tensors. The supported data format is ND.
 * The data type and data format need to be consistent with self.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAvgPool3dGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* kernelSize,
    const aclIntArray* stride, const aclIntArray* padding, bool ceilMode,
    bool countIncludePad, int64_t divisorOverride, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnAvgPool3d is used to perform calculations.
 *
 * Operator function: Perform average pooling operation on the input self.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAtan2GetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAvgPool3d(void* workspace, uint64_t workspaceSize,
                                     aclOpExecutor* executor,
                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_AVGPOOL3D_H_
