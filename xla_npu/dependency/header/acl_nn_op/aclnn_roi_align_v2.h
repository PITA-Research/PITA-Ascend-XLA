/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_V2_H_
#define OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_V2_H_

#include <cstring>
#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnRoiAlignV2 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: ROIAlign is a pooling layer that is used to input feature maps of non-uniform size and output feature maps of fixed size.
 *
 * @param [in] self: aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, non-continuous Tensor, and the data format supports NCHW.
 * @param [in] boxes: aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, non-continuous Tensor, and the data format supports ND.
 * @param [in] pooledHeight: int type on the host side, H of the output image after forward RoiAlign pooling.
 * @param [in] pooledWidth: int type on the host side, W of the output image after forward RoiAlign pooling.
 * @param [in] spatialScale: float type on the host side, scaling factor, used to convert ROI coordinates into input feature maps.
 * @param [in] samplingRatio: int type on the host side, used to calculate the number of bins above the sum W of each output element.
 * @param [in] aligned: Bool type on the host side, used to determine whether the pixel frame is offset by -0.5 for better alignment.
 * @param [out] out: npu
 * The data type of aclTensor on the device side supports FLOAT16 and FLOAT32, which is consistent with self. It supports non-continuous Tensor and the data format supports NCHW.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRoiAlignV2GetWorkspaceSize(
    const aclTensor* self, const aclTensor* boxes, int64_t pooledHeight,
    int64_t pooledWidth, float spatialScale, int64_t samplingRatio,
    bool aligned, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnRoiAlignV2 is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnRoiAlignV2GetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnRoiAlignV2(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_BACKWARD_H_