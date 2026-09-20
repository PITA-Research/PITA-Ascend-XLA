/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_H_
#define OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_H_

#include <cstring>
#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnRoiAlign calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: ROIAlign is a pooling layer that is used to input feature maps of non-uniform size and output feature maps of fixed size.
 *
 * @param [in] self: aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, non-continuous Tensor, and the data format supports NCHW.
 * @param [in] rois: aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT32, non-continuous Tensor, and the data format supports ND.
 * @param [in] batchIndices: aclTensor on the NPU device side, the data type supports INT64, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] out: npu
 * The data type of aclTensor on the device side supports FLOAT16 and FLOAT32, which is consistent with self. It supports non-continuous Tensor and the data format supports NCHW.
 * @param [in] mode: string type on the host side, pooling mode, supports "avg" or "max".
 * @param [in] outputHeight: int type on the host side, H of the ROI output feature map.
 * @param [in] outputWidth: int type on the host side, W of the ROI output feature map.
 * @param [in] samplingRatio: int type on the host side, used to calculate the number of bins above the sum W of each output element.
 * @param [in] spatialScale: float type on the host side, scaling factor, used to convert ROI coordinates into input feature maps.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRoiAlignGetWorkspaceSize(
    const aclTensor* self, const aclTensor* rois, const aclTensor* batchIndices,
    const char* mode, int outputHeight, int outputWidth, int samplingRatio,
    float spatialScale, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnRoiAlign is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnRoiAlignGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnRoiAlign(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ROI_ALIGN_H_