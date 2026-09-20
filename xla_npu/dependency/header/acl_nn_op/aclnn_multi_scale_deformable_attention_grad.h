/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MULTI_SCALE_DEFORMABLE_ATTENTION_GRAD_H_
#define OP_API_INC_MULTI_SCALE_DEFORMABLE_ATTENTION_GRAD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnMultiScaleDeformableAttentionGrad calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * @param [in] value:
 * aclTensor on the Device side, the feature value of the feature map, the data type supports FLOAT, FLOAT16, BFLOAT16, supports non-continuous Tensor, and the data format supports ND
 * @param [in] spatialShape:
 * The aclTensor on the device side stores the height and width of each scale feature map. The data type supports INT32, INT64, non-continuous Tensor, and the data format supports ND.
 * @param [in] levelStartIndex:
 * aclTensor on the device side, the starting index of each feature map, the data type supports INT32, INT64, non-continuous Tensor, and the data format supports ND.
 * @param [in] location:
 * The aclTensor on the Device side stores the location of the sampling point. The data type supports FLOAT, FLOAT16, and BFLOAT16. It supports non-continuous Tensor. The data format supports ND.
 * @param [in] attnWeight:
 * The aclTensor on the Device side stores the weight of each sampling point. The data type supports FLOAT, FLOAT16, BFLOAT16, non-continuous Tensor, and the data format supports ND.
 * @param [in] gradOutput:
 * aclTensor on the Device side, the initial gradient of the reverse operator, the data type supports FLOAT, FLOAT16, BFLOAT16, non-continuous Tensor, and the data format supports ND.
 * @param [in] gradValue:
 * For aclTensor on the Device side, input the gradient corresponding to the value. The data type supports FLOAT, FLOAT16, and BFLOAT16. It supports non-continuous Tensor and the data format supports ND.
 * @param [in] gradLocation:
 * For aclTensor on the Device side, enter the gradient corresponding to the location. The data type supports FLOAT, FLOAT16, and BFLOAT16. It supports non-continuous Tensor. The data format supports ND.
 * @param [in] gradAttnWeight:
 * For aclTensor on the Device side, input the gradient corresponding to attnWeight. The data type supports FLOAT, FLOAT16, BFLOAT16, non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize(
    const aclTensor* value, const aclTensor* spatialShape,
    const aclTensor* levelStartIndex, const aclTensor* location,
    const aclTensor* attnWeight, const aclTensor* gradOutput,
    aclTensor* gradValue, aclTensor* gradLocation, aclTensor* gradAttnWeight,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMultiScaleDeformableAttentionGrad's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceMishGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMultiScaleDeformableAttentionGrad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MULTI_SCALE_DEFORMABLE_ATTENTION_GRAD_H_
