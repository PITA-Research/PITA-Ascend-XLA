/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_MAX_POOL2D_WITH_INDICES_H_
#define OP_API_INC_LEVEL2_ACLNN_MAX_POOL2D_WITH_INDICES_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnMaxPool2dWithMask calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * For the input channel of the input signal, a 2-dimensional pooling operation is provided:
 *
 * @param [in] self: aclTensor on NPU device side, data type supports float16, float32, bfloat16 (only Ascend 910B AI processor,
 * Ascend 910_93 AI processor supports), the data format supports NCHW (ND), and supports non-continuous Tensor.
 * @param [in] kernelSize: aclIntArray type, indicating the maximum pooling window size.
 * @param [in] stride: aclIntArray type, the step size of the window movement.
 * @param [in] padding: aclIntArray type, the number of added layers for each edge, and negative infinity for the added position.
 * @param [in] dilation: aclIntArray type, controls the stride of elements in the window.
 * @param [in] ceilMode: aclIntArray type, when true, the output shape is calculated by rounding up, and the default is rounding down.
 * @param [in] out: aclTensor on NPU device side, data type supports float16, float32, bfloat16 (only Shengteng 910B
 * AI processor, Shengteng 910_93 AI processor supports), the data format supports NCHW (ND), and supports non-continuous Tensor.
 * @param [in] indices: npu
 * The aclTensor on the device side is a Tensor composed of bit values whose maximum value is at the kernel position of the mask. The data type supports int8 and the data format supports NCHW(ND).
 * Supports non-contiguous Tensors.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMaxPool2dWithMaskGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* kernelSize,
    const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool ceilMode, aclTensor* out,
    aclTensor* indices, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnMaxPool2dWithMask is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnMaxPool2dWithMaskGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMaxPool2dWithMask(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

/**
 * The first interface of @brief aclnnMaxPool2dWithIndices calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * For the input channel of the input signal, a 2-dimensional pooling operation is provided:
 *
 * @param [in] self: aclTensor on the NPU device side, the data type supports float32 (only supported by Ascend 910B AI processor and Ascend 910_93 AI processor),
 * The data format supports NCHW(ND) and non-continuous Tensor.
 * @param [in] kernelSize: aclIntArray type, indicating the maximum pooling window size.
 * @param [in] stride: aclIntArray type, the step size of the window movement.
 * @param [in] padding: aclIntArray type, the number of added layers for each edge, and negative infinity for the added position.
 * @param [in] dilation: aclIntArray type, controls the stride of elements in the window.
 * @param [in] ceilMode: aclIntArray type, when true, the output shape is calculated by rounding up, and the default is rounding down.
 * @param [in] out: aclTensor on the NPU device side, the data type supports float32 (only supported by Ascend 910B AI processor), and the data format supports NCHW(ND).
 * Supports non-contiguous Tensors.
 * @param [in] indices: aclTensor on the NPU device side, tensor composed of maximum index position, data type supports int32, data format supports NCHW(ND),
 * Supports non-contiguous Tensors.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMaxPool2dWithIndicesGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* kernelSize,
    const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool ceilMode, aclTensor* out,
    aclTensor* indices, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnMaxPool2dWithIndices is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnMaxPool2dWithIndicesGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMaxPool2dWithIndices(void* workspace,
                                                uint64_t workspaceSize,
                                                aclOpExecutor* executor,
                                                aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_MAX_POOL2D_WITH_INDICES_H_
