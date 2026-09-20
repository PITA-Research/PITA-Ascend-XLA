/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_CONVOLUTION_BACKWARD_H_
#define OP_API_INC_CONVOLUTION_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnConvolutionBackward, calculates and obtains the workspace size
 * @domain aclnn_ops_train
 *
 * @param [in] gradOutput: NPU, convolution output gradient.
 * The aclTensor on the device side supports non-continuous Tensor, and requires that the input and weight satisfy the derivation relationship of the convolution input and output shape.
 * @param [in] input: NPU, convolution input.
 * The aclTensor on the device side supports non-continuous Tensor, and is required to satisfy the derivation relationship of the convolution input and output shape with gradOutput and weight.
 * @param [in] weight: NPU, convolution weight.
 * The aclTensor on the device side supports non-continuous Tensor and is required to satisfy the derivation relationship of the convolution input and output shape with gradOutput and input.
 * @param [in] biasSizes: NPU, the shape of the bias (bias) in the forward process of convolution.
 * aclIntArray。
 * @param [in] stride: The step size of the convolution kernel moving on the input during backpropagation.
 * aclIntArray, the array length is the weight dimension minus 2, and the value must be greater than 0.
 * @param [in] padding: input padding during backpropagation.
 * aclIntArray, the array length can be the weight dimension minus 2. In a 2D scenario, the array length can be 4. The value must be greater than or equal to 0.
 * @param [in] dilation: dilation parameter during backpropagation.
 * aclIntArray, the array length can be the weight dimension minus 2. The value must be greater than 0.
 * @param [in] transposed: Transposed convolution enable flag.
 * bool, when its value is True, transposed convolution is enabled.
 * @param [in] outputPadding: Output padding during backpropagation.
 * aclIntArray, only takes effect when transposed is True. The length of the array can be the weight dimension minus 2, and the value must be greater than or equal to 0 and less than stride.
 * @param [in] groups: Number of groups of input channels during backpropagation.
 * int, the value must be greater than 0, the C dimension of groups*weight = the C dimension of input.
 * @param [in] outputMask: Output mask parameter, specifies whether the output contains the gradient of the input, weight, and bias.
 * aclBoolArray, the output mask parameter of the backpropagation process is the gradient of the position corresponding to True.
 * @param [in] cubeMathType: used to determine which calculation logic the Cube unit should use for operations.
 * int8_t, Cube unit calculates logical judgment parameters.
 * @param [out] gradInput: convolution input gradient aclTensor on the NPU device side.
 * @param [out] gradWeight: The aclTensor of the convolution weight gradient on the NPU device side.
 * @param [out] gradBias: convolution bias gradient aclTensor on the NPU device side.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnConvolutionBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* input,
    const aclTensor* weight, const aclIntArray* biasSizes,
    const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool transposed,
    const aclIntArray* outputPadding, int groups,
    const aclBoolArray* outputMask, int8_t cubeMathType, aclTensor* gradInput,
    aclTensor* gradWeight, aclTensor* gradBias, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnConvTbcBackward's first interface, calculates and obtains the workspace size
 * @domain aclnn_ops_train
 *
 * @param [in] self: NPU, convolution output gradient
 * aclTensor on the device side, data type floating point type FLOAT16, FLOAT32
 * Supports non-continuous Tensor, data format supports ND, NCHW
 * @param [in] input: NPU, convolution input
 * aclTensor on the device side, data type floating point type FLOAT16, FLOAT32
 * Supports non-continuous Tensor, data format supports ND, NCHW
 * @param [in] weight: NPU, convolution weight
 * aclTensor on the device side, the data type is consistent with input
 * Supports non-continuous Tensor, the data format is consistent with input
 * @param [in] bias: NPU, convolution bias
 * aclTensor on the device side, the data type is consistent with input
 * @param [in] pad: padding
 * int64_t, (also equal to kernel size -1), for example: the effective length of the padding array of 2D convolution is 2 bits
 * @param [in] dilation: The interval between elements in the kernel, >1 represents atrous convolution
 * aclIntArray, the array length must be equal to the input dimension -2 (also equal to kernel size -1). For example: the effective length of the dilation array of 2D convolution is 2 bits
 * @param [out] grad_input: convolution input gradient aclTensor on the NPU device side
 * @param [out] grad_input: aclTensor of the convolution weight gradient on the NPU device side
 * @param [out] grad_bias: convolution bias gradient aclTensor on the NPU device side
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnConvTbcBackwardGetWorkspaceSize(
    const aclTensor* self, const aclTensor* input, const aclTensor* weight,
    const aclTensor* bias, int64_t pad, int8_t cubeMathType,
    aclTensor* gradInput, aclTensor* gradWeight, aclTensor* gradBias,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnConvolutionBackward is used to perform calculations.
 *
 * Operator function: Complete convolution reverse calculation
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnConvolutionBackwardGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnConvolutionBackward(void* workspace,
                                               uint64_t workspaceSize,
                                               aclOpExecutor* executor,
                                               const aclrtStream stream);

/**
 * The second interface of @brief aclnnConvTbcBackward is used to perform calculations.
 *
 * Operator function: Complete convolution reverse calculation
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnConvTbcbackwardGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnConvTbcBackward(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_CONVOLUTION_BACKWARD_H_
