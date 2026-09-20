/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_convolution.h
 * \brief
 */

#ifndef OP_API_INC_CONVOLUTION_H
#define OP_API_INC_CONVOLUTION_H

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * The first interface of @brief aclnnConvolution, calculates and obtains the workspace size
 * @domain aclnn_ops_infer
 *
 * @param [in] input: NPU, convolution input.
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, and FLOAT8_E4M3FN.
 * Supports non-continuous Tensor, and the data format supports NCL, NCHW, and NCDHW.
 * @param [in] weight: NPU, convolution weight.
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, and FLOAT8_E4M3FN.
 * Supports non-continuous Tensor, and the data format supports NCL, NCHW, and NCDHW.
 * @param [in] bias: NPU, bias.
 * The data type of aclTensor on the device side supports BFLOAT16, FLOAT16, and FLOAT.
 * Supports non-continuous Tensor, and the data format supports NCL, NCHW, NCDHW, and ND.
 * @param [in] stride: convolution scan stride.
 * aclIntArray, the array length must be equal to the input dimension minus 2. Its value should be greater than 0.
 * @param [in] padding: padding for input.
 * aclIntArray, for conv1d non-transposed convolution, the array length can be 1 or 2; for conv2d array length, it can be 2 or 4; and conv3d array length can be 3. Its value should be greater than or equal to 0.
 * @param [in] dilation: The interval between elements in the convolution kernel.
 * aclIntArray, the array length must be equal to the input dimension minus 2. Its value should be greater than 0.
 * @param [in] transposed: Whether it is transposed convolution.
 * bool, True represents transposed convolution.
 * @param [in] outputPadding: effective when transposed convolution, supplementary edge for output.
 * aclIntArray, the array length must be equal to the input dimension minus 2. Its value should be greater than or equal to 0 and less than the value of the corresponding dimension of stride or dilation. In the case of non-transposed convolution, this attribute configuration is ignored.
 * @param [in] groups: Indicates the number of block links from input channels to output channels.
 * int64_t, the value must be greater than 0, and satisfy the C dimension of groups*weight = the C dimension of input.
 * @param [out] output: convolution output.
 * For aclTensor on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16, HIFLOAT8, FLOAT8_E4M3FN, and the data format supports NCL, NCHW, and NCDHW.
 * @param [in] cubeMathType: used to determine which calculation logic the Cube unit should use for operations.
 * int8_t, Cube unit calculates logical judgment parameters.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnConvolutionGetWorkspaceSize(
    const aclTensor* input, const aclTensor* weight, const aclTensor* bias,
    const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, bool transposed,
    const aclIntArray* outputPadding, const int64_t groups, aclTensor* output,
    int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The first interface of aclnnConvTbc calculates and obtains the workspace size
 * @domain aclnn_ops_infer
 *
 * @param [in] self: NPU, Tbc convolution input.
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, BFLOAT16, and HIFLOAT8.
 * Supports non-continuous Tensor, and the data format supports ND and NCL.
 * @param [in] weight: NPU, Tbc convolution weight.
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, BFLOAT16, and HIFLOAT8.
 * Supports non-continuous Tensor, and the data format supports ND and NCL.
 * @param [in] bias: NPU, bias.
 * The data type of aclTensor on the device side supports BFLOAT16, FLOAT16, and FLOAT.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] pad: padding for self.
 * int64_t, its value should be greater than or equal to 0.
 * @param [out] output: Tbc convolution output.
 * For aclTensor on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16, and HIFLOAT8, and the data format supports NCL.
 * @param [in] cubeMathType: used to determine which calculation logic the Cube unit should use for operations.
 * int8_t, Cube unit calculates logical judgment parameters.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnConvTbcGetWorkspaceSize(
    const aclTensor* self, const aclTensor* weight, const aclTensor* bias,
    const int64_t pad, aclTensor* output, int8_t cubeMathType,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The first interface of aclnnConvDepthwise2d calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * @param [in] self: NPU, depthwise2d convolution input.
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, BFLOAT16, and HIFLOAT8.
 * Supports non-continuous Tensor, and the data format supports NCHW.
 * @param [in] weight: NPU, depthwise2d convolution weight.
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, BFLOAT16, and HIFLOAT8.
 * Supports non-continuous Tensor, and the data format supports NCHW.
 * @param [in] kernelSize: NPU, depthwise2d convolution kernel size.
 * The aclTensor on the device side supports non-continuous Tensor.
 * @param [in] bias: NPU, bias.
 * The aclTensor on the device side supports non-continuous Tensor, supports one dimension and the value is equal to the first dimension of weight.
 * @param [in] stride: depthwise2d convolution scan stride.
 * aclIntArray, the length of the array must be equal to the dimension of self minus 2. Its value should be greater than 0.
 * @param [in] padding: padding for self.
 * aclIntArray, the length of the array must be equal to the dimension of self minus 2. Its value should be greater than or equal to 0.
 * @param [in] dilation: The interval between elements in the convolution kernel.
 * aclIntArray, the length of the array must be equal to the dimension of self minus 2. Its value should be greater than 0.
 * @param [out] out: depthwise2d convolution output.
 * The data type of aclTensor on the device side supports FLOAT, FLOAT16, BFLOAT16, and HIFLOAT8. The data format supports NCHW.
 * @param [in] cubeMathType: used to determine which calculation logic the Cube unit should use for operations.
 * int8_t, Cube unit calculates logical judgment parameters.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnConvDepthwise2dGetWorkspaceSize(
    const aclTensor* self, const aclTensor* weight,
    const aclIntArray* kernelSize, const aclTensor* bias,
    const aclIntArray* stride, const aclIntArray* padding,
    const aclIntArray* dilation, aclTensor* out, int8_t cubeMathType,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnConvolution is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by aclnnConvolutionGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process. After calling this interface, the executor is no longer available
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnConvolution(void* workspace,
                                       const uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream);

/**
 * @brief The second interface of aclnnConvTbc is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by aclnnConvTbcGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process. After calling this interface, the executor is no longer available
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnConvTbc(void* workspace,
                                   const uint64_t workspaceSize,
                                   aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief The second interface of aclnnConvDepthwise2d is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by aclnnConvDepthwise2dGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process. After calling this interface, the executor is no longer available
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnConvDepthwise2d(void* workspace,
                                           const uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_CONVOLUTION_H
