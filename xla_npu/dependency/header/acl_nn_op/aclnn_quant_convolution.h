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
 * \file aclnn_quant_convolution.h
 * \brief
 */

#ifndef OP_API_INC_QUANT_CONVOLUTION_H
#define OP_API_INC_QUANT_CONVOLUTION_H

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief quant convolution interface, calculate and obtain the workspace size
 * @domain aclnn_ops_infer
 *
 * @param [in] input: npu，feature map
 * aclTensor on the device side, the data type supports INT8, HIFLOAT8, FLOAT8_E4M3FN
 * Supports non-continuous Tensor, data format supports NCHW, NCDHW
 * @param [in] weight: npu, kernels
 * aclTensor on the device side, the data type is consistent with input
 * Supports non-continuous Tensor, the data format is consistent with input
 * @param [in] bias: NPU, bias
 * aclTensor on device side
 * When the input data type is INT8, the data type supports BFLOAT16, FLOAT, and INT32.
 * When the input data type is HIFLOAT8 or FLOAT8_E4M3FN, the data type supports FLOAT.
 * Supports non-continuous Tensor, the data format is ND
 * @param [in] scale: npu, scale
 * The data type of aclTensor on the device side supports INT64, UINT64, and FLOAT. Data format supports ND
 * @param [in] offset: npu, offset
 * aclTensor on the device side is not supported yet.
 * @param [in] stride: stride length
 * For an array of int64, the array length must be equal to input_dim - 2. For example: the effective length of the step array of quant conv2d is 2 dimensions.
 * @param [in] padding: padding
 * Int64 array, the array length supports input_dim - 2 or (input_dim - 2) * 2. For example: the effective length of the padding array of quant conv2d supports 2 dimensions or 4 dimensions.
 * @param [in] dilation: The interval between elements in the kernel, >1 represents atrous convolution
 * For an int64 array, the array length must be equal to input_dim - 2. For example: the effective length of the dilation array of quant conv2d is 2 dimensions.
 * @param [in] groups: Number of groups, indicating the number of block links from input channels to output channels
 * int64, greater than 0, and can be evenly divided by the number of input and output channels, and the number of input channels = the number of weight channels * groups
 * @param [in] offsetx: quantization factor, indicating the offset to the value range during quantization scaling
 * int32, maximum offset to the right is 127, offset to the left is -128
 * @param [in] roundMode: rounding mode
 * When the output data type is INT8 or FLOAT8_E4M3FN, rint is supported.
 * When the output data type is HIFLOAT8, round is supported; other scenarios do not require it.
 * @param [out] output: npu
 * aclTensor on device side
 * When the input data type is INT8, the data type is FLOAT16;
 * When the input data type is HIFLOAT8, the data type supports FLOAT16, FLOAT32, BFLOAT16, HIFLOAT8
 * When the input data type is FLOAT8_E4M3FN, the data type supports FLOAT16, FLOAT32, BFLOAT16, FLOAT8_E4M3FN
 * The shape after broadcast has the same data format as input.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */

ACLNN_API aclnnStatus aclnnQuantConvolutionGetWorkspaceSize(
    const aclTensor* input, const aclTensor* weight, const aclTensor* bias,
    const aclTensor* scale, const aclTensor* offset, const aclIntArray* stride,
    const aclIntArray* padding, const aclIntArray* dilation, bool transposed,
    const aclIntArray* outputPadding, int64_t groups, int32_t offsetx,
    const char* roundMode, aclTensor* output, uint64_t* workspaceSize,
    aclOpExecutor** executor);
/**
 * @brief quant convolution interface for kernellaunch
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by aclnnQuantConvolutionGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process. After calling this interface, the executor is no longer available
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnQuantConvolution(void* workspace,
                                            const uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANT_CONVOLUTION_H