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
#ifndef OP_API_INC_TRANS_CONVOLUTION_WEIGHT_H
#define OP_API_INC_TRANS_CONVOLUTION_WEIGHT_H

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * aclnnCalculateConvolutionWeightSize is used to calculate the size of elements that need to be occupied by the weighttensor passed in when calling aclnnconvolution.
 *
 *
 * @param [in] tensorShape: Shape used to express the weight of the Convolution loading matrix
 * @param [in] transposed: used to express the Convolution type, whether to reverse
 * @param [in] groups: group used to express this Convolution
 * @param [in] dataType: Enter the Datatype of Weight, currently only supports Float16
 * @param [out] weightTensorSize: According to the internal processing logic of Convolution, calculate how many elements of data are needed after Weight conversion under the input
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnCalculateConvolutionWeightSize(
    const aclIntArray* tensorShape, bool transposed, int64_t groups,
    aclDataType dataType, uint64_t* weightTensorSize);

/**
 * The first interface of @brief aclnnTransConvolutionWeight calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Convert the input tensor to the specified dtype\format type.
 *
 * @param [in] weightIn: The input is a weightTensor of Convolution to be processed. The format is normal ND input. The data type supports Float16/Float32.
 * After being processed by this interface, the tensor is refreshed into the preprocessed weightTensor format and converted into a private format based on affinity, and the cast is a float16 data type.
 * @param [in] transposed: used to express the Convolution type, whether to reverse
 * @param [in] groups: group used to express this Convolution
 * @param [out] weightOut: Returns the tensor after the input weight is converted into a private format
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTransConvolutionWeightGetWorkspaceSize(
    const aclTensor* weightIn, bool transposed, const int64_t groups,
    aclTensor* weightOut, uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * The second interface of @brief aclnnTransConvolutionWeight is used to perform calculations.
 *
 * Operator function: Convert the input tensor to the specified dtype\format type.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnTransConvolutionWeightGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnTransConvolutionWeight(void* workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_TRANS_CONVOLUTION_WEIGHT_H_
