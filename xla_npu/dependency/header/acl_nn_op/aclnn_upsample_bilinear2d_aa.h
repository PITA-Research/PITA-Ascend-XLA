/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 *Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at
*    http://www.apache.org/licenses/LICENSE-2.0
*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an &quot;AS IS&quot; BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
 */

#ifndef OP_API_INC_UPSAMPLE_BILINEAR2D_AA_H_
#define OP_API_INC_UPSAMPLE_BILINEAR2D_AA_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnUpsampleBilinear2dAA calculates the workspace size according to the specific calculation process.
 * Function description: Apply 2D bilinear anti-aliasing sampling to an input signal consisting of multiple input channels.
 * @domain aclnn_ops_infer
 * Parameter description:
 * @Param [in] input
 * Input Tensor, the data type supports FLOAT, FLOAT16, and BFLOAT16. Supports non-continuous Tensor, the data format supports NCHW, and the shape dimension only supports 4-dimensional Tensor. The data type is consistent with the data type of the outgoing parameter.
 * @param [in] outputSize The size of the output space. It is required to be a two-dimensional array. The data type supports INT64. The value is the same as the H and W dimensions of out.
 * @Param [in] alignCorners
 * Whether it is corner aligned. If set to true, the input and output tensors are aligned at the center points of their corner pixels, preserving the values ​​at the corner pixels. If set to false, the input and output tensors are aligned by the corner points of their corner pixels, and values ​​outside the bounds are padded with edge values.
 * @param [in] scalesH height dimension multiplier of space size.
 * @param [in] scalesW The width dimension multiplier of the space size.
 * @Param [in] out
 * The data type supports FLOAT, FLOAT16, and BFLOAT16. Supports non-continuous Tensor, and the data format supports NCHW. The shape dimension only supports 4-dimensional Tensor. The data type is consistent with the data type of the input parameter.
 * @param [out] workspaceSize returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor returns the op executor, including the operator calculation process.
 */
ACLNN_API aclnnStatus aclnnUpsampleBilinear2dAAGetWorkspaceSize(
    const aclTensor *input, const aclIntArray *outputSize, bool alignCorners,
    double scalesH, double scalesW, aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnUpsampleBilinear2dAA's second interface is used to perform calculations.
 * Function description: Apply 2D bilinear anti-aliasing sampling to an input signal consisting of multiple input channels.
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnUpsampleBilinear2dAA(void *workspace,
                                                uint64_t workspaceSize,
                                                aclOpExecutor *executor,
                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_UPSAMPLE_BILINEAR2D_AA_H_