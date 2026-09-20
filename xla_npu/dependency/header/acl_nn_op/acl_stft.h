/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/license/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OP_API_INC_LEVEL2_ACL_STFT_H_
#define OP_API_INC_LEVEL2_ACL_STFT_H_

#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclStft calculates the workspace size according to the specific calculation process.
 * @domain aclnn_fft
 * Operator function: Returns the result of inputting Tensor and doing Stft
 * Calculation formula:
 * $$ out = Stft(input) $$
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *   A[(self)]--->B([l0op::Stft])
 *   B--->C([l0op::ViewCopy])
 *   C--->D[(out)]
 * ```
 *
 * @param [in] self: aclTensor on the NPU device side. The input to be calculated is required to be a 1D/2D Tensor with a shape of (L)/(B, L).
 * Among them, L is the length of the time series sampling sequence, and B is the number of time series sampling sequences. Data types support FLOAT32, DOUBLE, COMPLEX64, COMPLEX128,
 * Supports non-continuous Tensor, the data format requirement is ND.
 * @param [in] windowOptional: aclTensor on the NPU device side, optional parameter, required to be a 1D Tensor with shape (winLength),
 * winLength is the length of the STFT window function. The data type supports FLOAT32, DOUBLE, COMPLEX64, and COMPLEX128, and the data type is consistent with self, and the data format requirement is ND.
 * @param [in] out: aclTensor on the NPU device side, the Fourier transform result of self in the window, which is required to be a 2D/3D/4D Tensor.
 * The data type supports FLOAT32, DOUBLE, COMPLEX64, COMPLEX128, supports non-continuous Tensor, and the data format requirement is ND.
 * @param [in] nFft: required parameter, int on Host side, number of FFT points (greater than 0).
 * @param [in] hopLength: required parameter, int on Host side, interval of sliding window (greater than 0).
 * @param [in] winLength: required parameter, int on Host side, window size (greater than 0).
 * @param [in] normalized: required parameter, bool on the Host side, whether to normalize the Fourier transform results.
 * @param [in] onesided: Required parameter, bool on the Host side, whether to return all results or half of the results.
 * @param [in] returnComplex: required parameter, bool on the Host side, confirm that the return value is a complex tensor or a tensor with separate real and imaginary parts.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process
 * @return aclnnStatus: return status code
 */
aclnnStatus aclStftGetWorkspaceSize(const aclTensor* self,
                                    const aclTensor* windowOptional,
                                    aclTensor* out, int64_t nFft,
                                    int64_t hopLength, int64_t winLength,
                                    bool normalized, bool onesided,
                                    bool returnComplex, uint64_t* workspaceSize,
                                    aclOpExecutor** executor);

/**
 * @brief The second interface of aclStft is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAbsGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
aclnnStatus aclStft(void* workspace, uint64_t workspaceSize,
                    aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACL_STFT_H_