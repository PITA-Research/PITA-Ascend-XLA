/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_LINALG_QR_H_
#define OP_API_INC_LINALG_QR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnLinalgQr calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete QR operation on input Tensor
 * @param [in] self: calculation input, data type supports FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128, data format supports ND,
 * Support non-continuous Tensors
 * @param [in] mode: Calculate input, specify whether the output tensor shape is complete, the data type supports int64_t
 * @param [out] Q: Calculation output,
 * The orthogonal matrix obtained after Qr decomposition, the data type supports FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128, and is the same as self. The data format supports ND,
 * Supports non-contiguous Tensors.
 * @param [out] R: Calculation output,
 * The upper triangular matrix obtained after Qr decomposition, the data type supports FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128, and is the same as self. The data format supports ND,
 * Supports non-contiguous Tensors.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnLinalgQrGetWorkspaceSize(const aclTensor* self,
                                                    int64_t mode, aclTensor* Q,
                                                    aclTensor* R,
                                                    uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnLinalgQr, used to perform calculations
 *
 * Operator function: Complete Qr operation on input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnLinalgQrGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLinalgQr(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LINALG_QR_H_
