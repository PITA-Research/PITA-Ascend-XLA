/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
#ifndef OP_API_INC_QUANT_GROUPED_MATMUL_DEQUANT_H_
#define OP_API_INC_QUANT_GROUPED_MATMUL_DEQUANT_H_

#include <string>

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnQuantGroupedMatmulDequant calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Function description: Quantize, matrix multiply and inverse quantize the input x.
 * @param [in] x: The aclTensor on the NPU device side supports non-continuous Tensor, the data type supports FLOAT16, and the data format supports ND.
 * @param [in] weight: aclTensor on the NPU device side, the data type supports INT8, and the data format supports ND and NZ.
 * @param [in] weightScale: aclTensor on the NPU device side, supports non-continuous Tensor, the data type supports FLOAT and INT64, and the data format supports ND.
 * @param [in] groupList: aclTensor on the NPU device side, supports non-continuous Tensor, the data type supports INT64, and the data format supports ND.
 * @param [in] biasOptional: aclTensor on the NPU device side supports non-continuous Tensor, the data type supports INT32, and the data format supports ND.
 * @param [in] xScaleOptional: aclTensor on the NPU device side, supports non-continuous Tensor, the data type supports FLOAT and FLOAT16, and the data format supports ND.
 * @param [in] xOffsetOptional: aclTensor on the NPU device side, supports non-continuous Tensor, the data type supports FLOAT and FLOAT16, and the data format supports ND.
 * @param [in] smoothScaleOptional: aclTensor on the NPU device side, supports non-continuous Tensor, the data type supports FLOAT16, and the data format supports ND.
 * @param [in] xQuantMode: char* type on the host side, indicating the quantization mode of x.
 * @param [in] transposeWeight: bool type on the host side, indicating whether the weight is transposed.
 * @param [out] out: aclTensor on the NPU device side only supports continuous Tensor, the data type supports FLOAT16, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnQuantGroupedMatmulDequantGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, const aclTensor *weightScale,
    const aclTensor *groupList, const aclTensor *biasOptional,
    const aclTensor *xScaleOptional, const aclTensor *xOffsetOptional,
    const aclTensor *smoothScaleOptional, char *xQuantMode,
    bool transposeWeight, const aclTensor *out, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnQuantGroupedMatmulDequant's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnQuantGroupedMatmulDequantGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnQuantGroupedMatmulDequant(void *workspace,
                                                     uint64_t workspaceSize,
                                                     aclOpExecutor *executor,
                                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_QUANT_GROUPED_MATMUL_DEQUANT_H_