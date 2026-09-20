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

#ifndef OP_API_INC_SWI_GLU_QUANT_H_
#define OP_API_INC_SWI_GLU_QUANT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnSwiGluQuant calculates the workspace size according to the specific calculation process.
 * Function description: Add quant operation after SwiGlu activation function to realize SwiGluQuant calculation of input x
 * Dynamic quantification calculation formula:
 * Activation = SwiGLU(x) = Swish(A)*B \\
 * Y_{tmp}^i = Activation[i:group\_index[i], :] *  smooth\_scales[i:group\_index[i], :], i \in [0, G) \cap \mathbb{Z}\\
 * scale=row\_max(abs(Y_{tmp}))/127
 * Y = Cast(Mul(Y_{tmp}, Scale))
 * Among them, A represents the first half of the input x, B represents the second half of the input x, and G is the number of groups in group_index.
 * Static quantization calculation formula:
 * Y=AscendQuant(SwiGLU(x), smooth\_scales, offsets)
 * Among them, A represents the first half of the input x, and B represents the second half of the input x.

 * @domain aclnnop_ops_infer
 * Parameter description:
 * @param [in]   x
 * Enter aclTensor, the data type supports FLOAT, FLOAT16, and BFLOAT16. The data format supports ND. Supports non-continuous Tensors. The last dimension needs to be a multiple of 2, and the dimension of x must be greater than 1 dimension. Currently, only the length of the last dimension of input x does not exceed 8192.
 * @param [in]   smoothScalesOptional
 * Enter aclTensor, the data type supports FLOAT, and the threshold value is calculated by threshold. The data format supports ND. Supports non-continuous Tensors. shape supports [G, N], [G, ], where G represents the number of groupIndex groups, and N is half the size of the last dimension of the calculated input x.
 * @param [in]   offsetsOptional
 * Output aclTensor. This parameter does not take effect in dynamic quantization scenarios. The user can just pass in a null pointer. In the static quantization scenario: the data type supports FLOAT, the shape supports [G, N] in per_channel mode, and the shape supports [G, ] in per_tensor mode, and the data type and shape need to be consistent with smoothScalesOptional. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in]   groupIndexOptional
 * The data type supports INT32, group_index required for MoE grouping, corresponding to the formula group_index variable, and aclTensor on the Device side. The data format supports ND. Supports non-continuous Tensors. shape supports [G, ].
 * @param [in]   activateLeft
 * The data type supports bool, indicating whether the left matrix participates in the operation, and the user must pass parameters.
 * @param [in]   quantModeOptional
 * The data type supports String, indicating whether the left matrix participates in the operation, and the user must pass parameters. Users must pass parameters, "static" means static quantization, "dynamic" means dynamic quantization, and "dynamic_msd" means dynamic MSD quantization. Currently only "dynamic" dynamic quantization and "static" static quantization are supported. Static quantization only supports per_tensor quantization and per_channel quantization.
 * @param [in]   yOut
 * The data type supports INT8 and the data format supports ND. Supports non-continuous Tensors. The size of the last dimension of the shape of the calculation output yOut is half of the last dimension of the calculation input x, and the remaining dimensions are consistent with x.
 * @param [in]   scaleOut
 * The data type supports FLOAT, and the data format supports ND. Supports non-continuous Tensors. Compared with the calculation input x, the shape of the calculation output scaleOut does not have the last dimension, and the remaining dimensions are consistent with the calculation input x.
 * @param [out] workspaceSize returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor returns the op executor, including the operator calculation process.
 * @return aclnnStatus return status code
 */

ACLNN_API aclnnStatus aclnnSwiGluQuantGetWorkspaceSize(
    const aclTensor *x, const aclTensor *smoothScalesOptional,
    const aclTensor *offsetsOptional, const aclTensor *groupIndexOptional,
    bool activateLeft, char *quantModeOptional, const aclTensor *yOut,
    const aclTensor *scaleOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnSwiGluQuant's second interface is used to perform calculations.
 * Function description: Add quant operation after SwiGlu activation function to realize SwiGluQuant calculation of input x
 * Dynamic quantification calculation formula:
 * Activation = SwiGLU(x) = Swish(A)*B \\
 * Y_{tmp}^i = Activation[i:group\_index[i], :] *  smooth\_scales[i:group\_index[i], :], i \in [0, G) \cap \mathbb{Z}\\
 * scale=row\_max(abs(Y_{tmp}))/127
 * Y = Cast(Mul(Y_{tmp}, Scale))
 * Among them, A represents the first half of the input x, B represents the second half of the input x, and G is the number of groups in group_index.
 * Static quantization calculation formula:
 * Y=AscendQuant(SwiGLU(x), smooth\_scales, offsets)
 * Among them, A represents the first half of the input x, and B represents the second half of the input x.

 * @domain aclnnop_ops_infer
 * Parameter description:
 * param [in] workspace: the starting address of the workspace memory applied for on the NPU device side.
 * param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnSwiGluQuantGetWorkspaceSize.
 * param [in] stream: ACL stream.
 * param [in] executor: op executor, including the operator calculation process.
 * return aclnnStatus: Return status code.
 */

ACLNN_API aclnnStatus aclnnSwiGluQuant(void *workspace, uint64_t workspaceSize,
                                       aclOpExecutor *executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SWI_GLU_QUANT_H_