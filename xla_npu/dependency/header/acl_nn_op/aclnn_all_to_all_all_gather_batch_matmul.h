/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ALL_TO_ALL_ALL_GATHER_BATCH_MATMUL_
#define OP_API_INC_ALL_TO_ALL_ALL_GATHER_BATCH_MATMUL_

#include <string>

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: implement alltoall + allGather + bmm fusion calculation
 * The first interface of @brief aclnnAlltoAllAllGatherBatchMatMul calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] x: Calculation input, Tensor, data type supports float16, bfloat16. This input is used for AllToAll and AllGather set communication. It must be 3-dimensional. The data format supports ND. The result after communication is used as the left matrix calculated by BatchMatMul.
 * @param [in] weight: Calculation input, Tensor, data type supports float16, bfloat16, type must be consistent with x, must be 3 dimensions, data format supports ND, right matrix calculated by BatchMatMul
 * @param [in] biasOptional: Calculation input, Tensor, data type supports float16, float32. When x is float16, bias needs to be float16; when x is bfloat16, bias needs to be float32, which must be two-dimensional or three-dimensional. The data format supports ND, and the bias calculated by BatchMatMul
 * @param [in] groupEp: calculation input, str. ep communication domain name, expert parallel communication domain
 * @param [in] groupTp: calculation input, str. tp communication domain name, Tensor parallel communication domain
 * @param [in] epWorldSize: calculation input, int. ep communication domain size, supports 2/4/8/16/32
 * @param [in] tpWorldSize: calculation input, int. tp communication domain size, supports 2/4/8/16/32
 * @param [in] xShardType: calculation input, int, the default value is 0, 0 means allgather according to the tp domain in the H dimension, 1 means allgather according to the tp domain in the C dimension.
 * @param [in] actType: calculation input, int, activation function type, the default value is 0, indicating no activation function. Supports passing in 0/1/2/3/4, the corresponding relationship is [0: None, 1: GELU, 2: Silu, 3: Relu, 4: FastGELU], etc.
 * @param [out] y1Out: Calculation output, Tensor, data type supports float16, bfloat16, only supports 3 dimensions. The final calculation result, if there is an activation function, is the output of the activation function, otherwise it is the output of BatchMatMul. The data type remains the same as input x
 * @param [out] y2OutOptional: Calculation output, Tensor, optional output, data type supports float16, bfloat16, only supports 3 dimensions. The output data type of AllGather is consistent with the input x. Reverse may require only the results of allgather communication operations, data type: same as input.
 * @param [out] y3OutOptional: Calculation output, Tensor, optional output, data type supports float16, bfloat16, only supports 3 dimensions. When there is an activation function, the output type of BatchMatMul is consistent with the input x. When ctType is non-0, only the result of bmm calculation is used. The data type is the same as the input.
 * @param [out] workspaceSize: Output parameter, returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Out parameter, returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return value, return status code.
 *
 * Because required for collective communication and BatchMatMul calculations, the input and output shapes need to satisfy the following mathematical relationship: (where ep=epWorldSize, tp=tpWorldSize)
 * Perform the AllGather scene according to the H axis, that is, the scene where xShardType is 0:
 * x: (E, C, H/tp)；
 * weight：(E/ep, H, M/tp)；
 * bias: (E/ep, 1, M/tp); supports two dimensions or three dimensions. In two dimensions, the shape is: (E/ep, M/tp);
 * y1Out：(E/ep, ep * C, M/tp)；
 * y2OutOptional：(E/ep, ep * C, H)；
 * y3OutOptional：(E/ep, ep * C, M/tp)；
 * Press the C axis to perform the AllGather scene, that is, the xShardType is 0 scene:
 * x: (E, C/tp, H)；
 * weight：(E/ep, H, M/tp)；
 * bias: (E/ep, 1, M/tp); supports two dimensions or three dimensions. In two dimensions, the shape is: (E/ep, M/tp);
 * y1：(E/ep, ep * tp * C/tp, M/tp)；
 * y2：(E/ep, ep * tp * C/tp, H)；
 * y3：(E/ep, ep * tp * C/tp, M/tp)
 * Data relationship description:
 * For example, x.size(0) is equal to E and weight.size(0) is equal to E/ep, which means that x.size(0) = ep * weight.size(0), x.size(0) is an integer multiple of ep; other relationships are similar
 * The value range of E is [2, 512], and E is an integer multiple of ep;
 * The value range of H is: [1, 65535]. When xShardType is 0, H is an integer multiple of tp;
 * The value range of M/tp is: [1, 65535];
 * The value range of E/ep is: [1, 32];
 * Both ep and tp only support 2, 4, 8, 16, and 32;
 * The names of groupEp and groupTp cannot be the same;
 * C is greater than 0, and the upper limit is the upper limit of the operator device memory. When xShardType is 1, C is an integer multiple of tp;
 * Cross-supernodes are not supported, only within supernodes.
 */
ACLNN_API aclnnStatus aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize(
    const aclTensor* x, const aclTensor* weight, const aclTensor* biasOptional,
    const char* groupEp, const char* groupTp, int64_t epWorldSize,
    int64_t tpWorldSize, int64_t xShardType, int64_t actType, aclTensor* y1Out,
    aclTensor* y2OutOptional, aclTensor* y3OutOptional, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnAlltoAllAllGatherBatchMatMul's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnAlltoAllAllGatherBatchMatMul(void* workspace,
                                                        uint64_t workspaceSize,
                                                        aclOpExecutor* executor,
                                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ALL_TO_ALL_ALL_GATHER_BATCH_MATMUL_