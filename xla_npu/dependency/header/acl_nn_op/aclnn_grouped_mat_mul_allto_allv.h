/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_ALL_TO_ALLV_
#define OP_API_INC_GROUPED_MATMUL_ALL_TO_ALLV_

#include <string>

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: implement grouped matmul + alltoallv fusion calculation
 * The first interface of @brief aclnnGroupedMatMulAlltoAllv calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] gmmX: Calculation input, Tensor, data type supports float16, bfloat16. This input performs AllToAll communication and only supports two dimensions.
 * The data format supports ND, and the result after communication is used as the left matrix calculated by GroupedMatMul
 * @param [in] gmmWeight: Calculation input, Tensor, data type supports float16, bfloat16, the type must be consistent with x, only supports three dimensions.
 * The data format supports ND, the right matrix calculated by GroupedMatMul
 * @param [in] sendCountsTensorOptional: Optional parameters, calculation input, Tensor, data type supports int32,
 * int64, the type must be consistent with x, and the data format supports ND
 * @param [in] recvCountsTensorOptional: Optional parameters, calculation input, Tensor, data type supports int32,
 * int64, the type must be consistent with x, and the data format supports ND
 * @param [in] mmXOptional: optional input parameter, calculation input, left matrix in shared expert matmul calculation in parallel. *Currently not supported
 * @param [in] mmWeightOptional: Optional input parameter, calculation input, right matrix in shared expert matmul calculation in parallel. *Currently not supported
 * @param [in] group: calculation input, str. ep communication domain name, expert parallel communication domain
 * @param [in] epWorldSize: calculation input, int. ep communication domain size
 * @param [in] sendCounts: calculation input, list int. The amount of data sent by communication
 * @param [in] recvCounts: calculation input, list int. Amount of data received by communication
 * @param [in] transGmmWeight: Optional parameter, calculation input. Indicates whether the right matrix of gmm needs to be transposed, the default is false
 * @param [in] transMmWeight: optional parameter, calculation input. Indicates whether the right matrix of the shared expert mm needs to be transposed. *Currently not supported
 * @param [out] y: Calculation output, Tensor, data type supports float16, bfloat16. The final calculation result, the data type is consistent with the input gmmX
 * @param [out] mmYOptional: optional output, calculation output, Tensor, data type supports float16,
 * bfloat16, the output of shared expert matmul, is output only when mmX and mmWeight are passed in, and the data type is consistent with mmX. *Currently not supported
 * @param [out] workspaceSize: Output parameter, returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Out parameter, returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return value, return status code.
 *
 * Because required for collective communication and BatchMatMul calculations, the input and output shapes need to satisfy the following mathematical relationship: (where ep=epWorldSize, tp=tpWorldSize)
 * gmmX: (BSK, H);
 * gmmWeight: (e, H, N1);
 * groupList: (e);
 * mmX: (BS, H);
 * mmWeight: (H, N2);
 * y: (A, N1);
 * mmY: (BS, N2);
 *
 * Data relationship description:
 * e represents the number of experts on a single card;
 * A = cumulative sum of recvCounts;
 */
ACLNN_API aclnnStatus aclnnGroupedMatMulAlltoAllvGetWorkspaceSize(
    const aclTensor* gmmX, const aclTensor* gmmWeight,
    const aclTensor* sendCountsTensorOptional,
    const aclTensor* recvCountsTensorOptional, const aclTensor* mmXOptional,
    const aclTensor* mmWeightOptional, const char* group, int64_t epWorldSize,
    const aclIntArray* sendCounts, const aclIntArray* recvCounts,
    bool transGmmWeight, bool transMmWeight, aclTensor* y,
    aclTensor* mmYOptional, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnGroupedMatMulAlltoAllv is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnGroupedMatMulAlltoAllvGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnGroupedMatMulAlltoAllv(void* workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GROUPED_MATMUL_ALL_TO_ALLV_