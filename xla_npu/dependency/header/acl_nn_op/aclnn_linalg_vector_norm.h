/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LINALG_VECTOR_NORM_H_
#define OP_API_INC_LINALG_VECTOR_NORM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnLinalgVectorNorm calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] self: aclTensor on NPU device side, data type supports floating point data type
 * @param [in] ord: host side aclScalar, data type supports floating point data type
 * @param [in] dims: aclIntArray on the NPU device side, representing the information of the corresponding dimensionality reduction axis Axis
 * @param [in] keepDims: bool on the host side, indicating whether to retain the information corresponding to the dim dimension
 * @param [in] dtype: Specify the data class when calculating self. Before calculation, convert self into the type specified by dtype for calculation. The data type supports FLOAT and FLOAT16.
 * @param [in] out: output tensor
 * @param [in] workspaceSize: Returns the workspace size requested by the user on the NPU device side.
 * @param [in] executor: Returns the op executor, including the operator calculation process
 * @return aclnnStatus: return status code
 * */
ACLNN_API aclnnStatus aclnnLinalgVectorNormGetWorkspaceSize(
    const aclTensor* self, const aclScalar* ord, const aclIntArray* dims,
    bool keepDims, const aclDataType dtype, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of aclnnLinalgVectorNorm is used to perform calculations
 * @param [in] workspace: the workspace memory address applied for on the NPU device side
 * @param [in] workspaceSize: the workspace size applied for on the NPU device side,
 * Obtained by the first interface aclnnLinalgVectorNormGetWorkspaceSize
 * @param [in] executor: Returns the op executor, including the operator calculation process
 * @param [in] stream: ACL stream
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnLinalgVectorNorm(void* workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LINALG_VECTOR_NORM_H_