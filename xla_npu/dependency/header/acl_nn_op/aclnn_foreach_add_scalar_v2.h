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
 * \file aclnn_foreach_add_scalar_v2.h
 * \brief
 */

#ifndef OP_API_INC_ACLNN_FOREACH_ADD_SCALAR_V2_H_
#define OP_API_INC_ACLNN_FOREACH_ADD_SCALAR_V2_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnForeachAddScalarV2 calculates the workspace size according to the specific calculation process.
 * Function description: Add each element of the input matrix to scalar and output it.
 * Calculation formula:
 * out_{i}=x_{i}+scalar
 * @domain aclnnop_math
 * Parameter description:
 * @param [in]   x
 * Input Tensor, the data type supports FLOAT, FLOAT16, BFLOAT16, INT32. The data format supports ND.
 * @param [in]   scalar
 * Enter Scalar, the data type supports FLOAT, FLOAT16 and INT32. The data format supports ND.
 * @param [in]   out
 * Output Tensor, data type supports FLOAT, FLOAT16, BFLOAT16, INT32. The data format supports ND.
 * @param [out] workspaceSize returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor returns the op executor, including the operator calculation process.
 * @return aclnnStatus return status code
 */
ACLNN_API aclnnStatus aclnnForeachAddScalarV2GetWorkspaceSize(
    const aclTensorList *x, const aclScalar *scalar, aclTensorList *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnForeachAddScalarV2 is used to perform calculations.
 * Function description: Add each element of the input matrix to scalar and output it.
 * Calculation formula:
 * out_{i}=x_{i}+scalar
 * @domain aclnnop_math
 * Parameter description:
 * param [in] workspace: the starting address of the workspace memory applied for on the NPU device side.
 * param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnForeachAddScalarV2GetWorkspaceSize.
 * param [in] stream: ACL stream.
 * param [in] executor: op executor, including the operator calculation process.
 * return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnForeachAddScalarV2(void *workspace,
                                              uint64_t workspaceSize,
                                              aclOpExecutor *executor,
                                              aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
