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
#ifndef OP_API_INC_LEVEL2_ACLNN_NANTONUM_H_
#define OP_API_INC_LEVEL2_ACLNN_NANTONUM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnNanToNum calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Replace NaN, positive infinity and negative infinity in the input with the values ​​specified by nan, posinf and neginf respectively.
 *
 * @param [in] self: Input parameter to be calculated for NanToNum. aclTensor on the NPU device side,
 * Data types support FLOAT32, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, BOOL,
 * The data format supports ND, and the data format needs to be consistent with out. Non-continuous Tensors are supported.
 * @param [in] nan: input parameter. The data type supports FLOAT, replacing NaN values ​​in tensor elements.
 * @param [in] posinf: input parameter. The data type supports FLOAT, which replaces positive infinity values ​​in tensor elements.
 * @param [in] neginf: input parameters. The data type supports FLOAT, which replaces negative infinity values ​​in tensor elements.
 * @param [out] out: The output parameter calculated by NanToNum. aclTensor on the NPU device side,
 * Data types support FLOAT32, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8, BOOL,
 * The data format supports ND, and the data format needs to be consistent with self. Non-continuous Tensors are supported.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnNanToNumGetWorkspaceSize(
    const aclTensor* self, float nan, float posinf, float neginf,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnNanToNum is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnNanToNumGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnNanToNum(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceNanToNum calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * @param [in] selfRef: input parameter to be calculated for NanToNum. aclTensor on the NPU device side,
 * The data type supports FLOAT64, FLOAT32, FLOAT16, and BFLOAT16. The data format supports ND, and the data format needs to be consistent with out. Non-continuous Tensors are supported.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceNanToNumGetWorkspaceSize(
    aclTensor* selfRef, float nan, float posinf, float neginf,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnInplaceNanToNum is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnNanToNumGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceNanToNum(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_NANTONUM_H_