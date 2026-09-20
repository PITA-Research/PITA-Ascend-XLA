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
#ifndef OP_API_INC_LOG_H_
#define OP_API_INC_LOG_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnLog calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: complete the calculation of natural logarithms
 * Calculation formula:
 * $$ output = log_e(self) $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C -->D([Log])
 *     E[(Out)] -->F([Contiguous])
 *     F --> G([Cast])
 *     G -->D([Log])
 *     D --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 *
 * @param [in] self: aclTensor on the NPU device side. The data type supports FLOAT, FLOAT16, DOUBLE, COMPLEX64, and COMPLEX128.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] out: aclTensor on the NPU device side, data type supports FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128. The data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLogGetWorkspaceSize(const aclTensor* self,
                                               aclTensor* out,
                                               uint64_t* workspaceSize,
                                               aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnLog is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnLogGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnLog(void* workspace, uint64_t workspaceSize,
                               aclOpExecutor* executor, aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceLog calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete the inplace calculation of natural logarithms.
 * Calculation formula:
 * $$ selfRef_i = log_e(selfRef_i) $$
 *
 * @param [in] selfRef:
 * Input `selfRef` in the formula, the data type supports FLOAT, FLOAT16, DOUBLE, COMPLEX64, COMPLEX128, non-continuous Tensor is supported, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceLogGetWorkspaceSize(aclTensor* selfRef,
                                                      uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceLog is used to perform calculations.
 *
 * Operator function: Complete the inplace calculation of natural logarithms.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceAddGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceLog(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor,
                                      aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LOG_H_
