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

#ifndef OP_API_INC_GCD_H_
#define OP_API_INC_GCD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnGcd calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Calculate the greatest common divisor of element-wise dimensions for the given self and other.
 *
 * @param [in] self:
 * Enter `self` and satisfy the data type derivation rules with other. After derivation, the data type supports INT32 and INT16 (Ascend910B). Shape needs to satisfy the broadcast rules with other.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] other:
 * Enter `other` and satisfy the data type derivation rules with self. After derivation, the data type supports INT32 and INT16 (Ascend910B). Shape needs to satisfy the broadcast rules with self.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] out: Output `out`, the data type supports INT32, INT16 (Ascend910B). The shape needs to be consistent with the shape after broadcasting self and other.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGcdGetWorkspaceSize(const aclTensor* self,
                                               const aclTensor* other,
                                               aclTensor* out,
                                               uint64_t* workspaceSize,
                                               aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnGcd is used to perform calculations.
 *
 * Operator function: Calculate the greatest common divisor of element-wise dimensions for the given self and other.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGcdGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGcd(void* workspace, uint64_t workspaceSize,
                               aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_GCD_H_
