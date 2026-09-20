/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_SWISH_H_
#define OP_API_INC_LEVEL2_ACLNN_SWISH_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnSwish calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * Operator function: Swish activation function
 * @param [in] self: aclTensor on the Device side, input in the formula. Supports non-continuous Tensors, the data format supports ND, and the shape and data type of self and out are consistent.
 * @param [in] betaOptional: aclScalar on the Host side, beta in the formula. The data type needs to be a data type convertible to FLOAT.
 * When betaOptional is a null pointer, the default value is 1.0.
 * @param [out] out: aclTensor on the Device side, output in the formula. Supports non-continuous Tensor, and the data format supports ND.
 * The shape and data type of self and out are consistent.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSwishGetWorkspaceSize(const aclTensor* self,
                                                 const aclScalar* betaOptional,
                                                 aclTensor* out,
                                                 uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnSwish is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAcosGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnSwish(void* workspace, uint64_t workspaceSize,
                                 aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
