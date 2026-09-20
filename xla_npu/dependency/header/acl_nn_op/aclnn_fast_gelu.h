/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_FAST_GELU_H_
#define OP_API_INC_LEVEL2_ACLNN_FAST_GELU_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFastGelu's first interface calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Returns the calculation result of input FastGelu
 * Calculation formula:
 * $$ FastGelu(x_i) = \frac {x_i} {1 + \exp(-1.702 * \left| x_i \right|)} * \exp(0851 * (x_i - \left| x_i \right|)) $$
 *
 *
 * @param [in] self: input tensor, aclTensor on the NPU device side,
 * The data type supports FLOAT16, FLOAT32, and BFLOAT16, the data format supports ND, and non-continuous Tensors are supported.
 * @param [in] out: output tensor, aclTensor on the NPU device side,
 * The data type supports FLOAT16, FLOAT32, and BFLOAT16, the data format supports ND, and non-continuous Tensors are supported.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnFastGeluGetWorkspaceSize(const aclTensor* self,
                                                    aclTensor* out,
                                                    uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnFastGelu is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnFastGeluGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnFastGelu(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif /* OP_API_INC_LEVEL2_ACLNN_FAST_GELU_H_ */