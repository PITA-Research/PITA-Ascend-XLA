/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_HARDSIGMOID_BACKWARD_H_
#define OP_API_INC_HARDSIGMOID_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnHardsigmoidBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: reverse calculation of activation function hardsigmoid.
 *
 * @param [in] gradOutput(aclTensor*, calculation input): The data type supports FLOAT and FLOAT16, supports non-continuous Tensor, and the data format supports ND.
 * @param [in] self(aclTensor*, calculation input): The data type supports FLOAT and FLOAT16, supports non-continuous Tensor, and the data format supports ND.
 * @param [out] out(aclTensor*, calculation output): The data type supports FLOAT and FLOAT16, supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspace_size(uint64_t*, out parameter): Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor(aclOpExecutor**, out parameter): Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnHardsigmoidBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnHardsigmoidBackward performs calculations.
 *
 * Operator function: reverse calculation of activation function hardsigmoid.
 *
 * @param [in] workspace(void*, input parameter): The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize(uint64_t, input parameter): on NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnHardsigmoidBackwardGetWorkspaceSize.
 * @param [in] executor(aclOpExecutor*, input parameter): op executor, including the operator calculation process.
 * @param [in] stream(aclrtStream, input parameter): ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnHardsigmoidBackward(void* workspace,
                                               uint64_t workspaceSize,
                                               aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_HARDSIGMOID_BACKWARD_H_
