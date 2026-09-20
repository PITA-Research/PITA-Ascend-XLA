/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_HARDSHRINK_BACKWARD_H_
#define OP_API_INC_HARDSHRINK_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnHardshrinkBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: Complete the hardShrink backward operation on the input Tensor
 * @param [in] gradOutput: Calculation input, data type supports FLOAT, FLOAT16, data format supports ND, supports non-continuous Tensor.
 * @param [in] self: Calculation input, data type supports FLOAT, FLOAT16, data format supports ND, supports non-continuous Tensor.
 * @param [in] lambd: Calculation input, specifies the threshold of hardShrinkBackward segmentation, the data type supports FLOAT type.
 * @param [in] gradInput: Calculation output, data type supports FLOAT, FLOAT16, data format supports ND, supports non-continuous Tensor.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Returns the status code. ACLNN_SUCCESS is returned on success, and the corresponding error code is returned on failure.
 */
ACLNN_API aclnnStatus aclnnHardshrinkBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclScalar* lambd,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnHardshrinkBackward is used to perform calculations
 *
 * Operator function: Complete the hardShrink backward operation on the input Tensor
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnHardshrinkBackwardGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Returns the status code, returns ACLNN_SUCCESS if successful, and returns the corresponding error code if failed.
 */
ACLNN_API aclnnStatus aclnnHardshrinkBackward(void* workspace,
                                              uint64_t workspaceSize,
                                              aclOpExecutor* executor,
                                              const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_HARDSHRINK_BACKWARD_H_