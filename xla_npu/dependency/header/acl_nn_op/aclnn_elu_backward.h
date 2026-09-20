/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_ELU_BACKWARD_H_
#define OP_API_INC_LEVEL2_ACLNN_ELU_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: Complete the reverse calculation of the ELU activation function and output the gradient of the forward input of the ELU activation function.
 * Calculation formula: None
 *
 * Implementation instructions: as follows
 *
 * Calculation diagram: as follows
 *
 * ```mermaid
 * graph LR
 *     A[(gradOutput)] --> B([l0op::Contiguous])
 *     B --> C([l0op::EluGradV2])
 *     C --> D([l0op::Cast])
 *     D --> E([l0op::ViewCopy])
 *     E --> F[(gradInput)]
 *     G((alpha)) --> C
 *     H((scale)) --> C
 *     I((inputScale)) --> C
 *     L((isResult)) --> C
 *     J((selfOrResult)) --> K([l0op::Contiguous])
 *     K --> C
 * ```
 */

/**
 * The first interface of @brief aclnnEluBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 * @param [in] gradOutput: represents the gradient of the forward output of the ELU activation function, customizable, similar to weight, NPU
 * aclTensor on the device side, the data type supports FLOAT,
 * FLOAT16, BFLOAT16 (supported by 910B), support non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in] alpha: Represents the activation coefficient of the ELU activation function, aclScalar on the host side, the data type needs to be a data type convertible to FLOAT.
 * @param [in] scale: Represents the scaling coefficient of the ELU activation function, aclScalar on the host side, the data type needs to be a data type that can be converted to FLOAT.
 * @param [in] inputScale: Represents the scaling coefficient of the input of the ELU activation function, aclScalar on the host side, the data type needs to be a data type that can be converted to FLOAT.
 * @param [in] isResult: Indicates whether the input passed to ELU for reverse calculation is the forward output of ELU. The data type supports BOOL.
 * @param [in]
 * selfOrResult: When isResult is True, it represents the forward output of the ELU activation function. When isResult is False, it represents the forward input of the ELU activation function, NPU
 * The data type of aclTensor on the device side supports FLOAT and FLOAT16, and the data type needs to be consistent with gradOutput. The shape needs to be consistent with the shape of gradOutput, supported
 * For non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in] gradInput: represents the gradient of the forward input of the ELU activation function, that is, the result of derivation of the input, NPU
 * aclTensor on the device side, the data type needs to be
 * The data type that gradOutput can convert, the shape needs to be consistent with the shape of gradOutput, supports non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEluBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclScalar* alpha, const aclScalar* scale,
    const aclScalar* inputScale, bool isResult, const aclTensor* selfOrResult,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnEluBackward is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnEluBackwardGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEluBackward(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ELU_BACKWARD_H_
