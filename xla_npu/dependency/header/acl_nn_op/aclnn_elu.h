/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_ELU_H_
#define OP_API_INC_LEVEL2_ACLNN_ELU_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: Call the exponential linear unit activation function ELU for each element x in the input tensor self, and store the result in the output tensor out.
 * Calculation formula: as follows
 * $$
 * ELU(x) =
 * \begin{cases}
 * scale \ast x, \quad x > 0\\
 * \alpha \ast scale \ast (exp(x \ast inputScale)-1), \quad x \leq 0
 * \end{cases}
 * $$
 *
 * Implementation instructions: as follows
 *
 * Calculation diagram: as follows
 *
 * ```mermaid
 * graph LR
 *     A[(Self)] --> B([l0op::Contiguous])
 *     B --> C([l0op::Elu])
 *     C --> D([l0op::Cast])
 *     D --> E([l0op::ViewCopy])
 *     E --> F[(out)]
 *     G((alpha)) --> C
 *     H((scale)) --> C
 *     I((inputScale)) --> C
 * ```
 */

/**
 * The first interface of @brief aclnnElu calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] self: represents the input of the ELU activation function, NPU
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, BFLOAT16, and supports non-continuous
 * Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in] alpha: Represents the activation coefficient of the ELU activation function, aclScalar on the host side, the data type needs to be a data type convertible to FLOAT.
 * @param [in] scale: Represents the scaling coefficient of the ELU activation function, aclScalar on the host side, the data type needs to be a data type that can be converted to FLOAT.
 * @param [in] inputScale: Represents the scaling coefficient of the input of the ELU activation function, aclScalar on the host side, the data type needs to be a data type that can be converted to FLOAT.
 * @param [in] out: Represents the output of the ELU activation function, NPU
 * For aclTensor on the device side, the data type needs to be a data type that can be converted by self, and the shape needs to be consistent with self.
 * Supports non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnEluGetWorkspaceSize(
    const aclTensor* self, const aclScalar* alpha, const aclScalar* scale,
    const aclScalar* inputScale, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnElu is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnEluGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnElu(void* workspace, uint64_t workspaceSize,
                               aclOpExecutor* executor, aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceElu calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer。
 * @param [in] selfRef: represents the input of the ELU activation function, NPU
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, DOUBLE, BFLOAT16, and non-
 * Continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in] alpha: Represents the activation coefficient of the ELU activation function, aclScalar on the host side, the data type needs to be a data type convertible to FLOAT.
 * @param [in] scale: Represents the scaling coefficient of the ELU activation function, aclScalar on the host side, the data type needs to be a data type that can be converted to FLOAT.
 * @param [in] inputScale: Represents the scaling coefficient of the input of the ELU activation function, aclScalar on the host side, the data type needs to be a data type that can be converted to FLOAT.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceEluGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* alpha, const aclScalar* scale,
    const aclScalar* inputScale, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceElu is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceEluGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceElu(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ELU_H_
