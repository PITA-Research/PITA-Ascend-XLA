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

#ifndef OP_API_INC_LEVEL2_ACLNN_FMOD_SCALAR_H_
#define OP_API_INC_LEVEL2_ACLNN_FMOD_SCALAR_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnFmodScalar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Returns the remainder of self divided by other.
 * Calculation formula: $$ out_{i} = self_{i} - (other_{i} *\left \lfloor (self_{i}/other_{i}) \right \rfloor) $$
 *
 * Implementation description: Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(self)] -->B([l0op::Contiguous])
 * B -->C([l0op::Cast])
 * C -->D([l0op::Mod])
 * E[(other)]-->F([l0op::Cast])
 * F --> D
 * D--> G([l0op::Cast])
 * G --> I([l0op::ViewCopy])
 * I --> J[(out)]
 * ```
 *
 * @param [in] self: aclTensor on NPU device side
 * Data types support DOUBLE, BFLOAT16, FLOAT16, FLOAT32, INT32, INT64, INT8, UNIT8
 * And the data type and the data type of other need to satisfy the data type derivation rules. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] other: aclScalar on the NPU device side,
 * Data types support DOUBLE, BFLOAT16, FLOAT16, FLOAT32, INT32, INT64, INT8, UNIT8
 * And the data type and the data type of self need to satisfy the data type derivation rules. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] out: aclTensor on the NPU device side, data type supports DOUBLE, BFLOAT16, FLOAT16, FLOAT32, INT32, INT64, INT8, UNIT8,
 * Shape needs to be the shape after self and other broadcast. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */

ACLNN_API aclnnStatus aclnnFmodScalarGetWorkspaceSize(const aclTensor* self,
                                                      const aclScalar* other,
                                                      aclTensor* out,
                                                      uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnFmodScalar is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnFmodScalarGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFmodScalar(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor,
                                      aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceFmodScalar calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Returns the remainder of selfRef divided by other.
 * Calculation formula: $$ out_{i} = self_{i} - (other_{i} *\left \lfloor (self_{i}/other_{i}) \right \rfloor) $$
 *
 * Implementation description: Basic path for API calculation:
 * ```mermaid
 * graph LR
 * A[(self)] -->B([l0op::Contiguous])
 * B -->C([l0op::Cast])
 * C -->D([l0op::Mod])
 * E[(other)]-->F([l0op::Cast])
 * F --> D
 * D--> G([l0op::Cast])
 * G --> I([l0op::ViewCopy])
 * I --> J[(out)]
 * ```
 *
 * @param [in] selfRef: aclTensor on NPU device side
 * Data types support DOUBLE, FLOAT16, FLOAT32, INT32, INT64, INT8, UNIT8
 * And the data type and the data type of other need to satisfy the data type derivation rules. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] other: aclScalar on the NPU device side,
 * Data types support DOUBLE, FLOAT16, FLOAT32, INT32, INT64, INT8, UNIT8
 * And the data type and the data type of self need to satisfy the data type derivation rules. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */

ACLNN_API aclnnStatus aclnnInplaceFmodScalarGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnInplaceFmodScalar is used to perform calculations.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnInplaceFmodScalarGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceFmodScalar(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_FMOD_SCALAR_H_
