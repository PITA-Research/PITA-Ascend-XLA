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

/*!
 * \file aclnn_isposinf.h
 * \brief
 */

#ifndef OP_API_INC_ISPOSINF_H_
#define OP_API_INC_ISPOSINF_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnIsPosInf calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Operator function: Determine whether the elements of the input tensor are positive infinity.
 * Implementation Description - Computational Graph
 * Calculation diagram: as follows
 * Scenario 1: Floating point scenario
 * ```mermaid
 * graph LR
 *   A[(self)] -->B([l0op::Contiguous])
 *   B --> D([l0op::IsPosInf])
 *   D --> I([l0op::ViewCopy])
 *   I --> J[(out)]
 * ```
 * Scenario 2: Non-floating point number, bounded, returns False
 * ```mermaid
 * graph LR
 *   A[(self)] -->B([l0op::Fill])
 *   B --> I([l0op::ViewCopy])
 *   I --> J[(out)]
 * ```
 *
 * @param [in] self: Input `self`, the data type supports FLOAT, FLOAT16, BFLOAT16 (Ascend910B),
 * INT32, INT64, INT16, INT8, UINT8, BOOL. Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [out] out: Input `out`, the data type supports BOOL, and the shape needs to be consistent with self.
 * Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnIsPosInfGetWorkspaceSize(const aclTensor* self,
                                                    aclTensor* out,
                                                    uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnIsPosInf, used to perform calculations
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnIsPosInfGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnIsPosInf(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    const aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ISPOSINF_H_
