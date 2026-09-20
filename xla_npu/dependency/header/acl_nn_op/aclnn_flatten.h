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

#ifndef OP_API_INC_FLATTEN_H_
#define OP_API_INC_FLATTEN_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnFlatten calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: Flatten the input Tensor into a 2D Tensor based on the given axis.
 * If the shape of self is (d_0,d_1,...,d_n), then the shape of output out is (d_0 X d_1... X d_(axis-1), d_axis X d_(axis+1)... X d_n).
 * If the value of axis is 0, the shape of the output out is (1, d_0 X d_1... X d_n).
 * Implementation Description - Computational Graph
 *
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0op::Contiguous])
 *   B --> C([l0op::Flatten])
 *   D[(axis)] --> C
 *   C --> I([l0op::ViewCopy])
 *   I --> J[(out)]
 * ```
 *
 * @param [in]
 * self: Input `self`, data type supports INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16 (Ascend910B), FLOAT, FLOAT16.
 * Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [in] axis: input `axis`, int64_t type integer. Represents the datum axis for flatten calculation. The value range is [-self.dim(),self.dim()).
 * @param [out]
 * out: Output `out`, data type supports INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16 (Ascend910B), FLOAT, FLOAT16.
 * shape is 2D. Supports [non-continuous Tensor](#), and the data format supports ND ([Reference](#)).
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFlattenGetWorkspaceSize(const aclTensor* self,
                                                   int64_t axis, aclTensor* out,
                                                   uint64_t* workspaceSize,
                                                   aclOpExecutor** executor);

/**
 * @brief aclnnFlatten's second interface, used to perform calculations
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnFlattenGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFlatten(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor,
                                   const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_FLATTEN_H_
