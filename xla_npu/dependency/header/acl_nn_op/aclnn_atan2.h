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
#ifndef OP_API_INC_ATAN2_H_
#define OP_API_INC_ATAN2_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnAtan2 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Function description: Perform element-by-element arctangent operation on the input tensors self and other, note (self represents the y coordinate, other represents the x coordinate).
 * Calculation formula:
 * out_{i}=tan^{-1}(self_{i}/other_{i})
 * Parameter description:
 * @param [in] self: aclTensor on NPU device side, data type supports INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16,
 * FLOAT, FLOAT16, DOUBLE. Supports non-continuous Tensor, the data format supports ND, the dimension is not greater than 8, and the shape needs to satisfy the broadcast relationship with other.
 * @param [in] other: aclTensor on NPU device side, data type supports INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16,
 * FLOAT, FLOAT16, DOUBLE. Supports non-continuous Tensor, the data format supports ND, the dimension is not greater than 8, and the shape needs to satisfy the broadcast relationship with self.
 * @param [out] out: aclTensor on the NPU device side, the data type supports FLOAT, FLOAT16, and DOUBLE. Support non-continuous Tensor,
 * The data format supports ND, the dimension is not greater than 8, and the shape is the shape after self and other broadcast.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnAtan2GetWorkspaceSize(const aclTensor* self,
                                                 const aclTensor* other,
                                                 aclTensor* out,
                                                 uint64_t* workspaceSize,
                                                 aclOpExecutor** executor);
/**
 * @brief The second interface of aclnnAtan2 is used to perform calculations.
 * Function description: Perform element-by-element arctangent operation on the input tensors self and other, note (self represents the y coordinate, other represents the x coordinate).
 * Calculation formula:
 * out_{i}=tan^{-1}(self_{i}/other_{i})
 * Implementation instructions:
 * Basic path for API calculation:
```mermaid
graph LR
    A[(Self)] -->B([l0op::Contiguous])
    B --> C([l0op::Atan])
    C --> G([l0op::Cast])
    G --> E([l0op::ViewCopy])
    E --> S[(Out)]
```
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAtan2GetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAtan2(void* workspace, uint64_t workspaceSize,
                                 aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief The first interface of aclnnInplaceAtan2 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Function description: Perform element-by-element arctangent operation on the input tensors self and other, note (self represents the y coordinate, other represents the x coordinate).
 * Calculation formula:
 * out_{i}=tan^{-1}(self_{i}/other_{i})
 * Parameter description:
 * @param [in] self: aclTensor on NPU device side, data type supports INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16,
 * FLOAT, FLOAT16, DOUBLE. Supports non-continuous Tensor, the data format supports ND, the dimension is not greater than 8, and the shape needs to satisfy the broadcast relationship with other.
 * @param [in] other: aclTensor on NPU device side, data type supports INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16,
 * FLOAT, FLOAT16, DOUBLE. Supports non-continuous Tensor, the data format supports ND, the dimension is not greater than 8, and the shape needs to satisfy the broadcast relationship with self.
 * @param [out] workspaceSize returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor returns the op executor, including the operator calculation process.
 * @return aclnnStatus return status code
 */
ACLNN_API aclnnStatus aclnnInplaceAtan2GetWorkspaceSize(
    aclTensor* selfRef, aclTensor* other, uint64_t* workspace_size,
    aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnInplaceAtan2 is used to perform calculations
 *
 * Operator function: Perform element-by-element arctangent operation on the input tensors self and other, note (self represents the y coordinate, other represents the x coordinate).
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAtan2GetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceAtan2(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif
