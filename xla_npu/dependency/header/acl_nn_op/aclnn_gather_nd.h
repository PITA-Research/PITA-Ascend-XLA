/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_GATHER_ND_H_
#define OP_API_INC_LEVEL2_ACLNN_GATHER_ND_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: For the input tensor `self` with dimension **r≥1**, and the input tensor `indices` with dimension **q≥1**, collect data slices into dimensions **(q-1) + (r -
 * indices_shape[-1])**
 * The output tensor out. indices is a **q**-dimensional integer tensor, which can be regarded as a **q-1**-dimensional special tensor composed of **index pairs** (each **index pair** is a length of **indices_shape[-1]**
 * A one-dimensional tensor, each **index pair** points to a slice in **self**), the specific calculation logic is:
 * 1) If **indices_shape[-1]** > **r**, it is an illegal scenario.
 * 2) If **indices_shape[-1]** = **r**, then the dimension of the output tensor out is **q-1**, that is, the shape of out is **[indices_shape[0:q-2]]**,
 * The element in out is the element at the ** index of self. (see example 1)
 * 3) If **indices_shape[-1]** < **r**, then the dimension of the output tensor out is **(q-1) + (r -
 * indices_shape[-1])**, let **c**=**r**-**indices_shape[-1]**, that is, the shape of out is
 * **[indices_shape[0:q-2],self_shape[r-c:r-1]]**, `out` consists of the ** index of `self` and the slice of ** position. (See examples 2,3,4)
 * Some restrictions on **r**, **q**, **indices_shape[-1]** are as follows:
 * 1) Must satisfy **r**≥1, **q**≥1.
 * 2) The value of **indices_shape[-1]** must be between 1 (inclusive) and **r** (inclusive).
 * 3) Each element of `indices` must be within the range of [-**s**, **s-1**] (**s** is the value on each axis of **self_shape**),
 * That is -**self_shape[i]**≤indices[...,i]≤**self_shape[i]**-1.
 * Example
 * Example 1:
 *   self: [[0, 1],[2, 3]]       # self_shape=[2, 2], r=2
 *   indices: [[0, 0], [1, 1]]   # indices_shape=[2, 2], q=2, indices_shape[-1]=2
 *   out: [0, 3]                 # out_shape=[2]
 * Example 2:
 *   self: [[0, 1],[2, 3]]       # self_shape=[2, 2], r=2
 *   indices: [[1], [0]]         # indices_shape=[2, 1], q=2, indices_shape[-1]=1
 *   out: [[2, 3], [0, 1]]       # out_shape=[2, 2]
 * Example 3:
 *   self: [[[0, 1],[2, 3]], [[4, 5],[6, 7]]]   # self_shape=[2, 2, 2], r=3
 *   indices: [[0, 1], [1, 0]]                  # indices_shape=[2, 2], q=2, indices_shape[-1]=2
 *   out: [[2, 3], [4, 5]]                      # out_shape=[2, 2]
 * Example 4:
 *   self: [[[0, 1],[2, 3]], [[4, 5],[6, 7]]]   # self_shape=[2, 2, 2], r=3
 *   indices: [[[0, 1]], [[1, 0]]]              # indices_shape=[2, 1, 2], q=3, indices_shape[-1]=2
 *   out: [[[2, 3]], [[4, 5]]]                  # out_shape=[2, 1, 2]
 *
 * Calculation graph:
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0op::Contiguous])
 *   C[(indices)] --> D([l0op::Contiguous])
 *   B --> E([l0op::GatherNd])
 *   D --> E
 *   E --> F([l0op::ViewCopy])
 *   F --> G[(out)]
 * ```
 */

/**
 * The first interface of @brief aclnnGatherNd calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * @param [in] self: aclTensor on the NPU device side. The data type supports INT64, INT32, INT8, UINT8, BOOL, FLOAT, FLOAT16, and BFLOAT16.
 * It supports non-continuous Tensor, the data format supports ND, and the data dimension does not support more than 8 dimensions.
 * @param [in] index: npu
 * The aclTensor on the device side supports INT64 and INT32 as data type and non-continuous Tensor. The data format supports ND. The data dimension does not support more than 8 dimensions.
 * @param [in] negativeIndexSupport: attribute, indicating whether to support negative number scenarios, the data type supports bool.
 * @param [in] out: aclTensor on the NPU device side, data type supports INT64, INT32, INT8, UINT8, BOOL, FLOAT, FLOAT16, BFLOAT16, data
 * The type needs to be consistent with self, and the data format supports ND.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGatherNdGetWorkspaceSize(
    const aclTensor* self, const aclTensor* indices, bool negativeIndexSupport,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnGatherNd is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGatherNdGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGatherNd(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_GATHER_ND_H_
