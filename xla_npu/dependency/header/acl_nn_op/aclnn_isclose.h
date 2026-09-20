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
#ifndef OP_API_INC_LEVEL2_ISCLOSE_H_
#define OP_API_INC_LEVEL2_ISCLOSE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnIsClose calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Returns a new tensor with Boolean elements to determine whether self and other are equal within epsilon.
 * Calculation formula
 * $$ \left | input-other\right | = atol + rtol\times \left | other \right | $$
 *
 * @param [in] self: npu
 * The aclTensor on the device side supports 1 to 8 dimensions, and the data type supports FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL, and DOUBLE.
 * To support non-continuous Tensors, the dtype must be consistent with the dtype of other, and the shape must satisfy the broadcast relationship with other. The data format supports ND.
 * @param [in] other: npu
 * The aclTensor on the device side supports 1 to 8 dimensions, and the data type supports FLOAT, FLOAT16, INT32, INT64, INT16, INT8, UINT8, BOOL, and DOUBLE.
 * To support non-continuous Tensors, the dtype must be consistent with the dtype of self, and the shape must satisfy the broadcast relationship with self. The data format supports ND.
 * @param [in] rtol: Absolutely tolerant. The data type supports DOUBLE.
 * @param [in] atol: Relative tolerance. The data type supports DOUBLE.
 * @param [in] equal_nan:
 * NaN value comparison option. If True, two NaNs will be considered equal; if False, two NaNs will be considered unequal. The data type supports BOOL.
 * @param [out] out: npu
 * The aclTensor on the device side supports 1 to 8 dimensions, the data type supports BOOL, and non-continuous Tensor. The shape needs to be self and other.
 * The shape and data format after broadcast support ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnIsCloseGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, double rtol, double atol,
    bool equal_nan, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnIsClose is used to perform calculations.
 *
 * Operator function: Returns a new tensor with Boolean elements to determine whether self and other are equal within epsilon.
 * Calculation formula
 * $$ \left | input-other\right | = atol + rtol\times \left | other \right | $$
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnIsCloseGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnIsClose(void* workspace, uint64_t workspaceSize,
                                   aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ISCLOSE_H_
