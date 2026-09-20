/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_KTHVALUE_H_
#define OP_API_INC_KTHVALUE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnKthvalue calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: Complete calculation of the k-th minimum value and subscript of the input.
 *
 * @param [in] self: npu
 * The data type of aclTensor on the NPU device side supports INT32, FLOAT16, and FLOAT32.
 * Supports continuous and non-continuous Tensor, and the data format supports ND.
 * @param [in] k:
 * int64_t type integer. Indicates the minimum value output in the calculation dimension.
 * @param [in] dim:
 * int64_t type integer. Represents the calculated dimension.
 * @param [in] keepdim:
 * bool type data. True means that the size of the output tensor is the same as self,
 * False means that dim will be compressed and the resulting tensor dimension will be 1 less than input.
 * @param [in] valuesOut:
 * The data type of aclTensor on the dnpu device side supports INT32, FLOAT16, and FLOAT32, and the data type is consistent with self.
 * Supports continuous and non-continuous Tensor, and the data format supports ND.
 * @param [in] indicesOut:
 * The aclTensor on the NPU device side supports INT64 as the data type. Supports continuous and non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnKthvalueGetWorkspaceSize(
    const aclTensor* self, int64_t k, int64_t dim, bool keepdim,
    aclTensor* valuesOut, aclTensor* indicesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);
/**
 * The second interface of @brief aclnnKthvalue is used to perform calculations.
 *
 * Operator function: Complete calculation of the k-th minimum value and subscript of the input.
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnKthvalueGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnKthvalue(void* workspace, uint64_t workspaceSize,
                                    aclOpExecutor* executor,
                                    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_KTHVALUE_H_
