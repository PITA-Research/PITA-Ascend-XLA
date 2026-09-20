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
#ifndef OP_API_INC_MASKED_SCALE_H_
#define OP_API_INC_MASKED_SCALE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnMaskedScale calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Operator function: complete elementwise calculation.
 *
 * Implementation instructions:
 * $$ out = self \times mask \times scale $$
 *
 * @param [in] self: aclTensor on the NPU device side, the data type supports FLOAT16, BFLOAT16, and FLOAT.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] mask: aclTensor on the NPU device side. The data type supports UINT8, INT8, FLOAT16, and FLOAT. shape needs to be consistent with self
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] scale: Perform data scaling, the data type supports FLOAT.
 * @param [in] y: aclTensor on the NPU device side, the data type and shape need to be consistent with self.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMaskedScaleGetWorkspaceSize(
    const aclTensor* self, const aclTensor* mask, float scale, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief The second interface of aclnnMaskedScale is used to perform calculations.
 *
 * Operator function: complete elementwise calculation.
 * Implementation instructions:
 * $$ out = self \times mask \times scale $$
 * 
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMaskedScatterGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMaskedScale(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MASKED_SCALE_H_
