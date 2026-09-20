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
#ifndef OP_API_INC_FRAC_H_
#define OP_API_INC_FRAC_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnFrac calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Function description: Calculate the decimal part of each element in the input Tensor.
 * Calculation formula:
 * out_{i}=input_{i} - \lfloor \vert input_{i} \vert \rfloor * sgn(input_{i})
 * Parameter description:
 * @param [in]   input
 * Input Tensor, the data type supports FLOAT16, FLOAT, UINT8, INT8, INT16, INT32, and INT64. Supports non-continuous Tensor, and the data format supports ND.
 * @param [in]   out
 * Output Tensor, data type supports FLOAT16, FLOAT, UINT8, INT8, INT16, INT32, INT64. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor returns the op executor, including the operator calculation process.
 * @return aclnnStatus return status code
 */
ACLNN_API aclnnStatus aclnnFracGetWorkspaceSize(const aclTensor* input,
                                                aclTensor* out,
                                                uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnFrac is used to perform calculations.
 * Function description: Calculate the decimal part of each element in the input Tensor.
 * Calculation formula:
 * out_{i}=input_{i} - \lfloor \vert input_{i} \vert \rfloor * sgn(input_{i})
 * Implementation instructions:
 * Basic path for API calculation:
```mermaid
graph LR
    A[(Input)] -->B([l0op::Contiguous])
    B -->C([l0op::Sub])

    B -->F1([l0op::Trunc])
    F1 --> C
    C -->D([l0op::ViewCopy])
    D -->E[(Out)]
```
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnFracGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnFrac(void* workspace, uint64_t workspaceSize,
                                aclOpExecutor* executor, aclrtStream stream);

/**
 * The first interface of @brief aclnnInplaceFrac calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 * Function description: Calculate the decimal part of each element in the input Tensor.
 * Parameter description:
 * @param [in]   inputRef
 * Input Tensor, the data type supports FLOAT16, FLOAT, UINT8, INT8, INT16, INT32, and INT64. Supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspaceSize returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor returns the op executor, including the operator calculation process.
 * @return aclnnStatus return status code
 */
ACLNN_API aclnnStatus aclnnInplaceFracGetWorkspaceSize(
    aclTensor* inputRef, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnInplaceFrac is used to perform calculations
 * Function description: Calculate the decimal part of each element in the input Tensor.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceFracGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceFrac(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
