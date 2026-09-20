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
#ifndef OP_API_INC_RANGE_H_
#define OP_API_INC_RANGE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnRange calculates the workspace size according to the specific calculation process.
 * @domain aclnn_math
 *
 * Function description: From start to end, take the value according to the step interval, and return the size $ \lfloor \frac{end - start} {step} \rfloor + 1
 * 1D tensor of $. Among them, the step size step is the interval between two adjacent values ​​​​in the tensor.
 *
 * Calculation formula: $$ out_{i+1}=out_i+step $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(Start)]--> E([l0op::Arange])
 *     B[(End)]--> D[(Limit = End + Step)]
 *     C[(Step)]--> D
 *     D--> E
 *     C--> E
 *     E--> H([l0op::Cast])
 *     H--> M([l0op::ViewCopy])
 *     M--> N[(Out)]
 * ```
 *
 * Parameter description:
 * @param [in]   start
 * Get the starting position of the value range: aclScalar on the host side. The data type supports integer and floating-point data types. The data format supports ND. It needs to be satisfied that when the step is greater than 0, the input start is less than the end, or when the step is less than 0, the input start is greater than the end.
 * @param [in]   end
 * Get the end position of the value range: aclScalar on the host side. The data type supports integer and floating-point data types. The data format supports ND. It needs to be satisfied that when the step is greater than 0, the input start is less than the end, or when the step is less than 0, the input start is greater than the end.
 * @param [in]   step
 * The step size of obtaining the value: aclScalar on the host side. The data type supports integer and floating point data types. The data format supports ND. It needs to be satisfied that step is not equal to 0.
 * @param [in] out specified output tensor: NPU
 * The data type of aclTensor on the device side supports integer and floating-point data types, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRangeGetWorkspaceSize(
    const aclScalar* start, const aclScalar* end, const aclScalar* step,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * The second interface of @brief aclnnRange is used to perform calculations.
 *
 *
 * Function description: From start to end, take the value according to the step interval, and return the size $ \lfloor \frac{end - start} {step} \rfloor + 1
 * 1D tensor of $. Among them, the step size step is the interval between two adjacent values ​​​​in the tensor.
 *
 * Calculation formula: $$ out_{i+1}=out_i+step $$
 *
 * Implementation instructions:
 * Basic path for API calculation:
 * ```mermaid
 * graph LR
 *     A[(Start)]--> E([l0op::Arange])
 *     B[(End)]--> D[(Limit = End + Step)]
 *     C[(Step)]--> D
 *     D--> E
 *     C--> E
 *     E--> H([l0op::Cast])
 *     H--> M([l0op::ViewCopy])
 *     M--> N[(Out)]
 * ```
 *
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnRangeGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRange(void* workspace, uint64_t workspaceSize,
                                 aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif