/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_HARDTANH_BACKWARD_H_
#define OP_API_INC_HARDTANH_BACKWARD_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnHardtanhBackward calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_train
 *
 * Operator function: Complete the reverse calculation of Hardtanh
 *
 * Implementation instructions: API
 * The base path for calculation: as shown below
 * ```mermaid
 * graph LR
 *     A[(gradOutput)] -->B([l0op::Contiguous])
 *     B -->C([l0op::HardtanhGrad])
 *     D[(self)] -->E([l0op::Contiguous])
 *     E -->C([l0op::HardtanhGrad])
 *     F((min)) --> C([l0op::HardtanhGrad])
 *     G((max)) --> C([l0op::HardtanhGrad])
 *     C --> H([l0op::ViewCopy])
 *     H --> K[(out)]
 * ```
 *
 * @param [in] gradOutput: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16 (only Shengteng 910B and 910_93
 * AI processor supports), supports non-continuous Tensor, and the data format supports ND.
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16 (only Shengteng 910B and 910_93
 * AI processor supports), supports non-continuous Tensor, and the data format supports ND.
 * @param [in] min: Lower bound.
 * @param [in] max: upper bound.
 * @param [out] out: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16 (only Shengteng 910B and 910_93
 * AI processor supports), supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnHardtanhBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclScalar* min,
    const aclScalar* max, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The first interface of @brief aclnnHardtanhBackward calculates the workspace size according to the specific calculation process.
 *
 * Operator function: Complete the reverse calculation of Hardtanh
 *
 * Implementation instructions: API
 * The base path for calculation: as shown below
 * ```mermaid
 * graph LR
 *     A[(gradOutput)] -->B([l0op::Contiguous])
 *     B -->C([l0op::HardtanhGrad])
 *     D[(self)] -->E([l0op::Contiguous])
 *     E -->C([l0op::HardtanhGrad])
 *     F((min)) --> C([l0op::HardtanhGrad])
 *     G((max)) --> C([l0op::HardtanhGrad])
 *     C --> H([l0op::ViewCopy])
 *     H --> K[(out)]
 * ```
 *
 * @param [in] gradOutput: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16 (only Shengteng 910B and 910_93
 * AI processor supports), supports non-continuous Tensor, and the data format supports ND.
 * @param [in] self: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16 (only Shengteng 910B and 910_93
 * AI processor supports), supports non-continuous Tensor, and the data format supports ND.
 * @param [in] min: Lower bound.
 * @param [in] max: upper bound.
 * @param [out] out: npu
 * aclTensor on the device side, the data type supports FLOAT, FLOAT16, BFLOAT16 (only Shengteng 910B and 910_93
 * AI processor supports), supports non-continuous Tensor, and the data format supports ND.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnHardtanhBackward(void* workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_HARDTANH_BACKWARD_H_
