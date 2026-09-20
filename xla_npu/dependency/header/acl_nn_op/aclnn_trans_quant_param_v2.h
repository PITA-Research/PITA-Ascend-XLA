/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_TRANS_QUANT_PARAM_V2_H
#define OP_API_INC_TRANS_QUANT_PARAM_V2_H

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: implement transQuantParamV2 calculation
 * @brief The first interface of aclnnTransQuantParamV2 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] scale: quantization parameter, supported data type: float32.
 * @param [in] offset: quantization parameter, data type support: float32.
 * @param [out] out: calculation result, data type: uint64_t
 * @return aclnnStatus: return status code
 */

ACLNN_API aclnnStatus aclnnTransQuantParamV2GetWorkspaceSize(
    const aclTensor* scale, const aclTensor* offset, const aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnTransQuantParamV2 is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnTransQuantParamV2GetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnTransQuantParamV2(void* workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor* executor,
                                             const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_TRANS_QUANT_PARAM_V2_H