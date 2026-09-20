/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_moe_distribute_buffer_reset.h
 * \brief
 */
#ifndef OP_API_INC_MOE_DISTRIBUTE_BUFFER_RESET_
#define OP_API_INC_MOE_DISTRIBUTE_BUFFER_RESET_

#include <string>

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: implement aclnnMoeDistributeBufferReset data area and status area cleaning
 * The first interface of @brief aclnnMoeDistributeBufferReset calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] elasticInfo: Valid rank: int32_t.
 * @param [in] groupEp: calculation input, str. ep communication domain name.
 * @param [in] epWorldSize: calculation input, int. ep communication domain size
 * @param [in] needSync: calculation input, int. Identifies whether full card synchronization is required.
 * @param [out] workspaceSize: Returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnMoeDistributeBufferResetGetWorkspaceSize(
    const aclTensor* elasticInfo, const char* groupEp, int64_t epWorldSize,
    int64_t needSync, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnMoeDistributeBufferReset is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnAbsGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnMoeDistributeBufferReset(void* workspace,
                                                    uint64_t workspaceSize,
                                                    aclOpExecutor* executor,
                                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MOE_DISTRIBUTE_BUFFER_RESET_