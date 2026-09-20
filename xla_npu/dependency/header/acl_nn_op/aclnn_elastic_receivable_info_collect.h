/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ELASTIC_RECEIVABLE_INFO_COLLECT_H_
#define OP_API_INC_ELASTIC_RECEIVABLE_INFO_COLLECT_H_

#include <string>

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: Collect the modified data area results of the test and output them to the host for processing
 * The first interface of @brief aclnnElasticReceivableInfoCollect calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] group: calculation input, str. Communication domain name, expert parallel communication domain.
 * @param [in] worldSize: calculation input, int64_t. Communication domain size.
 * @param [in] y: Calculation output, Tensor, required output, data type supports int32, only supports 2 dimensions, data format supports ND.
 * @param [out] workspaceSize: Output parameter, returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Out parameter, returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return value, return status code.
 *
 */
ACLNN_API aclnnStatus aclnnElasticReceivableInfoCollectGetWorkspaceSize(
    const char* group, int64_t worldSize, aclTensor* y, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnElasticReceivableInfoCollect is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnElasticReceivableInfoCollectGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnElasticReceivableInfoCollect(void* workspace,
                                                        uint64_t workspaceSize,
                                                        aclOpExecutor* executor,
                                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ELASTIC_RECEIVABLE_INFO_COLLECT_H_