/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ELASTIC_RECEIVABLE_TEST_H_
#define OP_API_INC_ELASTIC_RECEIVABLE_TEST_H_

#include <string>

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: Send data and write status bits to all cards in a communication domain to detect whether the communication link is normal.
 * The first interface of @brief aclnnElasticReceivableTest calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * @param [in] dstRank: Calculation input, representing the communication card in the specified server in the same communication domain. The aclTensor on the Device side is required to be a 1D Tensor, with a shape of (,rankNum), the data type supports INT32, the data format is required to be ND, and non-continuous Tensors are supported.
 * @param [in] group: calculation input, str. ep communication domain name, expert parallel communication domain.
 * @param [in] worldSize: calculation input, int. Communication domain size.
 * @param [in] rank_num: calculation input, int. The number of cards in the peer server needs to be sent.
 * @param [out] workspaceSize: Output parameter, returns the workspace size that needs to be applied for on the NPU device side.
 * @param [out] executor: Out parameter, returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return value, return status code.
 *
 */
ACLNN_API aclnnStatus aclnnElasticReceivableTestGetWorkspaceSize(
    aclTensor* dstRank, const char* group, int64_t worldSize, int64_t rank_num,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnElasticReceivableTest is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnElasticReceivableTestGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnElasticReceivableTest(void* workspace,
                                                 uint64_t workspaceSize,
                                                 aclOpExecutor* executor,
                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ELASTIC_RECEIVABLE_TEST_H_