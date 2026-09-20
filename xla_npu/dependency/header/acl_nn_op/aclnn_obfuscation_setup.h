/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_OBFUSCATION_SETUP_H_
#define OP_API_INC_OBFUSCATION_SETUP_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * The first interface of @brief aclnnObfuscationSetup calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * 
 * Operator function: PMCC project initialization.
 * @param [in] fdToClose: int32_t on the host side, the socket connector to be closed. When cmd is 3, fill in the fd returned by this operator when cmd is 1, otherwise fill in 0.
 * @param [in] dataType: int32_t on the host side, represents the number of the Tensor data type. You only need to fill in a valid value when cmd is set to 1 or 2, otherwise fill in 0.
 * Ascend 310P AI processor: Select between {0, 1}, 0 means ACL_FLOAT, 1 means ACL_FLOAT16
 * Ascend 910B AI processor: Select from {0, 1, 2, 27}, 0 means ACL_FLOAT, 1 means ACL_FLOAT16, 2 means ACL_INT8, 27 means ACL_BF16
 * @param [in] hiddenSize: int32_t on the host side, hidden layer dimension. Valid value needs to be filled in only when cmd is set to 1 or 2, otherwise fill in with 0.
 * @param [in] tpRank: int32_t on the host side, TP Rank. A valid value needs to be filled in only when cmd is set to 1 or 2, otherwise fill in with 0.
 * @param [in] modelObfSeedId: int32_t on the host side, model confusion factor id, used by TA to query the model confusion factor from TEE KMC. A valid value needs to be filled in only when cmd is set to 1 or 2, otherwise fill in with 0.
 * @param [in] dataObfSeedId: int32_t on the host side, data obfuscation factor id, used by TA to query the data obfuscation factor from TEE KMC. A valid value needs to be filled in only when cmd is set to 1 or 2, otherwise fill in with 0.
 * @param [in] cmd: int32_t on the host side, setup command number, select among {1, 2, 3}. When set to 1, normal mode resource initialization is performed, when 2 is set, high-precision mode resource initialization is performed, and when set to 3, resources are released.
 * @param [in] threadNum: int32_t on the host side, the number of threads used by CA/TA for obfuscation processing. Select from {1, 2, 3, 4, 5, 6}. You only need to fill in a valid value when cmd is set to 1 or 2, otherwise fill in 0.
 * @param [out] fd: aclTensor on the NPU device side, which represents the socket connector. Choose among {1, 2, 16}. When set to 1, the normal mode resource is initialized, when it is 2, the high-precision mode resource is initialized, and when it is set to 16, the resource is released.
 * The shape is (1). The data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnObfuscationSetupGetWorkspaceSize(
    int32_t fdToClose, int32_t dataType, int32_t hiddenSize, int32_t tpRank,
    int32_t modelObfSeedId, int32_t dataObfSeedId, int32_t cmd,
    int32_t threadNum, aclTensor* fd, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnObfuscationSetup, used to perform calculations
 * @param [in] workspace: the workspace memory address applied for on the NPU device side
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnObfuscationSetupGetWorkspaceSize
 * @param [in] executor: Returns the op executor, including the operator calculation process
 * @param [in] stream: ACL stream
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnObfuscationSetup(void* workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor* executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif  // OP_API_INC_OBFUSCATION_SETUP_H_
