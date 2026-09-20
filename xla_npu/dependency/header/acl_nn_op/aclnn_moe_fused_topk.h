/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_moe_fused_topk.h
 * \brief
 */

#ifndef OP_API_INC_MOE_FUSED_TOPK_H_
#define OP_API_INC_MOE_FUSED_TOPK_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * The first interface of @brief aclnnMoeFusedTopk calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * @param [in] x: npu
 * For aclTensor on the device side, the data type supports FLOAT16, BFLOAT16 and FLOAT, and the data format supports ND.
 * @param [in] addNum: npu
 * The aclTensor on the device side has the same data type as x, and the data format supports ND.
 * @param [in] mappingNum: npu
 * The aclTensor on the device side supports INT32 as data type and ND as data format.
 * @param [in] mappingTable: npu
 * The aclTensor on the device side supports INT32 as data type and ND as data format.
 * @param [in] groupNum: Indicates the number of groups.
 * uint32_t, the value must be greater than 0, the number of expert groups.
 * @param [in] groupTopk: Indicates the number of selected groups.
 * uint32_t, the value must be greater than 0, the number of expert groups selected.
 * @param [in] topN: Indicates the number of experts selected in the group to calculate the scores of each group.
 * uint32_t, the value must be greater than 0, the number of selected experts in the group, used to calculate the score of each group.
 * @param [in] topK: The number of experts finally selected.
 * uint32_t, the value must be greater than 0.
 * @param [in] activateType: uint32_t, activation type, currently only supports 0 (ACTIVATION_SIGMOID).
 * uint32_t, activation type, currently only supports 0 (ACTIVATION_SIGMOID).
 * @param [in] isNorm: bool, whether to normalize the output.
 * bool, true means normalizing the output.
 * @param [in] scale: Normalized coefficient multiplication.
 * float, when isNorm is true, the output is multiplied by the normalized coefficient.
 * @param [in] enableExpertMapping: Whether to enable the mapping of physical experts to logical experts.
 * bool, true means enabling the mapping of physical experts to logical experts.
 * @param [out] y: npu
 * The aclTensor on the device side supports FLOAT32 as data type and ND as data format.
 * @param [out] indices: npu
 * The aclTensor on the device side supports INT32 as data type and ND as data format.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMoeFusedTopkGetWorkspaceSize(
    const aclTensor* x, const aclTensor* addNum, const aclTensor* mappingNum,
    const aclTensor* mappingTable, uint32_t groupNum, uint32_t groupTopk,
    uint32_t topN, uint32_t topK, uint32_t activateType, bool isNorm,
    float scale, bool enableExpertMapping, aclTensor* y, aclTensor* indices,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/* The second interface of @brief aclnnMoeFusedTopk is used to perform calculations.
 * @param [in] workspace: the starting address of the workspace memory applied for on the NPU device side
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnMoeFusedTopkGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnMoeFusedTopk(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MOE_FUSED_TOPK_H_
