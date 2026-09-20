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

/*!
 * \file aclnn_attention_update.h
 * \brief
 */

#ifndef OP_API_INC_ATTENTION_UPDATE_H_
#define OP_API_INC_ATTENTION_UPDATE_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnAttentionUpdate calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * @param [in] lse: npu
 * The aclTensorList on the device side supports FLOAT32 as data type and ND as data format.
 * @param [in] localOut: npu
 * The aclTensorList on the device side supports FLOAT32 as data type and ND as data format.
 * @param [in] updateType: Indicates the AttentionUpdate type.
 * int64_t, non-A5 operators currently only support 0 (DecodeUpdate); the A5 operator controls whether lseOut is output, and supports 0 and 1, which respectively means not to output lseOut and output lseOut.
 * @param [out] out: npu
 * For aclTensor on the device side, the non-A5 operator data type supports FLOAT32, and the data format supports ND; the A5 operator data type supports FLOAT32, FLOAT16, and BFLOAT16.
 * @param [out] lseOut: npu
 * For aclTensor on the device side, the non-A5 operator is a reserved parameter and is not used yet. It only supports nullptr; the A5 operator is an optional output of lse_m, and the data type supports FLOAT32.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAttentionUpdateGetWorkspaceSize(
    const aclTensorList* lse, const aclTensorList* localOut, int64_t updateType,
    aclTensor* out, aclTensor* lseOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/* The second interface of @brief aclnnAttentionUpdate is used to perform calculations.
 * @param [in] workspace: the starting address of the workspace memory applied for on the NPU device side
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnBitwiseNotGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnAttentionUpdate(void* workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
