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
 * \file aclnn_gather.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_GATHER_H_
#define OP_API_INC_LEVEL2_ACLNN_GATHER_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGather calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 * Operator function: perform data aggregation on the dimensions specified in the input tensor
 * Calculation formula: $$ gather(X,index,d)_{i_0,i_1,\cdots,i_{d-1},i_{d+1},\cdots,i_{n-1}} =
 * X_{i_0,i_1,\cdots,i_{d-1},index_{i_d},i_{d+1},\cdots,i_{n-1}} $$
 *
 * @param [in] self: input parameter to be calculated for gather, aclTensor on the NPU device side,
 * Data types support FLOAT16, FLOAT32, INT32, INT64, INT8, UINT8, DOUBLE, UINT16, UINT32, UINT64, BOOL, and data formats support ND,
 * Supports non-contiguous Tensors.
 * @param [in] dim: int64 on the host side.
 * @param [in] index: input parameter to be calculated for gather, aclTensor on the NPU device side,
 * The data type supports INT32 and INT64 data format supports ND.
 * Supports non-contiguous Tensors.
 * @param [in] out: The output parameters calculated by gather. aclTensor on the NPU device side,
 * The data type is the same as self, and the data format supports ND.
 * Supports non-contiguous Tensors.
 * @param [out] workspace_size: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGatherGetWorkspaceSize(
    const aclTensor* self, const int64_t dim, const aclTensor* index,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * The second interface of @brief aclnnGather is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGatherGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnGather(void* workspace, uint64_t workspaceSize,
                                  aclOpExecutor* executor,
                                  const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_GATHER_H_