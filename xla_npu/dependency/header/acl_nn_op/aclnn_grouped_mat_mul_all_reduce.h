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
 * \file aclnn_grouped_mat_mul_all_reduce.h
 * \brief
 */
#ifndef OP_API_INC_GROUPED_MATMUL_ALL_REDUCE_H_
#define OP_API_INC_GROUPED_MATMUL_ALL_REDUCE_H_
#include "aclnn_base.h"
#include "aclnn_util.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGroupedMatMulAllReduce calculates the workspace size according to the specific calculation process.
 * @domain aclnnop_ops_infer
 * Operator function: implement gmm + AllReduce fusion calculation
 * @param [in] x: represents x in the formula. The data type supports FLOAT16 and BFLOAT16 data types. The data format supports ND. The maximum length supported is 128.
 * @param [in] weight:
 * Indicates the weight in the formula. The data type supports FLOAT16 and BFLOAT16 data types. The data format supports ND. The maximum length supported is 128.
 * @param [in] bias: Indicates the bias in the formula. The data type supports FLOAT16 and BFLOAT16 data types. The data format supports ND. The maximum supported length is 128.
 * @param [in] groupListOptional:
 * Optional parameter, representing the matmul size distribution in the input and output M directions. The data type supports INT64 and the maximum supported length is 128.
 * @param [in] splitItemOptional:
 * Optional parameter, representing whether the input and output need to be tensor segmented, 1 means that the input needs to be segmented, and the output does not need to be segmented; 2 means that the input does not need to be segmented,
 * The output needs to be split; 3 means that both input and output need to be split.
 * @param [in] group: Optional parameter, a string identifying the column group.
 * @param [in] reduceOp: optional parameter, reduce operation type, default value: sum.
 * @param [in] commTurn: Optional parameter, communication data cut fraction, that is, total data volume/single communication volume, default value: 0.
 * @param [in] streamMode: optional parameter, enumeration of ACL stream mode, type support: 0/1.
 * @param [out] out: Indicates out in the formula. The data type supports FLOAT16 and BFLOAT16 data types. The data format supports ND. The maximum supported length is 128.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACL_DEPRECATED_MESSAGE(
    "aclnnGroupedMatMulAllReduceGetWorkspaceSize will be deprecated")
ACLNN_API aclnnStatus aclnnGroupedMatMulAllReduceGetWorkspaceSize(
    const aclTensorList* x, const aclTensorList* weight,
    const aclTensorList* bias, const aclIntArray* groupListOptional,
    int64_t splitItem, const char* group, const char* reduceOp,
    int64_t commTurn, int64_t streamMode, const aclTensorList* y,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGroupedMatMulAllReduce's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGtTensorGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
ACL_DEPRECATED_MESSAGE("aclnnGroupedMatMulAllReduce will be deprecated")
ACLNN_API aclnnStatus aclnnGroupedMatMulAllReduce(void* workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif  // OP_API_INC_GROUPED_MATMUL_ALL_REDUCE_H_