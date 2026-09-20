/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_ADD_V2_H
#define OP_API_INC_GROUPED_MATMUL_ADD_V2_H
#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnGroupedMatmulAddV2 calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * @param [in] x: represents x in the formula, the data type supports FLOAT16 and BFLOAT16 data types, and the data format supports ND.
 * @param [in] weight:
 * Indicates the weight in the formula. The data type supports FLOAT16 and BFLOAT16 data types, and the data format supports ND.
 * @param [in] groupList: represents the matmul size distribution of the input and output grouping axes. The data type supports INT64 and the data format supports ND.
 * @param [in/out] yRef: represents y in the formula, the data type supports FLOAT32 data type, and the data format supports ND.
 * @param [in] transposeX:
 * bool type, which can only take the value true, indicating the transpose of the x matrix.
 * @param [in] transposeWeight:
 * bool type, which can only take the value false, indicating that the weight matrix is not transposed.
 * @param [in] groupType:
 * Integer parameter, representing the axis that needs to be divided. It can only take the value 2, which means dividing the k-axis.
 * @param [in] groupListType:
 * Integer parameter, which can take the value 0 or 1. 0 represents the cumsum result (cumulative sum) in groupList whose value is the size of the group axis.
 * 1 means that the value in groupList is the size of each group on the grouping axis.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnGroupedMatmulAddV2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, const aclTensor *groupList,
    aclTensor *yRef, bool transposeX, bool transposeWeight, int64_t groupType,
    int64_t groupListType, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnGroupedMatmulAddV2 is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulAddV2(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif