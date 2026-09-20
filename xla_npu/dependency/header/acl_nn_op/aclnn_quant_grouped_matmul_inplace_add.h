/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_QUANT_GROUPED_MATMUL_INPLACE_ADD_H
#define OP_API_INC_QUANT_GROUPED_MATMUL_INPLACE_ADD_H
#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnQuantGroupedMatmulInplaceAdd calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * @param [in] x1: represents x1 in the formula. The data type supports FLOAT8_E4M3FN, FLOAT8_E5M2, and HIFLOAT8 data types. The data format supports ND.
 * @param [in] x2:
 * Represents x2 in the formula, the data type supports FLOAT8_E4M3FN, FLOAT8_E5M2, and HIFLOAT8 data types, and the data format supports ND.
 * @param [in] scale1Optional：
 * Represents the scaling factor introduced by input x1 quantization in the quantization parameter. The data type supports FLOAT32 and FLOAT8_E8M0, and the data format supports ND.
 * @param [in] scale2: Indicates the quantization parameter, the data type supports FLOAT32 and FLOAT8_E8M0 data types, and the data format supports ND.
 * @param [in] groupList: represents the matmul size distribution of the input and output grouping axes. The data type supports INT64 and the data format supports ND.
 * @param [in/out] yRef: represents y in the formula, the data type supports FLOAT32 data type, and the data format supports ND.
 * @param [in] groupListType:
 * Integer parameter, which can take the value 0 or 1. 0 represents the cumsum result (cumulative sum) in groupList whose value is the size of the group axis.
 * 1 means that the value in groupList is the size of each group on the grouping axis.
 * @param [in] groupSize:
 * Integer parameter, used to input the quantization group size in the m, n, k directions.
 * The groupSize input consists of three values: groupSizeM, groupSizeN, and groupSizeK in three directions.
 * Each value occupies 16 bits, occupying a total of the lower 48 bits of the int64_t type groupSize (the upper 16 bits of groupSize are invalid),
 * The calculation formula is: groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32.
 * Currently only supports 0
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(
    const aclTensor *x1, const aclTensor *x2, const aclTensor *scale1Optional,
    const aclTensor *scale2, const aclTensor *groupList, aclTensor *yRef,
    int64_t groupListType, int64_t groupSize, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnQuantGroupedMatmulInplaceAdd's second interface is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: in NPU
 * The workspace size applied for on the device side is obtained by the first interface aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnQuantGroupedMatmulInplaceAdd(void *workspace, uint64_t workspaceSize,
                                  aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif