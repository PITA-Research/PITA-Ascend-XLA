/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_GROUPED_MATMUL_H
#define OP_API_INC_GROUPED_MATMUL_H
#include "aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnGroupedMatmul calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * @param [in] x: represents x in the formula. The data type supports FLOAT16, BFLOAT16, INT8, and FLOAT32 data types. The data format supports ND. The maximum supported length is 128.
 * @param [in] weight:
 * Indicates the weight in the formula. The data type supports FLOAT16, BFLOAT16, INT8, and FLOAT32 data types. The data format supports ND. The maximum supported length is 128.
 * @param [in] biasOptional:
 * Indicates the bias in the formula. The data type supports BLOAT16, FLOAT16, FLOAT32, and INT32 data types. The data format supports ND. The maximum supported length is 128.
 * @param [in] scaleOptional: Indicates the quantization parameter, the data type supports UINT64 data type, the data format supports ND, and the maximum supported length is 128.
 * @param [in] offsetOptional: represents the quantization parameter, the data type supports FLOAT32 data type, the data format supports ND, and the maximum supported length is 128.
 * @param [in] antiquantScaleOptional:
 * Indicates pseudo quantization parameters. The data type supports FLOAT16 and BFLOAT16 data types. The data format supports ND. The maximum supported length is 128.
 * @param [in] antiquantOffsetOptional:
 * Indicates pseudo quantization parameters. The data type supports FLOAT16 and BFLOAT16 data types. The data format supports ND. The maximum supported length is 128.
 * @param [in] groupListOptional: Optional parameter, representing the index status on the input and output M-axis. The data type supports INT64, and the maximum supported length is 128.
 * @param [in] splitItem:
 * An integer parameter that represents whether the output needs to be divided into tensors. 0/1 represents the output as multiple tensors; 2/3 represents the output as a single tensor. The default value is 0.
 * @param [out] y: Indicates y in the formula. The data type supports FLOAT16, BFLOAT16, INT8, and FLOAT32 data types. The data format supports ND. The maximum supported length is 128.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnGroupedMatmulGetWorkspaceSize(
    const aclTensorList* x, const aclTensorList* weight,
    const aclTensorList* biasOptional, const aclTensorList* scaleOptional,
    const aclTensorList* offsetOptional,
    const aclTensorList* antiquantScaleOptional,
    const aclTensorList* antiquantOffsetOptional,
    const aclIntArray* groupListOptional, int64_t splitItem,
    const aclTensorList* y, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnGroupedMatmul is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnGroupedMatmulGetWorkspaceSize.
 * @param [in] executor: op executor, including the operator calculation process.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Return status code.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmul(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif