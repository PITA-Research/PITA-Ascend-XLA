/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_NPU_FORMAT_CAST_H_
#define OP_API_INC_NPU_FORMAT_CAST_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator function: realizes the conversion of ND to FRACTAL_NZ format with specified C0 size. C0 is the size of the last dimension of FRACTAL_NZ format.
 * The first interface of @brief aclnnNpuFormatCast is used to obtain the workspace size required for calculation and the executor containing the operator calculation process.
 * @domain aclnn_ops_infer
 * @param [in] srcTensor:
 * Input data, data format supports ND. Non-continuous Tensor is not supported. Supported data types: INT8, INT32, FLOAT16, BFLOAT16, FLOAT.
 * @param [in]
 * dstTensor: converted target data, data format supports ACL_FORMAT_FRACTAL_NZ(29), ACL_FORMAT_FRACTAL_NZ_C0_16(50), ACL_FORMAT_FRACTAL_NZ_C0_32(51). Data types support INT8, INT32, FLOAT16, BFLOAT16, and FLOAT.
 * @param [in] workspaceSize: The size of the worksapce that needs to be applied for on the Device side.
 * @param [in] executor: The op executor containing the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(
    const aclTensor* srcTensor, aclTensor* dstTensor, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief The second interface of aclnnNpuFormatCast is used to perform calculations.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspace_size: The workspace size applied for on the NPU device side, obtained by the first interface aclnnNpuFormatCastGetWorkspaceSize.
 * @param [in] exector: op executor, including the operator calculation process.
 * @param [in] stream: Specifies the AscendCL Stream to execute the task.
 * @return aclnnStatus: return status code
 */

ACLNN_API aclnnStatus aclnnNpuFormatCast(void* workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         aclrtStream stream);

/**
 * @brief Shape and data format used to calculate the conversion target data
 * @param [in] srcTensor: Input data, the data format supports ND. Supported data types: INT8, INT32, FLOAT16, BFLOAT16, FLOAT.
 * @param [in] dstFormat: The target data format for conversion. Supported data format: ACL_FORMAT_FRACTAL_NZ(29).
 * @param [in] additionalDtype:
 * The enumeration value of the data type used when converting to NZ format to infer the size of C0. Supported data types: ACL_FLOAT16(1), ACL_BF16(27), INT8(2), ACL_FLOAT(0).
 * @param [in] dstShape: Pointer to the target array Shape array. The memory pointed to is released by the caller in non-error scenarios.
 * @param [in] dstShapeSize: Pointer to the size of the target data Shape array.
 * @param [in] actualFormat:
 * Pointer to the data format of the target data. Only ACL_FORMAT_FRACTAL_NZ(29), ACL_FORMAT_FRACTAL_NZ_C0_16(50), ACL_FORMAT_FRACTAL_NZ_C0_32(51) are supported.
 * @return aclnnStatus: return status code
 */

ACLNN_API aclnnStatus aclnnNpuFormatCastCalculateSizeAndFormat(
    const aclTensor* srcTensor, const int dstFormat, int additionalDtype,
    int64_t** dstShape, uint64_t* dstShapeSize, int* actualFormat);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_NPU_FORMAT_CAST_H_