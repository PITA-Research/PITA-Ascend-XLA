/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_RENORM_H_
#define OP_API_INC_RENORM_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnRenorm calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Returns a tensor in which each sub-tensor of the input tensor self along dimension dim is normalized such that the p-norm of the sub-tensor is lower than the maxNorm value.
 * @param [in] self: aclTensor on the NPU device side. The data type supports FLOAT16, FLOAT, and the shape does not exceed 8 dimensions. Support non-continuous Tensor,
 * Supports empty Tensor input, and the data format supports ND.
 * @param [in] p: aclScalar on the host side, the data type supports FLOAT, indicating the norm, and the value is required to be greater than 0.
 * @param [in] dim:
 * The INT64 value on the host side indicates the dimension direction of the specified norm. If not specified, the default is -1, and the range is [-the number of dimensions of self, the number of dimensions of self-1].
 * @param [in] maxNorm: aclScalar on the host side. The data type supports FLOAT, indicating the maximum allowed normalized value. The required value is greater than or equal to 0.
 * @param [in] out: aclTensor on the NPU device side. The data type supports FLOAT and FLOAT16. The data type and shape are consistent with self.
 * Supports non-continuous Tensor, supports empty Tensor input, and the data format supports ND.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnRenormGetWorkspaceSize(
    const aclTensor* self, const aclScalar* p, int64_t dim,
    const aclScalar* maxNorm, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnRenorm is used to perform calculations
 *
 * Operator function: Returns a tensor in which each sub-tensor of the input tensor self along dimension dim is normalized such that the p-norm of the sub-tensor is lower than the maxNorm value.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnRenormGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnRenorm(void* workspace, uint64_t workspaceSize,
                                  aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief The first interface of aclnnInplaceRenorm calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Returns a tensor in which each sub-tensor of the input tensor selfRef along dimension dim is normalized such that the p-norm of the sub-tensor is lower than the maxNorm value.
 * @param [in] selfRef: aclTensor on the NPU device side, the data type supports FLOAT16, FLOAT, and the shape does not exceed 8 dimensions. Support non-continuous Tensor,
 * Supports empty Tensor input, and the data format supports ND.
 * @param [in] p: aclScalar on the host side, the data type supports FLOAT, indicating the norm, and the value is required to be greater than 0.
 * @param [in] dim:
 * The INT64 value on the host side indicates the dimension direction of the specified norm. If not specified, the default is -1, and the range is [-the number of dimensions of self, the number of dimensions of self-1].
 * @param [in] maxNorm: aclScalar on the host side. The data type supports FLOAT, indicating the maximum allowed normalized value. The required value is greater than or equal to 0.
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnInplaceRenormGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* p, int64_t dim,
    const aclScalar* maxNorm, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnInplaceRenorm is used to perform calculations
 *
 * Operator function: Returns a tensor in which each sub-tensor of the input tensor selfRef along dimension dim is normalized such that the p-norm of the sub-tensor is lower than the maxNorm value.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnInplaceRenormGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnInplaceRenorm(void* workspace,
                                         uint64_t workspaceSize,
                                         aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_RENORM_H_