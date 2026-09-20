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
 * \file aclnn_scatter_add.h
 * \brief
 */

#ifndef OP_API_INC_SCATTER_OUT_H_
#define OP_API_INC_SCATTER_OUT_H_

#include "aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The first interface of @brief aclnnScatterAdd calculates the workspace size according to the specific calculation process.
 * @domain aclnn_ops_infer
 *
 * Operator function: Fill the values in the source tensor into the output tensor one by one according to the specified axis direction and the position relationship in the index tensor.
 * If more than one src value is filled into the same position of self, then these values will be accumulated at this position.
 * @param [in] self: aclTensor on the NPU device side, data type supports FLOAT16, FLOAT32, INT32, INT8, UINT8,
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] dim: num on the host side, the data type supports INT64.
 * @param [in] index: aclTensor on the NPU device side. The data type supports INT32 and int64 types. The number of dim reverse dimensions needs to be the same as src.
 * Supports non-continuous Tensor, and the data format supports ND.
 * @param [in] src: aclTensor on the NPU device side, data type supports FLOAT16, FLOAT32, INT32, INT8,
 * UINT8 type, the number of dim reverse dimensions needs to be the same as src. Supports non-continuous Tensor, the data format supports ND, and the data type is consistent with self.
 * @param [in] out: aclTensor on the NPU device side, data type supports FLOAT16, FLOAT32, INT32, INT8, UINT8,
 * Data type, data format, and tensor shape need to be consistent with self
 * @param [out] workspaceSize: Returns the workspace size that the user needs to apply for on the NPU device side.
 * @param [out] executor: Returns the op executor, including the operator calculation process.
 * @return aclnnStatus: return status code
 */
ACLNN_API aclnnStatus aclnnScatterAddGetWorkspaceSize(
    const aclTensor* self, int64_t dim, const aclTensor* index,
    const aclTensor* src, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief: The second interface of aclnnScatterAdd is used to perform calculations
 *
 * Operator function: Fill the values in the source tensor into the output tensor one by one according to the specified axis direction and the position relationship in the index tensor.
 * If more than one src value is filled into the same position of self, then these values will be accumulated at this position.
 * @param [in] workspace: The starting address of the workspace memory applied for on the NPU device side.
 * @param [in] workspaceSize: The workspace size applied for on the NPU device side, obtained by the first interface aclnnScatterAddGetWorkspaceSize.
 * @param [in] stream: ACL stream.
 * @param [in] executor: op executor, including the operator calculation process.
 * @return aclnnStatus: Return status code.
 */
ACLNN_API aclnnStatus aclnnScatterAdd(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor,
                                      const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SCATTER_OUT_H_